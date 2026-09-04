"""E73 — the E72 experiment for the small-tier Qwen/Gemma "set 2" (per-model INDIVIDUAL proxy/ridge/SEP).

E71 measured cross-model transfer for set 2 (LOLO `q_resp_only` proxy + pooled aligned ridge). This
is the per-model supervised complement — the exact analog of E72 (big tier), one tier down. For EACH
of the 4 small-tier models (`Qwen3-8B`, `Qwen3.5-9B`, `gemma-7b-it`, `gemma-2-9b-it`), train that
model's OWN proxy / ridge / SEP on its OWN trivia n2000, evaluate ID (trivia n1000) + OOD (squad
n1000). All three methods read the IDENTICAL input per model:

    z input = concat([ TBG:L_tbg , SLT:L_slt ])   (raw hidden states, 2 positions stacked)

with (L_tbg, L_slt) = the per-position val-Spearman layers E71's `aligned_ridge` (its `ModelAligner`,
via `fit_probe`) already selected, reused verbatim so E73 sits on the same layers as E71:

    Qwen3-8B TBG:34 SLT:23 | Qwen3.5-9B TBG:31 SLT:31 | gemma-7b-it TBG:27 SLT:18 | gemma-2-9b-it TBG:41 SLT:28

Methods (identical to E72):
  proxy : frozen Llama-3.2-3B + LoRA, `z` arm (hidden-state-in, no text), trained per-model via
          `exp2_run.train_arm` (E37/E53/E65 recipe: batch 8 x grad_accum 4 = eff 32, projector 1024,
          k=4, 10 epochs), 3 seeds, h_in = 2H.
  ridge : own-model ridge on the same concat([TBG,SLT]) (`fit_probe`: StandardScaler + Ridge, alpha on val).
  SEP   : own-model logistic on best_split-binarised SE, same concat([TBG,SLT]), FIXED layers.

Reference columns joined per-id from E71: true 10-sample SE, cross-model `aligned_ridge`, cross-model
LOLO `q_resp_only`. Metrics: Spearman + AUROC_incorrect. 10k paired bootstrap on AUROC deltas.

  --stage train : GPU, ~40 min if split across 2 GPUs via --models. Resumable (skips a done model).
  --stage eval  : GPU for the proxy forward pass; ridge/SEP/table CPU.
  --stage all / push_wandb.

Env: amortized_stage2. Data on /data2 (Qwen 8B/9B -> _nothink, gemma 7b/9b -> _full).
Training curves are written ONE FILE PER MODEL (e73_train_curves/<model>.json) so parallel --models
processes never race on a shared read-modify-write JSON (the E72 bug).
"""
from __future__ import annotations

import os
import json
import glob
import pickle
import argparse

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.correctness_eval import load_accuracy, paired_bootstrap_auc, ci
from amortized_ue.stage2.data import best_split, binarize_entropy

DATA2 = "/data2/mn1025/stage1"
SET = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
SUFFIX = {"Qwen3-8B": "nothink", "Qwen3.5-9B": "nothink",
          "gemma-7b-it": "full", "gemma-2-9b-it": "full"}
LAYERS = {"Qwen3-8B": {"TBG": 34, "SLT": 23}, "Qwen3.5-9B": {"TBG": 31, "SLT": 31},
          "gemma-7b-it": {"TBG": 27, "SLT": 18}, "gemma-2-9b-it": {"TBG": 41, "SLT": 28}}
POSITIONS = ["TBG", "SLT"]
SEEDS = [0, 1, 2]
ARM = "z"

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(_HERE, "stage2", "runs", "E73_settwo_individual", "checkpoints")
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "e73_settwo_individual.json")
CURVES_DIR = os.path.join(RESULTS_DIR, "e73_train_curves")   # one file per model -> no cross-process race
E71_JSON = os.path.join(RESULTS_DIR, "e71_settwo_lolo_aligned_ridge.json")


def s1cfg(model, ds, n):
    return Stage1Config(model_name=model, dataset=ds, num_samples=n, output_dir=DATA2,
                        run_name=f"{model}_{ds}_n{n}_{SUFFIX[model]}")


def build_z(model, ds, n):
    hid, y, ids = load_matrix(s1cfg(model, ds, n), POSITIONS)
    L = LAYERS[model]
    Z = np.concatenate([hid["TBG"][L["TBG"]].astype(np.float64),
                        hid["SLT"][L["SLT"]].astype(np.float64)], axis=1)
    return Z, y.astype(np.float64), ids


def model_ckpt_dir(model):
    return os.path.join(CKPT_ROOT, model)


# ------------------------------------------------------------------------------------ train ---
def do_train(seeds, batch_size, grad_accum, models=None):
    models = models or SET
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z
    from amortized_ue.exp2_run import train_arm

    os.makedirs(CKPT_ROOT, exist_ok=True)
    os.makedirs(CURVES_DIR, exist_ok=True)
    cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4, epochs=10,
                       batch_size=batch_size, grad_accum=grad_accum)
    proxy = None

    for model in models:
        ck = model_ckpt_dir(model)
        if len(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt"))) >= len(seeds):
            print(f"[{model}] {len(seeds)} checkpoints present -> skip")
            continue
        print(f"\n{'='*80}\n[{model}] individual z-arm proxy  (layers {LAYERS[model]})\n{'='*80}")
        Z, y, ids = build_z(model, "trivia_qa", 2000)
        assert len(ids) == 2000, f"{model}: {len(ids)} != 2000"
        tr, va, te = splits(len(ids))
        scaler = StandardScaler().fit(Z[tr])
        Zs = scaler.transform(Z).astype(np.float32)
        mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
        ys = ((y - mu) / sd).astype(np.float32)

        def pack(rows):
            return {"z": Zs[rows], "y": ys[rows], "q": [""] * len(rows), "r": [""] * len(rows)}

        train, val = pack(tr), pack(va)
        h_in = Z.shape[1]
        if proxy is None or proxy.h_in != h_in:
            proxy = ProxyModel(cfg, h_in=h_in).to("cuda" if torch.cuda.is_available() else "cpu")
        os.makedirs(ck, exist_ok=True)
        res = train_arm(train, val, val, ARM, seeds, cfg, proxy, torch, nn,
                        get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                        ckpt_dir=ck, tag=f"ind-{model}")
        print(f"  val Spearman per seed: {[round(s, 3) for s in res['te_spearman']]}")

        with open(os.path.join(ck, "z_bundle.pkl"), "wb") as f:
            pickle.dump({"model": model, "layers": LAYERS[model], "h_in": h_in,
                         "feat_scaler": scaler, "y_mu": mu, "y_sd": sd,
                         "split_seed": 42, "n_train": int(len(tr)), "n_val": int(len(va))}, f)
        with open(os.path.join(CURVES_DIR, f"{model}.json"), "w") as f:      # per-model -> no race
            json.dump({"model": model, "layers": LAYERS[model], "seeds": list(seeds),
                       "val_spearman_by_seed": res["te_spearman"],
                       "curves_by_seed": res["curves_by_seed"],
                       "y_mu": mu, "y_sd": sd, "train_config": cfg.as_dict()}, f)
        print(f"  curves -> {CURVES_DIR}/{model}.json")


# ------------------------------------------------------------------------------------- eval ---
def proxy_predict(model, Z_eval_raw):
    from amortized_ue.stage2.checkpoint import load_checkpoint
    ck = model_ckpt_dir(model)
    bundle = pickle.load(open(os.path.join(ck, "z_bundle.pkl"), "rb"))
    Zs = bundle["feat_scaler"].transform(Z_eval_raw).astype(np.float32)
    zt = torch.from_numpy(Zs).float().unsqueeze(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt")))
    acc, pm = np.zeros(len(Zs)), None
    for p in paths:
        pm, meta, _ = load_checkpoint(p, model=pm)
        pm.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Zs), 64):
                out.append(pm(zt[i:i + 64].to(dev), None, None).float().cpu())
        acc += torch.cat(out).numpy()
    return acc / len(paths)


def own_ridge_sep(model, Z_eval_raw):
    Zf, yf, _ = build_z(model, "trivia_qa", 2000)
    tr, va, te = splits(len(yf))
    rm, rsc, ralpha, _ = fit_probe(Zf, yf, tr, va)
    ridge_pred = rm.predict(rsc.transform(Z_eval_raw))
    thr = best_split(torch.tensor(yf[tr]))
    ybf = binarize_entropy(torch.tensor(yf), thr).numpy()
    m = ybf[tr] >= 0
    ssc = StandardScaler().fit(Zf[tr][m])
    clf = LogisticRegression(max_iter=1000).fit(ssc.transform(Zf[tr][m]), ybf[tr][m])
    sep_pred = clf.predict_proba(ssc.transform(Z_eval_raw))[:, 1]
    return ridge_pred, float(ralpha), sep_pred, float(thr)


def e71_ref(model, split, key):
    """{id: pred} from E71's per-fold JSON. key in {aligned_ridge_pred, proxy_pred}."""
    f = json.load(open(E71_JSON))["folds"][model][split]
    return dict(zip(f["ids"], f[key]))


def score(model, tag, preds_by_id, se_by_id, acc_by_id, bootstrap):
    ids = sorted(set.intersection(*[set(p) for p in preds_by_id.values()], set(se_by_id)))
    se = np.array([se_by_id[i] for i in ids], dtype=float)
    incorrect = np.array([acc_by_id[i] < 0.5 for i in ids], dtype=int)
    P = {k: np.array([v[i] for i in ids], dtype=float) for k, v in preds_by_id.items()}
    P["true_semantic_entropy"] = se

    thr = best_split(torch.tensor(se))
    yb = binarize_entropy(torch.tensor(se), thr).numpy()
    vv = yb >= 0
    metrics = {}
    for k, s in P.items():
        au = float(roc_auc_score(incorrect, s)) if len(np.unique(incorrect)) == 2 else float("nan")
        ase = float(roc_auc_score(yb[vv], s[vv])) if len(np.unique(yb[vv])) == 2 else float("nan")
        metrics[k] = {"auroc_incorrect": au, "auroc_binarised_se": ase, "spearman_se": rho(s, se)}

    boot = paired_bootstrap_auc(P, incorrect, B=bootstrap)
    deltas = {}
    for a, b in [("proxy", "ridge"), ("proxy", "sep"), ("proxy", "true_semantic_entropy"),
                 ("ridge", "sep"), ("ridge", "true_semantic_entropy"),
                 ("proxy", "aligned_ridge_xmodel"), ("proxy", "lolo_qresp_xmodel")]:
        if a in boot and b in boot:
            c = ci(boot[a] - boot[b])
            deltas[f"{a}_minus_{b}"] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}

    print(f"\n  [{tag}] {model}  n={len(ids)}  incorrect_rate={incorrect.mean():.3f}")
    print(f"    {'predictor':22s}{'AUROC_inc':>11s}{'rho_SE':>9s}")
    for k in P:
        print(f"    {k:22s}{metrics[k]['auroc_incorrect']:>11.3f}{metrics[k]['spearman_se']:>9.3f}")
    return {"n": len(ids), "incorrect_rate": float(incorrect.mean()), "best_split": float(thr),
            "metrics": metrics, "bootstrap_deltas": deltas,
            "preds": {k: [float(x) for x in v] for k, v in P.items()}, "ids": ids}


def do_eval(bootstrap):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {"experiment": "E73", "set": SET, "layers": LAYERS, "folds": {}}
    for model in SET:
        print(f"\n{'#'*90}\n# E73 — {model}  (individual, layers {LAYERS[model]})\n{'#'*90}")
        fold = {"layers": LAYERS[model]}
        for split, tag, ds in [("id", "ID trivia n1000", "trivia_qa"), ("ood", "OOD squad n1000", "squad")]:
            Ze, ye, ids = build_z(model, ds, 1000)
            cfg_e = s1cfg(model, ds, 1000)
            recs = load_records(cfg_e)
            se_by_id = {i: float(recs[i]["labels"]["cluster_assignment_entropy"]) for i in sorted(recs)}
            acc_by_id = load_accuracy(cfg_e)

            proxy_pred = proxy_predict(model, Ze)
            ridge_pred, ralpha, sep_pred, _ = own_ridge_sep(model, Ze)
            pbi = {
                "proxy": dict(zip(ids, proxy_pred)),
                "ridge": dict(zip(ids, ridge_pred)),
                "sep": dict(zip(ids, sep_pred)),
                "aligned_ridge_xmodel": e71_ref(model, split, "aligned_ridge_pred"),
                "lolo_qresp_xmodel": e71_ref(model, split, "proxy_pred"),
            }
            fold[split] = score(model, tag, pbi, se_by_id, acc_by_id, bootstrap)
            fold[split]["ridge_alpha"] = ralpha
        out["folds"][model] = fold
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> saved {len(out['folds'])} model(s) to {OUT_PATH}")

    cols = ["proxy", "ridge", "sep", "aligned_ridge_xmodel", "lolo_qresp_xmodel", "true_semantic_entropy"]
    short = {"proxy": "proxy", "ridge": "ridge", "sep": "SEP", "aligned_ridge_xmodel": "aln_ridge(x)",
             "lolo_qresp_xmodel": "LOLO_qresp(x)", "true_semantic_entropy": "trueSE"}
    for split, tag in [("id", "ID  (trivia n1000)"), ("ood", "OOD (squad n1000)")]:
        for metric, ml in [("spearman_se", "SPEARMAN vs SE"), ("auroc_incorrect", "AUROC_incorrect")]:
            print(f"\n===== {tag} — {ml} =====")
            print(f"{'model':16s}" + "".join(f"{short[c]:>15s}" for c in cols))
            for m in SET:
                row = [out["folds"][m][split]["metrics"][c][metric] for c in cols]
                print(f"{m:16s}" + "".join(f"{x:>15.3f}" for x in row))
            means = [float(np.nanmean([out["folds"][m][split]["metrics"][c][metric] for m in SET])) for c in cols]
            print(f"{'MEAN':16s}" + "".join(f"{x:>15.3f}" for x in means))
            out["folds"].setdefault("_summary", {})[f"{split}_{metric}"] = dict(zip(cols, means))
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_PATH}")


WANDB_ARTIFACT = "e73_settwo_individual_ckpts"


def do_push_wandb():
    import wandb
    paths = sorted(glob.glob(os.path.join(CKPT_ROOT, "*", "*.pt"))) + \
        sorted(glob.glob(os.path.join(CKPT_ROOT, "*", "z_bundle.pkl")))
    assert len(paths) == 4 * len(SET), f"expected {4*len(SET)} files, found {len(paths)}"
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=WANDB_ARTIFACT, job_type="checkpoint",
                     config={"experiment": "E73", "set": SET, "layers": LAYERS, "arm": ARM,
                             "design": "per-model individual z-arm proxy, set 2 (2-position, E71 layers)"})
    art = wandb.Artifact(WANDB_ARTIFACT, type="model",
                         metadata={"set": SET, "layers": LAYERS, "n_seeds": len(SEEDS),
                                   "contents": "12 ind-<model>_z_seed{0,1,2}.pt + 4 z_bundle.pkl"})
    art.add_dir(CKPT_ROOT)
    run.log_artifact(art)
    run.finish()
    a = wandb.Api().artifact(f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{WANDB_ARTIFACT}:latest")
    print(f"pushed + verified {WANDB_ARTIFACT}:{a.version}  size={a.size} bytes  n_files={len(list(a.files()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--models", nargs="+", default=None, choices=SET,
                   help="restrict training to a subset (split across GPUs); eval always does all 4")
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    if a.stage == "push_wandb":
        do_push_wandb()
        return
    if a.stage in ("train", "all"):
        do_train(a.seeds, a.batch_size, a.grad_accum, a.models)
    if a.stage in ("eval", "all"):
        do_eval(a.bootstrap)


if __name__ == "__main__":
    main()
