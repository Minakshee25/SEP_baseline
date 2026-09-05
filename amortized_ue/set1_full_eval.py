"""SET-1 FULL EVAL — the original 4 small/7-8B-tier target LLMs (Llama-2-7b-chat,
Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct, deepseek-llm-7b-chat): per-model INDIVIDUAL
5-arm proxy vs own-target ridge vs SEP-single vs SEP-multi, ID (fresh trivia n1000) + OOD (squad
n1000). Analog of e72_bigtier_individual.py / e73_settwo_individual.py, one tier down, generalized
from the single `z` arm to all 5 (z, z_q, z_q_resp, q_only, q_resp_only).

Fixed 2-position layers (NOT re-tuned here; = exp2_run.BEST_TBG / E36 CV picks + brief's SLT picks):
    Llama-2:  TBG:30 SLT:13   Mistral: TBG:31 SLT:6
    Llama-3:  TBG:31 SLT:11   DeepSeek: TBG:28 SLT:16

z input = concat([TBG:L_tbg, SLT:L_slt]) raw hidden states, 2 positions stacked, per model.

Methods:
  proxy : frozen Llama-3.2-3B + LoRA (exp2_run.train_arm recipe: batch 8 x grad_accum 4 = eff 32,
          projector 1024, k=4, 10 epochs), 3 seeds x 5 arms, per model.
  ridge : StandardScaler + Ridge on the SAME concat([TBG,SLT]) (linear_ceiling_probe.fit_probe).
  SEP-single : correctness_eval.sep_single_fixed_layer at the model's fixed TBG layer.
  SEP-multi  : correctness_eval.sep_5layer_concat (paper's top-5-layer concat; position+layers
               chosen by FIT-VAL AUROC only, never touches eval).

Eval: ID = fresh trivia_qa n1000 (disjoint ids from the n2000 train set, verified 0 overlap).
      OOD = squad n1000. true 10-sample SE reported as a sampling reference (AUROC only; its own
      Spearman is trivially 1).

Resumable: --stage train skips a (model, arm) whose 3 checkpoints already exist.
Two-GPU split: pass --models to restrict training to a subset (run two processes, one per GPU,
CUDA_VISIBLE_DEVICES pinned by the caller). Curves are written ONE FILE PER (model, arm) so
parallel processes never race on a shared JSON (the E72 bug, fixed the E73 way).

    python -m amortized_ue.set1_full_eval --stage train --models Llama-2-7b-chat Mistral-7B-Instruct-v0.2
    python -m amortized_ue.set1_full_eval --stage eval
    python -m amortized_ue.set1_full_eval --stage push_wandb
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
from amortized_ue.correctness_eval import (load_accuracy, paired_bootstrap_auc, ci,
                                            sep_single_fixed_layer, sep_5layer_concat)
from amortized_ue.stage2.data import best_split, binarize_entropy

DATA2 = "/data2/mn1025/stage1"
MODELS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct",
          "deepseek-llm-7b-chat"]
SHORT = {"Llama-2-7b-chat": "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "DeepSeek"}
LAYERS = {"Llama-2-7b-chat":          {"TBG": 30, "SLT": 13},
          "Mistral-7B-Instruct-v0.2": {"TBG": 31, "SLT": 6},
          "Meta-Llama-3-8B-Instruct": {"TBG": 31, "SLT": 11},
          "deepseek-llm-7b-chat":     {"TBG": 28, "SLT": 16}}
POSITIONS = ["TBG", "SLT"]
SEEDS = [0, 1, 2]
ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(_HERE, "stage2", "runs", "SET1_full_eval", "checkpoints")
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "set1_full_eval.json")
CSV_PATH = os.path.join(RESULTS_DIR, "set1_full_eval.csv")
BUNDLES_DIR = os.path.join(RESULTS_DIR, "set1_full_eval_bundles")
CURVES_DIR = os.path.join(RESULTS_DIR, "set1_train_curves")   # one file per (model, arm) -> no race


def s1cfg(model, ds, n):
    return Stage1Config(model_name=model, dataset=ds, num_samples=n, output_dir=DATA2,
                        run_name=f"{model}_{ds}_n{n}_full")


def build_z(model, ds, n):
    hid, y, ids = load_matrix(s1cfg(model, ds, n), POSITIONS)
    L = LAYERS[model]
    Z = np.concatenate([hid["TBG"][L["TBG"]].astype(np.float64),
                        hid["SLT"][L["SLT"]].astype(np.float64)], axis=1)
    return Z, y.astype(np.float64), ids, hid


def load_qr(model, ds, n, ids):
    recs = load_records(s1cfg(model, ds, n))
    q = [recs[i]["question"] for i in ids]
    r = [recs[i]["canonical"]["response"] for i in ids]
    return q, r


def model_ckpt_dir(model):
    return os.path.join(CKPT_ROOT, model)


def curve_path(model, arm):
    return os.path.join(CURVES_DIR, f"{model}__{arm}.json")


# ------------------------------------------------------------------------------------ train ---
def do_train(seeds, batch_size, grad_accum, models=None, arms=None):
    models = models or MODELS
    arms = arms or ARMS
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
        print(f"\n{'='*80}\n[{model}]  layers {LAYERS[model]}\n{'='*80}")
        Z, y, ids, _ = build_z(model, "trivia_qa", 2000)
        assert len(ids) == 2000, f"{model}: {len(ids)} != 2000"
        q, r = load_qr(model, "trivia_qa", 2000, ids)
        tr, va, te = splits(len(ids))
        scaler = StandardScaler().fit(Z[tr])
        Zs = scaler.transform(Z).astype(np.float32)
        mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
        ys = ((y - mu) / sd).astype(np.float32)

        def pack(rows):
            return {"z": Zs[rows], "y": ys[rows],
                    "q": [q[i] for i in rows], "r": [r[i] for i in rows]}

        train, val = pack(tr), pack(va)
        h_in = Z.shape[1]

        os.makedirs(ck, exist_ok=True)
        with open(os.path.join(ck, "z_bundle.pkl"), "wb") as f:
            pickle.dump({"model": model, "layers": LAYERS[model], "h_in": h_in,
                         "feat_scaler": scaler, "y_mu": mu, "y_sd": sd,
                         "split_seed": 42, "n_train": int(len(tr)), "n_val": int(len(va))}, f)

        for arm in arms:
            n_done = len(glob.glob(os.path.join(ck, f"*{arm}_seed*.pt")))
            if n_done >= len(seeds):
                print(f"  [{arm}] {n_done} checkpoints present -> skip")
                continue
            print(f"\n  --- [{model}] arm={arm} ---")
            if proxy is None or proxy.h_in != h_in:
                proxy = ProxyModel(cfg, h_in=h_in).to("cuda" if torch.cuda.is_available() else "cpu")
            res = train_arm(train, val, val, arm, seeds, cfg, proxy, torch, nn,
                            get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                            ckpt_dir=ck, tag=f"ind-{model}-{arm}")
            print(f"    val Spearman per seed: {[round(s, 3) for s in res['te_spearman']]}")
            with open(curve_path(model, arm), "w") as f:      # per-(model,arm) -> no cross-process race
                json.dump({"model": model, "arm": arm, "layers": LAYERS[model], "seeds": list(seeds),
                           "val_spearman_by_seed": res["te_spearman"],
                           "curves_by_seed": res["curves_by_seed"],
                           "y_mu": mu, "y_sd": sd, "train_config": cfg.as_dict()}, f)
            print(f"    curves -> {curve_path(model, arm)}")


# ------------------------------------------------------------------------------------- eval ---
def proxy_predict_per_seed(model, arm, Z_eval_raw, q_eval, r_eval):
    """Returns list of per-seed prediction arrays (one per checkpoint/seed)."""
    from amortized_ue.stage2.checkpoint import load_checkpoint
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z
    ck = model_ckpt_dir(model)
    bundle = pickle.load(open(os.path.join(ck, "z_bundle.pkl"), "rb"))
    Zs = bundle["feat_scaler"].transform(Z_eval_raw).astype(np.float32)
    zt = torch.from_numpy(Zs).float().unsqueeze(1)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    paths = sorted(glob.glob(os.path.join(ck, f"*{arm}_seed*.pt")))
    assert paths, f"no checkpoints for {model}/{arm} in {ck}"
    uses_z = _arm_uses_z(arm)
    pm = None
    per_seed = []
    for p in paths:
        pm, meta, _ = load_checkpoint(p, model=pm)
        pm.eval()
        out = []
        with torch.no_grad():
            for i in range(0, len(Zs), 64):
                rows = list(range(i, min(i + 64, len(Zs))))
                z = zt[rows].to(dev) if uses_z else None
                ids, attn = _tokenize_arm(pm.tokenizer, [q_eval[j] for j in rows],
                                          [r_eval[j] for j in rows], arm, 256)
                if ids is not None:
                    ids, attn = ids.to(dev), attn.to(dev)
                out.append(pm(z, ids, attn).float().cpu())
        per_seed.append(torch.cat(out).numpy())
    return per_seed


def proxy_predict(model, arm, Z_eval_raw, q_eval, r_eval):
    """Seed-ensemble mean prediction (used as the scored predictor)."""
    per_seed = proxy_predict_per_seed(model, arm, Z_eval_raw, q_eval, r_eval)
    return np.mean(per_seed, axis=0), per_seed


def own_ridge(model, hid_fit, y_fit, tr, va, hid_eval):
    L = LAYERS[model]
    Zf = np.concatenate([hid_fit["TBG"][L["TBG"]].astype(np.float64),
                         hid_fit["SLT"][L["SLT"]].astype(np.float64)], axis=1)
    Ze = np.concatenate([hid_eval["TBG"][L["TBG"]].astype(np.float64),
                         hid_eval["SLT"][L["SLT"]].astype(np.float64)], axis=1)
    rm, rsc, ralpha, _ = fit_probe(Zf, y_fit, tr, va)
    return rm.predict(rsc.transform(Ze)), float(ralpha)


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
        metrics[k] = {"auroc_incorrect": au, "auroc_binarised_se": ase,
                      "spearman_se": (1.0 if k == "true_semantic_entropy" else rho(s, se))}

    boot = paired_bootstrap_auc(P, incorrect, B=bootstrap)
    deltas = {}
    keys = list(P.keys())
    for a in keys:
        for b in keys:
            if a >= b:
                continue
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


def do_eval(bootstrap, models=None):
    models = models or MODELS
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(BUNDLES_DIR, exist_ok=True)
    out = {}
    if os.path.exists(OUT_PATH):
        out = json.load(open(OUT_PATH))
    out.setdefault("experiment", "SET1_full_eval")
    out.setdefault("models", MODELS)
    out.setdefault("layers", LAYERS)
    out.setdefault("arms", ARMS)
    out.setdefault("folds", {})

    for model in models:
        print(f"\n{'#'*90}\n# SET-1 — {model}  (layers {LAYERS[model]})\n{'#'*90}")
        # own-model fit data (n2000 train split) for ridge + SEP
        Zf, yf, idsf, hidf = build_z(model, "trivia_qa", 2000)
        trf, vaf, tef = splits(len(idsf))

        fold = {"layers": LAYERS[model]}
        bundle_extra = {}
        for split, tag, ds in [("id", "ID trivia n1000", "trivia_qa"), ("ood", "OOD squad n1000", "squad")]:
            Ze, ye, idse, hide = build_z(model, ds, 1000)
            qe, re_ = load_qr(model, ds, 1000, idse)
            cfg_e = s1cfg(model, ds, 1000)
            recs = load_records(cfg_e)
            se_by_id = {i: float(recs[i]["labels"]["cluster_assignment_entropy"]) for i in sorted(recs)}
            acc_by_id = load_accuracy(cfg_e)

            pbi = {}
            seed_stats = {}
            for arm in ARMS:
                mean_pred, per_seed = proxy_predict(model, arm, Ze, qe, re_)
                pbi[arm] = dict(zip(idse, mean_pred))
                # per-seed Spearman/AUROC_incorrect for mean+-std reporting (ensemble mean is what's scored)
                se_arr = np.array([se_by_id[i] for i in idse], dtype=float)
                inc_arr = np.array([acc_by_id[i] < 0.5 for i in idse], dtype=int)
                sp_seed, au_seed = [], []
                for s in per_seed:
                    sp_seed.append(rho(s, se_arr))
                    au_seed.append(float(roc_auc_score(inc_arr, s)) if len(np.unique(inc_arr)) == 2 else float("nan"))
                seed_stats[arm] = {"spearman_by_seed": [float(x) for x in sp_seed],
                                   "auroc_incorrect_by_seed": [float(x) for x in au_seed],
                                   "spearman_mean": float(np.mean(sp_seed)), "spearman_std": float(np.std(sp_seed)),
                                   "auroc_incorrect_mean": float(np.nanmean(au_seed)),
                                   "auroc_incorrect_std": float(np.nanstd(au_seed))}

            ridge_pred, ralpha = own_ridge(model, hidf, yf, trf, vaf, hide)
            pbi["ridge"] = dict(zip(idse, ridge_pred))

            eval_rows = np.arange(len(idse))
            sep1_p, sep1_auc, sep1_layer, thr1, ybe1 = sep_single_fixed_layer(
                hidf, yf, trf, vaf, hide, ye, eval_rows, "TBG", LAYERS[model]["TBG"])
            pbi["sep_single"] = dict(zip(idse, sep1_p))

            sep5_p, sep5_auc, sep5_layers = sep_5layer_concat(
                hidf, yf, trf, vaf, hide, ye, eval_rows)
            pbi["sep_multi"] = dict(zip(idse, sep5_p))

            fold[split] = score(model, tag, pbi, se_by_id, acc_by_id, bootstrap)
            fold[split]["proxy_seed_stats"] = seed_stats
            fold[split]["ridge_alpha"] = ralpha
            fold[split]["sep_single_layer"] = list(sep1_layer)
            fold[split]["sep_multi_layers"] = [sep5_layers[0], list(sep5_layers[1])]
            bundle_extra[split] = {"ralpha": ralpha, "sep_single_layer": sep1_layer,
                                   "sep_multi_layers": sep5_layers}

        out["folds"][model] = fold
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        with open(os.path.join(BUNDLES_DIR, f"{model}_bundle.pkl"), "wb") as f:
            pickle.dump({"model": model, "layers": LAYERS[model], "extra": bundle_extra}, f)
        print(f"  -> saved {len(out['folds'])} model(s) to {OUT_PATH}")

    write_csv_and_summary(out)


def write_csv_and_summary(out):
    cols = ARMS + ["ridge", "sep_single", "sep_multi", "true_semantic_entropy"]
    rows = []
    for model in MODELS:
        if model not in out["folds"]:
            continue
        for split, tag in [("id", "ID"), ("ood", "OOD")]:
            fold = out["folds"][model][split]
            ss = fold.get("proxy_seed_stats", {})
            for c in cols:
                m = fold["metrics"][c]
                if c in ss:
                    row = {"model": model, "method": c, "split": tag,
                          "spearman": ss[c]["spearman_mean"], "spearman_std": ss[c]["spearman_std"],
                          "auroc_incorrect": ss[c]["auroc_incorrect_mean"],
                          "auroc_incorrect_std": ss[c]["auroc_incorrect_std"]}
                else:
                    row = {"model": model, "method": c, "split": tag,
                          "spearman": m["spearman_se"], "spearman_std": "",
                          "auroc_incorrect": m["auroc_incorrect"], "auroc_incorrect_std": ""}
                rows.append(row)
    import csv
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["model", "method", "split", "spearman", "spearman_std",
                                          "auroc_incorrect", "auroc_incorrect_std"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"\nwrote {CSV_PATH}")

    # summary means (across models actually present) per method/split
    summary = {}
    for split in ["id", "ood"]:
        for c in cols:
            vals = [out["folds"][m][split]["metrics"][c]["auroc_incorrect"]
                    for m in MODELS if m in out["folds"]]
            if vals:
                summary.setdefault(split, {})[c] = float(np.nanmean(vals))
    out["_summary_auroc_incorrect"] = summary
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"summary AUROC_incorrect: {json.dumps(summary, indent=1)}")


WANDB_ARTIFACT = "set1_full_eval_ckpts"


def do_push_wandb():
    import wandb
    paths = sorted(glob.glob(os.path.join(CKPT_ROOT, "*", "*.pt"))) + \
        sorted(glob.glob(os.path.join(CKPT_ROOT, "*", "z_bundle.pkl"))) + \
        sorted(glob.glob(os.path.join(BUNDLES_DIR, "*.pkl")))
    expected = len(MODELS) * len(ARMS) * len(SEEDS)
    print(f"found {len(paths)} files (expected >= {expected + len(MODELS)*2} incl. bundles)")
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=WANDB_ARTIFACT, job_type="checkpoint",
                     config={"experiment": "SET1_full_eval", "models": MODELS, "layers": LAYERS,
                             "arms": ARMS, "seeds": SEEDS})
    art = wandb.Artifact(WANDB_ARTIFACT, type="model",
                         metadata={"models": MODELS, "arms": ARMS, "n_seeds": len(SEEDS)})
    art.add_dir(CKPT_ROOT)
    if os.path.isdir(BUNDLES_DIR):
        art.add_dir(BUNDLES_DIR, name="eval_bundles")
    run.log_artifact(art)
    if os.path.exists(OUT_PATH):
        res_art = wandb.Artifact(f"{WANDB_ARTIFACT}_results", type="results")
        res_art.add_file(OUT_PATH)
        if os.path.exists(CSV_PATH):
            res_art.add_file(CSV_PATH)
        run.log_artifact(res_art)
    run.finish()
    a = wandb.Api().artifact(f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{WANDB_ARTIFACT}:latest")
    print(f"pushed + verified {WANDB_ARTIFACT}:{a.version}  size={a.size} bytes  n_files={len(list(a.files()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--models", nargs="+", default=None, choices=MODELS)
    p.add_argument("--arms", nargs="+", default=None, choices=ARMS)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    a = p.parse_args()
    if a.stage == "push_wandb":
        do_push_wandb()
        return
    if a.stage in ("train", "all"):
        do_train(a.seeds, a.batch_size, a.grad_accum, a.models, a.arms)
    if a.stage in ("eval", "all"):
        do_eval(a.bootstrap, a.models)


if __name__ == "__main__":
    main()
