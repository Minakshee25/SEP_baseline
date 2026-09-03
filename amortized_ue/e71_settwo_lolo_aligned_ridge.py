"""E71 — the E70 comparison for the small-tier Qwen/Gemma "set 2".

Set 2 = `Qwen3-8B`, `Qwen3.5-9B`, `gemma-7b-it`, `gemma-2-9b-it` (the E44–E54 line). Neither a clean
leave-one-of-4-out `q_resp_only` proxy NOR the aligned pooled ridge had ever been run for this set
(E45 flagged the missing z-arm; E45/E51 used the deploy proxy with the target in its own pool).
E71 builds BOTH and evaluates proxy + aligned_ridge + fuse + SEP + true 10-sample SE, ID + OOD —
the exact protocol of E70 (big tier), one tier down.

  --stage train : 4-fold LOLO `q_resp_only` proxy, frozen Llama-3.2-3B + LoRA, E37/E53/E63/E65 recipe
                  (3 seeds, batch 8 × grad_accum 4 = eff 32, projector 1024, k=4, 10 epochs). Trains
                  by patching `e65_bigtier_lolo`'s module constants and reusing its `do_train`
                  (load_pool + train_arm) verbatim — the sacred recipe path is not re-implemented.
                  Env: amortized_stage2 + a free GPU.
  --stage eval  : aligned_ridge (PCA(512)→orthogonal Procrustes into a Qwen3-8B anchor frame,
                  label-free, exactly E70's `ModelAligner`) pooled LOLO; proxy preds via `arm_preds`
                  on the E71 checkpoints; SEP via leak-free val-selected layer on the held-out
                  model's own trivia n2000; true SE + `incorrect` from the records. Scored on
                  trivia n1000 (ID) and squad n1000 (OOD, E49 builds), 10k paired bootstrap.
                  Env: amortized_stage2 (arm_preds needs the proxy stack) + GPU for the proxy pass.
  --stage all   : train then eval.

Data-gen conventions: Qwen 8B/9B use the E55 `_nothink` builds (same as E70's big-tier Qwen),
gemma-7b/9b use `_full`. All 4 share question ids exactly on trivia n2000 / trivia n1000 / squad
n1000 (asserted); trivia n2000 ∩ {n1000, squad} = 0 (leak-free). All records on `/data2`.
"""
from __future__ import annotations

import os
import json
import glob
import pickle
import argparse

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.correctness_eval import (
    load_accuracy, sep_single_val_selected, paired_bootstrap_auc, ci)
from amortized_ue.stage2.data import best_split, binarize_entropy

DATA2 = "/data2/mn1025/stage1"
SET2 = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
SUFFIX2 = {"Qwen3-8B": "nothink", "Qwen3.5-9B": "nothink",
           "gemma-7b-it": "full", "gemma-2-9b-it": "full"}
ANCHOR = "Qwen3-8B"
PCA_DIM = 512
POSITIONS = ["TBG", "SLT"]
TRAIN_N = 2000
SEEDS = [0, 1, 2]
ARM = "q_resp_only"

_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_ROOT = os.path.join(_HERE, "stage2", "runs", "E71_settwo_lolo_qresp", "checkpoints")
ALIGN_CKPT_DIR = os.path.join(_HERE, "stage2", "runs", "E71_settwo_aligned_ridge", "checkpoints")
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_PATH = os.path.join(RESULTS_DIR, "e71_settwo_lolo_aligned_ridge.json")
CURVES = os.path.join(RESULTS_DIR, "e71_settwo_lolo_train_curves.json")


def run_name(m, ds, n):
    return f"{m}_{ds}_n{n}_{SUFFIX2[m]}"


def s1cfg(m, ds, n, data_dir):
    return Stage1Config(model_name=m, dataset=ds, num_samples=n,
                        output_dir=data_dir, run_name=run_name(m, ds, n))


def fold_ckpt_dir(held):
    return os.path.join(CKPT_ROOT, held)


# --------------------------------------------------------------------------------- data readiness ---
def do_check(data_dir):
    ok = True
    for ds, n in [("trivia_qa", TRAIN_N), ("trivia_qa", 1000), ("squad", 1000)]:
        S = []
        for m in SET2:
            man = s1cfg(m, ds, n, data_dir).manifest_path()
            k = len(json.load(open(man))["records"]) if os.path.isfile(man) else 0
            S.append((m, k))
        good = all(k >= n for _, k in S)
        ok &= good
        print(f"  {ds:10s} n{n:<5d} " + "  ".join(f"{m}:{k}" for m, k in S) + f"   {'OK' if good else 'MISSING'}")
    return ok


# ------------------------------------------------------------------------------------- training ---
def do_train(data_dir, seeds, batch_size, grad_accum):
    """Reuse e65_bigtier_lolo.do_train verbatim by patching its module constants to set 2.
    e65's run_name/s1cfg/fold_ckpt_dir/load_pool read these globals dynamically at call time."""
    import amortized_ue.e65_bigtier_lolo as E65
    E65.BIGTIER = list(SET2)
    E65.SUFFIX = dict(SUFFIX2)
    E65.DEFAULT_CKPT_ROOT = CKPT_ROOT
    E65.RESULTS_DIR = RESULTS_DIR
    E65.OUT_CURVES = CURVES
    os.makedirs(CKPT_ROOT, exist_ok=True)
    E65.do_train(data_dir, seeds, batch_size, grad_accum)


# ----------------------------------------------------------------------- per-model alignment (E70) ---
class ModelAligner:
    """Verbatim from e70_bigtier_lolo_aligned_ridge.ModelAligner, with set-2 run-name resolution.
    PCA (per-model, fit on trivia-n2000 train) -> orthogonal Procrustes -> ANCHOR PCA frame. Label-free."""

    def __init__(self, model, data_dir, pca_dim):
        self.model = model
        self.pca_dim = pca_dim
        hid, y, ids = load_matrix(s1cfg(model, "trivia_qa", 2000, data_dir), POSITIONS)
        self.ids, self.y = ids, y
        tr, va, te = splits(len(ids))
        self.tr, self.va = tr, va
        self.layer, self.pca, self.center_src = {}, {}, {}
        for pos in POSITIONS:
            best = (-np.inf, None)
            for L in range(hid[pos].shape[0]):
                _, _, _, val_s = fit_probe(hid[pos][L], y.astype(float), tr, va)
                if val_s > best[0]:
                    best = (val_s, L)
            L = int(best[1])
            self.layer[pos] = L
            X = hid[pos][L].astype(np.float64)
            p = PCA(n_components=pca_dim, random_state=0).fit(X[tr])
            self.pca[pos] = p
            self.center_src[pos] = p.transform(X[tr]).mean(0, keepdims=True)
        self._hid_train = {pos: hid[pos][self.layer[pos]].astype(np.float64) for pos in POSITIONS}
        print(f"    {model:16s} layers TBG:{self.layer['TBG']} SLT:{self.layer['SLT']} (val-selected)")

    def pca_scores_train(self, pos):
        return self.pca[pos].transform(self._hid_train[pos][self.tr]) - self.center_src[pos]

    def free_train_hidden(self):
        self._hid_train = None

    def fit_W(self, anchor_scores_by_pos):
        self.W, self.center_anchor = {}, {}
        for pos in POSITIONS:
            src = self.pca_scores_train(pos)
            tgt = anchor_scores_by_pos[pos]
            self.center_anchor[pos] = tgt.mean(0, keepdims=True)
            self.W[pos] = (np.eye(self.pca_dim) if self.model == ANCHOR
                           else orthogonal_procrustes(src, tgt - self.center_anchor[pos])[0])

    def aligned_z(self, data_dir, dataset, n):
        hid, y, ids = load_matrix(s1cfg(self.model, dataset, n, data_dir), POSITIONS)
        parts = []
        for pos in POSITIONS:
            X = hid[pos][self.layer[pos]].astype(np.float64)
            s = self.pca[pos].transform(X) - self.center_src[pos]
            parts.append(s @ self.W[pos] + self.center_anchor[pos])
        return np.concatenate(parts, axis=1), y, ids


def ecdf(vals):
    s = np.sort(np.asarray(vals, dtype=float))
    return lambda x: np.searchsorted(s, np.asarray(x, dtype=float), side="right") / len(s)


def per_model_standardize(Z, ref_rows):
    mu = Z[ref_rows].mean(0, keepdims=True)
    sd = Z[ref_rows].std(0, keepdims=True) + 1e-8
    return (Z - mu) / sd


# ------------------------------------------------------------------------------------------ eval ---
def score_split(tag, held, ar_by_id, proxy_by_id, sep_by_id, se_by_id, acc_by_id, bootstrap):
    ids = sorted(i for i in se_by_id if i in ar_by_id and i in proxy_by_id and i in sep_by_id)
    ar = np.array([ar_by_id[i] for i in ids])
    proxy = np.array([proxy_by_id[i] for i in ids])
    sep = np.array([sep_by_id[i] for i in ids])
    se = np.array([se_by_id[i] for i in ids], dtype=float)
    incorrect = np.array([acc_by_id[i] < 0.5 for i in ids], dtype=int)

    fa, fp = ecdf(ar), ecdf(proxy)
    fuse = 0.5 * (fa(ar) + fp(proxy))

    preds = {"aligned_ridge": ar, "q_resp_only": proxy, "fuse": fuse,
             "true_semantic_entropy": se, "sep_single_val_selected": sep}
    thr = best_split(torch.tensor(se))
    yb = binarize_entropy(torch.tensor(se), thr).numpy()
    v = yb >= 0
    metrics = {}
    for k, s in preds.items():
        au_inc = float(roc_auc_score(incorrect, s)) if len(np.unique(incorrect)) == 2 else float("nan")
        au_se = float(roc_auc_score(yb[v], s[v])) if len(np.unique(yb[v])) == 2 else float("nan")
        metrics[k] = {"auroc_incorrect": au_inc, "auroc_binarised_se": au_se, "spearman_se": rho(s, se)}

    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    deltas = {}
    for a, b in [("aligned_ridge", "q_resp_only"), ("aligned_ridge", "true_semantic_entropy"),
                 ("aligned_ridge", "sep_single_val_selected"), ("q_resp_only", "sep_single_val_selected"),
                 ("q_resp_only", "true_semantic_entropy"), ("fuse", "q_resp_only"),
                 ("fuse", "true_semantic_entropy")]:
        c = ci(boot[a] - boot[b])
        deltas[f"{a}_minus_{b}"] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}

    print(f"\n  [{tag}] {held}  n={len(ids)}  incorrect_rate={incorrect.mean():.3f}")
    print(f"    {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}")
    for k in preds:
        m = metrics[k]
        print(f"    {k:26s}{m['auroc_incorrect']:>11.3f}{m['auroc_binarised_se']:>10.3f}{m['spearman_se']:>9.3f}")
    for kk, dd in deltas.items():
        print(f"    Δ {kk:44s} {dd['mean']:+.3f} [{dd['lo95']:+.3f}, {dd['hi95']:+.3f}]"
              f"{'  *' if dd['ci_excludes_zero'] else ''}")
    return {"n": len(ids), "ids": ids, "incorrect_rate": float(incorrect.mean()),
            "best_split": float(thr), "metrics": metrics, "bootstrap_deltas": deltas,
            "aligned_ridge_pred": [float(x) for x in ar], "proxy_pred": [float(x) for x in proxy],
            "sep_pred": [float(x) for x in sep], "true_se": [float(x) for x in se],
            "incorrect": [int(x) for x in incorrect]}


def do_eval(data_dir, pca_dim, bootstrap):
    from amortized_ue.procrustes_e27_rank_fusion import arm_preds

    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(ALIGN_CKPT_DIR, exist_ok=True)

    print(f"E71 eval — anchor={ANCHOR} pca_dim={pca_dim}\nbuilding per-model aligners...")
    aligners = {m: ModelAligner(m, data_dir, pca_dim) for m in SET2}
    anc = aligners[ANCHOR]
    assert all(a.ids == anc.ids for a in aligners.values()), "trivia n2000 ids differ across set 2"
    anchor_scores = {pos: anc.pca_scores_train(pos) for pos in POSITIONS}
    for a in aligners.values():
        a.fit_W(anchor_scores)
    for m, a in aligners.items():
        with open(os.path.join(ALIGN_CKPT_DIR, f"aligner_{m}.pkl"), "wb") as f:
            pickle.dump({"model": m, "anchor": ANCHOR, "pca_dim": pca_dim, "layers": a.layer,
                         "pca": a.pca, "W": a.W, "center_src": a.center_src,
                         "center_anchor": a.center_anchor}, f)
    for a in aligners.values():
        a.free_train_hidden()

    print("\nextracting aligned z (trivia n2000 / trivia n1000 / squad n1000)...")
    Z = {}
    for m, a in aligners.items():
        z2, y2, i2 = a.aligned_z(data_dir, "trivia_qa", 2000)
        z1, _, i1 = a.aligned_z(data_dir, "trivia_qa", 1000)
        zs, _, isq = a.aligned_z(data_dir, "squad", 1000)
        Z[m] = {"tr2k": (z2, y2, i2), "id": (z1, i1), "ood": (zs, isq)}

    out = {"experiment": "E71", "set": SET2, "anchor": ANCHOR, "pca_dim": pca_dim,
           "layers": {m: aligners[m].layer for m in SET2}, "folds": {}}

    for held in SET2:
        sources = [m for m in SET2 if m != held]
        print(f"\n{'#'*92}\n# E71 fold — held out {held}  (pooled from {sources})\n{'#'*92}")

        # pooled aligned-ridge (per-model feature-std + per-model SE z-score on own train rows)
        Xtr, Xva, Ytr, Yva = [], [], [], []
        for s in sources:
            z, y, _ = Z[s]["tr2k"]
            tr, va = aligners[s].tr, aligners[s].va
            zt = per_model_standardize(z, tr)
            mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
            Xtr.append(zt[tr]); Ytr.append((y[tr] - mu) / sd)
            Xva.append(zt[va]); Yva.append((y[va] - mu) / sd)
        Xall = np.concatenate(Xtr + Xva); Yall = np.concatenate(Ytr + Yva)
        n_tr = sum(len(a) for a in Xtr)
        trs, vas = np.arange(n_tr), np.arange(n_tr, len(Xall))
        ridge, scaler, alpha, val_s = fit_probe(Xall, Yall, trs, vas)
        print(f"  pooled ridge: {n_tr} tr / {len(Xall) - n_tr} va, alpha={alpha}, val Spearman={val_s:.3f}")

        z2k, _, _ = Z[held]["tr2k"]
        hmu = z2k[aligners[held].tr].mean(0, keepdims=True)
        hsd = z2k[aligners[held].tr].std(0, keepdims=True) + 1e-8
        with open(os.path.join(ALIGN_CKPT_DIR, f"fold_{held}.pkl"), "wb") as f:
            pickle.dump({"held_out": held, "sources": sources, "anchor": ANCHOR, "pca_dim": pca_dim,
                         "alpha": float(alpha), "pooled_ridge": ridge, "pooled_scaler": scaler,
                         "held_feat_mu": hmu, "held_feat_sd": hsd,
                         "pooled_val_spearman": float(val_s)}, f)

        ck = fold_ckpt_dir(held)
        assert len(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt"))) >= len(SEEDS), \
            f"{held}: missing proxy checkpoints (run --stage train)"

        fold = {"sources": sources, "alpha": float(alpha), "pooled_val_spearman": float(val_s)}
        for split_key, tag, dataset, n in [("id", "ID trivia n1000", "trivia_qa", 1000),
                                           ("ood", "OOD squad n1000", "squad", 1000)]:
            z_split, ids_split = Z[held][split_key]
            ar = dict(zip(ids_split, ridge.predict(scaler.transform((z_split - hmu) / hsd))))

            cfg_h = s1cfg(held, dataset, n, data_dir)
            recs = load_records(cfg_h)
            eids = sorted(recs.keys())
            se_by_id = {i: float(recs[i]["labels"]["cluster_assignment_entropy"]) for i in eids}
            acc_by_id = load_accuracy(cfg_h)

            proxy_by_id = arm_preds(ARM, held, dataset, n, ckpt_dir=ck, data_dir=data_dir,
                                    run_name=run_name(held, dataset, n))

            # SEP: fit on held's own trivia n2000 tr/va, predict on this split's rows
            fit_cfg = s1cfg(held, "trivia_qa", TRAIN_N, data_dir)
            hf, yf, idf = load_matrix(fit_cfg, POSITIONS)
            trf, vaf, tef = splits(len(idf))
            he, ye_mat, ide = load_matrix(cfg_h, POSITIONS)
            assert ide == eids
            sep_p, _, sep_choice, _, _, _ = sep_single_val_selected(
                hf, yf, trf, vaf, he, ye_mat, np.arange(len(eids)))
            sep_by_id = dict(zip(eids, sep_p))
            del hf, he

            fold[split_key] = score_split(tag, held, ar, proxy_by_id, sep_by_id,
                                          se_by_id, acc_by_id, bootstrap)
            fold[split_key]["sep_choice"] = list(sep_choice)
        out["folds"][held] = fold
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> saved {len(out['folds'])} fold(s) to {OUT_PATH}")

    # ---- summary ----
    print("\n" + "=" * 100 + "\nE71 SUMMARY — mean over the 4 held-out set-2 models\n" + "=" * 100)
    for sk, tag in [("id", "ID trivia n1000"), ("ood", "OOD squad n1000")]:
        preds = list(out["folds"][SET2[0]][sk]["metrics"])
        print(f"\n[{tag}]  {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}")
        summ = {}
        for k in preds:
            ai = float(np.nanmean([out["folds"][h][sk]["metrics"][k]["auroc_incorrect"] for h in SET2]))
            ae = float(np.nanmean([out["folds"][h][sk]["metrics"][k]["auroc_binarised_se"] for h in SET2]))
            rs = float(np.nanmean([out["folds"][h][sk]["metrics"][k]["spearman_se"] for h in SET2]))
            summ[k] = {"mean_auroc_incorrect": ai, "mean_auroc_se": ae, "mean_spearman_se": rs}
            print(f"           {k:26s}{ai:>11.3f}{ae:>10.3f}{rs:>9.3f}")
        out["folds"].setdefault("_summary", {})[sk] = summ
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_PATH}\naligners + fold ridges -> {ALIGN_CKPT_DIR}")


def do_push_wandb():
    """Push both checkpoint sets as W&B artifacts and verify by fetch (E65/E70 pattern)."""
    import wandb
    ent = os.environ["WANDB_ENT"]
    specs = [
        ("e71_settwo_lolo_qresp_ckpts", CKPT_ROOT, "model",
         {"contents": "4 folds x 3 seeds held-<held>_q_resp_only_seedN.pt (leave-one-of-4-out)"}),
        ("e71_settwo_aligned_ridge_bundles", ALIGN_CKPT_DIR, "model",
         {"contents": "4 aligner_<model>.pkl (PCA+W+centers+layers) + 4 fold_<held>.pkl "
                      "(pooled ridge+scaler+held feat stats)"}),
    ]
    for name, path, typ, meta in specs:
        n_files = sum(len(fs) for _, _, fs in os.walk(path))
        run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                         name=name, job_type="checkpoint",
                         config={"experiment": "E71", "set": SET2, "anchor": ANCHOR,
                                 "pca_dim": PCA_DIM, "recipe": "q_resp_only, 3 seeds, batch 8 x "
                                 "grad_accum 4, proj 1024, k=4, 10 epochs"})
        art = wandb.Artifact(name, type=typ, metadata={"set": SET2, **meta})
        art.add_dir(path)
        run.log_artifact(art)
        run.finish()
        a = wandb.Api().artifact(f"{ent}/amortized_ue_stage2/{name}:latest")
        print(f"pushed + verified {name}:{a.version}  size={a.size} bytes  n_files={len(list(a.files()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["check", "train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--data_dir", default=DATA2)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--pca_dim", type=int, default=PCA_DIM)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    if args.stage == "check":
        raise SystemExit(0 if do_check(args.data_dir) else 1)
    if args.stage == "push_wandb":
        do_push_wandb()
        return
    if not do_check(args.data_dir):
        raise SystemExit("STOP: set-2 data not all ready.")
    if args.stage in ("train", "all"):
        do_train(args.data_dir, args.seeds, args.batch_size, args.grad_accum)
    if args.stage in ("eval", "all"):
        do_eval(args.data_dir, args.pca_dim, args.bootstrap)


if __name__ == "__main__":
    main()
