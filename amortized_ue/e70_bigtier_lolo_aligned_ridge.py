"""E70 — big-tier (5×27B) leave-one-LLM-out ALIGNED-RIDGE, label-free, ID + OOD.

E65/E69 ran only the text arm (`q_resp_only`) LOLO over the 5 big-tier targets. This adds the
missing E37-style **aligned pooled ridge**: for held-out model h, pool the OTHER 4 models' hidden
states — each rotated label-free into a common frame — plus per-model z-scored SE labels, fit ONE
ridge, and read h's aligned hidden state through it. No target SE labels, no target sampling.

**Dimension wall + fix.** `exp2_run.py` uses square `orthogonal_procrustes`, which needs every model
at the anchor's hidden dim (true for the original-4 @ 4096). The big-tier dims differ
(Qwen3.5/3.6/3.8-27B 5120, gemma-2-27b-it 4608, gemma-3-27b-it 5376). Fix (user-chosen): per-model
**PCA to a common dim** (`--pca_dim`, default 512; fit on that model's trivia-n2000 train split),
then orthogonal Procrustes between PCA scores on the shared anchor questions. Anchor frame (user-
chosen): **Qwen3.5-27B** (`--anchor`). E34 established the shared uncertainty direction lives in the
top ~50–100 PCs, so 512 is generous.

**Leak-free by construction.** All 5 big-tier models share question ids exactly on trivia n2000,
trivia n1000 and squad n1000 (asserted); trivia n2000 ∩ n1000 = 0. PCA + Procrustes W_m are fit on
each model's trivia **n2000 train split**; the pooled ridge trains on trivia n2000 (4 sources);
the held-out model is scored on its trivia **n1000** (ID) and **squad n1000** (OOD), both disjoint
from n2000 — W_h is applied there, never fit there. h's SE labels are never used.

**Baselines** are reloaded per-id from the E65 (ID) and E69 (OOD) result JSONs — no recompute of the
`q_resp_only` proxy, true 10-sample SE, SEP, or `incorrect`. Adds `fuse` = label-free rank-fusion
(empirical-CDF average, CDF fit on pooled-train preds) of aligned-ridge ⊕ q_resp_only (the E37 arm).

Env: `amortized_stage2` (or `se_probes`) — CPU only, sklearn PCA + Ridge, no GPU, no proxy stack.
    python -m amortized_ue.e70_bigtier_lolo_aligned_ridge
"""
from __future__ import annotations

import os
import json
import argparse

import pickle

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.correctness_eval import (
    load_accuracy, paired_bootstrap_auc, ci)
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.e69_bigtier_lolo_squad_ood import run_name_ds, s1cfg_ds, BIGTIER

DATA2 = "/data2/mn1025/stage1"
ANCHOR = "Qwen3.5-27B"
PCA_DIM = 512
POSITIONS = ["TBG", "SLT"]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS_DIR, "e70_bigtier_lolo_aligned_ridge.json")
CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "stage2", "runs", "E70_bigtier_lolo_aligned_ridge", "checkpoints")
E65_JSON = os.path.join(RESULTS_DIR, "e65_bigtier_lolo_n1000.json")
E69_JSON = os.path.join(RESULTS_DIR, "e69_bigtier_lolo_squad_ood.json")


def ecdf(train_vals):
    """Empirical-CDF transform (value -> normalized rank), fit on the given values. Label-free.
    Verbatim from procrustes_e27_rank_fusion.ecdf; inlined so this script needs no proxy stack."""
    s = np.sort(np.asarray(train_vals, dtype=float))
    return lambda x: np.searchsorted(s, np.asarray(x, dtype=float), side="right") / len(s)


# --------------------------------------------------------------------------- per-model alignment ---
class ModelAligner:
    """PCA (per-model, fit on trivia-n2000 train) -> orthogonal Procrustes -> ANCHOR PCA frame.
    Label-free. One (PCA, W, centers) pair per position."""

    def __init__(self, model, data_dir, pca_dim):
        self.model = model
        self.pca_dim = pca_dim
        cfg = s1cfg_ds(model, "trivia_qa", 2000, data_dir)
        hid, y, ids = load_matrix(cfg, POSITIONS)          # hid[pos]: [L+1, 2000, H]
        self.ids = ids
        self.y = y
        tr, va, te = splits(len(ids))
        self.tr, self.va = tr, va
        # per-position best layer by val Spearman of a raw ridge (leak-free)
        self.layer = {}
        self.pca = {}
        self.center_src = {}
        for pos in POSITIONS:
            best = (-np.inf, None)
            for L in range(hid[pos].shape[0]):
                _, _, _, val_s = fit_probe(hid[pos][L], y.astype(float), tr, va)
                if val_s > best[0]:
                    best = (val_s, L)
            L = best[1]
            self.layer[pos] = int(L)
            X = hid[pos][L].astype(np.float64)             # [2000, H]
            p = PCA(n_components=pca_dim, random_state=0).fit(X[tr])
            self.pca[pos] = p
            self.center_src[pos] = p.transform(X[tr]).mean(0, keepdims=True)
        self._hid_train = {pos: hid[pos][self.layer[pos]].astype(np.float64) for pos in POSITIONS}
        print(f"    {model:16s} layers TBG:{self.layer['TBG']} SLT:{self.layer['SLT']} "
              f"(val-selected)  n={len(ids)}")

    def pca_scores_train(self, pos):
        """Centered PCA scores on this model's trivia-n2000 TRAIN rows (for fitting W against anchor)."""
        return self.pca[pos].transform(self._hid_train[pos][self.tr]) - self.center_src[pos]

    def free_train_hidden(self):
        self._hid_train = None

    def fit_W(self, anchor_scores_by_pos):
        """anchor_scores_by_pos[pos] = ANCHOR centered PCA train scores on the SAME shared rows."""
        self.W = {}
        self.center_anchor = {}
        for pos in POSITIONS:
            src = self.pca_scores_train(pos)               # [n_tr, pca_dim]
            tgt = anchor_scores_by_pos[pos]
            self.center_anchor[pos] = tgt.mean(0, keepdims=True)  # ~0 (already centered) but keep explicit
            if self.model == ANCHOR:
                self.W[pos] = np.eye(self.pca_dim)
            else:
                W, _ = orthogonal_procrustes(src, tgt - self.center_anchor[pos])
                self.W[pos] = W

    def aligned_z(self, data_dir, dataset, n):
        """Aligned 2·pca_dim feature matrix for this model's <dataset> n<n> records, id-sorted."""
        cfg = s1cfg_ds(self.model, dataset, n, data_dir)
        hid, y, ids = load_matrix(cfg, POSITIONS)
        parts = []
        for pos in POSITIONS:
            X = hid[pos][self.layer[pos]].astype(np.float64)
            s = self.pca[pos].transform(X) - self.center_src[pos]
            parts.append(s @ self.W[pos] + self.center_anchor[pos])
        return np.concatenate(parts, axis=1), y, ids


# ------------------------------------------------------------------------------- pooled LOLO ridge ---
def per_model_standardize(Z, ref_rows):
    mu = Z[ref_rows].mean(0, keepdims=True)
    sd = Z[ref_rows].std(0, keepdims=True) + 1e-8
    return (Z - mu) / sd


def load_baseline_block(js_path, held):
    with open(js_path) as f:
        d = json.load(f)
    b = d[held]
    return dict(zip(b["te_ids"], zip(b["proxy_te_preds"], b["true_se_te"],
                                    b["sep_te_preds"], b["incorrect_te"])))


def score_split(name_split, held, aligned_pred_by_id, base_by_id, se_by_id, bootstrap):
    ids = [i for i in base_by_id if i in aligned_pred_by_id]
    ids.sort()
    ar = np.array([aligned_pred_by_id[i] for i in ids])
    proxy = np.array([base_by_id[i][0] for i in ids])
    true_se = np.array([base_by_id[i][1] for i in ids])
    sep = np.array([base_by_id[i][2] for i in ids])
    incorrect = np.array([base_by_id[i][3] for i in ids], dtype=int)
    se = np.array([se_by_id[i] for i in ids], dtype=float)

    # label-free rank fusion of aligned-ridge + proxy (CDF fit on ALL rows of this split — no labels)
    fa, fp = ecdf(ar), ecdf(proxy)
    fuse = 0.5 * (fa(ar) + fp(proxy))

    preds = {"aligned_ridge": ar, "q_resp_only": proxy, "fuse": fuse,
             "true_semantic_entropy": true_se, "sep_single_val_selected": sep}
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
                 ("aligned_ridge", "sep_single_val_selected"), ("fuse", "q_resp_only"),
                 ("fuse", "true_semantic_entropy")]:
        c = ci(boot[a] - boot[b])
        deltas[f"{a}_minus_{b}"] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}

    print(f"\n  [{name_split}] {held}  n={len(ids)}  incorrect_rate={incorrect.mean():.3f}")
    print(f"    {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}")
    for k in preds:
        m = metrics[k]
        print(f"    {k:26s}{m['auroc_incorrect']:>11.3f}{m['auroc_binarised_se']:>10.3f}{m['spearman_se']:>9.3f}")
    for kk, dd in deltas.items():
        print(f"    Δ {kk:42s} {dd['mean']:+.3f} [{dd['lo95']:+.3f}, {dd['hi95']:+.3f}]"
              f"{'  *' if dd['ci_excludes_zero'] else ''}")
    return {"n": len(ids), "ids": ids, "incorrect_rate": float(incorrect.mean()),
            "best_split": float(thr), "metrics": metrics, "bootstrap_deltas": deltas,
            "aligned_ridge_pred": [float(x) for x in ar]}


def se_map(data_dir, model, dataset, n):
    recs = load_records(s1cfg_ds(model, dataset, n, data_dir))
    return {i: float(recs[i]["labels"]["cluster_assignment_entropy"]) for i in recs}


WANDB_ARTIFACT = "e70_bigtier_lolo_aligned_ridge_bundles"


def do_push_wandb():
    import glob
    import wandb
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, "*.pkl")))
    assert len(paths) == 2 * len(BIGTIER), f"expected {2 * len(BIGTIER)} pkls, found {len(paths)}"
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=WANDB_ARTIFACT, job_type="checkpoint",
                     config={"experiment": "E70", "anchor": ANCHOR, "pca_dim": PCA_DIM,
                             "design": "big-tier 5x27B LOLO aligned-ridge, label-free, PCA->Procrustes"})
    art = wandb.Artifact(WANDB_ARTIFACT, type="model",
                         metadata={"bigtier": BIGTIER, "anchor": ANCHOR, "pca_dim": PCA_DIM,
                                   "contents": "5 aligner_<model>.pkl (PCA+W+centers+layers) + "
                                               "5 fold_<held>.pkl (pooled ridge+scaler+held feat stats)"})
    art.add_dir(CKPT_DIR)
    run.log_artifact(art)
    run.finish()
    a = wandb.Api().artifact(
        f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{WANDB_ARTIFACT}:latest")
    print(f"pushed + verified {WANDB_ARTIFACT}:{a.version}  size={a.size} bytes  "
          f"n_files={len(list(a.files()))}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--push_wandb", action="store_true", help="push saved bundles as a W&B artifact and exit")
    p.add_argument("--data_dir", default=DATA2)
    p.add_argument("--anchor", default=ANCHOR, choices=BIGTIER)
    p.add_argument("--pca_dim", type=int, default=PCA_DIM)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()
    if args.push_wandb:
        do_push_wandb()
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print(f"E70 — big-tier LOLO aligned-ridge | anchor={args.anchor} | pca_dim={args.pca_dim}")
    print("building per-model aligners (PCA on trivia-n2000 train, val-selected layers)...")
    aligners = {m: ModelAligner(m, args.data_dir, args.pca_dim) for m in BIGTIER}

    # shared-row id order is identical across models (asserted) -> anchor train PCA scores align 1:1
    anc = aligners[args.anchor]
    assert all(a.ids == anc.ids for a in aligners.values()), "trivia n2000 ids differ across big-tier"
    anchor_scores = {pos: anc.pca_scores_train(pos) for pos in POSITIONS}
    for a in aligners.values():
        a.fit_W(anchor_scores)

    # ---- SAVE per-model alignment bundle (label-free; PCA + Procrustes W + centers + layers) ----
    os.makedirs(CKPT_DIR, exist_ok=True)
    for m, a in aligners.items():
        with open(os.path.join(CKPT_DIR, f"aligner_{m}.pkl"), "wb") as f:
            pickle.dump({"model": m, "anchor": args.anchor, "pca_dim": args.pca_dim,
                         "layers": a.layer, "pca": a.pca, "W": a.W,
                         "center_src": a.center_src, "center_anchor": a.center_anchor}, f)
    for a in aligners.values():
        a.free_train_hidden()

    # cache each model's aligned z for the 3 splits it can appear in
    print("\nextracting aligned z (trivia n2000 / trivia n1000 / squad n1000)...")
    Z = {}
    for m, a in aligners.items():
        z2k, y2k, id2k = a.aligned_z(args.data_dir, "trivia_qa", 2000)
        z1k, _, id1k = a.aligned_z(args.data_dir, "trivia_qa", 1000)
        zsq, _, idsq = a.aligned_z(args.data_dir, "squad", 1000)
        Z[m] = {"tr2k": (z2k, y2k, id2k), "id": (z1k, id1k), "ood": (zsq, idsq)}
        print(f"    {m:16s} z-dim {z2k.shape[1]}")

    out = {"experiment": "E70", "anchor": args.anchor, "pca_dim": args.pca_dim,
           "layers": {m: aligners[m].layer for m in BIGTIER}, "folds": {}}

    for held in BIGTIER:
        sources = [m for m in BIGTIER if m != held]
        print(f"\n{'#'*92}\n# E70 fold — held out {held}  (aligned ridge pooled from {sources})\n{'#'*92}")

        # pool sources' trivia n2000: per-model feature standardize + per-model SE z-score, on each
        # model's own train rows; alpha picked on pooled val.
        Xtr, Xva, Ytr, Yva = [], [], [], []
        for s in sources:
            z, y, _ = Z[s]["tr2k"]
            tr, va = aligners[s].tr, aligners[s].va
            zt = per_model_standardize(z, tr)
            mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
            Xtr.append(zt[tr]); Ytr.append((y[tr] - mu) / sd)
            Xva.append(zt[va]); Yva.append((y[va] - mu) / sd)
        Xtr, Ytr = np.concatenate(Xtr), np.concatenate(Ytr)
        Xva, Yva = np.concatenate(Xva), np.concatenate(Yva)
        Xall = np.concatenate([Xtr, Xva]); Yall = np.concatenate([Ytr, Yva])
        trs = np.arange(len(Xtr)); vas = np.arange(len(Xtr), len(Xall))
        ridge, scaler, alpha, val_s = fit_probe(Xall, Yall, trs, vas)
        print(f"  pooled ridge: {len(Xtr)} tr / {len(Xva)} va rows, alpha={alpha}, val Spearman={val_s:.3f}")

        # held-out feature standardisation stats on ITS OWN trivia-n2000 train rows (label-free)
        _z2k, _, _ = Z[held]["tr2k"]
        held_mu = _z2k[aligners[held].tr].mean(0, keepdims=True)
        held_sd = _z2k[aligners[held].tr].std(0, keepdims=True) + 1e-8

        def predict_held(split_key, dataset, n):
            z, ids = Z[held][split_key]
            pred = ridge.predict(scaler.transform((z - held_mu) / held_sd))
            return dict(zip(ids, pred))

        with open(os.path.join(CKPT_DIR, f"fold_{held}.pkl"), "wb") as f:
            pickle.dump({"held_out": held, "sources": sources, "anchor": args.anchor,
                         "pca_dim": args.pca_dim, "alpha": float(alpha),
                         "pooled_ridge": ridge, "pooled_scaler": scaler,
                         "held_feat_mu": held_mu, "held_feat_sd": held_sd,
                         "pooled_val_spearman": float(val_s)}, f)

        fold = {"sources": sources, "alpha": float(alpha), "pooled_val_spearman": float(val_s)}
        for split_key, tag, dataset, n, base_json in [
                ("id", "ID trivia n1000", "trivia_qa", 1000, E65_JSON),
                ("ood", "OOD squad n1000", "squad", 1000, E69_JSON)]:
            base = load_baseline_block(base_json, held)
            se_by_id = se_map(args.data_dir, held, dataset, n)
            ap = predict_held(split_key, dataset, n)
            fold[split_key] = score_split(tag, held, ap, base, se_by_id, args.bootstrap)
        out["folds"][held] = fold
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> saved {len(out['folds'])} fold(s) to {OUT_PATH}")

    # ---- summary ----
    print("\n" + "=" * 100)
    print("E70 SUMMARY — mean over 5 held-out big-tier models")
    print("=" * 100)
    for split_key, tag in [("id", "ID trivia n1000"), ("ood", "OOD squad n1000")]:
        preds = list(out["folds"][BIGTIER[0]][split_key]["metrics"])
        print(f"\n[{tag}]  {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}")
        summ = {}
        for k in preds:
            ai = np.mean([out["folds"][h][split_key]["metrics"][k]["auroc_incorrect"] for h in BIGTIER])
            ase = np.mean([out["folds"][h][split_key]["metrics"][k]["auroc_binarised_se"] for h in BIGTIER])
            rs = np.mean([out["folds"][h][split_key]["metrics"][k]["spearman_se"] for h in BIGTIER])
            summ[k] = {"mean_auroc_incorrect": float(ai), "mean_auroc_se": float(ase),
                       "mean_spearman_se": float(rs)}
            print(f"           {k:26s}{ai:>11.3f}{ase:>10.3f}{rs:>9.3f}")
        out["folds"].setdefault("_summary", {})[split_key] = summ
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_PATH}")
    print(f"checkpoints (5 aligner_*.pkl + 5 fold_*.pkl) -> {CKPT_DIR}")


if __name__ == "__main__":
    main()
