"""E29 — label-free ensemble (aligned-z ridge + q_resp_only) vs the supervised SEP baseline, for a
NEW/unseen target LLM, in the WITHIN-n1000 regime (the only regime DeepSeek's data supports; run for
Mistral too as a same-regime calibration). Mirrors the E27 label-free ensemble but fits and evaluates
inside one shared n1000 split (fit on train, eval on the 100-row test split) because DeepSeek has no
disjoint n2000 to fit on.

For source S (the unseen target), reference = Llama-2 at its reference layers (TBG:22, SLT:15):
  - Procrustes W: S -> Llama-2 at the ref layers, TRAIN split only, NO SE labels.
  - Llama-2 FROZEN ridge R on the ref stacked z (TBG:22 | SLT:15) -> Llama-2 SE, TRAIN split.
  - aligned-z  = R(align(S))                      (label-free wrt S; z transfers only after alignment)
  - q_resp_only = the REFERENCE (Llama-2-trained) proxy run on S's records (text transfers)   [GPU]
  - std-avg / rank-fusion ensembles of {aligned-z, q_resp_only}, normalizers from TRAIN preds (label-free)
  - SEP baseline = S's OWN supervised probe: per-(pos,layer) LogisticRegression on best_split-binarized
    S SE; layer chosen by VAL AUROC (leak-free); predicted on TEST. This USES S labels -> the target we
    ask the label-free ensemble to match.
  - paired bootstrap (E25/E26 convention): (ensemble - SEP) Spearman + AUROC, 95% CI, for std-avg and
    rank-fusion.

Additive / read-only reuse: arm_preds/ecdf/boot_delta (procrustes_e27_rank_fusion), fit_probe/load_matrix/
splits/rho (linear_ceiling_probe), best_split/binarize_entropy (stage2.data), orthogonal_procrustes.
Touches no training logic. GPU (q_resp via the frozen proxy). Run in `amortized_stage2`.

    python -m amortized_ue.procrustes_e29_ensemble_sep --source deepseek-llm-7b-chat
    python -m amortized_ue.procrustes_e29_ensemble_sep --source Mistral-7B-Instruct-v0.2
"""
from __future__ import annotations

import json
import argparse

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.procrustes_e27_rank_fusion import arm_preds, ecdf, boot_delta


def sep_baseline(hidden, y, tr, va, te):
    """S's OWN supervised SEP: per-(position,layer) LogisticRegression on best_split-binarized SE.
    Threshold + fit on TRAIN, best (pos,layer) by VAL AUROC (leak-free), predicted prob on TEST.
    Returns (test_prob[all te rows], test_auroc, (pos,layer), threshold, yb[all rows])."""
    thr = best_split(torch.tensor(y[tr]))                          # train-only threshold
    yb = binarize_entropy(torch.tensor(y), thr).numpy()            # -1 == excluded (== threshold)
    best = (-np.inf, None, None)
    for pos in hidden:
        for L in range(hidden[pos].shape[0]):
            X = hidden[pos][L]
            trv, vav = tr[yb[tr] >= 0], va[yb[va] >= 0]
            if len(np.unique(yb[trv])) < 2 or len(np.unique(yb[vav])) < 2:
                continue
            sc = StandardScaler().fit(X[trv])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(X[trv]), yb[trv])
            va_au = roc_auc_score(yb[vav], clf.predict_proba(sc.transform(X[vav]))[:, 1])
            if va_au > best[0]:
                best = (va_au, (pos, int(L)), clf.predict_proba(sc.transform(X[te]))[:, 1])
    tev = te[yb[te] >= 0]
    te_au = roc_auc_score(yb[tev], best[2][yb[te] >= 0])
    return best[2], float(te_au), best[1], float(thr), yb


def run(source="deepseek-llm-7b-chat", ref="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=1000, tbg=22, slt=15, out=None):
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    rh, r_y, r_ids = load_matrix(Stage1Config(model_name=ref, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    assert s_ids == r_ids, "source/ref ids differ -- not the same questions"
    tr, va, te = splits(len(s_ids))
    print(f"source={source}  ref={ref}  n={len(s_ids)} (train {len(tr)}/val {len(va)}/test {len(te)})  ref layers TBG:{tbg} SLT:{slt}")

    # --- Procrustes W: source -> ref at the ref layers, TRAIN only, NO labels -----------------
    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = rh[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, rh[pos][L][tr] - l); return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R, sc, _, _ = fit_probe(np.concatenate([rh["TBG"][tbg], rh["SLT"][slt]], 1), r_y, tr, va)   # ref frozen ridge

    def zpred(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R.predict(sc.transform(a))
    z_all = zpred(sh["TBG"][tbg], sh["SLT"][slt])

    # --- q_resp_only from the REFERENCE (Llama-2) proxy run on SOURCE's records (GPU) ---------
    qr_map = arm_preds("q_resp_only", source, dataset, num_samples)
    qr_all = np.array([qr_map[i] for i in s_ids])

    # --- SEP baseline (source's OWN supervised probe) ----------------------------------------
    sep_te, sep_au, sep_choice, thr, yb = sep_baseline({"TBG": sh["TBG"], "SLT": sh["SLT"]}, s_y, tr, va, te)

    # --- assemble predictors on the TEST split -----------------------------------------------
    y_te, yb_te = s_y[te], yb[te]; v = yb_te >= 0
    z_te, qr_te, z_tr, qr_tr = z_all[te], qr_all[te], z_all[tr], qr_all[tr]
    zs = lambda x, refv: (x - refv.mean()) / (refv.std() + 1e-8)                 # standardise by TRAIN preds
    cz, cr = ecdf(z_tr), ecdf(qr_tr)                                             # CDF from TRAIN preds (label-free)
    preds = {
        "aligned-z ridge (label-free)":        z_te,
        "q_resp_only (label-free)":            qr_te,
        "avg standardized (label-free)":       0.5 * (zs(z_te, z_tr) + zs(qr_te, qr_tr)),
        "RANK FUSION (label-free)":            0.5 * (cz(z_te) + cr(qr_te)),
        "SEP supervised (uses S labels)":      sep_te,
    }

    def M(p): return rho(p, y_te), roc_auc_score(yb_te[v], p[v])
    print("\n" + "=" * 80)
    print(f"E29 LABEL-FREE ENSEMBLE vs SUPERVISED SEP — target {source} (Nvalid={int(v.sum())}/{len(te)}, best_split={thr:.3f})")
    print(f"  SEP best layer = {sep_choice}  (test AUROC {sep_au:.3f})")
    print("=" * 80)
    print(f"  {'predictor':38s}{'Spearman':>10s}{'AUROC':>9s}")
    metrics = {}
    for k, p in preds.items():
        spm, au = M(p); metrics[k] = {"spearman": float(spm), "auroc": float(au)}
        print(f"  {k:38s}{spm:>+10.3f}{au:>9.3f}")

    # --- paired bootstrap: (ensemble - SEP) Spearman + AUROC, 95% CI -------------------------
    deltas = {}
    print("  " + "-" * 76)
    for name in ["avg standardized (label-free)", "RANK FUSION (label-free)"]:
        bd = boot_delta(preds[name], sep_te, y_te, yb_te, v)
        deltas[f"{name} - SEP"] = {"spearman": {"mean": bd["spearman"][0], "lo95": bd["spearman"][1], "hi95": bd["spearman"][2]},
                                   "auroc": {"mean": bd["auroc"][0], "lo95": bd["auroc"][1], "hi95": bd["auroc"][2]}}
        s_ex = "excludes" if (bd["spearman"][1] > 0 or bd["spearman"][2] < 0) else "includes"
        a_ex = "excludes" if (bd["auroc"][1] > 0 or bd["auroc"][2] < 0) else "includes"
        print(f"  Δ({name.split(' (')[0]} − SEP)  Spearman {bd['spearman'][0]:+.3f} [{bd['spearman'][1]:+.3f}, {bd['spearman'][2]:+.3f}] ({s_ex} 0)"
              f"   AUROC {bd['auroc'][0]:+.3f} [{bd['auroc'][1]:+.3f}, {bd['auroc'][2]:+.3f}] ({a_ex} 0)")
    print("=" * 80 + "\n")

    result = {"source": source, "ref": ref, "dataset": dataset, "num_samples": num_samples,
              "regime": "within-n1000 (fit train / eval 100-row test split)", "ref_layers": {"TBG": tbg, "SLT": slt},
              "n_test": len(te), "n_valid": int(v.sum()), "best_split": thr,
              "sep_best_layer": sep_choice, "sep_test_auroc": sep_au,
              "metrics": metrics, "ensemble_minus_sep": deltas}
    out = out or f"amortized_ue/procrustes_e29_ensemble_sep_{source}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E29 label-free ensemble vs supervised SEP for a new target (CPU+GPU).")
    p.add_argument("--source", default="deepseek-llm-7b-chat")
    p.add_argument("--ref", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--tbg", type=int, default=22)
    p.add_argument("--slt", type=int, default=15)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.source, a.ref, a.dataset, a.num_samples, a.tbg, a.slt, a.out)
