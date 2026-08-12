"""E30 — FULL-POWER label-free ensemble (aligned-z + q_resp_only) vs supervised SEP, fitting on one
Stage-1 set and evaluating on a (possibly different, disjoint) set. Generalises the E29 within-n1000
script so DeepSeek can fit on its new n2000 and evaluate on the disjoint fresh n1000 (N=1000 power),
and Llama-3 can fit+eval on its n2000 (N=200 test split).

Same recipe as E29 (reference = Llama-2 at TBG:22/SLT:15): Procrustes W (source->ref, no labels) +
frozen ref ridge -> aligned-z; REFERENCE proxy on source text -> q_resp_only; std-avg & rank-fusion
ensembles (label-free); source's OWN supervised SEP (per-layer LogisticRegression, best_split) as the
baseline; paired-bootstrap (ensemble - SEP) Spearman + AUROC 95% CI.

  --fit_num_samples 2000 --eval_num_samples 1000   # DeepSeek: fit n2000, eval fresh n1000 (N=1000)
  --fit_num_samples 2000                           # Llama-3: fit+eval n2000 (eval = test split, N=200)

Additive; reuses arm_preds/ecdf/boot_delta (E27 rank fusion), fit_probe/load_matrix/splits/rho, and
best_split/binarize_entropy. Touches no training logic. GPU (q_resp via frozen proxy). `amortized_stage2`.
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


def sep_fit_eval(fit_hidden, fit_y, tr, va, eval_hidden, eval_y):
    """Source's OWN supervised SEP fit on the FIT set (train/val) and predicted on the EVAL set.
    Per-(pos,layer) LogisticRegression on best_split-binarised SE; best layer by FIT-val AUROC.
    Returns (eval_prob[all eval rows], eval_auroc, (pos,layer), threshold, yb_eval)."""
    thr = best_split(torch.tensor(fit_y[tr]))
    ybf = binarize_entropy(torch.tensor(fit_y), thr).numpy()
    ybe = binarize_entropy(torch.tensor(eval_y), thr).numpy()
    best = (-np.inf, None, None)
    for pos in fit_hidden:
        for L in range(fit_hidden[pos].shape[0]):
            Xf, Xe = fit_hidden[pos][L], eval_hidden[pos][L]
            trv, vav = tr[ybf[tr] >= 0], va[ybf[va] >= 0]
            if len(np.unique(ybf[trv])) < 2 or len(np.unique(ybf[vav])) < 2:
                continue
            sc = StandardScaler().fit(Xf[trv])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xf[trv]), ybf[trv])
            va_au = roc_auc_score(ybf[vav], clf.predict_proba(sc.transform(Xf[vav]))[:, 1])
            if va_au > best[0]:
                best = (va_au, (pos, int(L)), clf.predict_proba(sc.transform(Xe))[:, 1])
    ev = ybe >= 0
    return best[2], float(roc_auc_score(ybe[ev], best[2][ev])), best[1], float(thr), ybe


def run(source="deepseek-llm-7b-chat", ref="Llama-2-7b-chat", dataset="trivia_qa",
        fit_num_samples=2000, eval_num_samples=None, tbg=22, slt=15, out=None):
    # ---- fit set (source + ref, shared ids) ----
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fit_num_samples), ["TBG", "SLT"])
    rh, r_y, r_ids = load_matrix(Stage1Config(model_name=ref, dataset=dataset, num_samples=fit_num_samples), ["TBG", "SLT"])
    assert s_ids == r_ids, "fit source/ref ids differ"
    tr, va, te = splits(len(s_ids))

    # ---- eval set: separate disjoint dataset (all rows) OR the fit test split ----
    if eval_num_samples:
        esh, ey, e_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=eval_num_samples), ["TBG", "SLT"])
        erh, _, er_ids = load_matrix(Stage1Config(model_name=ref, dataset=dataset, num_samples=eval_num_samples), ["TBG", "SLT"])
        assert e_ids == er_ids
        eval_ids, eval_rows, regime = e_ids, np.arange(len(e_ids)), f"fit n{fit_num_samples} -> eval fresh n{eval_num_samples} (N={len(e_ids)})"
    else:
        esh, ey, e_ids = sh, s_y, s_ids
        eval_ids, eval_rows, regime = [s_ids[i] for i in te], te, f"fit+eval n{fit_num_samples} (eval=test split, N={len(te)})"
    print(f"source={source}  ref={ref}  {regime}  ref layers TBG:{tbg} SLT:{slt}")

    # ---- Procrustes W (source->ref, fit-train, no labels) + frozen ref ridge ----
    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = rh[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, rh[pos][L][tr] - l); return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R, sc, _, _ = fit_probe(np.concatenate([rh["TBG"][tbg], rh["SLT"][slt]], 1), r_y, tr, va)

    def zpred(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R.predict(sc.transform(a))
    z_tr = zpred(sh["TBG"][tbg], sh["SLT"][slt])[tr]                                   # train preds -> normalizers
    z_eval = zpred(esh["TBG"][tbg], esh["SLT"][slt])[eval_rows]

    # ---- q_resp_only from the REFERENCE proxy on SOURCE records (GPU) ----
    qr_fit = arm_preds("q_resp_only", source, dataset, fit_num_samples)
    qr_tr = np.array([qr_fit[i] for i in s_ids])[tr]
    if eval_num_samples:
        qr_ev_map = arm_preds("q_resp_only", source, dataset, eval_num_samples)
        qr_eval = np.array([qr_ev_map[i] for i in eval_ids])
    else:
        qr_eval = np.array([qr_fit[i] for i in eval_ids])

    # ---- SEP baseline (source supervised, fit-train -> eval) ----
    sep_all, _, sep_choice, thr, ybe = sep_fit_eval(
        {"TBG": sh["TBG"], "SLT": sh["SLT"]}, s_y, tr, va,
        {"TBG": esh["TBG"], "SLT": esh["SLT"]}, ey)

    # ---- predictors on eval ----
    y_ev, yb_ev = ey[eval_rows], ybe[eval_rows]; v = yb_ev >= 0
    sep_ev = sep_all[eval_rows] if not eval_num_samples else sep_all  # subset to eval rows (within-split path)
    sep_au = float(roc_auc_score(yb_ev[v], sep_ev[v]))               # honest eval-ONLY AUROC (no train leakage)
    zs = lambda x, refv: (x - refv.mean()) / (refv.std() + 1e-8)
    cz, cr = ecdf(z_tr), ecdf(qr_tr)
    preds = {
        "aligned-z ridge (label-free)":    z_eval,
        "q_resp_only (label-free)":        qr_eval,
        "avg standardized (label-free)":   0.5 * (zs(z_eval, z_tr) + zs(qr_eval, qr_tr)),
        "RANK FUSION (label-free)":        0.5 * (cz(z_eval) + cr(qr_eval)),
        "SEP supervised (uses S labels)":  sep_ev,
    }

    def M(p): return rho(p, y_ev), roc_auc_score(yb_ev[v], p[v])
    print("\n" + "=" * 82)
    print(f"E30 FULL-POWER ENSEMBLE vs SEP — target {source}  ({regime}, Nvalid={int(v.sum())}, split={thr:.3f})")
    print(f"  SEP best layer {sep_choice}  (eval AUROC {sep_au:.3f})")
    print("=" * 82)
    print(f"  {'predictor':38s}{'Spearman':>10s}{'AUROC':>9s}")
    metrics = {}
    for k, p in preds.items():
        spm, au = M(p); metrics[k] = {"spearman": float(spm), "auroc": float(au)}
        print(f"  {k:38s}{spm:>+10.3f}{au:>9.3f}")

    deltas = {}
    print("  " + "-" * 78)
    for name in ["avg standardized (label-free)", "RANK FUSION (label-free)"]:
        bd = boot_delta(preds[name], sep_ev, y_ev, yb_ev, v)
        deltas[f"{name} - SEP"] = {"spearman": {"mean": bd["spearman"][0], "lo95": bd["spearman"][1], "hi95": bd["spearman"][2]},
                                   "auroc": {"mean": bd["auroc"][0], "lo95": bd["auroc"][1], "hi95": bd["auroc"][2]}}
        se = "excludes" if (bd["spearman"][1] > 0 or bd["spearman"][2] < 0) else "includes"
        ae = "excludes" if (bd["auroc"][1] > 0 or bd["auroc"][2] < 0) else "includes"
        print(f"  Δ({name.split(' (')[0]} − SEP)  Spearman {bd['spearman'][0]:+.3f} [{bd['spearman'][1]:+.3f}, {bd['spearman'][2]:+.3f}] ({se} 0)"
              f"   AUROC {bd['auroc'][0]:+.3f} [{bd['auroc'][1]:+.3f}, {bd['auroc'][2]:+.3f}] ({ae} 0)")
    print("=" * 82 + "\n")

    result = {"source": source, "ref": ref, "dataset": dataset, "regime": regime,
              "fit_num_samples": fit_num_samples, "eval_num_samples": eval_num_samples,
              "ref_layers": {"TBG": tbg, "SLT": slt}, "n_eval": len(eval_rows), "n_valid": int(v.sum()),
              "best_split": thr, "sep_best_layer": sep_choice, "sep_eval_auroc": sep_au,
              "metrics": metrics, "ensemble_minus_sep": deltas}
    out = out or f"amortized_ue/procrustes_e30_ensemble_sep_{source}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="E30 full-power label-free ensemble vs SEP (CPU+GPU).")
    p.add_argument("--source", default="deepseek-llm-7b-chat")
    p.add_argument("--ref", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--fit_num_samples", type=int, default=2000)
    p.add_argument("--eval_num_samples", type=int, default=None, help="disjoint eval set size; omit to use the fit test split")
    p.add_argument("--tbg", type=int, default=22)
    p.add_argument("--slt", type=int, default=15)
    p.add_argument("--out", default=None)
    a = p.parse_args()
    run(a.source, a.ref, a.dataset, a.fit_num_samples, a.eval_num_samples, a.tbg, a.slt, a.out)
