"""E26 — decompose the aligned transfer: does the rotation carry info BEYOND shared difficulty?

E25 showed the aligned transfer (Mistral TBG → Procrustes → Llama-2 ridge, vs Mistral SE) beats the
Mechanism-A control (Llama-2's OWN TBG → Llama-2 ridge, vs Mistral SE) by a small, significant margin.
E26 pins down whether that margin is genuinely NEW model-specific information, two ways, on the E23
fresh n1000 batch (both models' TBG already on disk), reusing the SAME E25 fit (W + Llama-2 ridge on
the n2000 1440-train pairs, NO SE labels in W):

 (1) SEMI-PARTIAL CORRELATION. Regress the control prediction out of Mistral's SE (OLS) → residuals,
     then Spearman(aligned prediction, residuals). If clearly > 0, the rotation predicts variance in
     Mistral's SE that shared difficulty (the control) does NOT explain. Bootstrap 95% CI.
     (Also report the symmetric rank-based partial Spearman as a robustness check.)

 (2) ENSEMBLE. Combine control + aligned predictions — simple average, and a tiny 2-input ridge fit on
     the train split — and score vs Mistral SE on n1000. If the ensemble clearly beats control alone,
     the two carry complementary info; if it barely moves, they are redundant. Bootstrap the
     (ensemble − control) gap.

CPU-only, additive; reuses linear_ceiling_probe + procrustes_alignment helpers read-only; touches
nothing existing.

    python -m amortized_ue.procrustes_e26_decomposition        # se_probes env, no GPU
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata
from sklearn.linear_model import LinearRegression, Ridge

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.procrustes_alignment import best_tbg_layer


def semi_partial(aligned, y, control):
    """Spearman(aligned, residual of y after OLS-removing control). Semi-partial / part corr."""
    lr = LinearRegression().fit(control.reshape(-1, 1), y)
    resid = y - lr.predict(control.reshape(-1, 1))
    return rho(aligned, resid)


def partial_spearman(aligned, y, control):
    """Symmetric rank-based partial Spearman of (aligned, y) controlling for control."""
    ra, ry, rc = rankdata(aligned), rankdata(y), rankdata(control)
    ra_r = ra - LinearRegression().fit(rc.reshape(-1, 1), ra).predict(rc.reshape(-1, 1))
    ry_r = ry - LinearRegression().fit(rc.reshape(-1, 1), ry).predict(rc.reshape(-1, 1))
    if np.std(ra_r) < 1e-12 or np.std(ry_r) < 1e-12:
        return 0.0
    return float(np.corrcoef(ra_r, ry_r)[0, 1])


def boot_semi_partial(aligned, y, control, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); vals = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        vals[b] = semi_partial(aligned[idx], y[idx], control[idx])
    return float(vals.mean()), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)), float(np.mean(vals > 0))


def boot_diff(pred_a, pred_b, y, B=2000, seed=0):
    """Paired bootstrap of rho(pred_a,y) - rho(pred_b,y) over resampled rows."""
    rng = np.random.default_rng(seed); n = len(y); d = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        d[b] = rho(pred_a[idx], y[idx]) - rho(pred_b[idx], y[idx])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(np.mean(d > 0))


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, out="amortized_ue/procrustes_e26_decomposition.json"):
    # ---- load n2000 fit data, split, pick Llama-2 ridge layer ------------------
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG"])
    th, t_y, t_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG"])
    assert s_ids == t_ids, "source/target n2000 ids differ"
    S, T = sh["TBG"], th["TBG"]
    tr, va, te = splits(len(s_ids))
    L_a, _ = best_tbg_layer(T, t_y, tr, va)              # Llama-2 (target) ridge layer
    H = S.shape[2]
    print(f"fit on n{num_samples} train={len(tr)} | target ridge TBG L_a={L_a} | H={H}")

    # ---- frozen Llama-2 ridge + Procrustes W (train only, no SE labels in W) ---
    R_L, sc_L, _, _ = fit_probe(T[L_a], t_y, tr, va)
    m_mean = S[L_a][tr].mean(0, keepdims=True)
    l_mean = T[L_a][tr].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(S[L_a][tr] - m_mean, T[L_a][tr] - l_mean)

    def control_pred(tgt_La):                            # Mech-A: target's OWN states
        return R_L.predict(sc_L.transform(tgt_La))
    def aligned_pred(src_La):                            # rotated source states
        return R_L.predict(sc_L.transform((src_La - m_mean) @ W + l_mean))

    # ---- predictions on the TRAIN split (to fit the 2-input ensemble) ----------
    ctrl_tr, algn_tr, y_tr = control_pred(T[L_a][tr]), aligned_pred(S[L_a][tr]), s_y[tr]

    # ---- load the fresh n1000 batch (disjoint from n2000) ----------------------
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG"])
    fth, ft_y, ft_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=fresh_num_samples), ["TBG"])
    assert fs_ids == ft_ids, "fresh source/target ids differ"
    Sf, Tf = fsh["TBG"], fth["TBG"]
    ctrl_f = control_pred(Tf[L_a])                       # control prediction, fresh
    algn_f = aligned_pred(Sf[L_a])                       # aligned prediction, fresh
    y_f = fs_y                                           # Mistral SE, fresh
    n = len(y_f)
    sp_control = rho(ctrl_f, y_f)
    sp_aligned = rho(algn_f, y_f)
    print(f"fresh n{fresh_num_samples}: control={sp_control:+.3f}  aligned={sp_aligned:+.3f}")

    # ---- (1) SEMI-PARTIAL correlation on the fresh batch -----------------------
    sp_partial = semi_partial(algn_f, y_f, ctrl_f)
    bp = boot_semi_partial(algn_f, y_f, ctrl_f)
    sp_partial_full = partial_spearman(algn_f, y_f, ctrl_f)

    # ---- (2) ENSEMBLE (avg + tiny 2-input ridge fit on train) ------------------
    ens_avg_f = 0.5 * (ctrl_f + algn_f)
    sp_ens_avg = rho(ens_avg_f, y_f)
    meta = Ridge(alpha=1.0).fit(np.column_stack([ctrl_tr, algn_tr]), y_tr)   # tiny 2-input ridge
    ens_ridge_f = meta.predict(np.column_stack([ctrl_f, algn_f]))
    sp_ens_ridge = rho(ens_ridge_f, y_f)
    bd_avg = boot_diff(ens_avg_f, ctrl_f, y_f)
    bd_ridge = boot_diff(ens_ridge_f, ctrl_f, y_f)

    # ---- report ---------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"E26 DECOMPOSITION  {source} -> {target}, vs {source} SE, fresh n{fresh_num_samples} (N={n})")
    print("=" * 78)
    print("  (1) SEMI-PARTIAL CORRELATION  spearman(aligned, SE | control removed from SE):")
    sep = "ABOVE ZERO (rotation carries NEW info)" if bp[1] > 0 else "overlaps 0 (redundant with control)"
    print(f"        semi-partial = {sp_partial:+.3f}   bootstrap 95% CI [{bp[1]:+.3f}, {bp[2]:+.3f}]  P(>0)={bp[3]:.2f}  -> {sep}")
    print(f"        (robustness: symmetric rank-based partial Spearman = {sp_partial_full:+.3f})")
    print("  " + "-" * 74)
    print("  (2) ENSEMBLE vs single predictors (Spearman vs Mistral SE, fresh n1000):")
    print(f"        control alone            : {sp_control:+.3f}")
    print(f"        aligned alone            : {sp_aligned:+.3f}")
    print(f"        ensemble (avg)           : {sp_ens_avg:+.3f}   (avg - control) {bd_avg[0]:+.3f} [{bd_avg[1]:+.3f}, {bd_avg[2]:+.3f}] P(>0)={bd_avg[3]:.2f}")
    print(f"        ensemble (2-input ridge) : {sp_ens_ridge:+.3f}   (ridge - control) {bd_ridge[0]:+.3f} [{bd_ridge[1]:+.3f}, {bd_ridge[2]:+.3f}] P(>0)={bd_ridge[3]:.2f}")
    print(f"        ridge meta-weights [control, aligned] = [{meta.coef_[0]:+.3f}, {meta.coef_[1]:+.3f}]")
    print("=" * 78 + "\n")

    result = {
        "source": source, "target": target, "dataset": dataset, "fit_num_samples": num_samples,
        "fresh_num_samples": fresh_num_samples, "n_eval": n, "L_a_target_ridge": int(L_a),
        "control_spearman": sp_control, "aligned_spearman": sp_aligned,
        "semi_partial": {"value": sp_partial, "boot_mean": bp[0], "lo95": bp[1], "hi95": bp[2],
                         "frac_positive": bp[3], "symmetric_partial_spearman": sp_partial_full},
        "ensemble": {
            "avg_spearman": sp_ens_avg, "ridge_spearman": sp_ens_ridge,
            "avg_minus_control": {"mean": bd_avg[0], "lo95": bd_avg[1], "hi95": bd_avg[2], "frac_positive": bd_avg[3]},
            "ridge_minus_control": {"mean": bd_ridge[0], "lo95": bd_ridge[1], "hi95": bd_ridge[2], "frac_positive": bd_ridge[3]},
            "ridge_meta_weights": {"control": float(meta.coef_[0]), "aligned": float(meta.coef_[1])},
        },
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


def _parse():
    p = argparse.ArgumentParser(description="E26 decomposition of the aligned transfer (CPU).")
    p.add_argument("--source", default="Mistral-7B-Instruct-v0.2")
    p.add_argument("--target", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--fresh_num_samples", type=int, default=1000)
    p.add_argument("--out", default="amortized_ue/procrustes_e26_decomposition.json")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.source, a.target, a.dataset, a.num_samples, a.fresh_num_samples, a.out)
