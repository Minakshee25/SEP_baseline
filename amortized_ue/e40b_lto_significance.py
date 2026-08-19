"""E40b — fix the null, then put CIs on the clean test.

Why this exists: E40's `q_only` CONTROL failed. Its input is the question alone, identical for every
target, so its model-specific residual correlation had to be ~0 — it came out -0.097 (p=0.013).
That is not a coding bug, it is a structural property of the LEAVE-ONE-OUT design:

    for target T the probe is trained on the OTHER 3 models, so it estimates their SE. The
    model-specific residuals sum to zero across the 4 models, hence mean_{k!=T} s_k = -s_T/3.
    A predictor carrying NO information about T is therefore ANTI-correlated with s_T.

[1] proves it exactly: the perfect pure-difficulty LOO predictor D_T = mean_{k!=T} Y_k satisfies
    R_D = -(1/3) R_Y identically, so its residual correlation is -1.0, not 0. => every number in
    E40's [C]/[D]/[F] is biased DOWNWARD and "chance = 0" is wrong there.

[2] is the clean estimand. In LEAVE-TWO-OUT one ridge (trained on the other two models) scores BOTH
    members of the held-out pair, so dP = P_A - P_B uses the SAME weights and the fold-composition
    artifact cancels: a question-only predictor gives dP = 0 exactly => the null IS 0. This adds the
    bootstrap CIs and an exact sign-flip permutation test that E40's [G] lacked.

Reuses the SAVED ridge bundle (results/e40_ridge_checkpoints) — no refitting — which also serves as
the working proof that the checkpoint is loadable.

Env: se_probes (CPU). Run: python -m amortized_ue.e40b_lto_significance --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import pearsonr

from amortized_ue.exp2_run import MODELS, SHORT
from amortized_ue.e40_model_specificity import (load_all_plus, load_ridge_bundle, qnorm, rownorm,
                                                resid, cell_corr, PAIRS)


def boot_ci(fn, N, B=2000, seed=0):
    """Bootstrap over QUESTIONS (the unit of independence); fn(idx) -> statistic."""
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, N, N)) for _ in range(B)])
    return float(v.mean()), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))


def signflip_null(dP, dY, B=5000, seed=0):
    """Exact null for the LTO pair test: which member of the pair is 'A' is arbitrary, so flipping
    the sign of dP for a random subset of questions destroys any real A-vs-B signal and nothing else."""
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        s = rng.choice([-1.0, 1.0], size=len(dP))
        out[b] = pearsonr(dP * s, dY)[0]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data2/mn1025/stage1")
    ap.add_argument("--ckpt_dir", default="stage2/runs/E40_pooled_multimodel_ridge/checkpoints")
    ap.add_argument("--anchor_layer", type=int, default=30)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--perm", type=int, default=5000)
    ap.add_argument("--out", default="results/e40b_lto_significance.json")
    args = ap.parse_args()
    R = {"config": vars(args)}

    def dump():
        with open(args.out, "w") as f:
            json.dump(R, f, indent=1)

    print("=" * 92)
    print("E40b — the LOO null is NEGATIVE (proved), so the leave-TWO-out test is the clean one")
    print("=" * 92)

    data, ids, (tr, va, te) = load_all_plus(args.anchor_layer, args.data_dir)
    bundle = load_ridge_bundle(args.ckpt_dir)          # no refit: reuse the saved multi-model ridge
    print(f"loaded {len(ids)} questions; te={len(te)}; reusing SAVED bundle {args.ckpt_dir}")

    Y_te = np.vstack([data[m]["y"][te] for m in MODELS])
    Yn = rownorm(Y_te)
    Ry = resid(Yn)

    # ---- [1] the LOO null is not zero ---------------------------------------
    Dm = np.vstack([Yn[[j for j in range(4) if j != i]].mean(axis=0) for i in range(4)])
    r_oracle = cell_corr(resid(Dm), Ry)
    P_te = np.vstack([bundle["predict"](m, data[m]["H"][te], f"loo_{SHORT[m]}")
                      for m in MODELS])
    r_ridge = cell_corr(resid(rownorm(P_te)), Ry)
    R["loo_null"] = {
        "perfect_difficulty_oracle_residual_corr": r_oracle,
        "analytic_expectation": -1.0,
        "ridge_residual_corr_from_saved_ckpt": r_ridge,
        "note": ("A PERFECT pure-difficulty predictor scores -1.0 under leave-one-out, so 0 is NOT "
                 "the no-information baseline there; E40's [C]/[D]/[F] are biased downward."),
    }
    print(f"\n[1] perfect pure-difficulty LOO predictor (mean of the other 3 models' TRUE SE):")
    print(f"    residual corr = {r_oracle:+.4f}   (analytic: -1.0 exactly)")
    print(f"    => 'chance = 0' is WRONG for leave-one-out; E40 [C]/[D]/[F] are biased DOWN.")
    print(f"    (ridge from the SAVED checkpoint reproduces E40's [C]: {r_ridge:+.3f})")
    dump()

    # ---- [2] leave-two-out, with CIs and an exact null ------------------------
    print(f"\n[2] LEAVE-TWO-OUT (one probe scores both models of the pair => null IS 0):")
    print(f"    {'held-out pair':>22} | {'r(dP,dY)':>9} {'boot95':>18} {'perm p':>7} | "
          f"{'acc all':>8} {'boot95':>16}")
    G, dPs, dYs = {}, [], []
    for a, b in PAIRS:
        fold = f"lto_{SHORT[MODELS[a]]}_{SHORT[MODELS[b]]}"
        Pa = bundle["predict"](MODELS[a], data[MODELS[a]]["H"][te], fold)
        Pb = bundle["predict"](MODELS[b], data[MODELS[b]]["H"][te], fold)
        dY = qnorm(Y_te[a]) - qnorm(Y_te[b])
        dP = qnorm(Pa) - qnorm(Pb)
        dPs.append(dP); dYs.append(dY)

        def r_of(idx, dP=dP, dY=dY):
            if np.std(dP[idx]) < 1e-12 or np.std(dY[idx]) < 1e-12:
                return 0.0
            return float(pearsonr(dP[idx], dY[idx])[0])

        def acc_of(idx, dP=dP, dY=dY):
            k = dY[idx] != 0
            if k.sum() == 0:
                return 0.5
            h = np.where(dP[idx][k] == 0, 0.5,
                         (np.sign(dY[idx][k]) == np.sign(dP[idx][k])).astype(float))
            return float(h.mean())

        r_obs = r_of(np.arange(len(te)))
        acc_obs = acc_of(np.arange(len(te)))
        rb = boot_ci(r_of, len(te), args.boot)
        ab = boot_ci(acc_of, len(te), args.boot)
        nul = signflip_null(dP, dY, args.perm)
        p = float((np.abs(nul) >= abs(r_obs)).mean())
        G[f"{SHORT[MODELS[a]]}_vs_{SHORT[MODELS[b]]}"] = {
            "r_dP_dY": r_obs, "r_boot95": [rb[1], rb[2]], "perm_p": p,
            "pair_acc": acc_obs, "acc_boot95": [ab[1], ab[2]]}
        print(f"    {SHORT[MODELS[a]]+'_vs_'+SHORT[MODELS[b]]:>22} | {r_obs:>+9.3f} "
              f"[{rb[1]:+.3f},{rb[2]:+.3f}] {p:>7.4f} | {acc_obs:>8.3f} [{ab[1]:.3f},{ab[2]:.3f}]")
    R["G_lto"] = G
    dump()

    # ---- pooled across the 6 pairs (resample QUESTIONS once, recompute all pairs) -------
    dPs, dYs = np.array(dPs), np.array(dYs)

    def pooled_r(idx):
        A, B_ = dPs[:, idx].ravel(), dYs[:, idx].ravel()
        if np.std(A) < 1e-12 or np.std(B_) < 1e-12:
            return 0.0
        return float(pearsonr(A, B_)[0])

    def pooled_acc(idx):
        A, B_ = dPs[:, idx].ravel(), dYs[:, idx].ravel()
        k = B_ != 0
        h = np.where(A[k] == 0, 0.5, (np.sign(A[k]) == np.sign(B_[k])).astype(float))
        return float(h.mean())

    all_idx = np.arange(len(te))
    pr, pa = pooled_r(all_idx), pooled_acc(all_idx)
    prb = boot_ci(pooled_r, len(te), args.boot)
    pab = boot_ci(pooled_acc, len(te), args.boot)
    nul = signflip_null(dPs.ravel(), dYs.ravel(), args.perm)
    pp = float((np.abs(nul) >= abs(pr)).mean())
    R["pooled"] = {"r_dP_dY": pr, "r_boot95": [prb[1], prb[2]], "perm_p": pp,
                   "pair_acc": pa, "acc_boot95": [pab[1], pab[2]],
                   "n_pairs_scored": int(dPs.size)}
    print(f"\n    POOLED over all 6 pairs (n={dPs.size} comparisons, bootstrap over the 200 questions):")
    print(f"      r(dP,dY) = {pr:+.3f}  boot95 [{prb[1]:+.3f}, {prb[2]:+.3f}]  sign-flip p = {pp:.4f}")
    print(f"      pair-acc = {pa:.3f}   boot95 [{pab[1]:.3f}, {pab[2]:.3f}]   (chance 0.500)")
    dump()
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
