"""E40c — the ceiling MATCHED to the leave-two-out estimand.

E40's [B] measured the split-half reliability of the leave-ONE-out residual (0.870). The headline
now rests on the leave-TWO-out difference dY = qnorm(SE_A) - qnorm(SE_B), which is a different
quantity, so quoting "% of ceiling" off [B] would use the wrong denominator.

Same method, matched estimand: split each model's 10 samples into disjoint halves, recompute SE per
half, form the pair difference from each half independently, and correlate the two halves' difference
vectors -> r5; Spearman-Brown -> r10; a perfect predictor can attain at most sqrt(r10).

Env: se_probes (CPU). Run: python -m amortized_ue.e40c_lto_ceiling --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.stats import pearsonr

from amortized_ue.exp2_run import MODELS, SHORT
from amortized_ue.e40_model_specificity import load_all_plus, qnorm, cae, PAIRS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/data2/mn1025/stage1")
    ap.add_argument("--anchor_layer", type=int, default=30)
    ap.add_argument("--splits", type=int, default=200)
    ap.add_argument("--observed", type=float, default=0.110, help="E40b pooled r(dP,dY)")
    ap.add_argument("--out", default="results/e40c_lto_ceiling.json")
    args = ap.parse_args()

    data, ids, (tr, va, te) = load_all_plus(args.anchor_layer, args.data_dir)
    rng = np.random.default_rng(0)
    r5 = []
    for _ in range(args.splits):
        HA = np.zeros((len(MODELS), len(te)))
        HB = np.zeros((len(MODELS), len(te)))
        for mi, m in enumerate(MODELS):
            for k, idx in enumerate(te):
                s = np.asarray(data[m]["sids"][idx], dtype=int)
                p = rng.permutation(len(s))
                h = len(s) // 2
                HA[mi, k] = cae(s[p[:h]])
                HB[mi, k] = cae(s[p[h:2 * h]])
        HAn = np.vstack([qnorm(r) for r in HA])
        HBn = np.vstack([qnorm(r) for r in HB])
        dA = np.concatenate([HAn[a] - HAn[b] for a, b in PAIRS])
        dB = np.concatenate([HBn[a] - HBn[b] for a, b in PAIRS])
        r5.append(float(pearsonr(dA, dB)[0]))
    r5m = float(np.mean(r5))
    r10 = 2 * r5m / (1 + r5m)
    ceil = float(np.sqrt(max(r10, 0.0)))
    out = {"n_splits": args.splits, "pair_difference_r5": r5m,
           "pair_difference_r10_spearman_brown": r10, "ceiling_sqrt_r10": ceil,
           "observed_pooled_r": args.observed,
           "frac_of_ceiling": args.observed / ceil if ceil > 0 else None}
    print(f"LTO pair-difference reliability: r5={r5m:.3f} r10={r10:.3f} ceiling={ceil:.3f}")
    print(f"observed pooled r = {args.observed:+.3f}  =>  {args.observed / ceil:.1%} of attainable")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
