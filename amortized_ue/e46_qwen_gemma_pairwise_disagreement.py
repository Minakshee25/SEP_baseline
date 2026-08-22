"""E46 -- does the zero-shot q_resp_only proxy distinguish CROSS-MODEL disagreement on
Qwen/Gemma (additive, no retraining)?

E45 scored each of the 4 new targets (Qwen3-8B, Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it) against
its OWN correctness labels, in isolation -- strong AUROCs there are consistent with the proxy
just reading question difficulty + each model's own answer-text tells, without ever checking
whether it can tell "model A is uncertain here but model B isn't" on the SAME question. This
script asks that question directly, following E40's methodology (dP vs dY correlation + pairwise
accuracy on divergent rows, gated by disagreement size) -- but the leave-two-out complication that
drove E40's negative-null correction does NOT apply here: the deploy proxy was trained ONLY on
Llama-2/Mistral/Llama-3/DeepSeek, so every pair among these 4 new targets is symmetric (both
members equally unseen) -- no asymmetric train/test membership to bias the null toward negative.
`q_only`'s dP is included as a determinism sanity check (should be ~0 for every question, since
q_only reads only the question text, identical across models) -- NOT a statistical null here, just
a sanity check the two arms behave as expected.

Env: `amortized_stage2_v5` (the /data2 venv, see E45) + a free GPU. Run from the repo root:
    /data2/mn1025/conda_envs/amortized_stage2_v5/bin/python -m amortized_ue.e46_qwen_gemma_pairwise_disagreement
"""
from __future__ import annotations

import json
import argparse
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import load_accuracy
from amortized_ue.procrustes_e27_rank_fusion import arm_preds

TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
NUM_SAMPLES = 1000


def rho(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def ci(v, lo=2.5, hi=97.5):
    return {"mean": float(np.mean(v)), "lo95": float(np.percentile(v, lo)), "hi95": float(np.percentile(v, hi))}


def boot_rho(dP, dY, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dP)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(rho(dP[idx], dY[idx]))
    return ci(out)


def boot_acc(dP, dY, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(dP)
    out = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        out.append(float((np.sign(dP[idx]) == np.sign(dY[idx])).mean()))
    return ci(out)


def load_target(target: str) -> dict:
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
    recs = load_records(cfg)
    ids = sorted(recs.keys())
    y_se = {i: float(recs[i]["labels"]["cluster_assignment_entropy"]) for i in ids}
    acc = load_accuracy(cfg)
    print(f"  loading proxy preds (q_only, q_resp_only) for {target} ...")
    q_only = arm_preds("q_only", target, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    q_resp = arm_preds("q_resp_only", target, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    return {"ids": set(ids), "se": y_se, "acc": acc, "q_only": q_only, "q_resp_only": q_resp}


def pair_analysis(A_name, A, B_name, B, bootstrap):
    common = sorted(A["ids"] & B["ids"])
    n = len(common)
    dY_se = np.array([A["se"][i] - B["se"][i] for i in common])
    incorrect_A = np.array([1.0 - A["acc"][i] for i in common])
    incorrect_B = np.array([1.0 - B["acc"][i] for i in common])
    dY_correct = incorrect_A - incorrect_B          # +1: A wrong/B right, -1: A right/B wrong, 0: same

    out = {"pair": f"{A_name} vs {B_name}", "n_common": n}
    for arm in ["q_only", "q_resp_only"]:
        dP = np.array([A[arm][i] - B[arm][i] for i in common])

        # --- SE-fidelity: does dP track the continuous SE gap? (magnitude-weighted) ---
        r_se = rho(dP, dY_se)
        r_se_ci = boot_rho(dP, dY_se, B=bootstrap)

        # --- correctness: on rows where the two targets DISAGREE on correctness, does sign(dP)
        # pick the actually-more-wrong model? (unweighted pairwise accuracy, large-gap subset too) ---
        divergent = dY_correct != 0
        nd = int(divergent.sum())
        if nd >= 20:
            acc_all = float((np.sign(dP[divergent]) == np.sign(dY_correct[divergent])).mean())
            acc_all_ci = boot_acc(dP[divergent], dY_correct[divergent], B=bootstrap)
        else:
            acc_all, acc_all_ci = None, None

        # large-SE-gap subset (top quartile |dY_se|, matches E40's gating)
        keep = dY_se != 0
        if keep.sum() >= 20:
            thr = np.quantile(np.abs(dY_se[keep]), 0.75)
            top = keep & (np.abs(dY_se) >= thr)
            r_se_top = rho(dP[top], dY_se[top])
        else:
            r_se_top = None

        out[arm] = {
            "se_gap_corr": r_se, "se_gap_corr_ci": r_se_ci, "se_gap_corr_top_quartile": r_se_top,
            "n_divergent_correctness": nd, "pair_accuracy_on_divergent": acc_all,
            "pair_accuracy_ci": acc_all_ci,
            "dP_std": float(dP.std()), "dP_near_zero_frac": float((np.abs(dP) < 1e-6).mean()),
        }
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/e46_qwen_gemma_pairwise.json")
    args = p.parse_args()

    print("loading all 4 targets' records + proxy predictions ...")
    data = {t: load_target(t) for t in TARGETS}

    results = []
    for A_name, B_name in combinations(TARGETS, 2):
        print(f"\n=== {A_name} vs {B_name} ===")
        r = pair_analysis(A_name, data[A_name], B_name, data[B_name], args.bootstrap)
        results.append(r)
        for arm in ["q_only", "q_resp_only"]:
            m = r[arm]
            print(f"  {arm:12s}  SE-gap corr={m['se_gap_corr']:+.3f} "
                  f"[{m['se_gap_corr_ci']['lo95']:+.3f},{m['se_gap_corr_ci']['hi95']:+.3f}]  "
                  f"top-quartile={m['se_gap_corr_top_quartile']}  "
                  f"pair_acc(n={m['n_divergent_correctness']})="
                  f"{m['pair_accuracy_on_divergent']}  dP_std={m['dP_std']:.4f}  "
                  f"dP~0 frac={m['dP_near_zero_frac']:.3f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
