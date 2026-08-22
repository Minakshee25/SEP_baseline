"""E47 -- SE-FIDELITY (not correctness) of the zero-shot deploy proxy on Qwen/Gemma (additive).

E45 measured whether q_resp_only detects WRONG ANSWERS (AUROC vs correctness). E46 measured
whether it detects WHICH model is more uncertain (pairwise, on the SE gap). Neither directly
answers "how well does the proxy's raw score track the actual continuous SE value" -- the
question this script answers, matching the project's primary SE-fidelity metric (Spearman
rho(pred, true_SE), used throughout E12-E37) but for the 4 new Qwen/Gemma targets, zero-shot.

Per E31's established finding (SE-fidelity != correctness), expect this number to differ from
E45's AUROC_incorrect -- that gap IS the point, not noise.

Env: `amortized_stage2_v5` + free GPU. Run from the repo root:
    /data2/mn1025/conda_envs/amortized_stage2_v5/bin/python -m amortized_ue.e47_qwen_gemma_se_fidelity
"""
from __future__ import annotations

import json
import numpy as np
from scipy.stats import spearmanr

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.procrustes_e27_rank_fusion import arm_preds

TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
NUM_SAMPLES = 1000


def rho(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def boot_ci(a, b, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(a)
    vals = [rho(a[idx], b[idx]) for idx in (rng.integers(0, n, n) for _ in range(B))]
    return {"mean": float(np.mean(vals)), "lo95": float(np.percentile(vals, 2.5)),
            "hi95": float(np.percentile(vals, 97.5))}


def main():
    results = {}
    print(f"{'target':16s}{'q_only rho':>12s}{'q_resp_only rho':>18s}{'95% CI':>24s}")
    for t in TARGETS:
        cfg = Stage1Config(model_name=t, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
        recs = load_records(cfg)
        ids = sorted(recs.keys())
        y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids])

        q_only = arm_preds("q_only", t, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
        q_resp = arm_preds("q_resp_only", t, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
        p_only = np.array([q_only[i] for i in ids])
        p_resp = np.array([q_resp[i] for i in ids])

        r_only = rho(p_only, y_se)
        r_resp = rho(p_resp, y_se)
        ci = boot_ci(p_resp, y_se)

        print(f"{t:16s}{r_only:>12.3f}{r_resp:>18.3f}   [{ci['lo95']:+.3f},{ci['hi95']:+.3f}]")
        results[t] = {"n": len(ids), "q_only_rho": r_only, "q_resp_only_rho": r_resp,
                      "q_resp_only_rho_ci": ci, "mean_se": float(y_se.mean()), "std_se": float(y_se.std())}

    with open("amortized_ue/results/e47_qwen_gemma_se_fidelity.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote amortized_ue/results/e47_qwen_gemma_se_fidelity.json")


if __name__ == "__main__":
    main()
