"""Concrete examples for E47: for questions where Qwen3-8B and gemma-7b-it's TRUE semantic
entropy diverges sharply (one confident, one not), show the proxy's predicted SE alongside the
true SE, plus correctness. Additive, read-only.

IMPORTANT SCALE CAVEAT: the proxy's raw output is a normalized score, not literal SE units (the
deploy checkpoint is missing its original train-time decode stats -- see E45's checkpoint.py
identity-transform fallback). For readability here ONLY, each model's raw proxy scores are
rescaled (z-score then re-scale) onto THAT SAME model's own true-SE mean/std, so the numbers sit
in comparable units. This is a display convenience computed WITH eval-time information the proxy
itself never sees -- it makes the rank-correct predictions readable in nats, it does NOT mean the
proxy is calibrated in an absolute sense. Rank order and relative gaps are the trustworthy part
(matches E47's Spearman framing); the exact decoded number is illustrative only.
"""
from __future__ import annotations

import json
import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import load_accuracy
from amortized_ue.procrustes_e27_rank_fusion import arm_preds

DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
A_NAME, B_NAME = "Qwen3-8B", "gemma-7b-it"


def load(target):
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000, output_dir=DATA_DIR)
    recs = load_records(cfg)
    acc = load_accuracy(cfg)
    raw = arm_preds("q_resp_only", target, "trivia_qa", 1000, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    ids = sorted(recs.keys())
    y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids])
    p_raw = np.array([raw[i] for i in ids])
    # rescale proxy's raw score onto THIS model's own true-SE mean/std (readability only, see docstring)
    p_z = (p_raw - p_raw.mean()) / (p_raw.std() + 1e-8)
    p_rescaled = p_z * y_se.std() + y_se.mean()
    pred_se = dict(zip(ids, p_rescaled))
    true_se = dict(zip(ids, y_se))
    return recs, acc, pred_se, true_se


def main():
    print(f"loading {A_NAME} ...")
    recA, accA, predA, trueA = load(A_NAME)
    print(f"loading {B_NAME} ...")
    recB, accB, predB, trueB = load(B_NAME)

    common = sorted(set(recA) & set(recB))
    rows = []
    for i in common:
        gap = trueA[i] - trueB[i]
        rows.append({
            "id": i, "question": recA[i]["question"], "true_se_gap": gap,
            "A_response": recA[i]["canonical"]["response"], "A_correct": bool(accA[i] >= 0.5),
            "A_true_se": trueA[i], "A_pred_se": predA[i],
            "B_response": recB[i]["canonical"]["response"], "B_correct": bool(accB[i] >= 0.5),
            "B_true_se": trueB[i], "B_pred_se": predB[i],
        })
    rows.sort(key=lambda r: -abs(r["true_se_gap"]))
    top = rows[:10]

    print(f"\n=== TOP 10 by |true SE gap| between {A_NAME} and {B_NAME} ===\n")
    for r in top:
        a_err = r["A_pred_se"] - r["A_true_se"]
        b_err = r["B_pred_se"] - r["B_true_se"]
        print(f"Q: {r['question']}")
        print(f"  {A_NAME:14s} response=\"{r['A_response']}\"  correct={r['A_correct']}")
        print(f"  {'':14s} true SE={r['A_true_se']:.3f}   proxy predicted SE={r['A_pred_se']:.3f}   (error={a_err:+.3f})")
        print(f"  {B_NAME:14s} response=\"{r['B_response']}\"  correct={r['B_correct']}")
        print(f"  {'':14s} true SE={r['B_true_se']:.3f}   proxy predicted SE={r['B_pred_se']:.3f}   (error={b_err:+.3f})")
        print()

    with open("amortized_ue/results/e47_examples.json", "w") as f:
        json.dump(top, f, indent=2)
    print("wrote amortized_ue/results/e47_examples.json")


if __name__ == "__main__":
    main()
