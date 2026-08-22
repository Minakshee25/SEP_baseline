"""Pull concrete illustrative examples for E46: questions where Qwen3-8B and gemma-7b-it
genuinely disagree on correctness, and the zero-shot q_resp_only proxy correctly identified
which one was more likely wrong. Additive, read-only, no retraining. Prints to stdout + JSON.
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
    q_resp = arm_preds("q_resp_only", target, "trivia_qa", 1000, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
    return recs, acc, q_resp


def main():
    print(f"loading {A_NAME} ...")
    recA, accA, predA = load(A_NAME)
    print(f"loading {B_NAME} ...")
    recB, accB, predB = load(B_NAME)

    common = sorted(set(recA) & set(recB))
    rows = []
    for i in common:
        a_correct, b_correct = accA[i] >= 0.5, accB[i] >= 0.5
        if a_correct == b_correct:
            continue  # not a disagreement case
        dP = predA[i] - predB[i]              # >0 means proxy thinks A more uncertain
        # proxy is "right" if it points at whichever one was actually wrong
        a_more_uncertain_actual = not a_correct   # A is the wrong one
        proxy_says_a_more_uncertain = dP > 0
        correct_call = proxy_says_a_more_uncertain == a_more_uncertain_actual
        rows.append({
            "id": i, "question": recA[i]["question"],
            "A_response": recA[i]["canonical"]["response"], "A_correct": bool(a_correct),
            "A_se": recA[i]["labels"]["cluster_assignment_entropy"], "A_proxy_score": float(predA[i]),
            "B_response": recB[i]["canonical"]["response"], "B_correct": bool(b_correct),
            "B_se": recB[i]["labels"]["cluster_assignment_entropy"], "B_proxy_score": float(predB[i]),
            "dP": float(dP), "proxy_correct_call": bool(correct_call),
        })

    n = len(rows)
    n_right = sum(r["proxy_correct_call"] for r in rows)
    print(f"\n{n} divergent rows, proxy got {n_right}/{n} = {n_right/n:.3f} right")

    # pick the clearest illustrative cases: proxy correct, big |dP|, and the wrong model also had
    # high SE (uncertain) while the right model had low SE (confident) -- the intuitive story.
    clean = [r for r in rows if r["proxy_correct_call"]]
    clean.sort(key=lambda r: -abs(r["dP"]))
    top = clean[:8]

    print("\n=== TOP CLEAR EXAMPLES (proxy correctly identified the more-uncertain/wrong model) ===")
    for r in top:
        wrong_model, right_model = (A_NAME, B_NAME) if not r["A_correct"] else (B_NAME, A_NAME)
        wrong_resp = r["A_response"] if not r["A_correct"] else r["B_response"]
        right_resp = r["B_response"] if not r["A_correct"] else r["A_response"]
        wrong_se = r["A_se"] if not r["A_correct"] else r["B_se"]
        right_se = r["B_se"] if not r["A_correct"] else r["A_se"]
        print(f"\nQ: {r['question']}")
        print(f"  {right_model} (CORRECT): \"{right_resp}\"  [true SE={right_se:.3f}]")
        print(f"  {wrong_model} (WRONG):   \"{wrong_resp}\"  [true SE={wrong_se:.3f}]")
        print(f"  proxy dP={r['dP']:+.3f} (correctly points at {wrong_model} as more uncertain)")

    out = {"pair": f"{A_NAME} vs {B_NAME}", "n_divergent": n, "n_proxy_correct": n_right,
           "accuracy": n_right / n, "examples": top, "all_rows": rows}
    with open("amortized_ue/results/e46_examples.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote amortized_ue/results/e46_examples.json")


if __name__ == "__main__":
    main()
