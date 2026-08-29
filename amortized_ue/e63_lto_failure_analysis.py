"""E63 follow-up — failure analysis of the leave-TWO-out proxy on the DeepSeek vs Qwen3-8B
disagreement test. Read-only post-processing of the already-committed
`results/e63_lto_disagreement_table.json` (no GPU, no retraining).

Splits the 294 mis-called disagreements into:
  * OPPOSITE-DIRECTION errors (predicted_diff sign != true_diff sign, predicted_diff != 0) --
    the proxy actively pointed at the wrong model.
  * NO-DIRECTION cases (predicted_diff exactly 0) -- both models produced the IDENTICAL canonical
    answer string, so the text proxy sees the same input for both and cannot distinguish them,
    even though their sampled SE differed. Scored 0.5 (coin flip) by the E63 [a] sign-agreement
    convention; scored as a miss by the strict hit-rate.

Also the failure count per |true_diff| quartile, and the full failure question lists.

    python -m amortized_ue.e63_lto_failure_analysis
"""
from __future__ import annotations

import os
import json

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
TABLE = os.path.join(_HERE, "results", "e63_lto_disagreement_table.json")
OUT = os.path.join(_HERE, "results", "e63_lto_failures.json")


def _slim(r):
    return {k: r[k] for k in ("id", "question", "deepseek_response", "qwen3_8b_response",
                              "deepseek_true_se", "qwen3_8b_true_se", "deepseek_correct",
                              "qwen3_8b_correct", "true_diff", "predicted_diff", "abs_true_diff")}


def main():
    d = json.load(open(TABLE))
    rows = d["rows"]
    tie = [r for r in rows if r["correct"] is None]                 # true_diff == 0
    nontie = [r for r in rows if r["correct"] is not None]
    hit = [r for r in nontie if r["correct"] is True]
    miss = [r for r in nontie if r["correct"] is False]
    no_dir = [r for r in miss if r["predicted_diff"] == 0.0]
    opp = [r for r in miss if r["predicted_diff"] != 0.0]
    # cross-check: no-direction rows must have identical canonical answers
    id_ans = [r for r in no_dir if r["deepseek_response"].strip() == r["qwen3_8b_response"].strip()]

    dP = np.array([r["predicted_diff"] for r in nontie])
    dY = np.array([r["true_diff"] for r in nontie])
    h = np.where(dP == 0, 0.5, (np.sign(dP) == np.sign(dY)).astype(float))
    ag = np.abs(dY)
    order = np.argsort(ag)
    n = len(order)
    quart = []
    for lab, lo, hi in [("Q4_largest", 3, 4), ("Q3", 2, 3), ("Q2", 1, 2), ("Q1_smallest", 0, 1)]:
        idx = order[lo * n // 4:hi * n // 4]
        real = idx[dY[idx] != 0]
        fails = int(sum(1 for i in real if np.sign(dP[i]) != np.sign(dY[i])))
        quart.append({"quartile": lab, "abs_true_diff_range": [float(ag[idx].min()), float(ag[idx].max())],
                      "n_real_disagreements": int(len(real)), "n_failures": fails,
                      "hit_rate": float(1 - fails / len(real)) if len(real) else None})

    opp.sort(key=lambda r: -r["abs_true_diff"])
    no_dir.sort(key=lambda r: -r["abs_true_diff"])

    summary = {
        "n_shared_questions": len(rows),
        "n_tied_true_se_nothing_to_predict": len(tie),
        "n_real_disagreements": len(nontie),
        "n_direction_correct": len(hit),
        "n_failures_total": len(miss),
        "n_failures_opposite_direction": len(opp),
        "n_failures_no_direction_predicted_diff_zero": len(no_dir),
        "no_direction_rows_with_identical_canonical_answer": len(id_ans),
        "sign_agreement_predtie_half": float(h.mean()),
        "strict_hit_rate_predtie_as_miss": len(hit) / len(nontie),
        "failure_rate_total": len(miss) / len(nontie),
        "per_quartile": quart,
    }

    print("=" * 78)
    print("E63 failure analysis — DeepSeek vs Qwen3-8B leave-TWO-out disagreement test")
    print("=" * 78)
    for k, v in summary.items():
        if k != "per_quartile":
            print(f"  {k:52s} {v}")
    print("\n  per |true_diff| quartile:")
    for q in quart:
        print(f"    {q['quartile']:12s} |dY| [{q['abs_true_diff_range'][0]:.3f}, {q['abs_true_diff_range'][1]:.3f}]  "
              f"failures {q['n_failures']:3d}/{q['n_real_disagreements']:3d}  hit-rate {q['hit_rate']:.3f}")

    print("\n  top 10 OPPOSITE-DIRECTION failures (proxy pointed at the wrong model), by |true_diff|:")
    for r in opp[:10]:
        hi = "DeepSeek" if r["deepseek_true_se"] > r["qwen3_8b_true_se"] else "Qwen3-8B"
        print(f"    Q: {r['question']}")
        print(f"       DeepSeek {r['deepseek_response']!r} SE={r['deepseek_true_se']:.2f} | "
              f"Qwen3-8B {r['qwen3_8b_response']!r} SE={r['qwen3_8b_true_se']:.2f}")
        print(f"       true gap {r['true_diff']:+.2f} (more uncertain: {hi})  "
              f"proxy said {r['predicted_diff']:+.2f}  -> WRONG WAY")

    print("\n  top 10 NO-DIRECTION failures (identical canonical answer, proxy predicted_diff = 0):")
    for r in no_dir[:10]:
        print(f"    Q: {r['question']}")
        print(f"       both answered {r['deepseek_response']!r}  |  "
              f"DeepSeek SE={r['deepseek_true_se']:.2f}  Qwen3-8B SE={r['qwen3_8b_true_se']:.2f}  "
              f"(true gap {r['true_diff']:+.2f})")

    out = {
        "source_table": os.path.relpath(TABLE, _HERE),
        "diff_convention": "A - B  where A = deepseek-llm-7b-chat, B = Qwen3-8B",
        "summary": summary,
        "failures_opposite_direction": [_slim(r) for r in opp],
        "failures_no_direction_identical_answer": [_slim(r) for r in no_dir],
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT}  ({len(opp)} opposite-direction + {len(no_dir)} no-direction failure rows)")


if __name__ == "__main__":
    main()
