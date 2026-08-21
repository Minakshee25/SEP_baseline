"""E45 -- zero-shot flagship-proxy transfer test on Qwen/Gemma (additive, no retraining).

Scores the DEPLOY proxy (frozen Llama-3.2-3B + LoRA, trained on the pooled Llama-2/Mistral/
Llama-3/DeepSeek n2000 -- `amortized_ue/results/deploy_checkpoints/`, text-only arms) on 4 NEW
target-LLM families it never saw during training: Qwen3-8B, Qwen3.5-9B, gemma-7b-it,
gemma-2-9b-it. Uses each target's existing n1000 eval records (`*_trivia_qa_n1000_full`,
shared question ids, already on /data2 -- built for E44, zero overlap with the n2000 the deploy
proxy trained on, verified earlier this session).

Only the TEXT arms (q_only, q_resp_only) are run: they need no target hidden states, no
Procrustes alignment, no per-target fitting -- the sharpest, cheapest test of whether the
model-agnostic pathway (E20-23, E37-39) extends to genuinely new architectures/training data,
not just the 4 already-related targets. z-arms are deferred to a follow-up once alignment
(Procrustes W) is fit for these families.

Mirrors mistral_trained_proxy_ood.py's structure (arm_preds + correctness_eval helpers) but
computes its OWN baselines (true SE, random) directly instead of reusing a frozen file, since
these targets have no prior baseline on record.

Env: `amortized_stage2` + a free GPU (proxy forward pass). Run from the repo root:
    python -m amortized_ue.e45_qwen_gemma_zeroshot
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import (
    load_accuracy, accuracy_coverage, prediction_rejection_ratio, paired_bootstrap_auc, ci, COVERAGES)
from amortized_ue.procrustes_e27_rank_fusion import arm_preds

TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"  # copied off NFS (see amortized_ue/CLAUDE.md NFS note)
NUM_SAMPLES = 1000


def eval_target(target: str, bootstrap: int) -> dict:
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
    recs = load_records(cfg)
    ids = sorted(recs.keys())
    n = len(ids)
    y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=float)
    acc_map = load_accuracy(cfg)
    assert set(ids).issubset(acc_map), f"{target}: ids missing from accuracy manifest"
    acc = np.array([acc_map[i] for i in ids], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    print(f"\n=== {target}  N={n}  mean_acc={acc.mean():.3f}  incorrect_rate={incorrect.mean():.3f} ===")

    preds = {"true_semantic_entropy": y_se}
    rng = np.random.default_rng(0)
    preds["random"] = rng.random(n)

    for arm in ["q_only", "q_resp_only"]:
        print(f"  running deploy proxy arm={arm} on {target} (zero-shot, never trained on this family) ...")
        mp = arm_preds(arm, target, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
        preds[arm] = np.array([mp[i] for i in ids], dtype=float)

    metrics = {}
    print(f"\n  {'predictor':22s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}{'acc@.50':>8s}")
    for name, s in preds.items():
        au = float(roc_auc_score(incorrect, s))
        ap = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        metrics[name] = {"auroc_incorrect": au, "auprc_incorrect": ap, "prr": pr,
                          "accuracy_coverage": {str(c): cov[c] for c in COVERAGES}}
        print(f"  {name:22s}{au:>10.3f}{ap:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}{cov[0.5]:>8.3f}")

    boot = paired_bootstrap_auc(
        {"q_resp_only": preds["q_resp_only"], "q_only": preds["q_only"],
         "true_semantic_entropy": preds["true_semantic_entropy"]},
        incorrect, B=bootstrap)
    deltas = {}
    for a, b in [("q_resp_only", "true_semantic_entropy"), ("q_resp_only", "q_only")]:
        c = ci(boot[a] - boot[b])
        deltas[f"{a}_minus_{b}"] = c
        excl = c["lo95"] > 0 or c["hi95"] < 0
        print(f"  Delta({a} - {b}) = {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
              f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": target, "n_test": n, "mean_accuracy": float(acc.mean()),
            "positive_rate_incorrect": float(incorrect.mean()), "metrics": metrics,
            "bootstrap_deltas": deltas}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/e45_qwen_gemma_zeroshot.json")
    args = p.parse_args()

    results = {t: eval_target(t, args.bootstrap) for t in args.targets}

    print(f"\n{'='*70}\nSUMMARY (AUROC_incorrect)\n{'='*70}")
    print(f"{'target':16s}{'true_SE':>10s}{'q_only':>10s}{'q_resp_only':>12s}")
    for t, r in results.items():
        m = r["metrics"]
        print(f"{t:16s}{m['true_semantic_entropy']['auroc_incorrect']:>10.3f}"
              f"{m['q_only']['auroc_incorrect']:>10.3f}{m['q_resp_only']['auroc_incorrect']:>12.3f}")

    with open(args.out, "w") as f:
        json.dump({"deploy_ckpt": DEPLOY_CKPT, "data_dir": DATA_DIR, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
