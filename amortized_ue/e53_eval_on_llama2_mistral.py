"""E53b — zero-shot eval of the E53 Qwen/Gemma-pooled q_resp_only proxy on Llama-2 and Mistral.

Reverse direction of E45 (which trained on Llama-2/Mistral/Llama-3/DeepSeek and tested
zero-shot on Qwen3-8B/Qwen3.5-9B/gemma-7b-it/gemma-2-9b-it). Here the proxy is trained ONLY on
those 4 Qwen/Gemma models' pooled question+response text (`e53_train_qwengemma_deploy.py`,
`q_resp_only` arm only — no hidden states, no alignment) and scored with zero retraining on
Llama-2's and Mistral's existing fresh n1000 trivia_qa records (`*_trivia_qa_n1000_full`,
disjoint from the n2000 sets any prior proxy trained on).

Mirrors e45_qwen_gemma_zeroshot.py's eval_target structure (arm_preds + correctness_eval
helpers), computing true SE / random baselines directly, plus Spearman(pred, true SE) for the
project's primary SE-fidelity metric. The SEP numbers for these two targets are carried along
for context — read from the canonical `results/sep_reference_values.json`
(`build_sep_reference.py`; independently re-verified from scratch on 2026-08-25), not recomputed
or hand-copied here.

Env: amortized_stage2 (GPU, proxy forward pass). Run from the repo root:
    python -m amortized_ue.e53_eval_on_llama2_mistral
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import (
    load_accuracy, accuracy_coverage, prediction_rejection_ratio, paired_bootstrap_auc, ci, COVERAGES)
from amortized_ue.procrustes_e27_rank_fusion import arm_preds
from amortized_ue.e53_train_qwengemma_deploy import ARM, NEW_MODELS, DEFAULT_CKPT_DIR, DEFAULT_DATA_DIR

TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]
NUM_SAMPLES = 1000
SEP_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sep_reference_values.json")
SEP_REF_SETTING = "trivia_qa_fresh_n1000"   # matches this script's eval regime (fresh disjoint n1000)


def load_sep_reference(target: str) -> dict:
    """Canonical SEP-vs-SE numbers (build_sep_reference.py) -- context only, not recomputed here."""
    with open(SEP_REF_PATH) as f:
        ref = json.load(f)
    return ref["targets"][target]["settings"][SEP_REF_SETTING]


def eval_target(target: str, ckpt_dir: str, data_dir: str, bootstrap: int) -> dict:
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=data_dir)
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

    print(f"  running Qwen/Gemma-pooled-deploy proxy arm={ARM} on {target} "
          f"(zero-shot — never trained on this model) ...")
    mp = arm_preds(ARM, target, "trivia_qa", NUM_SAMPLES, ckpt_dir=ckpt_dir, data_dir=data_dir)
    pred = np.array([mp[i] for i in ids], dtype=float)

    preds = {"true_semantic_entropy": y_se, "proxy_q_resp_only": pred}
    rng = np.random.default_rng(0)
    preds["random"] = rng.random(n)

    sp_vs_se = float(spearmanr(pred, y_se).correlation)
    sep_ref = load_sep_reference(target)

    metrics = {}
    print(f"\n  {'predictor':24s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}{'acc@.50':>8s}")
    for name, s in preds.items():
        au = float(roc_auc_score(incorrect, s))
        ap = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        metrics[name] = {"auroc_incorrect": au, "auprc_incorrect": ap, "prr": pr,
                          "accuracy_coverage": {str(c): cov[c] for c in COVERAGES}}
        print(f"  {name:24s}{au:>10.3f}{ap:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}{cov[0.5]:>8.3f}")
    print(f"  Spearman(proxy, true SE) = {sp_vs_se:+.3f}   "
          f"[for context, SEP ({sep_ref['position']}:{sep_ref['layer']}, {sep_ref['selection']}): "
          f"{sep_ref['spearman']:+.3f}]   (delta = {sp_vs_se - sep_ref['spearman']:+.3f})")

    boot = paired_bootstrap_auc(
        {"proxy": pred, "true_semantic_entropy": y_se, "random": preds["random"]},
        incorrect, B=bootstrap)
    deltas = {}
    for a, b in [("proxy", "true_semantic_entropy"), ("proxy", "random")]:
        c = ci(boot[a] - boot[b])
        deltas[f"{a}_minus_{b}"] = c
        excl = c["lo95"] > 0 or c["hi95"] < 0
        print(f"  Delta({a} - {b}) AUROC_inc = {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
              f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": target, "n_test": n, "mean_accuracy": float(acc.mean()),
            "positive_rate_incorrect": float(incorrect.mean()),
            "spearman_proxy_vs_true_se": sp_vs_se,
            "metrics": metrics, "bootstrap_deltas": deltas,
            "sep_reference": sep_ref, "sep_reference_setting": SEP_REF_SETTING,
            "sep_reference_source": os.path.relpath(SEP_REF_PATH)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/e53_qwengemma_deploy_qresp_on_llama2_mistral.json")
    args = p.parse_args()

    results = {t: eval_target(t, args.ckpt_dir, args.data_dir, args.bootstrap) for t in args.targets}

    print(f"\n{'='*78}\nSUMMARY (AUROC_incorrect / Spearman-vs-true-SE)\n{'='*78}")
    print(f"{'target':22s}{'proxy AUROC':>13s}{'true_SE AUROC':>15s}{'proxy rho':>11s}{'SEP rho':>9s}")
    for t, r in results.items():
        m = r["metrics"]
        print(f"{t:22s}{m['proxy_q_resp_only']['auroc_incorrect']:>13.3f}"
              f"{m['true_semantic_entropy']['auroc_incorrect']:>15.3f}"
              f"{r['spearman_proxy_vs_true_se']:>11.3f}"
              f"{r['sep_reference']['spearman']:>9.3f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ckpt_dir": args.ckpt_dir, "data_dir": args.data_dir,
                    "train_models": NEW_MODELS, "arm": ARM, "results": results}, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
