"""E53b -- zero-shot eval of the E53 Qwen/Gemma-pooled `q_resp_only` proxy on Llama-2 and Mistral
SQuAD (n1000).

Simultaneous cross-family + cross-dataset transfer test: the proxy was trained ONLY on
Qwen3-8B / Qwen3.5-9B / gemma-7b-it / gemma-2-9b-it trivia_qa question+response text
(`e53_train_qwengemma_deploy.py`, `q_resp_only` arm -- no hidden states, no alignment). Here it is
scored with zero retraining on Llama-2's and Mistral's existing `*_squad_n1000_full` records
(never Llama-2/Mistral, never SQuAD).

Mirrors `e53_eval_on_llama2_mistral.py` (the TriviaQA reverse-transfer eval) as closely as
possible, only swapping `dataset="trivia_qa"` -> `"squad"`, and adds:
  - SEP computed on-the-fly via `se_fidelity_proxy_vs_sep.compute_sep` (trivia-fit -> squad,
    E41/E36 fixed TBG layer), exactly as `correctness_eval_lolo_squad.py` does, so we get SEP's
    per-row predictions -> Spearman-vs-SE, AUROC_incorrect, and a paired bootstrap proxy-SEP.
    Cross-checked against the canonical `results/sep_reference_values.json[squad_ood_n1000]`.
  - the existing comparable SQuAD ridge result (pooled 3-source aligned ridge, trivia-fit ->
    squad) read from `results/correctness_eval_e41_ood_fixedlayer.json` -- context only, carried
    along verbatim (not recomputed).

Env: amortized_stage2(_v5) (GPU, proxy forward pass). Run from the repo root:
    python -m amortized_ue.e53b_eval_on_llama2_mistral_squad --data_dir /data2/mn1025/stage1
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
from amortized_ue.se_fidelity_proxy_vs_sep import compute_sep
from amortized_ue.e53_train_qwengemma_deploy import ARM, NEW_MODELS, DEFAULT_CKPT_DIR, DEFAULT_DATA_DIR

TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]
NUM_SAMPLES = 1000
DATASET = "squad"
SEP_LAYER = {"Llama-2-7b-chat": 30, "Mistral-7B-Instruct-v0.2": 31}   # E41/E36 fixed TBG layer
HERE = os.path.dirname(os.path.abspath(__file__))
SEP_REF_PATH = os.path.join(HERE, "results", "sep_reference_values.json")
SEP_REF_SETTING = "squad_ood_n1000"
RIDGE_CTX_PATH = os.path.join(HERE, "results", "correctness_eval_e41_ood_fixedlayer.json")


def load_sep_reference(target: str) -> dict:
    with open(SEP_REF_PATH) as f:
        return json.load(f)["targets"][target]["settings"][SEP_REF_SETTING]


def load_ridge_context(target: str) -> dict:
    """Existing comparable SQuAD ridge: pooled 3-source aligned ridge, trivia-fit -> squad
    (E39/E41). Context only -- needs full target access + alignment, not a zero-shot peer."""
    with open(RIDGE_CTX_PATH) as f:
        m = json.load(f)[target]["metrics"]["ridge_z"]
    return {"auroc_incorrect": m["auroc_incorrect"], "spearman_se": m["spearman_se"],
            "provenance": m["provenance"], "source": os.path.relpath(RIDGE_CTX_PATH)}


def eval_target(target: str, ckpt_dir: str, data_dir: str, bootstrap: int) -> dict:
    cfg = Stage1Config(model_name=target, dataset=DATASET, num_samples=NUM_SAMPLES, output_dir=data_dir)
    recs = load_records(cfg)
    ids = sorted(recs.keys())
    n = len(ids)
    y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=float)
    acc_map = load_accuracy(cfg)
    assert set(ids).issubset(acc_map), f"{target}: ids missing from accuracy manifest"
    acc = np.array([acc_map[i] for i in ids], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    print(f"\n=== {target}  SQuAD  N={n}  mean_acc={acc.mean():.3f}  incorrect_rate={incorrect.mean():.3f} ===")

    print(f"  running Qwen/Gemma-pooled-deploy proxy arm={ARM} on {target} SQuAD "
          f"(zero-shot -- never this model, never SQuAD) ...")
    mp = arm_preds(ARM, target, DATASET, NUM_SAMPLES, ckpt_dir=ckpt_dir, data_dir=data_dir)
    pred = np.array([mp[i] for i in ids], dtype=float)

    # SEP: trivia-fit -> squad, E41/E36 fixed TBG layer (same recipe as correctness_eval_lolo_squad.py)
    layer = SEP_LAYER[target]
    sep = compute_sep(target, eval_dataset=DATASET, eval_num_samples=NUM_SAMPLES, data_dir=data_dir,
                      eval_data_dir=data_dir, fit_num_samples=2000, use_test_split_as_eval=False,
                      layer=layer)
    sep_col = {i: c for c, i in enumerate(sep["ids"])}
    assert set(ids).issubset(sep_col), f"{target}: SEP missing ids"
    sep_pred = np.array([sep["pred"][sep_col[i]] for i in ids], dtype=float)

    sep_sp = float(spearmanr(sep_pred, y_se).correlation)
    sep_auroc_inc = float(roc_auc_score(incorrect, sep_pred))
    sep_ref = load_sep_reference(target)
    # cross-check the fresh recompute against the canonical reference file
    if abs(sep_ref["spearman"] - sep_sp) > 1e-3:
        print(f"  [warn] fresh SEP Spearman {sep_sp:+.4f} vs reference {sep_ref['spearman']:+.4f}")

    ridge_ctx = load_ridge_context(target)

    preds = {"true_semantic_entropy": y_se, "proxy_q_resp_only": pred, "sep": sep_pred}
    rng = np.random.default_rng(0)
    preds["random"] = rng.random(n)

    sp_vs_se = float(spearmanr(pred, y_se).correlation)

    metrics = {}
    print(f"\n  {'predictor':24s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}{'acc@.50':>8s}{'rho_SE':>9s}")
    for name, s in preds.items():
        au = float(roc_auc_score(incorrect, s))
        ap = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        rho = float(spearmanr(s, y_se).correlation)
        metrics[name] = {"auroc_incorrect": au, "auprc_incorrect": ap, "prr": pr,
                         "accuracy_coverage": {str(c): cov[c] for c in COVERAGES}, "spearman_se": rho}
        print(f"  {name:24s}{au:>10.3f}{ap:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}{cov[0.5]:>8.3f}{rho:>9.3f}")

    print(f"\n  Spearman(proxy, true SE)      = {sp_vs_se:+.3f}")
    print(f"  Spearman(SEP,   true SE)      = {sep_sp:+.3f}   (ref {sep_ref['spearman']:+.3f})")
    print(f"  proxy AUROC_inc {metrics['proxy_q_resp_only']['auroc_incorrect']:.3f} | "
          f"true-SE {metrics['true_semantic_entropy']['auroc_incorrect']:.3f} | "
          f"SEP {sep_auroc_inc:.3f} | ridge(ctx) {ridge_ctx['auroc_incorrect']:.3f}")

    boot = paired_bootstrap_auc(
        {"proxy": pred, "true_semantic_entropy": y_se, "sep": sep_pred, "random": preds["random"]},
        incorrect, B=bootstrap)
    deltas = {}
    for a, b in [("proxy", "true_semantic_entropy"), ("proxy", "sep"), ("proxy", "random")]:
        c = ci(boot[a] - boot[b])
        excl = c["lo95"] > 0 or c["hi95"] < 0
        deltas[f"{a}_minus_{b}"] = {**c, "ci_excludes_zero": bool(excl)}
        print(f"  Delta({a} - {b}) AUROC_inc = {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
              f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": target, "dataset": DATASET, "n_test": n, "mean_accuracy": float(acc.mean()),
            "positive_rate_incorrect": float(incorrect.mean()),
            "spearman_proxy_vs_true_se": sp_vs_se,
            "sep": {"layer": ["TBG", layer], "selection": sep["selection"],
                    "spearman_vs_se": sep_sp, "auroc_incorrect": sep_auroc_inc,
                    "reference": sep_ref, "reference_setting": SEP_REF_SETTING},
            "ridge_context": ridge_ctx,
            "metrics": metrics, "bootstrap_deltas": deltas}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/e53b_qwengemma_deploy_qresp_on_llama2_mistral_squad.json")
    args = p.parse_args()

    results = {}
    for t in args.targets:
        results[t] = eval_target(t, args.ckpt_dir, args.data_dir, args.bootstrap)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"ckpt_dir": args.ckpt_dir, "data_dir": args.data_dir,
                       "train_models": NEW_MODELS, "arm": ARM, "dataset": DATASET,
                       "results": results}, f, indent=2)
        print(f"  -> wrote {args.out} ({len(results)} target(s))")

    print(f"\n{'='*90}\nSUMMARY -- E53b Qwen/Gemma proxy zero-shot on Llama-2/Mistral SQuAD\n{'='*90}")
    print(f"{'target':22s}{'proxy AUROC':>12s}{'trueSE AUROC':>13s}{'SEP AUROC':>11s}"
          f"{'proxy rho':>11s}{'SEP rho':>9s}{'ridge rho':>11s}")
    for t, r in results.items():
        m = r["metrics"]
        print(f"{t:22s}{m['proxy_q_resp_only']['auroc_incorrect']:>12.3f}"
              f"{m['true_semantic_entropy']['auroc_incorrect']:>13.3f}"
              f"{r['sep']['auroc_incorrect']:>11.3f}{r['spearman_proxy_vs_true_se']:>11.3f}"
              f"{r['sep']['spearman_vs_se']:>9.3f}{r['ridge_context']['spearman_se']:>11.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
