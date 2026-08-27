"""Diagnostic: which cheap supervision signal best flags a WRONG canonical answer?

Semantic entropy (the Stage-2 training target) needs an entailment-clustered
10-sample set. This script asks how it compares, as a wrong-answer detector, to
three signals that fall straight out of the same Stage-1 records at no extra cost:

  SE                 -- labels.cluster_assignment_entropy (the continuous label)
  SE_binary          -- SE thresholded via stage2.data.best_split (the SEP/Stage-2
                        binarisation; threshold reused, not reinvented)
  n_clusters         -- labels.n_clusters (raw integer, no transform)
  MC_sequence_entropy-- mean over the 10 high-temp samples of each sample's
                        length-normalised mean token log-likelihood, negated
  perplexity         -- exp(-mean(canonical.token_log_likelihoods))

Each is scored by AUROC against `incorrect` (1 - binarised canonical.accuracy),
on the FULL n2000 set (no split). Read-only over Stage-1: no GPU, no target-LLM
calls, no new generation, nothing under semantic_uncertainty/ touched.

Loading / ordering follows linear_ceiling_probe.py (load_records, id-sorted).

Run from the repo root in the `se_probes` env:
    python -m amortized_ue.supervision_signal_compare
    python -m amortized_ue.supervision_signal_compare --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import json
import argparse
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
# Reuse the exact paired-bootstrap convention from E25/E26/E31/E38 (shared resample
# indices across all predictors, Mann-Whitney fast_auc, B=10000, seed=0).
from amortized_ue.correctness_eval import paired_bootstrap_auc, ci
# Reuse the SEP / Stage-2 entropy binarisation verbatim (don't reinvent the threshold).
from amortized_ue.stage2.data import best_split, binarize_entropy

warnings.filterwarnings("ignore")

BOOTSTRAP_B = 10000
BOOTSTRAP_SEED = 0
# deltas reported per target: (a - b), CI checked against zero
DELTAS = [("SE", "SE_binary"), ("SE", "n_clusters"),
          ("SE", "MC_sequence_entropy"), ("SE", "perplexity")]

# canonical.accuracy is the squad (F1-based) metric; on Llama-2/trivia_qa it comes
# out already binary {0,1}. The repo's fixed convention for turning it into a
# correctness label (correctness_eval.py) is a 0.5 threshold, NOT best_split
# (best_split is only ever applied to entropy). We follow that convention and
# assert the data really is binary so the threshold is a no-op.
ACC_THRESHOLD = 0.5


def mean_or_nan(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--data_dir", default=None,
                   help="Stage-1 output_dir override (e.g. /data2/mn1025/stage1 to dodge the NFS)")
    p.add_argument("--out", default="amortized_ue/results/supervision_signal_compare_llama2_trivia.json")
    args = p.parse_args()

    cfg = Stage1Config(
        model_name=args.model_name, dataset=args.dataset, num_samples=args.num_samples,
        **({"output_dir": args.data_dir} if args.data_dir else {}),
    )
    records = load_records(cfg)
    ids = sorted(records.keys())                      # same ordering as linear_ceiling_probe
    print(f"\n{args.model_name} / {args.dataset}  n={len(ids)}  from {cfg.run_dir()}")

    se = np.empty(len(ids), dtype=np.float64)
    n_clusters = np.empty(len(ids), dtype=np.float64)
    mc_seq_entropy = np.empty(len(ids), dtype=np.float64)
    perplexity = np.empty(len(ids), dtype=np.float64)
    acc = np.empty(len(ids), dtype=np.float64)

    for k, i in enumerate(ids):
        r = records[i]
        se[k] = r["labels"]["cluster_assignment_entropy"]
        n_clusters[k] = r["labels"]["n_clusters"]

        # MC sequence entropy: per-sample length-normalised mean log-likelihood,
        # averaged over the 10 samples, then negated (high => uncertain).
        per_sample_mean_ll = [mean_or_nan(s["token_log_likelihoods"]) for s in r["samples"]]
        mc_seq_entropy[k] = -float(np.mean(per_sample_mean_ll))

        # Perplexity of the canonical answer.
        perplexity[k] = float(np.exp(-mean_or_nan(r["canonical"]["token_log_likelihoods"])))

        acc[k] = r["canonical"]["accuracy"]

    uniq = np.unique(np.round(acc, 6))
    is_binary = set(uniq.tolist()).issubset({0.0, 1.0})
    assert is_binary, f"canonical.accuracy is not binary (unique={uniq[:10]}); revisit thresholding"
    incorrect = (acc < ACC_THRESHOLD).astype(int)      # 1 == wrong canonical answer
    incorrect_rate = float(incorrect.mean())
    print(f"accuracy already binary {{0,1}}; threshold {ACC_THRESHOLD} -> "
          f"incorrect rate {incorrect_rate:.4f}")

    # SE_binary: threshold SE with stage2.data.best_split (fit on this same n2000 SE
    # array, the only SE set this script loads), then binarise 0/1 via binarize_entropy.
    se_thr = best_split(torch.from_numpy(se))
    se_binary = binarize_entropy(torch.from_numpy(se), se_thr).numpy().astype(np.float64)
    assert not np.any(se_binary < 0), "binarize_entropy left ties (-1) at the threshold"
    print(f"SE_binary: best_split threshold {se_thr:.4f} -> positive rate {se_binary.mean():.4f}")

    scores = {
        "SE": se,
        "SE_binary": se_binary,
        "n_clusters": n_clusters,
        "MC_sequence_entropy": mc_seq_entropy,
        "perplexity": perplexity,
    }

    rows = {}
    print(f"\n{'signal':<22}{'AUROC':>9}{'N':>8}{'incorrect_rate':>16}")
    for name, s in scores.items():
        auroc = float(roc_auc_score(incorrect, s))
        rows[name] = {"auroc": auroc, "N": int(len(ids)), "incorrect_rate": incorrect_rate}
        print(f"{name:<22}{auroc:>9.4f}{len(ids):>8d}{incorrect_rate:>16.4f}")

    # ---- paired bootstrap: one set of resampled row indices, reused for all 5 signals ----
    boot = paired_bootstrap_auc(scores, incorrect, B=BOOTSTRAP_B, seed=BOOTSTRAP_SEED)
    deltas = {}
    print(f"\npaired bootstrap ({BOOTSTRAP_B} resamples, shared indices)")
    print(f"{'delta':<28}{'mean':>9}{'lo95':>9}{'hi95':>9}   excludes 0?")
    for a, b in DELTAS:
        d = ci(boot[a] - boot[b])
        excl = (d["lo95"] > 0.0) or (d["hi95"] < 0.0)
        deltas[f"{a} - {b}"] = {**d, "excludes_zero": bool(excl)}
        print(f"{a + ' - ' + b:<28}{d['mean']:>9.4f}{d['lo95']:>9.4f}{d['hi95']:>9.4f}   "
              f"{'YES' if excl else 'no'}")

    payload = {
        "model_name": args.model_name,
        "dataset": args.dataset,
        "num_samples": args.num_samples,
        "n": len(ids),
        "run_dir": cfg.run_dir(),
        "acc_threshold": ACC_THRESHOLD,
        "acc_is_binary": bool(is_binary),
        "incorrect_rate": incorrect_rate,
        "target": "incorrect (1 - canonical accuracy)",
        "eval_set": "full n2000, no split",
        "results": rows,
        "bootstrap": {
            "method": "paired_bootstrap_auc (amortized_ue.correctness_eval); shared resample "
                      "indices across all 5 signals; Mann-Whitney fast_auc",
            "se_binary_threshold": float(se_thr),
            "n_resamples": BOOTSTRAP_B,
            "seed": BOOTSTRAP_SEED,
            "deltas": deltas,
        },
    }
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote {args.out}\n")


if __name__ == "__main__":
    main()
