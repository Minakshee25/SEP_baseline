"""E42 — a proxy trained ONLY on Mistral's TriviaQA data, tested on Mistral's squad (additive).

Mirrors what the REFERENCE proxy already gives us for Llama-2 (correctness_eval_ood.py's
`reference_*` rows: Llama-2-trained proxy -> Llama-2's own squad = pure dataset-shift test), but for
Mistral, using the E22 role-swap proxy (`stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints`,
5 seeds, trained on Mistral trivia_qa n2000, arms z/z_q/z_q_resp/q_only/q_resp_only) -- built for the
E22/E23 role-swap experiment but never evaluated on squad. Only the TEXT arms are run here (q_only,
q_resp_only): the z arm is in Mistral's own native basis and squad-vs-trivia is a within-model
dataset shift only, so z COULD be run too, but E39/E41 already established the aligned-z pathway is
weak/not worth it relative to text (E33) -- kept text-only to match the REFERENCE row it's compared
against apples-to-apples.

Baselines are NOT recomputed -- SEP (E41-fixed layer) and true SE for Mistral/squad were already
fit+audited in `correctness_eval_e41_ood_fixedlayer.json` (E41); this script re-derives its own
squad ids/accuracy/SE labels independently (via `loaders.load_records`, not `load_matrix`) and
asserts them against the same audit convention before reusing the frozen baseline numbers, so the
comparison is honest without repeating an hour of SEP layer-fitting.

Env: `amortized_stage2` + a free GPU (proxy forward pass). Run from the repo root:
    python -m amortized_ue.mistral_trained_proxy_ood
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
from amortized_ue import exp2_run as E2

MISTRAL = "Mistral-7B-Instruct-v0.2"
MISTRAL_CKPT = "amortized_ue/stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints"
FROZEN_BASELINES = "amortized_ue/results/correctness_eval_e41_ood_fixedlayer.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/e42_mistral_trained_proxy_ood.json")
    args = p.parse_args()

    cfg = Stage1Config(model_name=MISTRAL, dataset="squad", num_samples=1000)
    recs = load_records(cfg)
    ids = sorted(recs.keys())
    n = len(ids)
    y_se = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=float)
    acc_map = load_accuracy(cfg)
    assert set(ids).issubset(acc_map), "squad ids missing from the accuracy manifest"
    acc = np.array([acc_map[i] for i in ids], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    print(f"Mistral squad: N={n}  mean_acc={acc.mean():.3f}  incorrect_rate={incorrect.mean():.3f}")

    # ---- id-mapping audit vs the frozen baselines (same convention as E38/E39/E41) ----------------
    with open(FROZEN_BASELINES) as f:
        frozen = json.load(f)[MISTRAL]
    assert frozen["n_test"] == n, f"row-count mismatch: frozen n_test={frozen['n_test']} vs {n}"
    # correctness_eval_ood loads squad ids via load_matrix; this script uses load_records directly.
    # Both must agree on which 1000 rows and their SE labels, or the two evals aren't comparable.
    from amortized_ue.linear_ceiling_probe import load_matrix
    _, y_lm, ids_lm = load_matrix(cfg, ["TBG"])
    assert ids_lm == ids, "id order differs between load_matrix and load_records"
    max_dev = float(np.max(np.abs(y_lm.astype(float) - y_se)))
    print(f"ID-MAPPING AUDIT: max|SE(load_records) - SE(load_matrix)| = {max_dev:.3e}  "
          f"({'MATCH' if max_dev < 1e-5 else 'MISMATCH -- STOP'})")
    assert max_dev < 1e-5

    # ---- Mistral-trained proxy (E22 checkpoints), text arms only -----------------------------------
    preds, provenance = {}, {}
    print("running Mistral-trained proxy (E22 ckpts) on Mistral squad ...")
    for arm in ["q_only", "q_resp_only"]:
        mp = arm_preds(arm, MISTRAL, "squad", 1000, ckpt_dir=MISTRAL_CKPT)
        preds[arm] = np.array([mp[i] for i in ids], dtype=float)
        provenance[arm] = "Mistral-trained proxy (E22, 5 seeds) -> Mistral squad (in-model, cross-dataset only)"
        print(f"  {arm}: done")

    # ---- frozen baselines, reused not recomputed ---------------------------------------------------
    for name in ["true_semantic_entropy", "sep_single_e36_layer", "sep_single_best_layer"]:
        m = frozen["metrics"][name]
        preds[name] = None                      # per-example preds not stored for these in E41; scalar reuse below
        provenance[name] = f"REUSED from {FROZEN_BASELINES} (E41): {m['provenance'] if 'provenance' in m else name}"

    preds["reference_q_resp_only"] = None        # Llama-2-trained proxy row, for 3-way comparison
    provenance["reference_q_resp_only"] = f"REUSED from {FROZEN_BASELINES} (E41)"

    # ---- score the Mistral-trained proxy's own predictions ------------------------------------------
    print(f"\n{'predictor':30s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}{'acc@.50':>8s}")
    metrics = {}
    for name in ["q_only", "q_resp_only"]:
        s = preds[name]
        au = float(roc_auc_score(incorrect, s))
        ap = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        metrics[name] = {"auroc_incorrect": au, "auprc_incorrect": ap, "prr": pr,
                          "accuracy_coverage": {str(c): cov[c] for c in COVERAGES},
                          "provenance": provenance[name]}
        print(f"{name:30s}{au:>10.3f}{ap:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}{cov[0.5]:>8.3f}")

    for name in ["true_semantic_entropy", "sep_single_e36_layer", "sep_single_best_layer",
                 "reference_q_resp_only"]:
        au = frozen["metrics"][name]["auroc_incorrect"]
        print(f"{name + ' [frozen]':30s}{au:>10.3f}")

    # ---- paired bootstrap: Mistral-trained proxy vs its OWN frozen SEP/SE ---------------------------
    # (paired bootstrap needs per-example preds for both sides; SEP/SE per-example preds for Mistral
    # aren't in the E41 json, only scalars -- so this bootstraps q_resp_only against q_only as an
    # internal check, and reports the SEP/SE deltas as point estimates only, flagged as such.)
    boot_local = paired_bootstrap_auc({"q_only": preds["q_only"], "q_resp_only": preds["q_resp_only"]},
                                       incorrect, B=args.bootstrap)
    c = ci(boot_local["q_resp_only"] - boot_local["q_only"])
    print(f"\nΔ(q_resp_only - q_only) = {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
          f"({'excludes 0' if c['lo95'] > 0 or c['hi95'] < 0 else 'includes 0'})")

    out = {
        "target": MISTRAL, "eval_dataset": "squad", "fit_dataset": "trivia_qa (Mistral only)",
        "n_test": n, "mean_accuracy": float(acc.mean()), "positive_rate_incorrect": float(incorrect.mean()),
        "id_mapping_max_dev": max_dev,
        "metrics": metrics,
        "frozen_baselines_reused_from": FROZEN_BASELINES,
        "frozen_baseline_auroc_incorrect": {n2: frozen["metrics"][n2]["auroc_incorrect"]
                                            for n2 in ["true_semantic_entropy", "sep_single_e36_layer",
                                                       "sep_single_best_layer", "reference_q_resp_only",
                                                       "deploy_q_resp_only"]},
        "bootstrap_q_resp_vs_q_only_internal": c,
    }
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
