"""Correctness eval of the TRUE LOLO proxy on squad OOD (Llama-2 + Mistral). Additive; fills a gap
E39/E52 left open: E39's squad correctness eval used the DEPLOY proxy (all-4-pooled, target WAS in
the training pool) because at the time E37's LOLO run had saved no checkpoints. E52 later scored the
TRUE LOLO proxy (checkpoints now exist: `stage2/runs/E37_LOLO_ckpt/checkpoints/`, trained on the
OTHER 3 targets, zero exposure to this target's data OR to squad) on squad -- but only against the
continuous SE label, never against actual wrong answers. This closes that: same LOLO `q_resp_only`
checkpoints, same squad n1000 rows/ids, scored against `incorrect` (E38/E39's convention), with a
paired bootstrap vs the E41 fixed-layer SEP and vs true 10-sample SE.

Trains nothing. Reuses `se_fidelity_proxy_vs_sep.{compute_sep, arm_preds_per_seed_prefixed}` for the
SEP fit + LOLO forward pass, and `correctness_eval.{load_accuracy, accuracy_coverage,
prediction_rejection_ratio, paired_bootstrap_auc, ci}` for the correctness scoring -- the same recipe
`correctness_eval_ood.py` uses for the DEPLOY/REFERENCE proxies, applied here to the LOLO one.

Env: amortized_stage2 (or amortized_stage2_v5) + a free GPU (LOLO proxy forward pass, 3 seeds).
    python -m amortized_ue.correctness_eval_lolo_squad
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.correctness_eval import (
    load_accuracy, accuracy_coverage, prediction_rejection_ratio,
    paired_bootstrap_auc, ci, COVERAGES)
from amortized_ue.se_fidelity_proxy_vs_sep import (
    compute_sep, arm_preds_per_seed_prefixed, LOLO_CKPT_DIR, LOLO_SQUAD_TARGETS)
from amortized_ue import exp2_run as E2

OUT = "amortized_ue/results/correctness_eval_lolo_squad.json"
BASES = ["sep_single_e36_layer", "true_semantic_entropy"]


def evaluate_target(target, bootstrap=10000, trivia_dir=None):
    short = E2.SHORT[target]
    layer = E2.BEST_TBG[target] if target != E2.ANCHOR else 30

    # SEP (E41 fixed layer) + true SE, fit on trivia n2000, eval on squad n1000 -- identical recipe
    # and ids to E51/E52's `squad`/`lolo_squad` settings, so directly comparable to those tables.
    # squad now has a local /data2 copy too (staged 2026-08-25 to unblock this exact NFS stall) --
    # route both fit and eval loads through the same override, not just trivia.
    sep = compute_sep(target, eval_dataset="squad", eval_num_samples=1000, data_dir=trivia_dir,
                       eval_data_dir=trivia_dir, fit_num_samples=2000, use_test_split_as_eval=False,
                       layer=layer)
    ids = sep["ids"]

    # correctness label, id-joined (never positional) -- same convention as correctness_eval_ood.py
    acc_map = load_accuracy(Stage1Config(model_name=target, dataset="squad", num_samples=1000,
                                         **({"output_dir": trivia_dir} if trivia_dir else {})))
    assert set(ids).issubset(acc_map), "squad ids missing from the accuracy manifest"
    acc = np.array([acc_map[i] for i in ids], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    pos_rate = float(incorrect.mean())

    # true LOLO proxy: trained on the OTHER 3 targets, never this target's data OR squad
    lolo_ids, P = arm_preds_per_seed_prefixed("q_resp_only", short, target, "squad", 1000,
                                              ckpt_dir=LOLO_CKPT_DIR, data_dir=trivia_dir)
    id_to_col = {i: c for c, i in enumerate(lolo_ids)}
    missing = [i for i in ids if i not in id_to_col]
    assert not missing, f"{short}: {len(missing)} eval ids missing a LOLO prediction"
    P = P[:, [id_to_col[i] for i in ids]]     # [n_seeds, N] aligned to `ids`
    ens = P.mean(0)

    preds = {"true_semantic_entropy": sep["y"], "sep_single_e36_layer": sep["pred"],
             "lolo_q_resp_only": ens}
    for s in range(P.shape[0]):
        preds[f"lolo_q_resp_only_seed{s}"] = P[s]

    yb = sep["yb"]
    v = yb >= 0

    print(f"\n{'#' * 92}\n# LOLO-on-squad CORRECTNESS -- held out {short} "
          f"(proxy trained on the other 3, trivia only; never squad)\n{'#' * 92}")
    print(f"  squad N={len(ids)}  mean_acc={acc.mean():.3f}  incorrect_rate={pos_rate:.3f}")

    metrics = {}
    print(f"\n  {'predictor':28s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}"
          f"{'acc@.50':>8s}{'AUROC_SE':>10s}{'rho_SE':>8s}")
    for name, s in preds.items():
        s = np.asarray(s, dtype=float)
        au_inc = float(roc_auc_score(incorrect, s))
        ap_inc = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        au_se = float(roc_auc_score(yb[v], s[v])) if len(np.unique(yb[v])) == 2 else float("nan")
        rho = float(spearmanr(s, sep["y"]).correlation)
        metrics[name] = {"auroc_incorrect": au_inc, "auprc_incorrect": ap_inc, "prr": pr,
                         "accuracy_coverage": {str(c): cov[c] for c in COVERAGES},
                         "auroc_binarised_se": au_se, "spearman_se": rho}
        print(f"  {name:28s}{au_inc:>10.3f}{ap_inc:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}"
              f"{cov[0.5]:>8.3f}{au_se:>10.3f}{rho:>8.3f}")

    aus = [metrics[f"lolo_q_resp_only_seed{s}"]["auroc_incorrect"] for s in range(P.shape[0])]
    metrics["lolo_q_resp_only"]["auroc_incorrect_per_seed"] = aus
    metrics["lolo_q_resp_only"]["auroc_incorrect_seed_mean"] = float(np.mean(aus))
    metrics["lolo_q_resp_only"]["auroc_incorrect_seed_std"] = float(np.std(aus))

    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    vs = {}
    print(f"\n  paired bootstrap (B={bootstrap}, shared indices) -- delta AUROC_incorrect [N={len(ids)}]")
    for base in BASES:
        vs[base] = {}
        print(f"    vs {base}:")
        for name in preds:
            if name == base:
                continue
            c = ci(boot[name] - boot[base])
            excl = c["lo95"] > 0 or c["hi95"] < 0
            vs[base][name] = {**c, "ci_excludes_zero": bool(excl)}
            print(f"      d({name:26s}) {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
                  f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": short, "target_full": target, "eval_dataset": "squad", "fit_dataset": "trivia_qa",
            "n_test": len(ids), "mean_accuracy": float(acc.mean()), "positive_rate_incorrect": pos_rate,
            "sep_choice": list(sep["choice"]), "sep_selection": sep["selection"],
            "proxy_provenance": ("E37/E43 LOLO q_resp_only (trained on the OTHER 3 models' trivia_qa, "
                                 "never this target's data OR squad)"),
            "bootstrap_resamples": bootstrap, "metrics": metrics, "bootstrap_auroc_incorrect": vs}


def main():
    p = argparse.ArgumentParser(description="LOLO-on-squad correctness eval. Additive.")
    p.add_argument("--targets", nargs="+", default=LOLO_SQUAD_TARGETS, choices=LOLO_SQUAD_TARGETS)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--trivia_dir", default=None,
                   help="output_dir override for the trivia n2000 loads only (squad always reads the NFS default)")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    out = {}
    for t in args.targets:
        out[E2.SHORT[t]] = evaluate_target(t, bootstrap=args.bootstrap, trivia_dir=args.trivia_dir)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"    -> saved {len(out)} target(s) to {args.out}")

    print("\n" + "=" * 92)
    print("SUMMARY -- LOLO-on-squad AUROC_incorrect")
    print("=" * 92)
    for t, block in out.items():
        m = block["metrics"]
        print(f"{t:10s}  true_SE={m['true_semantic_entropy']['auroc_incorrect']:.3f}  "
              f"SEP={m['sep_single_e36_layer']['auroc_incorrect']:.3f}  "
              f"LOLO_qresp={m['lolo_q_resp_only']['auroc_incorrect']:.3f}")


if __name__ == "__main__":
    main()
