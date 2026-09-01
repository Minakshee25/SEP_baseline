"""E68 — extend the TRUE LOLO-proxy squad OOD evaluation from Llama-2/Mistral to Llama-3 + DeepSeek.

Additive. Trains nothing. Reuses:
  * the E37/E43 leave-one-LLM-out `q_resp_only` checkpoints
    (`stage2/runs/E37_LOLO_ckpt/checkpoints/{Llama-3,DeepSeek}_q_resp_only_seed*.pt`) — the held-out
    target was NEVER in that fold's training pool (trained on the OTHER 3 targets, trivia only,
    never squad);
  * the E55 squad n1000 builds for both targets (`/data2/mn1025/stage1/<model>_squad_n1000_full`,
    1000/1000 records, same question selection as Llama-2/Mistral's squad sets);
  * the E41 fixed SEP layers (`exp2_run.BEST_TBG`: Llama-3 TBG:31, DeepSeek TBG:28).

Protocol is byte-identical to:
  * E52 (`se_fidelity_proxy_vs_sep.run_lolo_squad`) for SE-fidelity — Spearman(pred, continuous SE)
    plus a paired-bootstrap CI on (proxy_ensemble − SEP);
  * E54 (`correctness_eval_lolo_squad.evaluate_target`) for correctness — AUROC_incorrect plus
    paired-bootstrap CIs on (proxy − SEP) and (proxy − true 10-sample SE).

Does NOT touch `results/se_fidelity_proxy_vs_sep.json` or `results/correctness_eval_lolo_squad.json`
(the E52/E54 outputs). Writes only e68_* files.

Env: amortized_stage2 + a free GPU (LOLO proxy forward pass, 3 seeds × 2 targets, squad n1000).
    python -m amortized_ue.e68_lolo_squad_llama3_deepseek
"""
from __future__ import annotations

import json
import argparse

import numpy as np

from amortized_ue import exp2_run as E2
from amortized_ue.se_fidelity_proxy_vs_sep import (
    compute_sep, arm_preds_per_seed_prefixed, score_block, print_block, LOLO_CKPT_DIR)
from amortized_ue.correctness_eval_lolo_squad import evaluate_target

DATA2 = "/data2/mn1025/stage1"
TARGETS = ["Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
OUT_FID = "amortized_ue/results/e68_lolo_squad_sefidelity.json"
OUT_COR = "amortized_ue/results/e68_lolo_squad_correctness.json"
OUT_TAB = "amortized_ue/results/e68_combined_table.json"


def se_fidelity_block(target, bootstrap):
    """E52 `run_lolo_squad` loop body, squad routed to /data2 (E55 builds live there, not the NFS)."""
    short = E2.SHORT[target]
    layer = E2.BEST_TBG[target] if target != E2.ANCHOR else 30
    sep = compute_sep(target, eval_dataset="squad", eval_num_samples=1000, data_dir=DATA2,
                      eval_data_dir=DATA2, fit_num_samples=2000, use_test_split_as_eval=False,
                      layer=layer)
    ids, P = arm_preds_per_seed_prefixed("q_resp_only", short, target, "squad", 1000,
                                         ckpt_dir=LOLO_CKPT_DIR, data_dir=DATA2)
    block = score_block(sep, ids, P, bootstrap=bootstrap, tag=f"e68_lolo_squad/{short}")
    block["target"] = short
    block["target_full"] = target
    block["proxy_provenance"] = ("E37/E43 LOLO q_resp_only (trained on the OTHER 3 models' trivia_qa, "
                                 "never this target's data OR squad) -> squad OOD")
    print_block(f"E68 SE-fidelity — LOLO squad OOD, held out {short}", block)
    return block


def main():
    p = argparse.ArgumentParser(description="E68: LOLO-proxy squad OOD for Llama-3 + DeepSeek. Additive.")
    p.add_argument("--targets", nargs="+", default=TARGETS, choices=TARGETS)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    fid, cor = {}, {}
    for target in args.targets:
        short = E2.SHORT[target]
        cor[short] = evaluate_target(target, bootstrap=args.bootstrap, trivia_dir=DATA2)
        with open(OUT_COR, "w") as f:
            json.dump(cor, f, indent=2)
        fid[short] = se_fidelity_block(target, args.bootstrap)
        with open(OUT_FID, "w") as f:
            json.dump(fid, f, indent=2)
        print(f"    -> saved {len(fid)} target(s) to {OUT_FID} / {OUT_COR}")

    # ---- combined table -------------------------------------------------------------------------
    rows = []
    for short in [E2.SHORT[t] for t in args.targets]:
        fb, cb = fid[short], cor[short]
        fm = fb["metrics"]
        cm = cb["metrics"]
        d_rho = fb["bootstrap_vs_sep"]["proxy_ensemble"]["spearman_delta"]
        d_sep = cb["bootstrap_auroc_incorrect"]["sep_single_e36_layer"]["lolo_q_resp_only"]
        d_tse = cb["bootstrap_auroc_incorrect"]["true_semantic_entropy"]["lolo_q_resp_only"]
        rows.append({
            "target": short,
            "n_test": cb["n_test"],
            "incorrect_rate": cb["positive_rate_incorrect"],
            "sep_layer": fb["sep_choice"],
            "proxy_spearman": fm["proxy_ensemble"]["spearman"],
            "sep_spearman": fm["sep"]["spearman"],
            "delta_rho_vs_sep": {k: d_rho[k] for k in ("mean", "lo95", "hi95", "ci_excludes_zero")},
            "proxy_auroc_incorrect": cm["lolo_q_resp_only"]["auroc_incorrect"],
            "sep_auroc_incorrect": cm["sep_single_e36_layer"]["auroc_incorrect"],
            "true_se_auroc_incorrect": cm["true_semantic_entropy"]["auroc_incorrect"],
            "delta_auroc_vs_sep": {k: d_sep[k] for k in ("mean", "lo95", "hi95", "ci_excludes_zero")},
            "delta_auroc_vs_true_se": {k: d_tse[k] for k in ("mean", "lo95", "hi95", "ci_excludes_zero")},
        })
    with open(OUT_TAB, "w") as f:
        json.dump({"experiment": "E68", "targets": rows,
                   "note": "true LOLO q_resp_only proxy on squad OOD; E52 protocol for Spearman, "
                           "E54 protocol for AUROC_incorrect. Held-out target excluded from LOLO "
                           "training. No retraining."}, f, indent=2)

    def fmt(c):
        return f"{c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}]{'*' if c['ci_excludes_zero'] else ''}"

    print("\n" + "=" * 130)
    print("E68 — TRUE LOLO q_resp_only proxy on squad OOD (held-out target never in LOLO training, never squad)")
    print("=" * 130)
    h = (f"{'target':10s}{'proxy_rho':>10s}{'SEP_rho':>9s}{'Δρ vs SEP [95% CI]':>26s}"
         f"{'proxy_AUC':>11s}{'SEP_AUC':>9s}{'trueSE_AUC':>11s}"
         f"{'Δ vs SEP [95% CI]':>24s}{'Δ vs trueSE [95% CI]':>26s}")
    print(h)
    for r in rows:
        print(f"{r['target']:10s}{r['proxy_spearman']:>10.3f}{r['sep_spearman']:>9.3f}"
              f"{fmt(r['delta_rho_vs_sep']):>26s}"
              f"{r['proxy_auroc_incorrect']:>11.3f}{r['sep_auroc_incorrect']:>9.3f}"
              f"{r['true_se_auroc_incorrect']:>11.3f}"
              f"{fmt(r['delta_auroc_vs_sep']):>24s}{fmt(r['delta_auroc_vs_true_se']):>26s}")
    print("=" * 130)
    print(f"saved: {OUT_FID}\n       {OUT_COR}\n       {OUT_TAB}")


if __name__ == "__main__":
    main()
