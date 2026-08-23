"""E50 — fill two gaps the user asked about directly: how does (a) the Mistral-ONLY-trained proxy
and (b) the Deploy (all-4-pooled) proxy score on trivia_qa ID (the SAME N=1000 fresh eval set used
throughout correctness_eval.py), for BOTH Llama-2 and Mistral. Neither had been run on trivia_qa ID
before (only squad/OOD existed for both, in E42 and E39 respectively). Additive: reuses `arm_preds`
(unchanged) with a different `ckpt_dir`, and reuses `load_accuracy` for the incorrect label -- no SEP
refit, no hidden-state loading beyond what arm_preds itself needs.

    python -m amortized_ue.e50_crossproxy_id_pull
"""
from __future__ import annotations
import json
import numpy as np
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.correctness_eval import load_accuracy

THRESH = 0.5  # matches correctness_eval.py's evaluate_target() binarisation exactly
from amortized_ue.procrustes_e27_rank_fusion import arm_preds

DATA_DIR = "/data2/mn1025/stage1"
EVAL_N = 1000
DATASET = "trivia_qa"

CKPTS = {
    "trained_on_llama2_only (REFERENCE)": "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints",
    "trained_on_mistral_only (E22)": "amortized_ue/stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints",
    "trained_on_all_4_pooled (deploy)": "amortized_ue/results/deploy_checkpoints",
}

TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]


def main():
    results = {}
    for target in TARGETS:
        acc_cfg = Stage1Config(model_name=target, dataset=DATASET, num_samples=EVAL_N, output_dir=DATA_DIR)
        acc_map = load_accuracy(acc_cfg)
        eval_ids = sorted(acc_map.keys())
        incorrect = (np.array([acc_map[i] for i in eval_ids]) < THRESH).astype(int)
        print(f"\n=== target={target}  N={len(eval_ids)}  incorrect_rate={incorrect.mean():.3f} ===")

        results[target] = {}
        for label, ckpt in CKPTS.items():
            preds = arm_preds("q_resp_only", target, DATASET, EVAL_N, ckpt_dir=ckpt, data_dir=DATA_DIR)
            s = np.array([preds[i] for i in eval_ids])
            auroc = float(roc_auc_score(incorrect, s))
            results[target][label] = auroc
            print(f"  {label:38s} AUROC_incorrect={auroc:.3f}")

    out = "amortized_ue/results/e50_crossproxy_id_pull.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
