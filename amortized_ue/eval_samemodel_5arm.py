"""Evaluate the EXISTING frozen 5-arm SAME-MODEL proxy checkpoints. Trains / refits NOTHING.

  Llama-2  proxy : amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints
  Mistral  proxy : amortized_ue/stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints

Arms: z, z_q, z_q_resp, q_only, q_resp_only   (5 saved seeds each)

SAME-MODEL ONLY:  Llama-2 proxy -> Llama-2 data,  Mistral proxy -> Mistral data.
Per source model, two evals:
  ID  : its fresh TriviaQA n1000
  OOD : its SQuAD  n1000

For every arm and every saved seed:  Spearman(pred, continuous SE)  and  AUROC(pred, incorrect),
incorrect = (canonical.accuracy < 0.5).  Reported value = MEAN +/- STD of each METRIC across the
5 seeds (metrics averaged, NOT predictions -- E12/E23 convention).

Reuses amortized_ue.se_fidelity_proxy_vs_sep.arm_preds_per_seed for the frozen forward pass
(same checkpoints, same Stage2Data, no retraining) and linear_ceiling_probe.load_matrix /
correctness_eval.load_accuracy for the id-keyed SE + accuracy labels.

Needs the `amortized_stage2` env + a GPU (frozen 3B backbone forward pass).

    <amortized_stage2 python> -m amortized_ue.eval_samemodel_5arm --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix
from amortized_ue.correctness_eval import load_accuracy
from amortized_ue.se_fidelity_proxy_vs_sep import arm_preds_per_seed

OUT = "amortized_ue/results/samemodel_5arm_id_ood.json"
ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]
INCORRECT_THRESH = 0.5

SOURCES = {
    "Llama-2-7b-chat": "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints",
    "Mistral-7B-Instruct-v0.2": "amortized_ue/stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints",
}

# approximate reproduction targets (E23 ID Spearman; E12/E18 Llama-2 OOD Spearman)
E23_ID = {
    "Llama-2-7b-chat": {"z": .562, "z_q": .561, "z_q_resp": .545, "q_only": .489, "q_resp_only": .558},
    "Mistral-7B-Instruct-v0.2": {"z": .628, "z_q": .604, "z_q_resp": .620, "q_only": .487, "q_resp_only": .572},
}
E12_OOD_LLAMA2 = {"z": .368, "z_q": .402, "z_q_resp": .398, "q_only": .259, "q_resp_only": .399}


def _spearman(a, b):
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def labels_for(model, dataset, n, data_dir):
    """id -> (continuous SE, incorrect) for (model, dataset, n). Tries data_dir then NFS default."""
    cfg = Stage1Config(model_name=model, dataset=dataset, num_samples=n,
                       **({"output_dir": data_dir} if data_dir else {}))
    if not os.path.exists(cfg.manifest_path()):
        cfg = Stage1Config(model_name=model, dataset=dataset, num_samples=n)
    _, y, ids = load_matrix(cfg, ["TBG"])                 # y = cluster_assignment_entropy, id order
    acc = load_accuracy(cfg)
    se = {i: float(v) for i, v in zip(ids, y)}
    inc = {i: (0 if acc[i] >= INCORRECT_THRESH else 1) for i in ids}
    return se, inc, cfg.run_dir()


def eval_split(model, ckpt_dir, dataset, n, data_dir, tag):
    se_map, inc_map, run_dir = labels_for(model, dataset, n, data_dir)
    print(f"\n  [{tag}] {model}  <-  {dataset} n{n}   ({run_dir})")
    per_arm = {}
    for arm in ARMS:
        ids, per_seed = arm_preds_per_seed(arm, model, dataset, n, ckpt_dir=ckpt_dir, data_dir=data_dir)
        y = np.array([se_map[i] for i in ids], float)
        inc = np.array([inc_map[i] for i in ids], int)
        single_class = len(np.unique(inc)) < 2
        sp, au = [], []
        for s in range(per_seed.shape[0]):
            p = per_seed[s]
            sp.append(_spearman(p, y))
            au.append(np.nan if single_class else float(roc_auc_score(inc, p)))
        sp, au = np.array(sp), np.array(au)
        per_arm[arm] = {
            "n_seeds": int(per_seed.shape[0]), "N": int(len(ids)),
            "incorrect_rate": float(inc.mean()),
            "spearman_per_seed": sp.tolist(), "auroc_incorrect_per_seed": au.tolist(),
            "spearman_mean": float(sp.mean()), "spearman_std": float(sp.std(ddof=0)),
            "auroc_incorrect_mean": float(np.nanmean(au)), "auroc_incorrect_std": float(np.nanstd(au)),
        }
        print(f"    {arm:12s} rho {sp.mean():.3f}±{sp.std():.3f}   auroc_inc {np.nanmean(au):.3f}±{np.nanstd(au):.3f}")
    return per_arm


def check_repro(results):
    print(f"\n{'='*80}\nREPRODUCTION CHECK (approx; |Δ| flagged > 0.03)\n{'='*80}")
    def cmp(label, got, ref):
        for arm in ARMS:
            d = got[arm]["spearman_mean"] - ref[arm]
            flag = "  <-- CHECK" if abs(d) > 0.03 else ""
            print(f"  {label:22s} {arm:12s} got {got[arm]['spearman_mean']:.3f}  ref {ref[arm]:.3f}  Δ {d:+.3f}{flag}")
    cmp("Llama-2 ID (E23)", results["Llama-2-7b-chat"]["ID"], E23_ID["Llama-2-7b-chat"])
    cmp("Mistral ID (E23)", results["Mistral-7B-Instruct-v0.2"]["ID"], E23_ID["Mistral-7B-Instruct-v0.2"])
    cmp("Llama-2 OOD (E12/E18)", results["Llama-2-7b-chat"]["OOD"], E12_OOD_LLAMA2)


def print_table(results):
    print(f"\n{'='*104}")
    print(f"{'Model':26s}{'Arm':13s}{'ID Spearman':>16s}{'ID AUROC inc':>16s}{'OOD Spearman':>16s}{'OOD AUROC inc':>16s}")
    print("-" * 104)
    for model, blk in results.items():
        for i, arm in enumerate(ARMS):
            idd, ood = blk["ID"][arm], blk["OOD"][arm]
            m = model if i == 0 else ""
            print(f"{m:26s}{arm:13s}"
                  f"{idd['spearman_mean']:>8.3f}±{idd['spearman_std']:<6.3f}"
                  f"{idd['auroc_incorrect_mean']:>8.3f}±{idd['auroc_incorrect_std']:<6.3f}"
                  f"{ood['spearman_mean']:>8.3f}±{ood['spearman_std']:<6.3f}"
                  f"{ood['auroc_incorrect_mean']:>8.3f}±{ood['auroc_incorrect_std']:<6.3f}")
        print()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data2/mn1025/stage1")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    results = {}
    for model, ckpt_dir in SOURCES.items():
        assert os.path.isdir(ckpt_dir), ckpt_dir
        results[model] = {
            "checkpoint_dir": ckpt_dir,
            "ID": eval_split(model, ckpt_dir, "trivia_qa", 1000, args.data_dir, "ID  fresh-trivia-n1000"),
            "OOD": eval_split(model, ckpt_dir, "squad", 1000, args.data_dir, "OOD squad-n1000"),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"_meta": {
                "description": "Frozen 5-arm SAME-MODEL proxy checkpoints evaluated on own ID fresh "
                               "TriviaQA n1000 and own OOD SQuAD n1000. No retrain/refit/selection. "
                               "Reported = mean±std of each metric across 5 seeds (metrics averaged).",
                "arms": ARMS, "incorrect": "accuracy < 0.5",
                "generated_by": "amortized_ue/eval_samemodel_5arm.py"},
                "results": results}, f, indent=2)

    print_table(results)
    check_repro(results)
    print(f"\n  -> saved to {args.out}")


if __name__ == "__main__":
    main()
