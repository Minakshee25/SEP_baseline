"""CANONICAL eval-only RAW cross-LLM transfer of the frozen single-model 5-arm proxies.

NO training / fine-tuning / refitting / layer-selection / Procrustes / target labels for fitting /
pooled or LOLO checkpoints. Each source proxy checkpoint is loaded frozen and run forward on the
TARGET model's hidden states + text; the source checkpoint's own (position, layer, z_inputs, k,
target-SE standardize transform) are used unchanged.

Source checkpoints (5 arms x 5 seeds each):
  Llama-2 : REFERENCE_multipos_p1024_5arm_ckpt/checkpoints   (z_inputs TBG:22,SLT:15 ; k=4)
  Mistral : E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints    (z_inputs TBG:31,SLT:20 ; k=4)

Pairs (source -> target):
  Mistral  -> Llama-2
  Llama-2  -> Mistral
  Llama-2  -> Meta-Llama-3-8B-Instruct
  Llama-2  -> deepseek-llm-7b-chat

Datasets, each n1000, id-disjoint from any n2000 the sources were trained on:
  ID  : trivia_qa   (fresh n1000)
  OOD : squad       (n1000)

Arms: z, z_q, z_q_resp, q_only, q_resp_only.

Per (pair, dataset, arm), FOR EACH of the 5 saved seeds INDEPENDENTLY:
  Spearman(pred_continuous_SE, target_continuous_SE)
  AUROC(pred_continuous_SE, incorrect),  incorrect = (accuracy < 0.5),
  higher pred => more uncertain => positive for incorrect.
MAIN reported number = mean +/- std of the 5 per-seed metric values.
Prediction-ensemble metrics (mean the 5 seeds' preds, then score) are ALSO reported but clearly
labelled "ensemble" and are NOT the main-table values.

Zero NFS dependency: checkpoints + backbone + data + env all on /data2 (see cross_llm_5arm_gpu_swap.sh).

    <amortized_stage2_v5 python> -m amortized_ue.cross_llm_5arm_eval --data_dir /data2/mn1025/stage1
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

OUT = "amortized_ue/results/cross_llm_5arm_fresh_n1000_correctness.json"
ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]
INCORRECT_THRESH = 0.5
N = 1000
# (dataset, regime-label)
DATASETS = [("trivia_qa", "ID"), ("squad", "OOD")]

CKPT = {
    "Llama-2-7b-chat": "/data2/mn1025/stage2_checkpoints/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints",
    "Mistral-7B-Instruct-v0.2": "/data2/mn1025/stage2_checkpoints/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints",
}
CKPT_NFS = {
    "Llama-2-7b-chat": "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints",
    "Mistral-7B-Instruct-v0.2": "amortized_ue/stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints",
}

PAIRS = [
    ("Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat"),
    ("Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"),
    ("Llama-2-7b-chat", "Meta-Llama-3-8B-Instruct"),
    ("Llama-2-7b-chat", "deepseek-llm-7b-chat"),
]

# E23 fresh-n1000 (ID) Spearman means for the two known transfers -- STOP if not reproduced ~0.05
E23 = {
    ("Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat"):
        {"z": 0.031, "z_q": 0.015, "z_q_resp": 0.042, "q_only": 0.477, "q_resp_only": 0.523},
    ("Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"):
        {"z": 0.014, "z_q": 0.073, "z_q_resp": 0.124, "q_only": 0.474, "q_resp_only": 0.531},
}
E23_TOL = 0.05
# E50 q_resp_only ENSEMBLE AUROC on the ID transfer (secondary sanity, warn only)
E50_ENS_AUROC = {
    ("Mistral-7B-Instruct-v0.2", "Llama-2-7b-chat"): 0.7197,
    ("Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"): 0.7253,
}
E50_TOL = 0.02


def _spearman(a, b) -> float:
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def target_labels(model, dataset, data_dir):
    cfg = Stage1Config(model_name=model, dataset=dataset, num_samples=N,
                       **({"output_dir": data_dir} if data_dir else {}))
    if not os.path.exists(cfg.manifest_path()):
        raise FileNotFoundError(f"no {dataset} n{N} manifest for {model} under {data_dir}")
    _, y, ids = load_matrix(cfg, ["TBG"])                        # y = continuous cluster_assignment_entropy
    acc = load_accuracy(cfg)
    se = {i: float(v) for i, v in zip(ids, y)}
    inc = {i: (0 if acc[i] >= INCORRECT_THRESH else 1) for i in ids}
    return se, inc, cfg.run_dir()


def eval_pair_dataset(source, target, dataset, regime, data_dir):
    ckpt_dir = CKPT[source] if os.path.isdir(CKPT[source]) else CKPT_NFS[source]
    se_map, inc_map, run_dir = target_labels(target, dataset, data_dir)
    print(f"\n{'='*96}\n[{regime}] {source}  ->  {target}   ({dataset} n{N})\n"
          f"  ckpt: {ckpt_dir}\n  data: {run_dir}\n{'='*96}")
    out_arms = {}
    for arm in ARMS:
        ids, per_seed = arm_preds_per_seed(arm, target, dataset, N, ckpt_dir=ckpt_dir, data_dir=data_dir)
        y = np.array([se_map[i] for i in ids], float)
        inc = np.array([inc_map[i] for i in ids], int)
        assert set(np.unique(inc)) == {0, 1}, f"{target}/{dataset}: single-class incorrect"
        n_seeds = per_seed.shape[0]

        sp = np.array([_spearman(per_seed[s], y) for s in range(n_seeds)])
        au = np.array([float(roc_auc_score(inc, per_seed[s])) for s in range(n_seeds)])
        ens_pred = per_seed.mean(0)
        ens_sp, ens_au = _spearman(ens_pred, y), float(roc_auc_score(inc, ens_pred))

        out_arms[arm] = {
            "arm": arm, "N": int(len(ids)), "n_seeds": int(n_seeds),
            "incorrect_rate": float(inc.mean()),
            "seed_spearman": sp.tolist(),
            "seed_auroc_incorrect": au.tolist(),
            "spearman_mean": float(sp.mean()), "spearman_std": float(sp.std(ddof=0)),
            "auroc_incorrect_mean": float(au.mean()), "auroc_incorrect_std": float(au.std(ddof=0)),
            "ensemble": {
                "note": "predictions averaged across the 5 seeds, THEN scored -- NOT the main value",
                "spearman": ens_sp, "auroc_incorrect": ens_au,
            },
        }
        print(f"  {arm:12s}  rho {sp.mean():.3f} ± {sp.std():.3f}   "
              f"auroc_inc {au.mean():.3f} ± {au.std():.3f}   "
              f"| ens rho {ens_sp:.3f}  auroc {ens_au:.3f}")
    return {
        "source_model": source, "target_model": target,
        "source_checkpoint_dir": ckpt_dir,
        "dataset": dataset, "regime": regime, "N": N,
        "target_run_dir": run_dir,
        "arms": out_arms,
    }


def sanity(results):
    print(f"\n{'#'*96}\n# SANITY CHECKS (ID / trivia_qa transfers only)\n{'#'*96}")
    hard_fail = []
    for r in results:
        if r["dataset"] != "trivia_qa":
            continue
        key = (r["source_model"], r["target_model"])
        if key in E23:
            print(f"\n  E23 Spearman-mean  {r['source_model']} -> {r['target_model']}  (tol {E23_TOL})")
            for arm, ref in E23[key].items():
                got = r["arms"][arm]["spearman_mean"]
                d = got - ref
                bad = abs(d) > E23_TOL
                print(f"    {arm:12s} got {got:.3f}  E23 {ref:.3f}  Δ {d:+.3f}  {'<-- FAIL' if bad else 'ok'}")
                if bad:
                    hard_fail.append((key, arm, got, ref))
        if key in E50_ENS_AUROC:
            got = r["arms"]["q_resp_only"]["ensemble"]["auroc_incorrect"]
            ref = E50_ENS_AUROC[key]
            d = got - ref
            print(f"  E50 q_resp_only ENSEMBLE AUROC  {r['source_model']} -> {r['target_model']}: "
                  f"got {got:.4f}  E50 {ref:.4f}  Δ {d:+.4f}  {'WARN' if abs(d) > E50_TOL else 'ok'}")
    if hard_fail:
        print(f"\n  *** {len(hard_fail)} E23 MISMATCH(es) > {E23_TOL} — STOPPING.")
        for f in hard_fail:
            print(f"    {f}")
        raise SystemExit(2)
    print("\n  E23 reproduced within tolerance for both known ID transfers. Proceeding.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data2/mn1025/stage1")
    p.add_argument("--out", default=OUT)
    p.add_argument("--datasets", nargs="+", default=[d for d, _ in DATASETS],
                   choices=[d for d, _ in DATASETS])
    args = p.parse_args()
    regime_of = dict(DATASETS)

    results = []
    for dataset in args.datasets:
        for s, t in PAIRS:
            results.append(eval_pair_dataset(s, t, dataset, regime_of[dataset], args.data_dir))

    sanity(results)

    payload = {
        "_meta": {
            "description": "Canonical eval-only RAW frozen cross-LLM transfer of the single-model "
                           "5-arm proxies (REFERENCE=Llama-2, E22=Mistral). ID = fresh trivia_qa "
                           "n1000, OOD = squad n1000. No retrain/refit/layer-select/Procrustes/"
                           "target-labels/pooled/LOLO. Main number = mean±std of the per-seed "
                           "metric (NOT metric of seed-averaged preds); ensemble reported separately.",
            "arms": ARMS, "N": N, "datasets": DATASETS,
            "incorrect_definition": "accuracy < 0.5",
            "pairs": [f"{s} -> {t}" for s, t in PAIRS],
            "sanity_e23_id": {f"{s}->{t}": v for (s, t), v in E23.items()},
            "sanity_e50_id_ensemble_auroc": {f"{s}->{t}": v for (s, t), v in E50_ENS_AUROC.items()},
            "no_nfs": "checkpoints /data2/mn1025/stage2_checkpoints, backbone /data2/mn1025/hf_cache, "
                      "data /data2/mn1025/stage1, env amortized_stage2_v5 (/data2)",
            "generated_by": "amortized_ue/cross_llm_5arm_eval.py",
        },
        "results": results,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  -> saved to {args.out}")

    # compact final table -- ID and OOD side by side
    print(f"\n{'='*118}\nMAIN TABLE (mean ± std over 5 seeds)   ID = fresh trivia n1000 | OOD = squad n1000\n{'='*118}")
    by = {(r["source_model"], r["target_model"], r["dataset"]): r for r in results}
    print(f"{'source -> target':44s}{'arm':13s}{'ID Spearman':>15s}{'ID AUROC inc':>15s}{'OOD Spearman':>15s}{'OOD AUROC inc':>15s}")
    for s, t in PAIRS:
        tag = f"{s} -> {t}"
        for i, arm in enumerate(ARMS):
            idr = by.get((s, t, "trivia_qa"), {}).get("arms", {}).get(arm)
            odr = by.get((s, t, "squad"), {}).get("arms", {}).get(arm)
            def c(d):
                return f"{d['spearman_mean']:.3f}±{d['spearman_std']:.3f}" if d else "     -    ", \
                       f"{d['auroc_incorrect_mean']:.3f}±{d['auroc_incorrect_std']:.3f}" if d else "     -    "
            i_sp, i_au = c(idr)
            o_sp, o_au = c(odr)
            print(f"{tag if i == 0 else '':44s}{arm:13s}{i_sp:>15s}{i_au:>15s}{o_sp:>15s}{o_au:>15s}")
        print()


if __name__ == "__main__":
    main()
