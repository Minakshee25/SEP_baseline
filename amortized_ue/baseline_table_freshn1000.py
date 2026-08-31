"""Recompute the baseline table for the original four TriviaQA target LLMs on the CLEAN fresh
n1000 eval sets -- NO new proxy training, NO new generation.

For each model, at ONE fixed TBG layer (E36/E41 CV-picked -- exp2_run.BEST_TBG):

    Llama-2-7b-chat            TBG:30
    Mistral-7B-Instruct-v0.2   TBG:31
    Meta-Llama-3-8B-Instruct   TBG:31
    deepseek-llm-7b-chat       TBG:28

three predictors are scored on the SAME fresh n1000 rows:

  1. True semantic entropy   -- the continuous cluster_assignment_entropy label itself.
       Spearman vs SE = 1.0 (trivially); AUROC vs `incorrect` reported.
  2. SEP at the fixed layer  -- LogisticRegression on best_split-binarised SE, trained ONLY on
       the existing n2000 TRAIN split (linear_ceiling_probe.splits), train-side binarisation
       threshold (stage2.data.best_split on the train rows). Reuses
       correctness_eval.sep_single_fixed_layer verbatim. Reports Spearman(continuous SEP
       probability, continuous SE) and AUROC(SEP probability, incorrect).
  3. Own-model single-layer Ridge at the SAME fixed layer -- Ridge to the CONTINUOUS SE label,
       fit on the n2000 TRAIN split, alpha chosen on the n2000 VAL split by val Spearman
       (linear_ceiling_probe.fit_probe). Reports Spearman(ridge prediction, SE) and
       AUROC(ridge prediction, incorrect).

NOT used: aligned_z_ridge, pooled ridge, Procrustes, TBG+SLT stacked ridge, val-selected layers.

Asserts zero id overlap between the n2000 fit set and the fresh n1000 eval set (per model).

Prints + saves one table (results/baseline_table_freshn1000.json) with, per model:
    model, layer, N, incorrect_rate,
    SE_spearman, SE_auroc_incorrect,
    SEP_spearman, SEP_auroc_incorrect,
    Ridge_spearman, Ridge_auroc_incorrect,
    ridge_alpha

Then compares every recomputed value against committed values found on disk
(results/sep_reference_values.json, results/se_fidelity_proxy_vs_sep.json ["fresh"]) and prints
any mismatch > 1e-4.

Run from the repo root in the `se_probes` env (CPU only):
    python -m amortized_ue.baseline_table_freshn1000 --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe
from amortized_ue.correctness_eval import sep_single_fixed_layer, load_accuracy
from amortized_ue import exp2_run as E2

OUT = "amortized_ue/results/baseline_table_freshn1000.json"
POS = "TBG"
FIT_DATASET = "trivia_qa"
FIT_N = 2000
EVAL_DATASET = "trivia_qa"
EVAL_N = 1000
INCORRECT_THRESH = 0.5   # correctness_eval convention: correct = accuracy >= 0.5

# fixed single layers (E36/E41 CV-picked; exp2_run.BEST_TBG)
LAYERS = {
    "Llama-2-7b-chat": 30,
    "Mistral-7B-Instruct-v0.2": 31,
    "Meta-Llama-3-8B-Instruct": 31,
    "deepseek-llm-7b-chat": 28,
}


def _spearman(a, b) -> float:
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def _cfg(model, dataset, n, data_dir):
    return Stage1Config(model_name=model, dataset=dataset, num_samples=n,
                        **({"output_dir": data_dir} if data_dir else {}))


def _resolve_dir(model, data_dir):
    """Prefer `data_dir`; fall back to the NFS default per model if a manifest is missing there."""
    if data_dir is None:
        return None
    for n, ds in ((FIT_N, FIT_DATASET), (EVAL_N, EVAL_DATASET)):
        if not os.path.exists(_cfg(model, ds, n, data_dir).manifest_path()):
            return None
    return data_dir


def run_model(model, layer, data_dir, bootstrap=10000):
    tdir = _resolve_dir(model, data_dir)

    fit_cfg = _cfg(model, FIT_DATASET, FIT_N, tdir)
    eval_cfg = _cfg(model, EVAL_DATASET, EVAL_N, tdir)

    # --- zero-overlap assertion (ids from the manifests, never a positional join) ---------------
    with open(fit_cfg.manifest_path()) as f:
        fit_ids = set(json.load(f)["records"].keys())
    with open(eval_cfg.manifest_path()) as f:
        eval_ids_manifest = set(json.load(f)["records"].keys())
    overlap = fit_ids & eval_ids_manifest
    assert not overlap, (f"{model}: fresh n1000 overlaps n2000 fit set by {len(overlap)} ids "
                         f"(e.g. {sorted(overlap)[:3]}) -- not a clean fresh eval")

    # --- load hidden states + continuous SE labels (TBG only; sep helper takes pos/layer) -------
    fit_hidden, fit_y, fit_ids_ord = load_matrix(fit_cfg, [POS])
    eval_hidden, eval_y, eval_ids = load_matrix(eval_cfg, [POS])
    tr, va, te = splits(len(fit_ids_ord))               # the EXISTING n2000 train/val/test split
    eval_rows = np.arange(len(eval_ids))
    n_layers = fit_hidden[POS].shape[0]
    assert 0 <= layer < n_layers, f"{model}: layer {layer} out of range (0..{n_layers - 1})"

    # --- correctness labels for the fresh n1000 rows (id-keyed from the manifest) ---------------
    acc_map = load_accuracy(eval_cfg)
    incorrect = np.array([0 if acc_map[i] >= INCORRECT_THRESH else 1 for i in eval_ids], dtype=int)
    assert set(np.unique(incorrect)) == {0, 1}, f"{model}: eval set is single-class on correctness"
    incorrect_rate = float(incorrect.mean())

    # ========== 1. TRUE SEMANTIC ENTROPY ==========
    se_spearman = _spearman(eval_y, eval_y)            # == 1.0 by construction
    se_auroc_incorrect = float(roc_auc_score(incorrect, eval_y))

    # ========== 2. SEP at the fixed layer (train on n2000 train split, train-side binarisation) =
    sep_prob, sep_auroc_se, sep_choice, sep_thr, _ybe = sep_single_fixed_layer(
        fit_hidden, fit_y, tr, va, eval_hidden, eval_y, eval_rows, POS, layer)
    sep_prob = np.asarray(sep_prob)[eval_rows]
    sep_spearman = _spearman(sep_prob, eval_y)         # continuous SEP prob vs continuous SE
    sep_auroc_incorrect = float(roc_auc_score(incorrect, sep_prob))

    # ========== 3. Own-model single-layer Ridge at the SAME fixed layer ========================
    X_fit = fit_hidden[POS][layer]                     # [N2000, H]
    X_eval = eval_hidden[POS][layer]                   # [N1000, H]
    ridge_m, ridge_sc, ridge_alpha, ridge_val_s = fit_probe(X_fit, fit_y, tr, va)  # alpha on val Spearman
    ridge_pred = ridge_m.predict(ridge_sc.transform(X_eval))
    ridge_spearman = _spearman(ridge_pred, eval_y)
    ridge_auroc_incorrect = float(roc_auc_score(incorrect, ridge_pred))

    return {
        "model": model,
        "layer": f"{POS}:{layer}",
        "layer_int": int(layer),
        "N": int(len(eval_ids)),
        "n_fit": int(len(fit_ids_ord)),
        "fit_split_sizes": [int(len(tr)), int(len(va)), int(len(te))],
        "incorrect_rate": incorrect_rate,
        "data_dir": tdir if tdir else "NFS-default",
        "SE_spearman": se_spearman,
        "SE_auroc_incorrect": se_auroc_incorrect,
        "SEP_spearman": sep_spearman,
        "SEP_auroc_incorrect": sep_auroc_incorrect,
        "SEP_auroc_se_binarised": float(sep_auroc_se),
        "SEP_train_binarise_threshold": float(sep_thr),
        "SEP_choice": [sep_choice[0], int(sep_choice[1])],
        "Ridge_spearman": ridge_spearman,
        "Ridge_auroc_incorrect": ridge_auroc_incorrect,
        "ridge_alpha": float(ridge_alpha),
        "ridge_val_spearman": float(ridge_val_s),
    }


# --------------------------------------------------------------------------------------------------
# compare against committed values on disk
# --------------------------------------------------------------------------------------------------
def committed_values():
    """{(model, key): (value, source)} for every recomputed quantity we have a committed copy of."""
    out = {}
    ref_path = "amortized_ue/results/sep_reference_values.json"
    if os.path.exists(ref_path):
        ref = json.load(open(ref_path))["targets"]
        for model, blk in ref.items():
            s = blk["settings"].get("trivia_qa_fresh_n1000")
            if s:
                out[(model, "SEP_spearman")] = (s["spearman"], f"{ref_path}:trivia_qa_fresh_n1000.spearman")
                out[(model, "SEP_auroc_se_binarised")] = (s["auroc_se"], f"{ref_path}:trivia_qa_fresh_n1000.auroc_se")

    sf_path = "amortized_ue/results/se_fidelity_proxy_vs_sep.json"
    if os.path.exists(sf_path):
        sf = json.load(open(sf_path)).get("fresh", {})
        long = {v: k for k, v in E2.SHORT.items()}
        for short, b in sf.items():
            model = long.get(short, short)
            m = b.get("metrics", {}).get("sep", {})
            if "spearman" in m:
                out.setdefault((model, "SEP_spearman"), (m["spearman"], f"{sf_path}:fresh.{short}.metrics.sep.spearman"))
            if "auroc_se" in m:
                out.setdefault((model, "SEP_auroc_se_binarised"),
                               (m["auroc_se"], f"{sf_path}:fresh.{short}.metrics.sep.auroc_se"))
    return out


def compare(rows, tol=1e-4):
    comm = committed_values()
    print(f"\n{'=' * 96}\nCOMPARISON vs committed values (mismatch threshold {tol})\n{'=' * 96}")
    if not comm:
        print("  no committed values found on disk to compare against")
        return []
    mism = []
    by_model = {r["model"]: r for r in rows}
    for (model, key), (cval, src) in sorted(comm.items()):
        if model not in by_model or key not in by_model[model]:
            continue
        rval = by_model[model][key]
        diff = abs(rval - cval)
        flag = "  MISMATCH" if diff > tol else "ok"
        print(f"  [{flag:>10s}] {model:26s} {key:24s} recomputed={rval:.6f}  committed={cval:.6f}  "
              f"|Δ|={diff:.2e}   ({src})")
        if diff > tol:
            mism.append({"model": model, "key": key, "recomputed": rval, "committed": cval,
                         "abs_diff": diff, "source": src})
    if not mism:
        print("\n  ALL committed values reproduced within tolerance.")
    else:
        print(f"\n  {len(mism)} MISMATCH(es) > {tol}")
    return mism


def print_table(rows):
    print(f"\n{'=' * 130}\nBASELINE TABLE -- fresh trivia_qa n1000, fixed-layer, no new training\n{'=' * 130}")
    hdr = (f"{'model':26s}{'layer':>8s}{'N':>6s}{'inc_rate':>9s}"
           f"{'SE_rho':>8s}{'SE_auc':>8s}{'SEP_rho':>9s}{'SEP_auc':>9s}"
           f"{'Rdg_rho':>9s}{'Rdg_auc':>9s}{'alpha':>10s}")
    print(hdr)
    print("-" * 130)
    for r in rows:
        print(f"{r['model']:26s}{r['layer']:>8s}{r['N']:>6d}{r['incorrect_rate']:>9.3f}"
              f"{r['SE_spearman']:>8.3f}{r['SE_auroc_incorrect']:>8.3f}"
              f"{r['SEP_spearman']:>9.3f}{r['SEP_auroc_incorrect']:>9.3f}"
              f"{r['Ridge_spearman']:>9.3f}{r['Ridge_auroc_incorrect']:>9.3f}{r['ridge_alpha']:>10.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None,
                   help="Stage1Config.output_dir override (e.g. /data2/mn1025/stage1); "
                        "falls back to the NFS default per model if a manifest is missing there")
    p.add_argument("--out", default=OUT)
    p.add_argument("--tol", type=float, default=1e-4)
    args = p.parse_args()

    rows = []
    for model, layer in LAYERS.items():
        print(f"\n>>> {model}  (fixed {POS}:{layer})")
        rows.append(run_model(model, layer, args.data_dir))

    print_table(rows)
    mism = compare(rows, tol=args.tol)

    payload = {
        "_meta": {
            "description": "Baseline table (True SE / SEP / own-model single-layer Ridge) on the clean "
                           "fresh trivia_qa n1000 eval sets, fixed CV-picked TBG layers, NO new training.",
            "fit": f"{FIT_DATASET} n{FIT_N} (existing train/val split from linear_ceiling_probe.splits)",
            "eval": f"{EVAL_DATASET} n{EVAL_N} (fresh, id-disjoint from fit -- asserted)",
            "incorrect_definition": f"incorrect = (canonical.accuracy < {INCORRECT_THRESH})",
            "sep": "correctness_eval.sep_single_fixed_layer (LogisticRegression on train-side "
                   "best_split-binarised SE, trained on the n2000 train split only)",
            "ridge": "linear_ceiling_probe.fit_probe (Ridge -> continuous SE, alpha chosen on n2000 "
                     "val Spearman), single fixed layer, own model only",
            "not_used": ["aligned_z_ridge", "pooled_ridge", "procrustes", "TBG+SLT_stacked_ridge",
                         "val_selected_layers"],
            "generated_by": "amortized_ue/baseline_table_freshn1000.py",
        },
        "layers": {m: f"{POS}:{L}" for m, L in LAYERS.items()},
        "rows": rows,
        "comparison_mismatches": mism,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  -> saved to {args.out}")


if __name__ == "__main__":
    main()
