"""Fit the clean ID baseline models (SEP + own-model single-layer Ridge) for the 4 original
TriviaQA targets and PERSIST every fitted object, so the SAME models can be reloaded and scored
on OOD (SQuAD) with NO refitting.

Fixed single layers (E36/E41 CV-picked; exp2_run.BEST_TBG):
    Llama-2-7b-chat            TBG:30
    Mistral-7B-Instruct-v0.2   TBG:31
    Meta-Llama-3-8B-Instruct   TBG:31
    deepseek-llm-7b-chat       TBG:28

Recipe (identical to amortized_ue/baseline_table_freshn1000.py, only now the fitted objects are
captured and pickled instead of thrown away):

  SEP   -- LogisticRegression on train-side best_split-binarised SE, StandardScaler on the n2000
           TRAIN rows only. Byte-for-byte the same fit as correctness_eval.sep_single_fixed_layer
           (asserted: predictions on the fresh n1000 set match that function to < 1e-12).
  Ridge -- linear_ceiling_probe.fit_probe: Ridge -> CONTINUOUS SE, alpha chosen by n2000 VAL
           Spearman, StandardScaler on the n2000 TRAIN rows only. Same single fixed layer.

Saved per model under  amortized_ue/checkpoints/baselines/<model>/ :
    sep.joblib      {scaler, clf, position, layer, se_threshold, binarise:"best_split on train SE",
                     train_pos_rate, ...}
    ridge.joblib    {scaler, model, alpha, alpha_grid, val_spearman, position, layer, ...}
    meta.json       model_name, position, layer, fit dataset/N, split sizes + SEED/TEST_SIZE/
                    VAL_SIZE, train/val/test id lists, data_dir, incorrect_threshold

Then loads those exact checkpoints back and scores them on the fresh TriviaQA n1000 set, and
verifies the numbers reproduce results/baseline_table_freshn1000.json (|Δ| < 1e-6).

CPU only, `se_probes` env. Run from the repo root:
    python -m amortized_ue.fit_save_baselines --data_dir /data2/mn1025/stage1

OOD later (SEPARATE script, no refit): load <model>/sep.joblib + <model>/ridge.joblib, build the
SQuAD n1000 hidden matrix at meta['position']/meta['layer'], apply scaler+model, done.
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
import joblib
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, ALPHAS, SEED, TEST_SIZE, VAL_SIZE
from amortized_ue.correctness_eval import sep_single_fixed_layer, load_accuracy
from amortized_ue.stage2.data import best_split, binarize_entropy

CKPT_ROOT = "amortized_ue/checkpoints/baselines"
BASELINE_TABLE = "amortized_ue/results/baseline_table_freshn1000.json"
POS = "TBG"
FIT_DATASET, FIT_N = "trivia_qa", 2000
EVAL_DATASET, EVAL_N = "trivia_qa", 1000
INCORRECT_THRESH = 0.5

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
    if data_dir is None:
        return None
    for n, ds in ((FIT_N, FIT_DATASET), (EVAL_N, EVAL_DATASET)):
        if not os.path.exists(_cfg(model, ds, n, data_dir).manifest_path()):
            return None
    return data_dir


# ------------------------------------------------------------------------------------------------
# FIT
# ------------------------------------------------------------------------------------------------
def fit_one(model, layer, data_dir):
    tdir = _resolve_dir(model, data_dir)
    fit_cfg = _cfg(model, FIT_DATASET, FIT_N, tdir)
    eval_cfg = _cfg(model, EVAL_DATASET, EVAL_N, tdir)

    with open(fit_cfg.manifest_path()) as f:
        fit_ids_manifest = set(json.load(f)["records"].keys())
    with open(eval_cfg.manifest_path()) as f:
        eval_ids_manifest = set(json.load(f)["records"].keys())
    overlap = fit_ids_manifest & eval_ids_manifest
    assert not overlap, f"{model}: n2000 fit / fresh n1000 overlap by {len(overlap)} ids"

    fit_hidden, fit_y, fit_ids = load_matrix(fit_cfg, [POS])
    tr, va, te = splits(len(fit_ids))
    n_layers = fit_hidden[POS].shape[0]
    assert 0 <= layer < n_layers, f"{model}: layer {layer} out of range (0..{n_layers-1})"

    Xf = fit_hidden[POS][layer]                                   # [N2000, H]

    # ---- SEP: replicate correctness_eval.sep_single_fixed_layer's fit, capturing the objects ----
    thr = best_split(torch.tensor(fit_y[tr]))
    ybf = binarize_entropy(torch.tensor(fit_y), float(thr)).numpy()
    trv = tr[ybf[tr] >= 0]
    sep_scaler = StandardScaler().fit(Xf[trv])
    sep_clf = LogisticRegression(max_iter=1000).fit(sep_scaler.transform(Xf[trv]), ybf[trv])

    # ---- Ridge: continuous SE, alpha on val Spearman (linear_ceiling_probe.fit_probe) ----------
    ridge_model, ridge_scaler, ridge_alpha, ridge_val_s = fit_probe(Xf, fit_y, tr, va)

    # ---- persist ------------------------------------------------------------------------------
    out_dir = os.path.join(CKPT_ROOT, model)
    os.makedirs(out_dir, exist_ok=True)

    joblib.dump({
        "kind": "SEP",
        "model_name": model, "position": POS, "layer": int(layer),
        "scaler": sep_scaler, "clf": sep_clf,
        "se_threshold": float(thr),
        "binarise": "stage2.data.best_split on the n2000 TRAIN-split continuous SE",
        "label_convention": "1 = high SE (SE > threshold), 0 = low SE; -1 rows dropped from fit",
        "train_rows_used": int(len(trv)), "train_pos_rate": float(ybf[trv].mean()),
        "fit_dataset": FIT_DATASET, "fit_num_samples": FIT_N,
        "predict": "clf.predict_proba(scaler.transform(X[:, layer]))[:, 1]  -> P(high SE)",
    }, os.path.join(out_dir, "sep.joblib"))

    joblib.dump({
        "kind": "Ridge",
        "model_name": model, "position": POS, "layer": int(layer),
        "scaler": ridge_scaler, "model": ridge_model,
        "alpha": float(ridge_alpha), "alpha_grid": list(map(float, ALPHAS)),
        "val_spearman": float(ridge_val_s),
        "target": "continuous cluster_assignment_entropy (SE)",
        "fit_dataset": FIT_DATASET, "fit_num_samples": FIT_N,
        "predict": "model.predict(scaler.transform(X[:, layer]))  -> continuous SE estimate",
    }, os.path.join(out_dir, "ridge.joblib"))

    meta = {
        "model_name": model, "position": POS, "layer": int(layer),
        "fit_dataset": FIT_DATASET, "fit_num_samples": FIT_N,
        "split": {"seed": SEED, "test_size": TEST_SIZE, "val_size": VAL_SIZE,
                  "sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
                  "note": "linear_ceiling_probe.splits(len(fit_ids)); indices are sorted"},
        "train_ids": [fit_ids[i] for i in tr],
        "val_ids": [fit_ids[i] for i in va],
        "test_ids": [fit_ids[i] for i in te],
        "data_dir": tdir if tdir else "NFS-default",
        "incorrect_threshold": INCORRECT_THRESH,
        "sep_recipe": "correctness_eval.sep_single_fixed_layer (LogisticRegression, train-side best_split)",
        "ridge_recipe": "linear_ceiling_probe.fit_probe (alpha on val Spearman)",
        "generated_by": "amortized_ue/fit_save_baselines.py",
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # ---- sanity: our captured SEP reproduces sep_single_fixed_layer on the fresh n1000 set -----
    eval_hidden, eval_y, eval_ids = load_matrix(eval_cfg, [POS])
    eval_rows = np.arange(len(eval_ids))
    ref_prob, _, ref_choice, ref_thr, _ = sep_single_fixed_layer(
        fit_hidden, fit_y, tr, va, eval_hidden, eval_y, eval_rows, POS, layer)
    ours = sep_clf.predict_proba(sep_scaler.transform(eval_hidden[POS][layer]))[:, 1]
    dev = float(np.max(np.abs(np.asarray(ref_prob)[eval_rows] - ours)))
    assert dev < 1e-12, f"{model}: captured SEP != sep_single_fixed_layer (max dev {dev:.2e})"
    assert abs(ref_thr - float(thr)) < 1e-12 and tuple(ref_choice) == (POS, layer)
    print(f"  [{model}] fitted + saved to {out_dir}/  "
          f"(SEP thr={thr:.4f} alpha_sep=n/a | Ridge alpha={ridge_alpha:.0f} val_rho={ridge_val_s:.4f}) "
          f"| SEP==sep_single_fixed_layer OK (dev {dev:.1e})")
    return out_dir


# ------------------------------------------------------------------------------------------------
# LOAD + EVAL (no refit) -- reused verbatim by the OOD script later
# ------------------------------------------------------------------------------------------------
def load_ckpts(model):
    d = os.path.join(CKPT_ROOT, model)
    return (joblib.load(os.path.join(d, "sep.joblib")),
            joblib.load(os.path.join(d, "ridge.joblib")),
            json.load(open(os.path.join(d, "meta.json"))))


def eval_saved(model, eval_dataset, eval_n, data_dir):
    """Load the saved SEP+Ridge and score them on (eval_dataset, eval_n). No fitting."""
    sep, ridge, meta = load_ckpts(model)
    pos, layer = meta["position"], meta["layer"]

    # allow eval data on a different dir than fit (SQuAD is NFS-only); try data_dir then NFS
    ecfg = _cfg(model, eval_dataset, eval_n, data_dir)
    if not os.path.exists(ecfg.manifest_path()):
        ecfg = _cfg(model, eval_dataset, eval_n, None)
    eh, ey, eids = load_matrix(ecfg, [pos])
    X = eh[pos][layer]

    acc = load_accuracy(ecfg)
    incorrect = np.array([0 if acc[i] >= meta["incorrect_threshold"] else 1 for i in eids], dtype=int)

    sep_prob = sep["clf"].predict_proba(sep["scaler"].transform(X))[:, 1]
    ridge_pred = ridge["model"].predict(ridge["scaler"].transform(X))

    single_class = len(np.unique(incorrect)) < 2
    return {
        "model": model, "eval_dataset": eval_dataset, "N": int(len(eids)),
        "layer": f"{pos}:{layer}",
        "incorrect_rate": float(incorrect.mean()),
        "SE_spearman": 1.0,
        "SE_auroc_incorrect": None if single_class else float(roc_auc_score(incorrect, ey)),
        "SEP_spearman": _spearman(sep_prob, ey),
        "SEP_auroc_incorrect": None if single_class else float(roc_auc_score(incorrect, sep_prob)),
        "Ridge_spearman": _spearman(ridge_pred, ey),
        "Ridge_auroc_incorrect": None if single_class else float(roc_auc_score(incorrect, ridge_pred)),
        "ridge_alpha": ridge["alpha"],
    }


def verify_against_table(rows, tol=1e-6):
    if not os.path.exists(BASELINE_TABLE):
        print(f"\n  [warn] {BASELINE_TABLE} not found -- skipping reproduce-check")
        return []
    tbl = {r["model"]: r for r in json.load(open(BASELINE_TABLE))["rows"]}
    keys = ["SE_spearman", "SE_auroc_incorrect", "SEP_spearman", "SEP_auroc_incorrect",
            "Ridge_spearman", "Ridge_auroc_incorrect", "ridge_alpha", "incorrect_rate"]
    print(f"\n{'=' * 96}\nVERIFY saved-model predictions vs {BASELINE_TABLE} (tol {tol})\n{'=' * 96}")
    mism = []
    for r in rows:
        ref = tbl.get(r["model"])
        if not ref:
            print(f"  [warn] no table row for {r['model']}"); continue
        for k in keys:
            a, b = r[k], ref[k]
            if a is None or b is None:
                continue
            diff = abs(a - b)
            tag = "MISMATCH" if diff > tol else "ok"
            if diff > tol:
                mism.append({"model": r["model"], "key": k, "saved_eval": a, "table": b, "abs_diff": diff})
            print(f"  [{tag:>8s}] {r['model']:26s} {k:22s} saved={a:.8f}  table={b:.8f}  |Δ|={diff:.1e}")
    print("\n  ALL reproduced within tolerance." if not mism else f"\n  {len(mism)} MISMATCH(es)")
    return mism


def print_table(rows, title):
    print(f"\n{'=' * 120}\n{title}\n{'=' * 120}")
    print(f"{'model':26s}{'eval':12s}{'layer':>8s}{'N':>6s}{'inc':>7s}"
          f"{'SE_auc':>8s}{'SEP_rho':>9s}{'SEP_auc':>9s}{'Rdg_rho':>9s}{'Rdg_auc':>9s}{'alpha':>9s}")
    for r in rows:
        f = lambda x: "  n/a  " if x is None else f"{x:.3f}"
        print(f"{r['model']:26s}{r['eval_dataset']:12s}{r['layer']:>8s}{r['N']:>6d}{r['incorrect_rate']:>7.3f}"
              f"{f(r['SE_auroc_incorrect']):>8s}{r['SEP_spearman']:>9.3f}{f(r['SEP_auroc_incorrect']):>9s}"
              f"{r['Ridge_spearman']:>9.3f}{f(r['Ridge_auroc_incorrect']):>9s}{r['ridge_alpha']:>9.0f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_dir", default=None, help="Stage1Config.output_dir override (e.g. /data2/mn1025/stage1)")
    p.add_argument("--skip_fit", action="store_true", help="only load existing checkpoints and re-verify")
    p.add_argument("--out", default="amortized_ue/results/fit_save_baselines_id_eval.json")
    args = p.parse_args()

    if not args.skip_fit:
        print("FITTING + SAVING baseline checkpoints\n" + "-" * 60)
        for model, layer in LAYERS.items():
            fit_one(model, layer, args.data_dir)

    print("\nLOADING saved checkpoints + scoring on fresh TriviaQA n1000 (NO refit)\n" + "-" * 60)
    rows = [eval_saved(m, EVAL_DATASET, EVAL_N, args.data_dir) for m in LAYERS]
    print_table(rows, "SAVED-MODEL ID EVAL -- fresh TriviaQA n1000")
    mism = verify_against_table(rows)

    payload = {
        "_meta": {
            "description": "ID eval of the SAVED baseline SEP+Ridge checkpoints on fresh TriviaQA n1000. "
                           "Checkpoints under amortized_ue/checkpoints/baselines/<model>/. NO refit for OOD.",
            "checkpoint_root": CKPT_ROOT,
            "verified_against": BASELINE_TABLE,
            "generated_by": "amortized_ue/fit_save_baselines.py",
        },
        "checkpoints": {m: os.path.join(CKPT_ROOT, m) for m in LAYERS},
        "rows": rows,
        "reproduce_mismatches": mism,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n  -> saved to {args.out}")
    assert not mism, "saved-model eval does NOT reproduce the committed ID table -- see mismatches above"


if __name__ == "__main__":
    main()
