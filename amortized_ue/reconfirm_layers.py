"""Reconfirm the best (position, layer) per model for Exp-2 / E35 alignment — LEAK-FREE.

Fixes the selection leak in `linear_ceiling_probe.main` (its `best_id` is chosen on
`id_test_spearman`, ~line 131 — the TEST set). Here the (position, layer) is selected on
VALIDATION Spearman (optionally k-fold CV over train+val for stability, which resolves the
Llama-3 single-split instability); the held-out TEST Spearman is only reported, never used to
pick. Reuses the leak-free pieces of `linear_ceiling_probe` (load_matrix / splits / fit_probe /
rho) so the ridge fit + alpha-on-val + id-join stay identical.

Leakage self-audit (printed at start): StandardScaler fit on TRAIN rows only (inside fit_probe);
alpha + layer selected on VAL/CV, never test; records joined by id via sorted keys; target is the
continuous `cluster_assignment_entropy`.

se_probes env, CPU only. Reads Stage-1 records read-only; writes one JSON per model + a summary.
    python -m amortized_ue.reconfirm_layers                       # all 4, single-split val
    python -m amortized_ue.reconfirm_layers --cv 5                # 5-fold CV layer selection
    python -m amortized_ue.reconfirm_layers --models Meta-Llama-3-8B-Instruct --cv 5
    python -m amortized_ue.reconfirm_layers --data_dir /data2/$USER/stage1   # dodge degraded NFS
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from sklearn.model_selection import KFold

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho, ALPHAS

DEFAULT_MODELS = [
    "Llama-2-7b-chat",
    "Mistral-7B-Instruct-v0.2",
    "Meta-Llama-3-8B-Instruct",
    "deepseek-llm-7b-chat",
]


def cv_layer_score(X, y, pool, k, seed=42):
    """Mean validation Spearman across k folds over `pool` (train+val union).

    Leak-free: each fold fits the scaler + ridge on the fold-train and scores the fold-val;
    the outer test rows are never in `pool`. Per-fold alpha is chosen on that fold's val
    (via fit_probe). Returns the mean fold-val Spearman (the layer's CV score).
    """
    kf = KFold(n_splits=k, shuffle=True, random_state=seed)
    scores = []
    for tr_i, va_i in kf.split(pool):
        m, sc, _, val_s = fit_probe(X, y, pool[tr_i], pool[va_i])
        scores.append(val_s)
    return float(np.mean(scores))


def pick_for_model(model, dataset, num_samples, positions, cv, topk, data_dir):
    kw = {"output_dir": data_dir} if data_dir else {}
    cfg = Stage1Config(model_name=model, dataset=dataset, num_samples=num_samples, **kw)
    hidden, y, ids = load_matrix(cfg, positions)          # ids = sorted keys -> join by id
    tr, va, te = splits(len(ids))
    pool = np.sort(np.concatenate([tr, va]))              # train+val for CV; te stays held out
    n_layers = hidden[positions[0]].shape[0]

    rows = []
    for pos in positions:
        for layer in range(n_layers):
            X = hidden[pos][layer]                         # [N, H]
            # SELECTION SCORE — val (single split) or CV mean-val; never touches te.
            if cv and cv > 1:
                sel = cv_layer_score(X, y, pool, cv)
            else:
                _, _, _, sel = fit_probe(X, y, tr, va)
            # REPORTED test score — refit on the original (tr, va), eval te. Report only.
            m, sc, alpha, val_s = fit_probe(X, y, tr, va)
            id_test = rho(m.predict(sc.transform(X[te])), y[te])
            rows.append({"position": pos, "layer": layer, "alpha": alpha,
                         "select_score": sel, "val_spearman": val_s,
                         "id_test_spearman": id_test})

    rows_sorted = sorted(rows, key=lambda r: -r["select_score"])
    best = rows_sorted[0]
    return {
        "model": model, "dataset": dataset, "num_samples": num_samples,
        "selection": f"cv{cv}" if cv and cv > 1 else "val_single_split",
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "best": {"position": best["position"], "layer": best["layer"],
                 "select_score": best["select_score"], "id_test_spearman": best["id_test_spearman"]},
        "topk": [{"position": r["position"], "layer": r["layer"],
                  "select_score": round(r["select_score"], 4),
                  "id_test_spearman": round(r["id_test_spearman"], 4)} for r in rows_sorted[:topk]],
        "results": rows,
    }


def main():
    p = argparse.ArgumentParser(description="Leak-free per-model best (pos,layer) reconfirmation.")
    p.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--positions", nargs="+", default=["TBG", "SLT"])
    p.add_argument("--cv", type=int, default=0, help="k for k-fold CV layer selection; 0/1 = single val split")
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--data_dir", default=None, help="override Stage-1 output_dir (e.g. /data2 staged copy)")
    p.add_argument("--out_dir", default="scratch_xllm")
    args = p.parse_args()

    print("=" * 78)
    print("LEAKAGE SELF-AUDIT")
    print("  [1] selection metric : %s  (never id_test)" % ("cv%d mean-val" % args.cv if args.cv > 1 else "val single split"))
    print("  [2] scaler           : StandardScaler fit on TRAIN rows only (inside fit_probe)")
    print("  [3] join             : records keyed by id (sorted keys), aligned across pos/layer")
    print("  [4] target           : continuous cluster_assignment_entropy; alpha chosen on val")
    print("=" * 78)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = []
    for model in args.models:
        print(f"\n--- {model} ---")
        out = pick_for_model(model, args.dataset, args.num_samples, args.positions,
                             args.cv, args.topk, args.data_dir)
        tag = out["selection"]
        path = os.path.join(args.out_dir, f"reconfirm_{model}_{tag}.json")
        with open(path, "w") as f:
            json.dump(out, f, indent=1)
        b = out["best"]
        print(f"  split {out['split']}  selection={tag}")
        print(f"  BEST (leak-free): {b['position']} L{b['layer']}  "
              f"select={b['select_score']:.4f}  id_test={b['id_test_spearman']:.4f}")
        print("  plateau (top-%d by selection score):" % args.topk)
        for r in out["topk"]:
            print(f"    {r['position']} L{r['layer']:<2}  select={r['select_score']:.4f}  id_test={r['id_test_spearman']:.4f}")
        print(f"  wrote {path}")
        summary.append((model, b["position"], b["layer"], b["select_score"], b["id_test_spearman"]))

    print("\n" + "=" * 78)
    print("SUMMARY — leak-free best (position, layer) per model")
    print(f"  {'model':<28}{'best':>10}{'select':>9}{'id_test':>9}")
    for model, pos, layer, sel, idt in summary:
        print(f"  {model:<28}{pos+' L'+str(layer):>10}{sel:>9.4f}{idt:>9.4f}")
    print("=" * 78)


if __name__ == "__main__":
    main()
