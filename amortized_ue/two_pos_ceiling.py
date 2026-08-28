"""Combined two-position ridge ceiling for Mistral-v0.2 / Llama-3-8B / DeepSeek-7B.

Mirrors `linear_ceiling_probe.py`'s ridge-sweep method exactly (same load_matrix / splits /
fit_probe / rho -> ridge from hidden state to continuous cluster_assignment_entropy, alpha
chosen on val Spearman, test Spearman reported), extended to TWO positions the way E8c formed
Llama-2's TBG:22 + SLT:15 ceiling: concatenate the two (position, layer) hidden-state vectors
feature-wise, then one ridge on the concatenation.

Procedure per model:
  1. Fix the first position at the model's LEAK-FREE best (position, layer) from
     reconfirm_layers.py (CV5): Mistral TBG:31, Llama-3 TBG:31, DeepSeek SLT:16.
  2. Sweep every layer of the COMPLEMENTARY position; for each, fit the concatenated ridge and
     score it on VALIDATION Spearman (never test). Pick the complement layer with the best val
     score -- leak-free, same selection philosophy as reconfirm_layers.py.
  3. Report the held-out TEST Spearman + Pearson r (ID) and the all-rows squad Spearman +
     Pearson r (OOD) at that val-selected two-position combo -- both correlations computed on
     the identical fitted predictions. Also report the best-OOD complement for context.

Read-only: reads Stage-1 records, trains only throw-away sklearn ridges, writes one JSON.

    python -m amortized_ue.two_pos_ceiling --data_dir /data2/$USER/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
from scipy.stats import pearsonr

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho


def prho(a, b) -> float:
    """Pearson r with the same degenerate-input guard as linear_ceiling_probe.rho."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = pearsonr(a, b)[0]
    return 0.0 if (r is None or np.isnan(r)) else float(r)

# leak-free best single (position, layer) per model -- reconfirm_layers.py --cv 5
FIRST_POS = {
    "Llama-2-7b-chat":          ("TBG", 30),
    "Mistral-7B-Instruct-v0.2": ("TBG", 31),
    "Meta-Llama-3-8B-Instruct": ("TBG", 31),
    "deepseek-llm-7b-chat":     ("SLT", 16),
}
OTHER = {"TBG": "SLT", "SLT": "TBG"}


def ridge_concat(hid, y, tr, va, te, hid_ood, y_ood, pos_layers):
    """One ridge on the feature-concatenation of pos_layers (same fit_probe as
    linear_ceiling_probe). Returns val Spearman, held-out test Spearman (ID), all-rows
    squad Spearman (OOD)."""
    X = np.concatenate([hid[p][l] for p, l in pos_layers], axis=1)     # [N, sum(H)]
    m, sc, alpha, val_s = fit_probe(X, y, tr, va)
    id_pred = m.predict(sc.transform(X[te]))                           # exact fitted preds
    id_s, id_p = rho(id_pred, y[te]), prho(id_pred, y[te])
    ood_s = ood_p = float("nan")
    if hid_ood is not None:
        Xo = np.concatenate([hid_ood[p][l] for p, l in pos_layers], axis=1)
        ood_pred = m.predict(sc.transform(Xo))                         # same predictions scored below
        ood_s, ood_p = rho(ood_pred, y_ood), prho(ood_pred, y_ood)
    return {"alpha": alpha, "val": float(val_s),
            "id": float(id_s), "id_pearson": float(id_p),
            "ood": float(ood_s), "ood_pearson": float(ood_p)}


def run_model(model, data_dir, id_ns, ood_ns):
    kw = {"output_dir": data_dir} if data_dir else {}
    positions = ["TBG", "SLT"]

    fit_cfg = Stage1Config(model_name=model, dataset="trivia_qa", num_samples=id_ns, **kw)
    hid, y, ids = load_matrix(fit_cfg, positions)
    tr, va, te = splits(len(ids))
    n_layers = {p: hid[p].shape[0] for p in positions}

    hid_ood = y_ood = None
    ood_dir = Stage1Config(model_name=model, dataset="squad", num_samples=ood_ns, **kw).run_dir()
    if os.path.isdir(ood_dir):
        ocfg = Stage1Config(model_name=model, dataset="squad", num_samples=ood_ns, **kw)
        hid_ood, y_ood, ood_ids = load_matrix(ocfg, positions)

    p1, l1 = FIRST_POS[model]
    p2 = OTHER[p1]

    # single-position baseline (first position alone), same split/metric
    base = ridge_concat(hid, y, tr, va, te, hid_ood, y_ood, [(p1, l1)])

    sweep = []
    for l2 in range(n_layers[p2]):
        r = ridge_concat(hid, y, tr, va, te, hid_ood, y_ood, [(p1, l1), (p2, l2)])
        r["complement_layer"] = l2
        sweep.append(r)

    by_val = max(sweep, key=lambda r: r["val"])
    by_ood = max((r for r in sweep if not np.isnan(r["ood"])), key=lambda r: r["ood"],
                 default=None)

    return {
        "model": model,
        "split": {"train": len(tr), "val": len(va), "test": len(te)},
        "id_dataset": f"trivia_qa n{id_ns}",
        "ood_dataset": (f"squad n{ood_ns}" if hid_ood is not None else "none"),
        "first_position": {"pos": p1, "layer": l1},
        "single_position_baseline": base,
        "val_selected": {
            "positions": [[p1, l1], [p2, by_val["complement_layer"]]],
            "val_spearman": by_val["val"], "id_spearman": by_val["id"],
            "id_pearson": by_val["id_pearson"],
            "ood_spearman": by_val["ood"], "ood_pearson": by_val["ood_pearson"],
            "alpha": by_val["alpha"],
        },
        "ood_selected_context": (None if by_ood is None else {
            "positions": [[p1, l1], [p2, by_ood["complement_layer"]]],
            "val_spearman": by_ood["val"], "id_spearman": by_ood["id"],
            "id_pearson": by_ood["id_pearson"],
            "ood_spearman": by_ood["ood"], "ood_pearson": by_ood["ood_pearson"],
        }),
        "sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(FIRST_POS))
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--id_num_samples", type=int, default=2000)
    ap.add_argument("--ood_num_samples", type=int, default=1000)
    ap.add_argument("--out", default="scratch_xllm/two_pos_ceiling.json")
    args = ap.parse_args()

    out = {}
    for model in args.models:
        print(f"\n=== {model} ===")
        res = run_model(model, args.data_dir, args.id_num_samples, args.ood_num_samples)
        out[model] = res
        b = res["single_position_baseline"]
        p1 = res["first_position"]
        vs = res["val_selected"]
        print(f"  first pos (leak-free best single): {p1['pos']}:{p1['layer']}  "
              f"ID rho {b['id']:.4f} / r {b['id_pearson']:.4f}  "
              f"OOD rho {b['ood']:.4f} / r {b['ood_pearson']:.4f}")
        print(f"  + val-selected complement {vs['positions'][1][0]}:{vs['positions'][1][1]}  "
              f"-> val rho {vs['val_spearman']:.4f}  "
              f"ID rho {vs['id_spearman']:.4f} / r {vs['id_pearson']:.4f}  "
              f"OOD rho {vs['ood_spearman']:.4f} / r {vs['ood_pearson']:.4f}")
        oc = res["ood_selected_context"]
        if oc:
            print(f"  (context) best-OOD complement {oc['positions'][1][0]}:{oc['positions'][1][1]}"
                  f"  ID rho {oc['id_spearman']:.4f} / r {oc['id_pearson']:.4f}  "
                  f"OOD rho {oc['ood_spearman']:.4f} / r {oc['ood_pearson']:.4f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
