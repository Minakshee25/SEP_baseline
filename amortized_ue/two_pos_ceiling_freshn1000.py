"""E57 two-position ridge ceiling, re-scored on the fresh trivia_qa n1000 held-out set.

Identical method to `two_pos_ceiling.py` (E57) in every respect -- same `load_matrix`,
`splits`, `fit_probe`, `ridge_concat`, same leak-free FIRST_POS (position, layer) per model,
same complement-layer sweep selected on the n2000 VALIDATION Spearman (never on the eval set),
same alpha selection. The ONLY change: the final held-out predictions are scored on the
model's fresh `trivia_qa_n1000_full` set (E23/E28/E33 held-out batch, disjoint from n2000)
instead of on the 200-row n2000 test split.

No new layer-selection logic: the first position is E57's fixed leak-free best single
(position, layer); the complement layer is still the argmax of the n2000-val Spearman. The
n1000 set is used ONLY to evaluate the already-selected two-position ridge.

Reads Stage-1 records, trains only throw-away sklearn ridges, writes one JSON.

    python -m amortized_ue.two_pos_ceiling_freshn1000 --data_dir /data2/$USER/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits
from amortized_ue.two_pos_ceiling import ridge_concat, FIRST_POS, OTHER


def run_model(model, data_dir, fit_ns, fresh_ns):
    kw = {"output_dir": data_dir} if data_dir else {}
    positions = ["TBG", "SLT"]

    fit_cfg = Stage1Config(model_name=model, dataset="trivia_qa", num_samples=fit_ns, **kw)
    hid, y, ids = load_matrix(fit_cfg, positions)
    tr, va, te = splits(len(ids))
    n_layers = {p: hid[p].shape[0] for p in positions}

    fresh_cfg = Stage1Config(model_name=model, dataset="trivia_qa", num_samples=fresh_ns, **kw)
    fresh_dir = fresh_cfg.run_dir()
    assert os.path.isdir(fresh_dir), f"missing fresh set: {fresh_dir}"
    assert os.path.abspath(fresh_dir) != os.path.abspath(fit_cfg.run_dir())
    hid_fr, y_fr, fresh_ids = load_matrix(fresh_cfg, positions)
    overlap = len(set(ids) & set(fresh_ids))

    p1, l1 = FIRST_POS[model]
    p2 = OTHER[p1]

    # single-position baseline (first position alone) -- ridge_concat scores the external
    # matrix in its "ood" fields; here the external matrix is the fresh trivia n1000 set.
    base = ridge_concat(hid, y, tr, va, te, hid_fr, y_fr, [(p1, l1)])

    sweep = []
    for l2 in range(n_layers[p2]):
        r = ridge_concat(hid, y, tr, va, te, hid_fr, y_fr, [(p1, l1), (p2, l2)])
        r["complement_layer"] = l2
        sweep.append(r)

    by_val = max(sweep, key=lambda r: r["val"])   # selection on n2000 val -- identical to E57

    return {
        "model": model,
        "fit_dataset": f"trivia_qa n{fit_ns}",
        "eval_dataset": f"trivia_qa n{fresh_ns} (fresh held-out)",
        "split": {"train": len(tr), "val": len(va), "n2000_test": len(te)},
        "n_eval": len(fresh_ids),
        "id_overlap_fit_vs_eval": overlap,
        "first_position": {"pos": p1, "layer": l1},
        "single_position_baseline": {
            "val_spearman": base["val"],
            "n2000_test_spearman": base["id"], "n2000_test_pearson": base["id_pearson"],
            "freshn1000_spearman": base["ood"], "freshn1000_pearson": base["ood_pearson"],
            "alpha": base["alpha"],
        },
        "val_selected_two_position": {
            "positions": [[p1, l1], [p2, by_val["complement_layer"]]],
            "val_spearman": by_val["val"],
            "n2000_test_spearman": by_val["id"], "n2000_test_pearson": by_val["id_pearson"],
            "freshn1000_spearman": by_val["ood"], "freshn1000_pearson": by_val["ood_pearson"],
            "alpha": by_val["alpha"],
        },
        "sweep": sweep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(FIRST_POS))
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--fit_num_samples", type=int, default=2000)
    ap.add_argument("--fresh_num_samples", type=int, default=1000)
    ap.add_argument("--out", default="scratch_xllm/two_pos_ceiling_freshn1000.json")
    args = ap.parse_args()

    out = {}
    for model in args.models:
        print(f"\n=== {model} ===")
        res = run_model(model, args.data_dir, args.fit_num_samples, args.fresh_num_samples)
        out[model] = res
        b = res["single_position_baseline"]
        p1 = res["first_position"]
        vs = res["val_selected_two_position"]
        print(f"  eval set: {res['eval_dataset']}  n={res['n_eval']}  "
              f"fit∩eval id overlap = {res['id_overlap_fit_vs_eval']}")
        print(f"  single pos {p1['pos']}:{p1['layer']}          "
              f"fresh-n1000 rho {b['freshn1000_spearman']:.4f} / r {b['freshn1000_pearson']:.4f}   "
              f"(n2000-test rho {b['n2000_test_spearman']:.4f} / r {b['n2000_test_pearson']:.4f})")
        print(f"  + val-sel complement {vs['positions'][1][0]}:{vs['positions'][1][1]}  "
              f"val rho {vs['val_spearman']:.4f}  "
              f"fresh-n1000 rho {vs['freshn1000_spearman']:.4f} / r {vs['freshn1000_pearson']:.4f}   "
              f"(n2000-test rho {vs['n2000_test_spearman']:.4f} / r {vs['n2000_test_pearson']:.4f})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
