"""Evaluate the EXACT TriviaQA-trained baseline checkpoints on SQuAD n1000 OOD -- NO refitting.

Loads only amortized_ue/checkpoints/baselines/<model>/{sep,ridge}.joblib via
amortized_ue.fit_save_baselines.{load_ckpts, eval_saved}. Nothing is fitted, tuned, alpha-selected,
layer-selected, or re-thresholded on SQuAD: the saved TriviaQA scaler / LogisticRegression / Ridge
/ SE threshold / alpha are applied as-is to ALL SQuAD n1000 rows.

    python -m amortized_ue.baseline_table_squad_ood --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import os
import json
import argparse

from amortized_ue.fit_save_baselines import load_ckpts, eval_saved, LAYERS

OUT = "amortized_ue/results/baseline_table_squad_ood_n1000.json"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default="/data2/mn1025/stage1")
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    rows = []
    print(f"\n{'AUDIT: loaded checkpoints (no refit)':<50}")
    print(f"{'model':26s}{'position':>10s}{'layer':>7s}{'ridge_alpha':>13s}{'sep_thr':>10s}")
    for model in LAYERS:
        sep, ridge, meta = load_ckpts(model)
        print(f"{model:26s}{meta['position']:>10s}{meta['layer']:>7d}{ridge['alpha']:>13.0f}"
              f"{sep['se_threshold']:>10.4f}")
        assert (meta["position"], meta["layer"]) == ("TBG", LAYERS[model]), f"{model}: unexpected layer"
        r = eval_saved(model, "squad", 1000, args.data_dir)
        rows.append(r)

    print(f"\n{'='*104}\nSQuAD n1000 OOD -- saved TriviaQA baseline checkpoints, no refit\n{'='*104}")
    print(f"{'Model':26s}{'Method':9s}{'Spearman vs SE':>16s}{'AUROC incorrect':>18s}{'  (N, inc_rate)'}")
    table = []
    for r in rows:
        meth = [("True SE", r["SE_spearman"], r["SE_auroc_incorrect"]),
                ("SEP", r["SEP_spearman"], r["SEP_auroc_incorrect"]),
                ("Ridge", r["Ridge_spearman"], r["Ridge_auroc_incorrect"])]
        for i, (name, sp, au) in enumerate(meth):
            mlab = r["model"] if i == 0 else ""
            extra = f"  (N={r['N']}, inc={r['incorrect_rate']:.3f})" if i == 0 else ""
            print(f"{mlab:26s}{name:9s}{sp:>16.3f}{au:>18.3f}{extra}")
            table.append({"model": r["model"], "method": name, "layer": r["layer"],
                          "ridge_alpha": r["ridge_alpha"] if name == "Ridge" else None,
                          "spearman_vs_se": sp, "auroc_incorrect": au,
                          "N": r["N"], "incorrect_rate": r["incorrect_rate"]})
        print()

    payload = {
        "_meta": {
            "description": "SQuAD n1000 OOD eval of the EXACT TriviaQA-trained baseline SEP+Ridge "
                           "checkpoints. No refit / no tuning / no alpha or layer selection / no "
                           "re-thresholding on SQuAD. All SQuAD n1000 rows evaluated.",
            "checkpoint_root": "amortized_ue/checkpoints/baselines",
            "loaded_via": "amortized_ue.fit_save_baselines.{load_ckpts, eval_saved}",
            "id_counterpart": "amortized_ue/results/baseline_table_freshn1000.json",
            "generated_by": "amortized_ue/baseline_table_squad_ood.py",
        },
        "rows": rows,
        "table": table,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  -> saved to {args.out}")


if __name__ == "__main__":
    main()
