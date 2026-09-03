"""E69 — the squad OOD counterpart of E65 (big-tier 5×27B leave-one-LLM-out q_resp_only proxy).

E65 scored the LOLO `q_resp_only` proxy on each held-out big-tier model's shared-ID **trivia**
n1000 set (in-distribution). This closes the OOD row — same 15 checkpoints, no retraining — by
scoring the proxy on the held-out model's **squad** n1000 build (E55 scope), while the supervised
baselines (SEP, own-model ridge) are FIT on that model's own **trivia** n2000 and PREDICTED onto
the squad rows. This is exactly the E68 setup (small-tier Llama-3/DeepSeek), lifted to the 5 27B
targets and E65's baseline convention (`sep_single_val_selected`, own-model TBG+SLT ridge ceiling).

Held-out target was never in that fold's training pool (trained on the OTHER 4 big-tier models,
trivia only) and never squad — a genuine model+dataset compound shift.

Additive. Trains nothing. Reuses E65's `run_name`/`fold_ckpt_dir`/`splits` and its exact eval
body (the `eval_n != TRAIN_N` branch of `e65_bigtier_lolo.do_eval`), only the eval dataset changes.
Writes `results/e69_bigtier_lolo_squad_ood.json` — E65's outputs are untouched.

Env: amortized_stage2 + a free GPU (proxy forward pass, 3 seeds × 5 targets, squad n1000).
    python -m amortized_ue.e69_bigtier_lolo_squad_ood
"""
from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, rho
from amortized_ue.e65_bigtier_lolo import (
    BIGTIER, SUFFIX, ARM, TRAIN_N, SEEDS, fold_ckpt_dir, DEFAULT_DATA_DIR, RESULTS_DIR)

EVAL_DATASET = "squad"
EVAL_N = 1000
OUT_PATH = os.path.join(RESULTS_DIR, "e69_bigtier_lolo_squad_ood.json")


def run_name_ds(model, dataset, n):
    return f"{model}_{dataset}_n{n}_{SUFFIX[model]}"


def s1cfg_ds(model, dataset, n, data_dir):
    return Stage1Config(model_name=model, dataset=dataset, num_samples=n,
                        output_dir=data_dir, run_name=run_name_ds(model, dataset, n))


def do_check(data_dir):
    ok = True
    print(f"{'model':16s}{'squad run_name':44s}{'n_pt':>7s}{'n_manifest':>12s}   ready")
    for m in BIGTIER:
        cfg = s1cfg_ds(m, EVAL_DATASET, EVAL_N, data_dir)
        rd = cfg.records_dir()
        n_pt = len(glob.glob(os.path.join(rd, "*.pt"))) if os.path.isdir(rd) else 0
        man = cfg.manifest_path()
        n_man = None
        if os.path.isfile(man):
            with open(man) as f:
                n_man = len(json.load(f).get("records", {}))
        done = n_pt >= EVAL_N and n_man is not None and n_man >= EVAL_N
        ok &= done
        print(f"{m:16s}{run_name_ds(m, EVAL_DATASET, EVAL_N):44s}{n_pt:>7d}{str(n_man):>12s}   {'YES' if done else 'no'}")
    print(f"\nALL 5 squad n{EVAL_N} READY: {ok}")
    return ok


def do_eval(data_dir, bootstrap):
    import torch  # noqa: F401  (arm_preds needs it)
    from sklearn.metrics import roc_auc_score
    from amortized_ue.procrustes_e27_rank_fusion import arm_preds
    from amortized_ue.correctness_eval import (
        load_accuracy, sep_single_val_selected, paired_bootstrap_auc, ci)
    from amortized_ue.linear_ceiling_probe import fit_probe

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    if os.path.isfile(OUT_PATH):
        with open(OUT_PATH) as f:
            out = json.load(f)

    for held in BIGTIER:
        ck = fold_ckpt_dir(held)
        if len(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt"))) < len(SEEDS):
            print(f"[{held}] checkpoints missing -> skip")
            continue
        print(f"\n{'#' * 92}\n# E69 LOLO squad OOD — held out {held}  (proxy trained on the other 4 big-tier, trivia only)"
              f"\n#   proxy scored on {held} squad n{EVAL_N};  SEP/ridge fit on {held} trivia n{TRAIN_N}\n{'#' * 92}")

        # baselines FIT on the held-out model's own trivia n2000 train/val
        fit_cfg = s1cfg_ds(held, "trivia_qa", TRAIN_N, data_dir)
        fit_recs = load_records(fit_cfg)
        fit_ids = sorted(fit_recs.keys())
        tr, va, te = splits(len(fit_ids))

        # rows every predictor is SCORED on: the held-out model's squad n1000
        eval_cfg = s1cfg_ds(held, EVAL_DATASET, EVAL_N, data_dir)
        eval_recs = load_records(eval_cfg)
        eval_ids = sorted(eval_recs.keys())

        se_eval = np.array([eval_recs[i]["labels"]["cluster_assignment_entropy"] for i in eval_ids], dtype=float)
        acc_map = load_accuracy(eval_cfg)
        acc = np.array([acc_map[i] for i in eval_ids], dtype=float)
        incorrect = (acc < 0.5).astype(int)
        pos_rate = float(incorrect.mean())

        # ---- proxy (q_resp_only), 3 seeds seed-averaged, on the squad rows ----
        mp = arm_preds(ARM, held, EVAL_DATASET, EVAL_N, ckpt_dir=ck, data_dir=data_dir,
                       run_name=run_name_ds(held, EVAL_DATASET, EVAL_N))
        proxy_te = np.array([mp[i] for i in eval_ids], dtype=float)

        # ---- supervised in-model SEP (leak-free val-selected) + own-model ridge ceiling ----
        # fit hidden = held-out model's trivia n2000 (tr/va); eval hidden = its squad rows.
        hid_fit, y_fit, ids_fit = load_matrix(fit_cfg, ["TBG", "SLT"])
        assert ids_fit == fit_ids, "load_matrix id order != manifest order (fit)"
        hid_eval, y_eval_mat, ids_eval = load_matrix(eval_cfg, ["TBG", "SLT"])
        assert ids_eval == eval_ids, "load_matrix id order != manifest order (eval)"
        eval_rows = np.arange(len(eval_ids))
        assert float(np.max(np.abs(y_eval_mat[eval_rows].astype(float) - se_eval))) < 1e-5, "eval SE mismatch"

        sep_p, sep_au_se, sep_choice, thr, ybe, _ = sep_single_val_selected(
            hid_fit, y_fit, tr, va, hid_eval, y_eval_mat, eval_rows)
        sep_te = sep_p[eval_rows]

        rbest = (-np.inf, None, None)
        for pos in ("TBG", "SLT"):
            for L in range(hid_fit[pos].shape[0]):
                m, sc, alpha, val_s = fit_probe(hid_fit[pos][L], y_fit.astype(float), tr, va)
                if val_s > rbest[0]:
                    rbest = (val_s, (pos, int(L), float(alpha)),
                             m.predict(sc.transform(hid_eval[pos][L][eval_rows])))
        ridge_te = rbest[2]
        ridge_choice, ridge_val = rbest[1], float(rbest[0])
        del hid_fit, hid_eval

        # ---- score ----
        yb_te = ybe[eval_rows]
        v = yb_te >= 0
        preds = {"proxy_q_resp_only": proxy_te, "true_semantic_entropy": se_eval,
                 "sep_single_val_selected": sep_te, "ridge_own_model_TBG": ridge_te}
        label_free = {"proxy_q_resp_only": True, "true_semantic_entropy": False,
                      "sep_single_val_selected": False, "ridge_own_model_TBG": False}
        metrics = {}
        print(f"  {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}  label-free")
        for name, s in preds.items():
            au_inc = float(roc_auc_score(incorrect, s)) if len(np.unique(incorrect)) == 2 else float("nan")
            au_se = float(roc_auc_score(yb_te[v], s[v])) if len(np.unique(yb_te[v])) == 2 else float("nan")
            metrics[name] = {"auroc_incorrect": au_inc, "auroc_binarised_se": au_se,
                             "spearman_se": rho(s, se_eval), "label_free_on_target": label_free[name]}
            print(f"  {name:26s}{au_inc:>11.3f}{au_se:>10.3f}{rho(s, se_eval):>9.3f}"
                  f"  {'yes' if label_free[name] else 'NO'}")

        boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
        vs = {}
        for base in ("sep_single_val_selected", "true_semantic_entropy"):
            c = ci(boot["proxy_q_resp_only"] - boot[base])
            vs[base] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}
            print(f"  Δ AUROC_inc (proxy − {base}): {c['mean']:+.3f} "
                  f"[{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
                  f"({'excludes 0' if vs[base]['ci_excludes_zero'] else 'includes 0'})")

        out[held] = {
            "held_out": held, "sources": [m for m in BIGTIER if m != held],
            "eval_dataset": EVAL_DATASET, "eval_n": EVAL_N, "fit_dataset": "trivia_qa", "fit_n": TRAIN_N,
            "n_test": len(eval_ids), "positive_rate_incorrect": pos_rate,
            "mean_accuracy": float(acc.mean()), "best_split": float(thr),
            "sep_choice": list(sep_choice), "sep_auroc_vs_se": float(sep_au_se),
            "ridge_choice_pos_layer_alpha": list(ridge_choice), "ridge_val_spearman": ridge_val,
            "bootstrap_resamples": bootstrap, "metrics": metrics,
            "bootstrap_delta_auroc_incorrect_vs": vs,
            "proxy_te_preds": [float(x) for x in proxy_te],
            "true_se_te": [float(x) for x in se_eval],
            "sep_te_preds": [float(x) for x in sep_te],
            "incorrect_te": [int(x) for x in incorrect],
            "te_ids": list(eval_ids),
        }
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> saved {len([k for k in out if not k.startswith('_')])} fold(s) to {OUT_PATH}")

    folds = [k for k in out if not k.startswith("_")]
    if folds:
        names = list(out[folds[0]]["metrics"])
        print("\n" + "=" * 92)
        print("E69 SUMMARY — squad OOD, AUROC_incorrect per held-out big-tier model")
        print("=" * 92)
        print(f"{'predictor':26s}" + "".join(f"{h[:11]:>12s}" for h in folds) + f"{'MEAN':>9s}")
        summary = {}
        for n in names:
            vals = [out[h]["metrics"][n]["auroc_incorrect"] for h in folds]
            summary[n] = {"per_fold": {h: out[h]["metrics"][n]["auroc_incorrect"] for h in folds},
                          "mean_auroc_incorrect": float(np.nanmean(vals)),
                          "mean_auroc_se": float(np.nanmean([out[h]["metrics"][n]["auroc_binarised_se"] for h in folds])),
                          "mean_spearman_se": float(np.nanmean([out[h]["metrics"][n]["spearman_se"] for h in folds])),
                          "label_free_on_target": all(out[h]["metrics"][n]["label_free_on_target"] for h in folds)}
            print(f"{n:26s}" + "".join(f"{x:>12.3f}" for x in vals) + f"{np.nanmean(vals):>9.3f}")
        out["_summary"] = summary
        with open(OUT_PATH, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {OUT_PATH}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["check", "eval", "all"], default="all")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    if args.stage == "check":
        raise SystemExit(0 if do_check(args.data_dir) else 1)
    if not do_check(args.data_dir):
        raise SystemExit("STOP: not all 5 big-tier squad n1000 datasets are ready.")
    if args.stage in ("eval", "all"):
        do_eval(args.data_dir, args.bootstrap)


if __name__ == "__main__":
    main()
