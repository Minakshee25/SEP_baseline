"""Correctness-based evaluation of the E37 multi-target LOLO proxy (E38 — strictly additive).

E37 scored the leave-one-LLM-out proxy against the *semantic-entropy* label only (Spearman).
E31 showed SE-fidelity != correctness (-0.10 to -0.15 AUROC) for the E27/E30-era predictors, but
the E37 proxy was never re-scored that way. This closes that gap on the SAME 200 held-out rows
E37 reported, so every number here is directly comparable to the E37 Spearman table.

Predictors, all on the held-out target's `te` (n2000 test split, N=200), oriented higher => more
uncertain => more likely INCORRECT:
    true_semantic_entropy        the 10-sample CAE label (sampling upper bound; needs N samples)
    sep_single_best_layer        supervised in-model SEP, leak-free val-selected (== E31/E30)
    sep_5layer_concat            supervised in-model SEP, top-5 layer concat (arXiv Tbl 4)
    ridge_z                      E37's baseline: ridge on the pooled ALIGNED z of the 3 sources
    z / z_q / z_q_resp           E37 proxy arms that consume the aligned hidden state
    q_only / q_resp_only         E37 proxy text arms (NO target hidden state)
    fuse_z_qresp                 E37's winner: label-free rank fusion of z + q_resp_only
    random_control

The proxy/ridge predictors are LABEL-FREE on the held-out target (trained on the other 3 models;
the target's own SE labels are never used) -- unlike the SEP baselines, which are supervised on the
target's own labels. That asymmetry is the point of the comparison and is flagged per predictor.

Reuses read-only: correctness_eval (metrics, SEP fits, bootstrap), exp2_run (alignment, fold build,
ridge), linear_ceiling_probe (load_matrix/splits). Trains nothing; reads E37's saved predictions.

Run from the repo root in `se_probes` (CPU only -- no proxy forward pass, predictions are on disk):
    python -m amortized_ue.correctness_eval_e37 --data_dir /data2/mn1025/stage1
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.correctness_eval import (
    load_accuracy, sep_single_val_selected, sep_5layer_concat,
    accuracy_coverage, prediction_rejection_ratio, paired_bootstrap_auc, ci, COVERAGES)
from amortized_ue import exp2_run as E2

# E37 arms in report order, then the fusion. Every one is label-free on the held-out target.
PROXY_ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]
LONG = {v: k for k, v in E2.SHORT.items()}          # "Mistral" -> "Mistral-7B-Instruct-v0.2"
BASELINES = ["sep_single_best_layer", "true_semantic_entropy"]


def ridge_preds_for_fold(data, sp, target_long, sources_long, al, fsc):
    """Recompute E37's ridge baseline AND keep its per-example te predictions (E37 saved only the
    scalar Spearman -- this is the missing piece the E37 paired bootstrap was blocked on)."""
    train, val, tgt, info = E2.build_fold(data, sp, target_long, sources_long, al, fsc)
    from sklearn.linear_model import Ridge
    best = None
    for a in E2.ALPHAS:
        r = Ridge(alpha=a).fit(train["z"], train["y"])
        s = E2.rho(r.predict(val["z"]), val["y"])
        if best is None or s > best[0]:
            best = (s, a, r)
    return best[2].predict(tgt["z"]), best[1], tgt["y"], info


def evaluate_fold(fold, data, sp, al, fsc, bootstrap=10000, data_dir=None):
    short = fold["info"]["target"]
    target = LONG[short]
    sources = [LONG[s] for s in fold["info"]["sources"]]
    print("\n" + "#" * 92)
    print(f"# E37 CORRECTNESS — held-out target {short}  (sources: {'+'.join(fold['info']['sources'])})")
    print("#" * 92)

    # ---- reproduce the exact te rows E37 scored on, and PROVE the id mapping is right -------------
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000,
                       **({"output_dir": data_dir} if data_dir else {}))
    with open(cfg.manifest_path()) as f:
        all_ids = sorted(json.load(f)["records"].keys())
    tr, va, te = splits(len(all_ids))
    te_ids = [all_ids[i] for i in te]
    y_saved = np.asarray(fold["target_y"], dtype=float)
    assert len(te_ids) == len(y_saved), f"te rows {len(te_ids)} != saved target_y {len(y_saved)}"

    # ---- accuracy + SE labels for those ids ------------------------------------------------------
    acc_map = load_accuracy(cfg)
    assert set(te_ids).issubset(acc_map), "te ids missing from the accuracy manifest"
    acc = np.array([acc_map[i] for i in te_ids], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    pos_rate = float(incorrect.mean())

    # ---- predictors ------------------------------------------------------------------------------
    preds, label_free, per_seed = {}, {}, {}

    preds["true_semantic_entropy"] = y_saved
    label_free["true_semantic_entropy"] = False          # IS the target's own N-sample label

    # supervised in-model SEP on the target's own n2000 (fit tr, select on va, score te) -- needs the
    # all-layer hidden states, so this is the one heavy load. Also re-derives the SE labels, which
    # double-checks the id mapping above against an independent code path.
    hid, y_all, ids_lm = load_matrix(cfg, ["TBG", "SLT"])
    assert ids_lm == all_ids, "load_matrix id order differs from the manifest order"
    y_te = y_all[te].astype(float)
    max_dev = float(np.max(np.abs(y_te - y_saved)))
    print(f"  ID-MAPPING AUDIT: max|SE(te ids) - E37 target_y| = {max_dev:.3e}  "
          f"({'MATCH' if max_dev < 1e-5 else 'MISMATCH -- STOP'})")
    assert max_dev < 1e-5, "te ids do not reproduce E37's target_y -- id mapping is wrong"

    sep1_p, sep1_au_se, sep1_choice, thr, ybe, _ = sep_single_val_selected(
        hid, y_all, tr, va, hid, y_all, te)
    sep5_p, sep5_au_se, sep5_choice = sep_5layer_concat(hid, y_all, tr, va, hid, y_all, te)
    preds["sep_single_best_layer"] = sep1_p[te]
    preds["sep_5layer_concat"] = sep5_p[te]
    label_free["sep_single_best_layer"] = label_free["sep_5layer_concat"] = False
    yb_eval = ybe[te]
    del hid
    print(f"  SEP single {sep1_choice} SE-AUROC {sep1_au_se:.3f} | 5-layer {sep5_choice[0]} "
          f"L{sep5_choice[1][0]}-{sep5_choice[1][-1]} SE-AUROC {sep5_au_se:.3f}")

    # E37 ridge baseline (recomputed; per-example preds were never saved)
    r_pred, r_alpha, r_y, _ = ridge_preds_for_fold(data, sp, target, sources, al, fsc)
    assert float(np.max(np.abs(np.asarray(r_y, dtype=float) - y_saved))) < 1e-5, \
        "rebuilt fold labels differ from E37's target_y"
    preds["ridge_z"] = r_pred
    label_free["ridge_z"] = True
    print(f"  ridge_z rebuilt (alpha={r_alpha:g}): Spearman {E2.rho(r_pred, y_saved):.3f} "
          f"vs E37-saved {fold['ridge_z']:.3f}")

    # E37 proxy arms: primary = mean over the 3 seeds' predictions; per-seed kept for the spread
    for arm in PROXY_ARMS:
        if arm not in fold["arms"]:
            continue
        P = np.asarray(fold["arms"][arm]["te_pred_by_seed"], dtype=float)     # [n_seeds, 200]
        preds[arm] = P.mean(0)
        label_free[arm] = True
        per_seed[arm] = P

    # E37's winner: rank fusion of z + q_resp_only, fused per seed then averaged (E37 convention)
    if "z" in per_seed and "q_resp_only" in per_seed:
        F = np.stack([E2.rank_fuse(per_seed["z"][s], per_seed["q_resp_only"][s])
                      for s in range(per_seed["z"].shape[0])])
        preds["fuse_z_qresp"] = F.mean(0)
        label_free["fuse_z_qresp"] = True
        per_seed["fuse_z_qresp"] = F

    preds["random_control"] = np.random.default_rng(0).random(len(te_ids))
    label_free["random_control"] = True

    # ---- score -----------------------------------------------------------------------------------
    metrics = {}
    print(f"\n  {'predictor':24s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}"
          f"{'acc@.75':>8s}{'acc@.50':>8s}{'AUROC_SE':>10s}{'rho_SE':>8s}  label-free")
    for name, s in preds.items():
        s = np.asarray(s, dtype=float)
        au_inc = float(roc_auc_score(incorrect, s))
        ap_inc = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        v = yb_eval >= 0
        au_se = float(roc_auc_score(yb_eval[v], s[v])) if len(np.unique(yb_eval[v])) == 2 else float("nan")
        m = {"auroc_incorrect": au_inc, "auprc_incorrect": ap_inc, "prr": pr,
             "accuracy_coverage": {str(c): cov[c] for c in COVERAGES},
             "auroc_binarised_se": au_se, "spearman_se": E2.rho(s, y_saved),
             "label_free_on_target": label_free[name]}
        if name in per_seed:                                     # honesty: seed spread, not just the mean
            aus = [float(roc_auc_score(incorrect, p)) for p in per_seed[name]]
            m["auroc_incorrect_per_seed"] = aus
            m["auroc_incorrect_seed_mean"] = float(np.mean(aus))
            m["auroc_incorrect_seed_std"] = float(np.std(aus))
        metrics[name] = m
        print(f"  {name:24s}{au_inc:>10.3f}{ap_inc:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}"
              f"{cov[0.75]:>8.3f}{cov[0.5]:>8.3f}{au_se:>10.3f}{m['spearman_se']:>8.3f}"
              f"  {'yes' if label_free[name] else 'NO'}")

    # ---- paired bootstrap vs the supervised SEP and vs the sampling upper bound -------------------
    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    vs = {}
    print(f"\n  paired bootstrap (B={bootstrap}, shared indices) — Δ AUROC_incorrect  "
          f"[N={len(te_ids)}, incorrect_rate={pos_rate:.3f}]")
    for base in BASELINES:
        vs[base] = {}
        print(f"    vs {base}:")
        for name in preds:
            if name == base:
                continue
            c = ci(boot[name] - boot[base])
            excl = c["lo95"] > 0 or c["hi95"] < 0
            vs[base][name] = {**c, "ci_excludes_zero": bool(excl)}
            print(f"      Δ({name:20s}) {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
                  f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": short, "sources": fold["info"]["sources"], "n_test": len(te_ids),
            "positive_rate_incorrect": pos_rate, "mean_accuracy": float(acc.mean()),
            "detection_label": "incorrect = 1", "best_split": float(thr),
            "sep_single_choice": list(sep1_choice), "sep_single_auroc_vs_se": sep1_au_se,
            "sep_5layer_auroc_vs_se": sep5_au_se, "ridge_alpha": float(r_alpha),
            "ridge_spearman_rebuilt": E2.rho(r_pred, y_saved), "ridge_spearman_e37": fold["ridge_z"],
            "id_mapping_max_dev": max_dev, "bootstrap_resamples": bootstrap,
            "metrics": metrics, "bootstrap_auroc_incorrect": vs,
            "ridge_te_preds": [float(v) for v in r_pred]}      # closes E37's "ridge preds not saved"


def main():
    p = argparse.ArgumentParser(description="Correctness eval of the E37 LOLO proxy (additive).")
    p.add_argument("--lolo", default="amortized_ue/results/exp2_lolo_full.json")
    p.add_argument("--data_dir", default=None, help="Stage-1 output_dir override (e.g. /data2/mn1025/stage1)")
    p.add_argument("--anchor_layer", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default="amortized_ue/results/correctness_eval_e37.json")
    args = p.parse_args()

    with open(args.lolo) as f:
        folds = json.load(f)
    print(f"loaded {len(folds)} LOLO folds from {args.lolo}")

    # one shared load + alignment for every fold (exactly E37's, same anchor layer / tr split)
    print(f"loading 4 models' best-TBG states from {args.data_dir or 'default (NFS)'} ...")
    data, sp = E2.load_all(anchor_layer=args.anchor_layer, data_dir=args.data_dir)
    al, fsc = E2.fit_alignment(data, sp[0], args.anchor_layer)

    out = {}
    for fold in folds:
        out[fold["info"]["target"]] = evaluate_fold(
            fold, data, sp, al, fsc, bootstrap=args.bootstrap, data_dir=args.data_dir)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:                          # incremental: keep completed folds
            json.dump(out, f, indent=2)
        print(f"    -> saved {len(out)} fold(s) to {args.out}")

    # ---- cross-target summary --------------------------------------------------------------------
    names = [n for n in next(iter(out.values()))["metrics"]]
    print("\n" + "=" * 92)
    print("SUMMARY — AUROC_incorrect (detect a WRONG answer) per held-out target")
    print("=" * 92)
    hdr = f"{'predictor':24s}" + "".join(f"{t:>11s}" for t in out) + f"{'MEAN':>9s}  label-free"
    print(hdr)
    summary = {}
    for n in names:
        vals = [out[t]["metrics"][n]["auroc_incorrect"] for t in out]
        lf = all(out[t]["metrics"][n]["label_free_on_target"] for t in out)
        summary[n] = {"per_target": {t: out[t]["metrics"][n]["auroc_incorrect"] for t in out},
                      "mean": float(np.mean(vals)), "label_free_on_target": lf}
        print(f"{n:24s}" + "".join(f"{v:>11.3f}" for v in vals) +
              f"{np.mean(vals):>9.3f}  {'yes' if lf else 'NO'}")

    print("\nSE-fidelity vs correctness (mean over targets) — does the E37 ordering survive?")
    for n in names:
        se = np.mean([out[t]["metrics"][n]["auroc_binarised_se"] for t in out])
        inc = summary[n]["mean"]
        summary[n]["mean_auroc_se"] = float(se)
        summary[n]["mean_se_minus_inc"] = float(se - inc)
        print(f"  {n:24s} AUROC_SE {se:.3f}   AUROC_inc {inc:.3f}   drop {se - inc:+.3f}")

    for t in out:
        m = out[t]["metrics"]
        common = [n for n in m if not np.isnan(m[n]["auroc_binarised_se"])]
        by_se = sorted(common, key=lambda k: m[k]["auroc_binarised_se"], reverse=True)
        by_inc = sorted(common, key=lambda k: m[k]["auroc_incorrect"], reverse=True)
        print(f"\n{t}: orderings {'MATCH' if by_se == by_inc else 'DIFFER'}")
        print(f"  by SE : {by_se}")
        print(f"  by inc: {by_inc}")

    out["_summary"] = summary
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
