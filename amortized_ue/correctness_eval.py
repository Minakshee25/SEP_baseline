"""Correctness-based evaluation of the amortized-UE predictors (E31 — strictly additive).

Every headline number in this repo is scored against the *semantic-entropy* label. Nothing is
scored against whether the target LLM's canonical answer was actually CORRECT. Stage-1 records
already store `canonical.accuracy`, so this needs no retraining and no new generation — only the
hidden states / labels / existing checkpoints already on disk.

This script measures the SAME predictors used in E27/E29/E30 against a *correctness* detection
target (`incorrect = 1`), alongside their SE-fidelity AUROC, so both live in one table:

  predictors (all oriented so higher score => more likely INCORRECT / higher SE):
    - true semantic entropy      (the 10-sample CAE label; sampling-based upper bound)
    - supervised SEP, single best layer         (best-on-eval, arXiv:2406.15927 convention)
    - supervised SEP, 5-layer concat            (Llama-2-7B layers 28-32; paper Table 4)
    - aligned-z ridge            (label-free: Procrustes source->Llama-2 ref + frozen ref ridge)
    - q_resp_only                (label-free: REFERENCE Llama-2 proxy run on the target's text)
    - rank-fusion ensemble       (label-free: empirical-CDF avg of aligned-z + q_resp_only)
    - random-score control

Regime mirrors E30 exactly (reference = Llama-2 at TBG:22/SLT:15), so the SEP numbers reproduce
`procrustes_e27_sep_comparison.json` and the eval rows/ids match E30:
    Mistral / DeepSeek : fit n2000 -> eval fresh n1000 (N=1000)
    Llama-3            : fit n2000 -> eval test split  (N=200, no fresh n1000 exists)
    Llama-2 (reference): fit n2000 -> eval fresh n1000 (aligned-z is self-supervised here; flagged)

Reuse (read-only): arm_preds/ecdf (procrustes_e27_rank_fusion), fit_probe/load_matrix/splits/rho
(linear_ceiling_probe), best_split/binarize_entropy (stage2.data), scipy orthogonal_procrustes.
Touches no training logic, no existing script, no prediction artifact.

The hidden-state-only predictors run under `se_probes` on CPU. `q_resp_only` / `rank-fusion` need
the REFERENCE proxy forward pass (arm_preds) => run the whole thing in `amortized_stage2` with a GPU;
if the proxy is unavailable those two predictors are skipped with a logged note and the rest still
write out.

    python -m amortized_ue.correctness_eval                       # all four targets
    python -m amortized_ue.correctness_eval --targets Mistral-7B-Instruct-v0.2
"""
from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.stage2.data import best_split, binarize_entropy

REF = "Llama-2-7b-chat"
REF_TBG, REF_SLT = 22, 15
COVERAGES = [1.0, 0.9, 0.75, 0.5]
SEP_BASELINES = ["sep_single_best_layer", "sep_5layer_concat"]

# (target, fit_num_samples, eval_num_samples or None=test-split, is_reference)
TARGETS = {
    "Mistral-7B-Instruct-v0.2":     (2000, 1000, False),
    "deepseek-llm-7b-chat":         (2000, 1000, False),
    "Meta-Llama-3-8B-Instruct":     (2000, None, False),
    "Llama-2-7b-chat":              (2000, 1000, True),
}


# ---- accuracy loaded from the manifest (no hidden states, so it is cheap and id-keyed) ----------
def load_accuracy(cfg: Stage1Config) -> dict:
    """{id: canonical.accuracy} straight from the Stage-1 manifest (tensor-free, exact ids)."""
    mp = os.path.join(cfg.run_dir(), "manifest.json")
    with open(mp) as f:
        recs = json.load(f)["records"]
    return {i: float(recs[i]["accuracy"]) for i in recs}


# ---- SEP predictors (best_split / binarize / LogisticRegression convention, verbatim) -----------
# The primary single-layer SEP reimplements procrustes_e30_ensemble_sep.sep_fit_eval EXACTLY
# (leak-free: layer chosen by FIT-VAL AUROC), so its AUROC-vs-SE reproduces the committed E30 numbers
# (Mistral 0.832 / DeepSeek 0.805 / Llama-3 0.839). Copied here, not imported, so the hidden-state-only
# path stays importable under `se_probes` (importing E30 pulls in the proxy/peft stack).
def _fit_predict(Xf, ybf, trv, Xe):
    sc = StandardScaler().fit(Xf[trv])
    clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xf[trv]), ybf[trv])
    return clf.predict_proba(sc.transform(Xe))[:, 1]


def sep_single_val_selected(fit_hidden, fit_y, tr, va, eval_hidden, eval_y, eval_rows):
    """Per-(pos,layer) LogisticRegression on best_split-binarised SE, fit on the n2000 TRAIN split;
    best single (pos,layer) chosen by FIT-VAL AUROC (leak-free; == E30 sep_fit_eval). All reported
    AUROCs are on eval_rows ONLY (so the test-split path never mixes in fit rows).
    Returns (eval_prob[all rows], eval_auroc_vs_SE, (pos,layer), thr, yb_eval, best_on_eval_auroc)."""
    thr = best_split(torch.tensor(fit_y[tr]))
    ybf = binarize_entropy(torch.tensor(fit_y), thr).numpy()
    ybe = binarize_entropy(torch.tensor(eval_y), thr).numpy()
    sel = eval_rows[ybe[eval_rows] >= 0]                           # valid eval rows only
    best = (-np.inf, None, None)                                   # (val_auroc, (pos,layer), eval_prob)
    best_on_eval = -np.inf                                         # optimistic arXiv/e27 selection, for reference
    for pos in fit_hidden:
        trv, vav = tr[ybf[tr] >= 0], va[ybf[va] >= 0]
        if len(np.unique(ybf[trv])) < 2 or len(np.unique(ybf[vav])) < 2:
            continue
        for L in range(fit_hidden[pos].shape[0]):
            sc = StandardScaler().fit(fit_hidden[pos][L][trv])
            clf = LogisticRegression(max_iter=1000).fit(sc.transform(fit_hidden[pos][L][trv]), ybf[trv])
            va_au = roc_auc_score(ybf[vav], clf.predict_proba(sc.transform(fit_hidden[pos][L][vav]))[:, 1])
            p = clf.predict_proba(sc.transform(eval_hidden[pos][L]))[:, 1]
            best_on_eval = max(best_on_eval, roc_auc_score(ybe[sel], p[sel]))
            if va_au > best[0]:
                best = (va_au, (pos, int(L)), p)
    eval_au = float(roc_auc_score(ybe[sel], best[2][sel]))
    return best[2], eval_au, best[1], float(thr), ybe, float(best_on_eval)


def sep_5layer_concat(fit_hidden, fit_y, tr, va, eval_hidden, eval_y, eval_rows, n_top=5):
    """SEP on the concatenation of the top-n_top layers (Llama-2-7B => layers 28-32; arXiv:2406.15927
    Table 4). Position (TBG|SLT) chosen by FIT-VAL AUROC (leak-free); AUROC on eval_rows only.
    Returns (eval_prob[all rows], eval_auroc, (pos,layers))."""
    thr = best_split(torch.tensor(fit_y[tr]))
    ybf = binarize_entropy(torch.tensor(fit_y), thr).numpy()
    ybe = binarize_entropy(torch.tensor(eval_y), thr).numpy()
    sel = eval_rows[ybe[eval_rows] >= 0]
    best = (-np.inf, None, None)
    for pos in fit_hidden:
        nL = fit_hidden[pos].shape[0]
        layers = list(range(nL - n_top, nL))                       # e.g. 28..32 for 33 hidden states
        Xf = np.concatenate([fit_hidden[pos][L] for L in layers], axis=1)
        Xe = np.concatenate([eval_hidden[pos][L] for L in layers], axis=1)
        trv, vav = tr[ybf[tr] >= 0], va[ybf[va] >= 0]
        if len(np.unique(ybf[trv])) < 2 or len(np.unique(ybf[vav])) < 2:
            continue
        sc = StandardScaler().fit(Xf[trv])
        clf = LogisticRegression(max_iter=1000).fit(sc.transform(Xf[trv]), ybf[trv])
        va_au = roc_auc_score(ybf[vav], clf.predict_proba(sc.transform(Xf[vav]))[:, 1])
        if va_au > best[0]:
            best = (va_au, (pos, layers), clf.predict_proba(sc.transform(Xe))[:, 1])
    eval_au = float(roc_auc_score(ybe[sel], best[2][sel]))
    return best[2], eval_au, best[1]


# ---- rejection-accuracy / prediction rejection ratio --------------------------------------------
def accuracy_coverage(uncert, correct, coverages=COVERAGES):
    """Retain the lowest-uncertainty `coverage` fraction; report accuracy over the retained set."""
    order = np.argsort(uncert, kind="stable")                      # most confident first
    c = correct[order].astype(float)
    n = len(c)
    return {cov: float(c[:max(1, int(round(cov * n)))].mean()) for cov in coverages}


def prediction_rejection_ratio(uncert, incorrect):
    """Malinin PRR: reject by descending uncertainty, area of retained-error curve vs rejection
    fraction, normalised between random (0) and oracle (1)."""
    n = len(incorrect)
    fr = np.arange(n) / n
    def ar(order):
        e = incorrect[order].astype(float)
        csum = np.concatenate([[0.0], np.cumsum(e)])
        ret_err = (csum[-1] - csum[:n]) / (n - np.arange(n))       # error rate on retained tail
        return float(np.trapz(ret_err, fr))
    ar_model = ar(np.argsort(-uncert, kind="stable"))
    ar_oracle = ar(np.argsort(-incorrect.astype(float), kind="stable"))
    ar_random = float(np.trapz(np.full(n, incorrect.mean()), fr))
    denom = ar_random - ar_oracle
    return float((ar_random - ar_model) / denom) if abs(denom) > 1e-12 else float("nan")


# ---- fast AUROC for the paired bootstrap (Mann-Whitney U; ==roc_auc_score with tie averaging) ----
def fast_auc(y, s):
    y = np.asarray(y); n1 = int(y.sum()); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return np.nan
    r = rankdata(s)
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0)


def paired_bootstrap_auc(preds: dict, incorrect, B=10000, seed=0):
    """Shared resample indices across ALL predictors. Returns {name: bootstrap AUC array [B]}."""
    rng = np.random.default_rng(seed)
    n = len(incorrect)
    idx = rng.integers(0, n, size=(B, n))                          # one index set, reused for every predictor
    boot = {}
    for name, s in preds.items():
        s = np.asarray(s)
        boot[name] = np.array([fast_auc(incorrect[ix], s[ix]) for ix in idx])
    return boot


def ci(delta):
    d = delta[~np.isnan(delta)]
    return {"mean": float(np.mean(d)), "lo95": float(np.percentile(d, 2.5)),
            "hi95": float(np.percentile(d, 97.5)), "n_valid_resamples": int(len(d))}


# ---- Procrustes-aligned-z ridge (E30 recipe) ----------------------------------------------------
def build_aligned_z(sh, rh, r_y, tr, va, eval_sh, eval_rows, tbg=REF_TBG, slt=REF_SLT):
    """Procrustes W (source->ref, train, NO labels) + frozen ref ridge -> aligned-z on train & eval."""
    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = rh[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, rh[pos][L][tr] - l)
        return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R, sc, _, _ = fit_probe(np.concatenate([rh["TBG"][tbg], rh["SLT"][slt]], 1), r_y, tr, va)

    def zpred(H):
        a = np.concatenate([(H["TBG"][tbg] - mT) @ WT + lT, (H["SLT"][slt] - mS) @ WS + lS], 1)
        return R.predict(sc.transform(a))
    return zpred(sh)[tr], zpred(eval_sh)[eval_rows]


# ---- one target --------------------------------------------------------------------------------
def evaluate_target(target, fit_n, eval_n, is_reference, dataset="trivia_qa",
                    bootstrap=10000, out=None):
    print("\n" + "#" * 90)
    print(f"# CORRECTNESS EVAL — target {target}  (fit n{fit_n}, "
          f"eval {'fresh n%d' % eval_n if eval_n else 'test split'})")
    print("#" * 90)

    # ---- fit set (source + Llama-2 ref, shared ids) ----
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=fit_n), ["TBG", "SLT"])
    rh, r_y, r_ids = load_matrix(Stage1Config(model_name=REF, dataset=dataset, num_samples=fit_n), ["TBG", "SLT"])
    assert s_ids == r_ids, "fit source/ref ids differ -- not the same questions"
    tr, va, te = splits(len(s_ids))

    # ---- eval set (fresh disjoint n OR the fit test split) ----
    if eval_n:
        esh, ey, eval_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=eval_n), ["TBG", "SLT"])
        eval_rows = np.arange(len(eval_ids))
        acc_cfg = Stage1Config(model_name=target, dataset=dataset, num_samples=eval_n)
    else:
        esh, ey, eval_rows = sh, s_y, te
        eval_ids = [s_ids[i] for i in te]
        acc_cfg = Stage1Config(model_name=target, dataset=dataset, num_samples=fit_n)

    # ---- accuracy / correctness label, id-matched to the prediction rows ----
    acc_map = load_accuracy(acc_cfg)
    assert set(eval_ids).issubset(acc_map), "eval ids missing from the accuracy manifest"
    acc = np.array([acc_map[i] for i in eval_ids], dtype=float)
    uniq = np.unique(acc)
    is_binary = set(np.round(uniq, 6).tolist()).issubset({0.0, 1.0})
    THRESH = 0.5
    correct = (acc >= THRESH).astype(int)
    incorrect = 1 - correct
    pos_rate = float(incorrect.mean())
    acc_report = {
        "metric": "squad (stored)", "is_binary": bool(is_binary), "n_unique": int(len(uniq)),
        "min": float(acc.min()), "max": float(acc.max()), "mean_accuracy": float(acc.mean()),
        "binarisation_threshold": THRESH, "positive_rate_incorrect": pos_rate}
    print(f"  accuracy: {'ALREADY BINARY {0,1}' if is_binary else 'continuous SQuAD F1'} "
          f"(n_unique={len(uniq)}, mean_acc={acc.mean():.3f}); threshold {THRESH} -> "
          f"incorrect rate {pos_rate:.3f}  (N_eval={len(eval_ids)})")

    # ---- predictors ----
    preds = {}
    label_free = {}
    needs_target_llm = {}

    # true SE (sampling upper bound; requires the target LLM's 10 samples). eval_rows indexes the
    # eval-set arrays uniformly: arange(all) for the fresh-n path, the test indices for the split path.
    preds["true_semantic_entropy"] = ey[eval_rows]
    label_free["true_semantic_entropy"] = False; needs_target_llm["true_semantic_entropy"] = True

    # supervised SEP (single best layer + 5-layer concat), leak-free val-selected (== E30 sep_fit_eval)
    sep1_p, sep1_au_se, sep1_choice, thr, ybe, sep1_boe = sep_single_val_selected(sh, s_y, tr, va, esh, ey, eval_rows)
    sep5_p, sep5_au_se, sep5_choice = sep_5layer_concat(sh, s_y, tr, va, esh, ey, eval_rows)
    preds["sep_single_best_layer"] = sep1_p[eval_rows]
    preds["sep_5layer_concat"] = sep5_p[eval_rows]
    for k in ("sep_single_best_layer", "sep_5layer_concat"):
        label_free[k] = False; needs_target_llm[k] = True       # uses target's hidden states + SE labels
    yb_eval = ybe[eval_rows]
    print(f"  SEP single (val-sel) {sep1_choice}  SE-AUROC {sep1_au_se:.3f} (best-on-eval {sep1_boe:.3f})   "
          f"5-layer {sep5_choice[0]} L{sep5_choice[1][0]}-{sep5_choice[1][-1]}  SE-AUROC {sep5_au_se:.3f}")

    # aligned-z ridge (label-free; source==ref for the reference => self-supervised, flagged)
    z_tr, z_eval = build_aligned_z(sh, rh, r_y, tr, va, esh, eval_rows)
    preds["aligned_z_ridge"] = z_eval
    label_free["aligned_z_ridge"] = not is_reference; needs_target_llm["aligned_z_ridge"] = True

    # q_resp_only + rank-fusion (need the REFERENCE proxy forward pass)
    have_text = True
    try:
        from amortized_ue.procrustes_e27_rank_fusion import arm_preds, ecdf
        qr_fit = arm_preds("q_resp_only", target, dataset, fit_n)
        qr_tr = np.array([qr_fit[i] for i in s_ids])[tr]
        if eval_n:
            qr_map = arm_preds("q_resp_only", target, dataset, eval_n)
            qr_eval = np.array([qr_map[i] for i in eval_ids])
        else:
            qr_eval = np.array([qr_fit[i] for i in eval_ids])
        preds["q_resp_only"] = qr_eval
        cz, cr = ecdf(z_tr), ecdf(qr_tr)
        preds["rank_fusion_ensemble"] = 0.5 * (cz(z_eval) + cr(qr_eval))
        label_free["q_resp_only"] = True; needs_target_llm["q_resp_only"] = False
        label_free["rank_fusion_ensemble"] = not is_reference; needs_target_llm["rank_fusion_ensemble"] = True
    except Exception as e:                                       # proxy env / GPU unavailable
        have_text = False
        print(f"  [WARN] q_resp_only / rank_fusion SKIPPED — proxy forward pass unavailable: "
              f"{type(e).__name__}: {e}")

    # random control
    preds["random_control"] = np.random.default_rng(0).random(len(eval_ids))
    label_free["random_control"] = True; needs_target_llm["random_control"] = False

    # ---- score every predictor: correctness + SE fidelity ----
    metrics = {}
    print(f"\n  {'predictor':26s}{'AUROC_inc':>10s}{'AUPRC_inc':>10s}{'PRR':>7s}"
          f"{'acc@1.0':>8s}{'acc@.90':>8s}{'acc@.75':>8s}{'acc@.50':>8s}{'AUROC_SE':>10s}")
    for name, s in preds.items():
        s = np.asarray(s, dtype=float)
        au_inc = float(roc_auc_score(incorrect, s)) if incorrect.sum() not in (0, len(incorrect)) else float("nan")
        ap_inc = float(average_precision_score(incorrect, s)) if incorrect.sum() else float("nan")
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        v = yb_eval >= 0
        au_se = float(roc_auc_score(yb_eval[v], s[v])) if len(np.unique(yb_eval[v])) == 2 else float("nan")
        metrics[name] = {
            "auroc_incorrect": au_inc, "auprc_incorrect": ap_inc, "prr": pr,
            "accuracy_coverage": {str(c): cov[c] for c in COVERAGES},
            "auroc_binarised_se": au_se,
            "label_free": label_free[name], "needs_target_llm": needs_target_llm[name]}
        print(f"  {name:26s}{au_inc:>10.3f}{ap_inc:>10.3f}{pr:>7.3f}"
              f"{cov[1.0]:>8.3f}{cov[0.9]:>8.3f}{cov[0.75]:>8.3f}{cov[0.5]:>8.3f}{au_se:>10.3f}")

    # ---- paired bootstrap of correctness-AUROC vs the SEP baselines (shared indices) ----
    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    boot_vs_sep = {}
    print(f"\n  paired bootstrap (B={bootstrap}, shared indices) — Δ AUROC_incorrect vs SEP baselines"
          f"   [N_test={len(eval_ids)}, incorrect_rate={pos_rate:.3f}]")
    for base in SEP_BASELINES:
        if base not in boot:
            continue
        boot_vs_sep[base] = {}
        print(f"    vs {base}:")
        for name in preds:
            if name == base:
                continue
            c = ci(boot[name] - boot[base])
            excl = "excludes 0" if (c["lo95"] > 0 or c["hi95"] < 0) else "includes 0"
            boot_vs_sep[base][name] = {**c, "ci_excludes_zero": excl == "excludes 0"}
            print(f"      Δ({name:24s} − {base:20s}) {c['mean']:+.3f} "
                  f"[{c['lo95']:+.3f}, {c['hi95']:+.3f}] ({excl})")

    result = {
        "target": target, "ref": REF, "dataset": dataset,
        "regime": (f"fit n{fit_n} -> eval fresh n{eval_n} (N={len(eval_ids)})" if eval_n
                   else f"fit+eval n{fit_n} (eval=test split, N={len(eval_ids)})"),
        "is_reference": is_reference, "ref_layers": {"TBG": REF_TBG, "SLT": REF_SLT},
        "n_test": len(eval_ids), "positive_rate_incorrect": pos_rate,
        "detection_label": "incorrect = 1", "accuracy_distribution": acc_report,
        "best_split": float(thr), "sep_single_choice": list(sep1_choice),
        "sep_5layer_choice": [sep5_choice[0], sep5_choice[1]],
        "sep_single_auroc_vs_se": sep1_au_se, "sep_5layer_auroc_vs_se": sep5_au_se,
        "sep_single_selection": "leak-free val-selected (== E30 sep_fit_eval)",
        "sep_single_best_on_eval_auroc_vs_se": sep1_boe,
        "text_predictors_available": have_text,
        "coverages": COVERAGES, "bootstrap_resamples": bootstrap,
        "metrics": metrics, "bootstrap_auroc_incorrect_vs_sep": boot_vs_sep,
    }
    out = out or f"amortized_ue/correctness_eval_{target}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\n  wrote {out}")
    return result


def main():
    p = argparse.ArgumentParser(description="Correctness-based eval of the amortized-UE predictors (additive).")
    p.add_argument("--targets", nargs="+", default=list(TARGETS), choices=list(TARGETS))
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    master = {}
    for t in args.targets:
        fit_n, eval_n, is_ref = TARGETS[t]
        try:
            master[t] = evaluate_target(t, fit_n, eval_n, is_ref, args.dataset, args.bootstrap)
        except Exception as e:
            print(f"\n[ERROR] target {t} failed: {type(e).__name__}: {e}")
            master[t] = {"error": f"{type(e).__name__}: {e}"}

    with open("amortized_ue/correctness_eval_master.json", "w") as f:
        json.dump(master, f, indent=2)
    print("\nwrote amortized_ue/correctness_eval_master.json")

    # ordering check: SE-AUROC vs correctness-AUROC (per target, over shared predictors)
    print("\n" + "=" * 70)
    print("ORDERING CHECK — methods ranked by SE-AUROC vs by correctness-AUROC")
    print("=" * 70)
    for t, r in master.items():
        if "metrics" not in r:
            continue
        m = r["metrics"]
        common = [k for k in m if not np.isnan(m[k]["auroc_binarised_se"]) and not np.isnan(m[k]["auroc_incorrect"])]
        by_se = sorted(common, key=lambda k: m[k]["auroc_binarised_se"], reverse=True)
        by_inc = sorted(common, key=lambda k: m[k]["auroc_incorrect"], reverse=True)
        same = by_se == by_inc
        print(f"\n{t}: orderings {'MATCH' if same else 'DIFFER'}")
        print(f"  by SE-AUROC : {by_se}")
        print(f"  by inc-AUROC: {by_inc}")


if __name__ == "__main__":
    main()
