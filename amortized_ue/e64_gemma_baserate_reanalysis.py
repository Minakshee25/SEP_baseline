"""E64 -- is E45's gemma-2-9b-it zero-shot loss a genuine cross-family transfer failure,
or an artefact of gemma-2-9b-it's higher base accuracy on trivia_qa?

Context (E45): the DEPLOY proxy (frozen Llama-3.2-3B + LoRA, trained by pooling
Llama-2/Mistral/Llama-3/DeepSeek n2000, text arms only) scored zero-shot on 4 Qwen/Gemma
targets. On gemma-2-9b-it it LOST to true 10-sample SE (AUROC_incorrect 0.722 vs 0.769,
-0.047, CI excludes 0) -- the only target anywhere in the project where q_resp_only is
clearly worse than sampling. gemma-2-9b-it is also a base-rate outlier: mean_acc 0.684 /
incorrect-rate 0.316, vs 0.42-0.56 / 0.44-0.58 for the other 3 targets. E45 flagged the
base-rate explanation as a hypothesis, not established.

This script tests it, read-only over existing E44/E45 records, NO retraining. The only
compute is a deterministic forward pass of the already-trained DEPLOY proxy to recover the
per-question q_only / q_resp_only scores (E45 saved only the aggregate AUROCs). Stage A
persists those per-id predictions so any future reanalysis needs no GPU.

Two checks:
  1. Matched-difficulty subset -- restrict all 4 targets to the questions gemma-2-9b-it got
     WRONG (its 316-question incorrect subset; all 4 targets share the same 1000 ids), and
     re-score q_resp_only / q_only vs true SE on that same subset. AUROC_incorrect is
     undefined for gemma-2-9b-it itself there (every row is incorrect) -- reported as NaN,
     with SE-fidelity Spearman(pred, true_SE) as the substitute; for the other 3 targets the
     hard-question subset still has both classes. If the proxy's gap to true SE closes (or
     the other 3 targets' deltas also go negative on this hard subset), the negative
     gemma-2-9b-it result is a difficulty effect, not a family-transfer failure.
  2. Base-rate matched by downsampling -- (2a, as asked) for each of the other 3 targets,
     keep every CORRECT question + a random 316-question sample of its WRONG questions
     (matching gemma-2-9b-it's wrong-answer COUNT), re-run the E45 comparison, repeat over
     many resamples. (2b, supplementary/cleaner) a fully balanced 316-wrong + 316-correct
     subset (incorrect-rate 0.5 for every target incl. a resampled gemma-2-9b-it), so
     difficulty AND base rate are held constant. If the other 3 targets' q_resp_only-vs-SE
     delta shifts toward gemma-2-9b-it's -0.047 when the base rate is matched, the E45
     result is a base-rate artefact.

Env: amortized_stage2(_v5) + a GPU for Stage A; Stage B is CPU/numpy only.
    python -m amortized_ue.e64_gemma_baserate_reanalysis
"""
from __future__ import annotations

import json
import argparse

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.correctness_eval import (
    load_accuracy, paired_bootstrap_auc, ci, prediction_rejection_ratio)

TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
GEMMA = "gemma-2-9b-it"
ARMS = ["q_only", "q_resp_only"]
DATA_DIR = "/data2/mn1025/stage1"
DEPLOY_CKPT = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
NUM_SAMPLES = 1000
PERID_OUT = "amortized_ue/results/e64_perid_preds.json"
OUT = "amortized_ue/results/e64_gemma_baserate_reanalysis.json"


# ------------------------------------------------------------------ Stage A: per-id preds
def build_perid_preds(out_path: str) -> dict:
    """Deterministic forward pass of the DEPLOY proxy (zero-shot) -> {target: {id: {arm: p,
    true_se: x, incorrect: 0/1}}}. Idempotent: reloads out_path if present."""
    import os
    if os.path.exists(out_path):
        print(f"[stageA] reusing {out_path}")
        with open(out_path) as f:
            return json.load(f)

    from amortized_ue.procrustes_e27_rank_fusion import arm_preds  # heavy (torch/peft) import

    blob = {}
    for t in TARGETS:
        cfg = Stage1Config(model_name=t, dataset="trivia_qa", num_samples=NUM_SAMPLES, output_dir=DATA_DIR)
        recs = load_records(cfg)
        ids = sorted(recs.keys())
        acc_map = load_accuracy(cfg)
        assert set(ids).issubset(acc_map), f"{t}: ids missing from accuracy manifest"
        rec = {i: {"true_se": float(recs[i]["labels"]["cluster_assignment_entropy"]),
                   "incorrect": int(float(acc_map[i]) < 0.5)} for i in ids}
        for arm in ARMS:
            print(f"[stageA] {t} arm={arm} -- deploy proxy forward pass (zero-shot) ...")
            mp = arm_preds(arm, t, "trivia_qa", NUM_SAMPLES, ckpt_dir=DEPLOY_CKPT, data_dir=DATA_DIR)
            for i in ids:
                rec[i][arm] = float(mp[i])
        blob[t] = rec
        n = len(ids); inc = sum(r["incorrect"] for r in rec.values())
        print(f"[stageA] {t}: N={n}  incorrect={inc}  rate={inc / n:.3f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(blob, f, indent=1)
    print(f"[stageA] wrote {out_path}")
    return blob


# ------------------------------------------------------------------ metric helpers
def _auroc(y, s):
    y = np.asarray(y)
    return float("nan") if len(np.unique(y)) < 2 else float(roc_auc_score(y, s))


def _predictors(rec_subset: list[dict]) -> dict:
    return {
        "true_semantic_entropy": np.array([r["true_se"] for r in rec_subset], float),
        "q_only":       np.array([r["q_only"] for r in rec_subset], float),
        "q_resp_only":  np.array([r["q_resp_only"] for r in rec_subset], float),
    }


def _score_block(rec_subset: list[dict], bootstrap: int, seed: int = 0) -> dict:
    """AUROC_incorrect + AUPRC + PRR for each predictor, SE-fidelity Spearman for the two
    proxy arms, and paired-bootstrap deltas (q_resp_only - true SE) and (q_resp_only - q_only)."""
    inc = np.array([r["incorrect"] for r in rec_subset], int)
    preds = _predictors(rec_subset)
    both_classes = len(np.unique(inc)) == 2
    y_se = preds["true_semantic_entropy"]

    out = {"n": len(rec_subset), "n_incorrect": int(inc.sum()),
           "incorrect_rate": float(inc.mean()), "mean_true_se": float(y_se.mean()),
           "both_classes": both_classes, "metrics": {}, "se_fidelity_spearman": {}}
    for name, s in preds.items():
        sep = {}
        if both_classes:
            sep = {"mean_on_incorrect": float(s[inc == 1].mean()),
                   "mean_on_correct": float(s[inc == 0].mean()),
                   "separation_gap": float(s[inc == 1].mean() - s[inc == 0].mean())}
        out["metrics"][name] = {
            "auroc_incorrect": _auroc(inc, s),
            "auprc_incorrect": float(average_precision_score(inc, s)) if both_classes else float("nan"),
            "prr": prediction_rejection_ratio(s, inc) if both_classes else float("nan"),
            **sep,
        }
    for arm in ("q_only", "q_resp_only"):
        rr = spearmanr(preds[arm], y_se)
        out["se_fidelity_spearman"][arm] = float(rr.statistic)

    out["bootstrap_deltas"] = {}
    if both_classes and bootstrap:
        boot = paired_bootstrap_auc(
            {"q_resp_only": preds["q_resp_only"], "q_only": preds["q_only"],
             "true_semantic_entropy": y_se}, inc, B=bootstrap, seed=seed)
        for a, b in [("q_resp_only", "true_semantic_entropy"), ("q_resp_only", "q_only")]:
            c = ci(boot[a] - boot[b])
            c["excludes_0"] = bool(c["lo95"] > 0 or c["hi95"] < 0)
            out["bootstrap_deltas"][f"{a}_minus_{b}"] = c
    return out


def _resample_delta(recs: dict, target: str, keep_ids, wrong_pool, correct_pool,
                    n_wrong: int, n_correct, R: int, seed: int) -> dict:
    """Repeatedly subsample (n_wrong wrong [+ n_correct correct if not None], + all keep_ids)
    and recompute point AUROCs + delta(q_resp_only - true SE). Return mean and 2.5/97.5
    percentiles ACROSS the R resamples (the downsampling uncertainty)."""
    rng = np.random.default_rng(seed)
    rec = recs[target]
    rows = {"true_semantic_entropy": [], "q_only": [], "q_resp_only": [],
            "delta_qresp_minus_trueSE": [], "delta_qresp_minus_qonly": [],
            "incorrect_rate": [], "n": []}
    for _ in range(R):
        w = rng.choice(wrong_pool, size=n_wrong, replace=False)
        ids = list(keep_ids) + list(w)
        if n_correct is not None:
            c = rng.choice(correct_pool, size=n_correct, replace=False)
            ids = list(w) + list(c)
        sub = [rec[i] for i in ids]
        inc = np.array([r["incorrect"] for r in sub], int)
        p = _predictors(sub)
        au = {k: _auroc(inc, v) for k, v in p.items()}
        for k in ("true_semantic_entropy", "q_only", "q_resp_only"):
            rows[k].append(au[k])
        rows["delta_qresp_minus_trueSE"].append(au["q_resp_only"] - au["true_semantic_entropy"])
        rows["delta_qresp_minus_qonly"].append(au["q_resp_only"] - au["q_only"])
        rows["incorrect_rate"].append(float(inc.mean()))
        rows["n"].append(len(sub))
    summ = {}
    for k, v in rows.items():
        v = np.asarray(v, float)
        summ[k] = {"mean": float(v.mean()), "lo95": float(np.percentile(v, 2.5)),
                   "hi95": float(np.percentile(v, 97.5))}
    d = np.asarray(rows["delta_qresp_minus_trueSE"], float)
    summ["delta_qresp_minus_trueSE"]["frac_below_0"] = float((d < 0).mean())
    summ["R"] = R
    return summ


# ------------------------------------------------------------------ Stage B: the analysis
def analyze(blob: dict, bootstrap: int, R: int) -> dict:
    # normalise json-string ids consistently; every target shares the same id set
    recs = {t: {k: v for k, v in blob[t].items()} for t in TARGETS}
    ids_all = sorted(recs[TARGETS[0]].keys())
    for t in TARGETS:
        assert sorted(recs[t].keys()) == ids_all, f"{t} id set differs"

    gemma_inc = recs[GEMMA]
    gemma_wrong_ids = [i for i in ids_all if gemma_inc[i]["incorrect"] == 1]
    print(f"\ngemma-2-9b-it wrong subset: {len(gemma_wrong_ids)} / {len(ids_all)} questions")

    result = {
        "meta": {"deploy_ckpt": DEPLOY_CKPT, "data_dir": DATA_DIR, "n_total": len(ids_all),
                 "gemma_wrong_subset_size": len(gemma_wrong_ids), "bootstrap": bootstrap,
                 "resamples": R,
                 "note": "q_only/q_resp_only are the DEPLOY proxy (pooled Llama-2/Mistral/"
                         "Llama-3/DeepSeek) run zero-shot; true_semantic_entropy is the stored "
                         "10-sample cluster_assignment_entropy label."},
        "original_full_1000": {},
        "check1_matched_difficulty_subset": {"subset": "gemma-2-9b-it incorrect (n=%d)" % len(gemma_wrong_ids),
                                             "targets": {}},
        "check2a_downsample_wrong_to_gemma_count": {
            "description": "keep ALL correct questions + random %d wrong (match gemma-2-9b-it "
                           "wrong COUNT); %d resamples; percentiles are across resamples." % (len(gemma_wrong_ids), R),
            "targets": {}},
        "check2b_balanced_subset": {
            "description": "random %d wrong + %d correct (incorrect-rate 0.5 for every target); "
                           "%d resamples." % (len(gemma_wrong_ids), len(gemma_wrong_ids), R),
            "targets": {}},
    }

    # ---- original E45 reproduction (full 1000) ----
    for t in TARGETS:
        sub = [recs[t][i] for i in ids_all]
        result["original_full_1000"][t] = _score_block(sub, bootstrap)

    # ---- check 1: gemma's wrong subset, all 4 targets ----
    for t in TARGETS:
        sub = [recs[t][i] for i in gemma_wrong_ids]
        blk = _score_block(sub, bootstrap)
        full_d = result["original_full_1000"][t]["bootstrap_deltas"].get(
            "q_resp_only_minus_true_semantic_entropy", {}).get("mean")
        sd = blk["bootstrap_deltas"].get("q_resp_only_minus_true_semantic_entropy", {}).get("mean")
        blk["delta_vs_full"] = None if (full_d is None or sd is None) else float(sd - full_d)
        result["check1_matched_difficulty_subset"]["targets"][t] = blk

    # ---- check 2a: downsample wrong to gemma's count ----
    n_wrong = len(gemma_wrong_ids)
    for si, t in enumerate(TARGETS):
        rec = recs[t]
        wrong = [i for i in ids_all if rec[i]["incorrect"] == 1]
        correct = [i for i in ids_all if rec[i]["incorrect"] == 0]
        if len(wrong) <= n_wrong:
            # gemma itself (or any target with <=316 wrong): nothing to downsample -> use full
            blk = dict(result["original_full_1000"][t])
            blk["note"] = "target already has <= gemma's wrong count; original full-1000 numbers"
            result["check2a_downsample_wrong_to_gemma_count"]["targets"][t] = blk
            continue
        result["check2a_downsample_wrong_to_gemma_count"]["targets"][t] = _resample_delta(
            recs, t, keep_ids=correct, wrong_pool=np.array(wrong), correct_pool=None,
            n_wrong=n_wrong, n_correct=None, R=R, seed=100 + si)

    # ---- check 2b: balanced subset, all 4 targets ----
    for si, t in enumerate(TARGETS):
        rec = recs[t]
        wrong = [i for i in ids_all if rec[i]["incorrect"] == 1]
        correct = [i for i in ids_all if rec[i]["incorrect"] == 0]
        result["check2b_balanced_subset"]["targets"][t] = _resample_delta(
            recs, t, keep_ids=[], wrong_pool=np.array(wrong), correct_pool=np.array(correct),
            n_wrong=n_wrong, n_correct=n_wrong, R=R, seed=200 + si)

    return result


def _print_summary(res: dict):
    def g(d, *ks):
        for k in ks:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        return d
    print("\n" + "=" * 78)
    print("ORIGINAL (full 1000)   AUROC_incorrect: true_SE / q_only / q_resp_only   d(qresp-SE)")
    for t, r in res["original_full_1000"].items():
        m = r["metrics"]; d = g(r, "bootstrap_deltas", "q_resp_only_minus_true_semantic_entropy")
        print(f"  {t:14s} {m['true_semantic_entropy']['auroc_incorrect']:.3f} / "
              f"{m['q_only']['auroc_incorrect']:.3f} / {m['q_resp_only']['auroc_incorrect']:.3f}"
              f"   {d.get('mean', float('nan')):+.3f} [{d.get('lo95', float('nan')):+.3f},"
              f"{d.get('hi95', float('nan')):+.3f}]")
    print("\nCHECK 1  (gemma-2-9b-it wrong subset)   AUROC / SE-fidelity rho(qresp) / d(qresp-SE) / dvsFull")
    for t, r in res["check1_matched_difficulty_subset"]["targets"].items():
        m = r["metrics"]; d = g(r, "bootstrap_deltas", "q_resp_only_minus_true_semantic_entropy")
        rho = r["se_fidelity_spearman"]["q_resp_only"]
        dm = d.get("mean")
        print(f"  {t:14s} n={r['n']:3d} rate={r['incorrect_rate']:.2f}  "
              f"qresp_AUROC={m['q_resp_only']['auroc_incorrect'] if not np.isnan(m['q_resp_only']['auroc_incorrect']) else float('nan'):.3f}"
              f"  rho={rho:+.3f}  d={dm if dm is not None else float('nan'):+.3f}"
              f"  dvsFull={r['delta_vs_full'] if r['delta_vs_full'] is not None else float('nan'):+.3f}")
    print("\nCHECK 2a  (all correct + 316 wrong)   d(qresp-SE) mean [95%] , frac<0")
    for t, r in res["check2a_downsample_wrong_to_gemma_count"]["targets"].items():
        d = r.get("delta_qresp_minus_trueSE") or g(r, "bootstrap_deltas", "q_resp_only_minus_true_semantic_entropy")
        rate = r.get("incorrect_rate", {}).get("mean") if isinstance(r.get("incorrect_rate"), dict) else r.get("incorrect_rate")
        print(f"  {t:14s} rate~{rate if rate is not None else float('nan'):.3f}  "
              f"d={d.get('mean', float('nan')):+.3f} [{d.get('lo95', float('nan')):+.3f},"
              f"{d.get('hi95', float('nan')):+.3f}]  frac<0={d.get('frac_below_0', float('nan'))}")
    print("\nCHECK 2b  (316 wrong + 316 correct, rate 0.5)   d(qresp-SE) mean [95%] , frac<0")
    for t, r in res["check2b_balanced_subset"]["targets"].items():
        d = r["delta_qresp_minus_trueSE"]
        print(f"  {t:14s} d={d['mean']:+.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}]  frac<0={d['frac_below_0']}")
    print("=" * 78)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--resamples", type=int, default=1000)
    p.add_argument("--perid_out", default=PERID_OUT)
    p.add_argument("--out", default=OUT)
    p.add_argument("--stage", choices=["all", "preds", "analyze"], default="all")
    args = p.parse_args()

    blob = build_perid_preds(args.perid_out)
    if args.stage == "preds":
        return
    res = analyze(blob, args.bootstrap, args.resamples)
    _print_summary(res)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=1)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
