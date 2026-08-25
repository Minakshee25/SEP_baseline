"""E53c — the complete, correctly-labeled comparison (true SE / SEP / ridge / E53 proxy) for the
Qwen/Gemma-trained deploy proxy on Llama-2 and Mistral. ONE script, ONE output file
(results/e53_full_comparison.json) -- consolidates what were three separate scripts/files in this
session (proxy eval, SEP correctness recompute, ridge context) so every number for this
comparison lives in one place instead of scattered across files.

Two metrics, kept in clearly separate columns (SE-fidelity ≠ correctness, E31):
  - SE-fidelity: Spearman(pred, continuous SE)  [primary metric, established E8]
  - correctness: AUROC(pred, incorrect)         [does it actually catch wrong answers, E31]

Three reference methods, each with a DIFFERENT access level to the target -- do not compare them
as if they were peers:
  - **SEP** (full target access: fit on the target's own hidden states + labels) -- the field's
    established prior method; single-layer LOGISTIC classifier on binarized SE, so its Spearman
    (a full-continuous-scale ranking question) is a repurposed use of a classifier's probability
    output, not the metric it was optimized for -- this is WHY it reads lower than ridge below,
    not a bug (see the conversation this script was written for).
  - **ridge** (full target access, CONTEXT ONLY -- not a fair opponent for the proxy): a plain
    ridge regression, this project's own in-distribution ceiling diagnostic (E8), directly
    optimized for the continuous SE value. Cannot run zero-shot BY CONSTRUCTION (has to be fit on
    the target's own hidden states) -- it answers "what's the best possible LINEAR read of this
    target's hidden states if you had full access", not "how does the zero-shot proxy compare to
    a real competing method". Two inputs: `same_as_sep` (single TBG layer, SEP's own layer --
    isolates the classifier-vs-regressor effect alone) and `reference_ceiling` (TBG+SLT stacked
    at this project's established reference architecture, E10).
  - **proxy** (E53, ZERO target access -- no hidden states, no labels, never saw Llama-2 or
    Mistral in any form; trained only on 4 different Qwen/Gemma models' text).

Reuses, CPU-only, no retraining:
  - the E53 proxy's already-computed numbers (results/e53_qwengemma_deploy_qresp_on_llama2_mistral.json)
  - `compute_sep` (se_fidelity_proxy_vs_sep.py) for SEP's raw fitted predictions on the SAME fresh
    n1000 rows the proxy was scored on, scored against BOTH SE and incorrect (SEP is always fit to
    predict SE -- the standard method; scoring that SAME fit against `incorrect` afterward is
    exactly how E31/E38/E41 measure its correctness skill).
  - the canonical `results/sep_reference_values.json` for SEP's SE-fidelity numbers (cross-checked
    against the fresh recompute here to 4 dp).
  - `linear_ceiling_probe.{load_matrix,splits,fit_probe,rho}` for the ridge context (fit on each
    target's own n2000 train, evaluated on the same fresh n1000 rows).

Env: se_probes (CPU only). Run from the repo root:
    python -m amortized_ue.e53_full_comparison
"""
from __future__ import annotations

import os
import json

import numpy as np
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.correctness_eval import load_accuracy
from amortized_ue.se_fidelity_proxy_vs_sep import compute_sep
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho

TARGETS = {"Llama-2-7b-chat": 30, "Mistral-7B-Instruct-v0.2": 31}   # E41 fixed TBG layer (also SEP's layer)
DATA_DIR = "/data2/mn1025/stage1"
PROXY_RESULTS = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "results", "e53_qwengemma_deploy_qresp_on_llama2_mistral.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "results", "e53_full_comparison.json")
# duplicated (not imported) from e53_eval_on_llama2_mistral.py -- that module eagerly imports
# arm_preds at the top level, which needs `peft` (amortized_stage2 only); this script is CPU-only
# (se_probes), so importing across that boundary would force a GPU-only dependency for no reason.
SEP_REF_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sep_reference_values.json")
SEP_REF_SETTING = "trivia_qa_fresh_n1000"

# ridge reference (TBG+SLT) layers per target -- E10's architecture for Llama-2; E22's picked
# layers for Mistral. `same_as_sep` reuses TARGETS' layer (single TBG, matches SEP exactly).
RIDGE_REF_LAYERS = {
    "Llama-2-7b-chat": {"ref_tbg": 22, "ref_slt": 15},
    "Mistral-7B-Instruct-v0.2": {"ref_tbg": 31, "ref_slt": 20},
}


def load_sep_reference(target: str) -> dict:
    with open(SEP_REF_PATH) as f:
        ref = json.load(f)
    return ref["targets"][target]["settings"][SEP_REF_SETTING]


def sep_on_fresh(target: str, layer: int) -> dict:
    """SEP's raw fitted predictions on the fresh n1000 eval rows, scored against BOTH SE and
    incorrect. `pred` is one fit (standard SEP method: fit to predict SE); the two scores below
    are two different questions asked of that SAME fit, not two different models."""
    sep = compute_sep(target, "trivia_qa", 1000, data_dir=DATA_DIR, layer=layer)
    cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000, output_dir=DATA_DIR)
    acc_map = load_accuracy(cfg)
    acc = np.array([acc_map[i] for i in sep["ids"]], dtype=float)
    incorrect = (acc < 0.5).astype(int)
    return {
        "position": sep["choice"][0], "layer": sep["choice"][1], "selection": sep["selection"],
        "n": len(sep["ids"]), "incorrect_rate": float(incorrect.mean()),
        "spearman_vs_se": float(sep_spearman(sep)), "auroc_vs_se": sep["auroc_se"],
        "auroc_vs_incorrect": float(roc_auc_score(incorrect, sep["pred"])),
    }


def sep_spearman(sep: dict) -> float:
    from scipy.stats import spearmanr
    return spearmanr(sep["pred"], sep["y"]).correlation


def ridge_at(hid_fit, y_fit, tr, va, hid_eval, y_eval, positions_layers):
    """positions_layers: list of (pos, layer). Concatenates features if more than one."""
    Xtr = np.concatenate([hid_fit[p][l] for p, l in positions_layers], axis=1)
    Xev = np.concatenate([hid_eval[p][l] for p, l in positions_layers], axis=1)
    model, scaler, alpha, val_rho = fit_probe(Xtr, y_fit, tr, va)
    pred = model.predict(scaler.transform(Xev))
    return {"spearman": float(rho(pred, y_eval)), "alpha": alpha, "val_spearman": float(val_rho)}


def ridge_context(target: str, sep_layer: int) -> dict:
    """CONTEXT ONLY -- ridge needs full target access (fit on its own n2000 hidden states), so it
    is not a fair opponent for the zero-access proxy; see module docstring."""
    ref = RIDGE_REF_LAYERS[target]
    fit_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000, output_dir=DATA_DIR)
    eval_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000, output_dir=DATA_DIR)
    positions = sorted({"TBG", "SLT"})
    hid_fit, y_fit, ids_fit = load_matrix(fit_cfg, positions)
    hid_eval, y_eval, ids_eval = load_matrix(eval_cfg, positions)
    tr, va, te = splits(len(ids_fit))

    same_as_sep = ridge_at(hid_fit, y_fit, tr, va, hid_eval, y_eval, [("TBG", sep_layer)])
    reference_ceiling = ridge_at(hid_fit, y_fit, tr, va, hid_eval, y_eval,
                                  [("TBG", ref["ref_tbg"]), ("SLT", ref["ref_slt"])])
    return {
        "same_as_sep": {**same_as_sep, "layer": ["TBG", sep_layer]},
        "reference_ceiling": {**reference_ceiling, "layers": [["TBG", ref["ref_tbg"]], ["SLT", ref["ref_slt"]]]},
    }


def main():
    with open(PROXY_RESULTS) as f:
        proxy = json.load(f)["results"]

    out = {}
    print(f"{'target':26s}{'metric':20s}{'true SE':>9s}{'SEP':>9s}{'ridge(ctx)':>11s}{'proxy':>9s}")
    for target, layer in TARGETS.items():
        p = proxy[target]
        sep = sep_on_fresh(target, layer)
        ridge = ridge_context(target, layer)

        # cross-check against the canonical reference file (should match to 4 dp)
        ref = load_sep_reference(target)
        assert abs(ref["spearman"] - sep["spearman_vs_se"]) < 1e-4, \
            f"{target}: fresh SEP recompute disagrees with {SEP_REF_PATH}[{SEP_REF_SETTING}]"

        row = {
            "n": p["n_test"],
            "spearman_vs_se": {"true_se": 1.0, "sep": sep["spearman_vs_se"],
                                "ridge_same_as_sep_CONTEXT_ONLY": ridge["same_as_sep"]["spearman"],
                                "ridge_reference_ceiling_CONTEXT_ONLY": ridge["reference_ceiling"]["spearman"],
                                "proxy": p["spearman_proxy_vs_true_se"]},
            "auroc_vs_incorrect": {
                "true_se": p["metrics"]["true_semantic_entropy"]["auroc_incorrect"],
                "sep": sep["auroc_vs_incorrect"],
                "proxy": p["metrics"]["proxy_q_resp_only"]["auroc_incorrect"],
            },
            "sep_detail": sep,
            "ridge_detail_CONTEXT_ONLY": ridge,
            "note": "ridge fields need FULL target access (fit on the target's own hidden "
                    "states) -- CONTEXT ONLY, not a fair opponent for the zero-access proxy.",
        }
        out[target] = row

        print(f"{target:26s}{'Spearman-vs-SE':20s}{row['spearman_vs_se']['true_se']:>9.3f}"
              f"{row['spearman_vs_se']['sep']:>9.3f}"
              f"{row['spearman_vs_se']['ridge_reference_ceiling_CONTEXT_ONLY']:>11.3f}"
              f"{row['spearman_vs_se']['proxy']:>9.3f}")
        print(f"{'':26s}{'AUROC-vs-incorrect':20s}{row['auroc_vs_incorrect']['true_se']:>9.3f}"
              f"{row['auroc_vs_incorrect']['sep']:>9.3f}{'':>11s}{row['auroc_vs_incorrect']['proxy']:>9.3f}")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
