"""Build ONE canonical SEP-vs-true-SE reference table, extracted (not hand-typed) from the
already-validated `results/se_fidelity_proxy_vs_sep.json` (E51/E52), so every future script that
wants "what does SEP get here" has one file to read instead of re-deriving or hand-copying numbers
into a local dict (which is what `e53_eval_on_llama2_mistral.py` did, and is exactly what this file
replaces). Independently re-verified for Llama-2/Mistral fresh-trivia on 2026-08-25 (recomputed via
`compute_sep` from scratch, CPU-only, matched to 4 dp) before trusting this extraction.

Metrics: `spearman` = Spearman(SEP prediction, continuous SE) -- this project's PRIMARY metric
(established E8: a threshold-free rank metric is the honest one for a continuous-SE regression).
`auroc_se` = AUROC of the same SEP prediction against SE binarized at a train-fit `best_split`
threshold -- a secondary, easier metric (only has to rank around one threshold, not the whole
scale), which is why it reads numerically higher than Spearman for the same fit -- NOT a
different/better model, a different/easier question. This file does NOT (yet) cover
AUROC-vs-`incorrect` (correctness) SEP numbers, which live in `correctness_eval_e41_fixedlayer.json`
/ `correctness_eval_ood.json` -- a natural v2 addition if that comparison is needed too.

Settings (3): `trivia_qa_fresh_n1000` (fit that target's OWN n2000 train -> eval a disjoint fresh
n1000, the standard ID comparison; merges E51's "fresh" (original 4 targets, E41 fixed-layer) and
"qwengemma" (4 new targets, leak-free val-selected layer -- no established CV layer for those yet)
since both are the same fit/eval recipe, just different targets/layer-selection maturity).
`trivia_qa_lolo_n200` (E37/E43 leave-one-LLM-out proxy regime's SEP baseline: fit target's own
n2000, eval its 200-row test split -- narrower CI, only the original 4 targets have this).
`squad_ood_n1000` (fit on trivia n2000, eval OOD on squad n1000 -- only Llama-2/Mistral have squad
records; `lolo_squad`'s SEP row in the source JSON is identical to this one, same recipe, so it is
not duplicated here).

Regenerate: `python -m amortized_ue.build_sep_reference`
"""
from __future__ import annotations

import os
import json

SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "se_fidelity_proxy_vs_sep.json")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "sep_reference_values.json")

# short name (as used in se_fidelity_proxy_vs_sep.json) -> canonical Stage1Config model_name
CANONICAL = {
    "Llama-2": "Llama-2-7b-chat", "Llama-2-7b-chat": "Llama-2-7b-chat",
    "Mistral": "Mistral-7B-Instruct-v0.2", "Mistral-7B-Instruct-v0.2": "Mistral-7B-Instruct-v0.2",
    "Llama-3": "Meta-Llama-3-8B-Instruct", "DeepSeek": "deepseek-llm-7b-chat",
    "Qwen3-8B": "Qwen3-8B", "Qwen3.5-9B": "Qwen3.5-9B",
    "gemma-7b-it": "gemma-7b-it", "gemma-2-9b-it": "gemma-2-9b-it",
}

# (source setting key, output setting key) -- "qwengemma" merges into the same output setting as
# "fresh" (both are fit-n2000 -> eval-fresh-n1000; only the layer-selection maturity differs, which
# is preserved per-target via the `selection` field, not folded into the setting name).
SETTING_MAP = [
    ("fresh", "trivia_qa_fresh_n1000"),
    ("qwengemma", "trivia_qa_fresh_n1000"),
    ("lolo", "trivia_qa_lolo_n200"),
    ("squad", "squad_ood_n1000"),
]


def build():
    with open(SRC) as f:
        src = json.load(f)

    targets: dict = {}
    for src_key, out_setting in SETTING_MAP:
        for short_name, block in src[src_key].items():
            model = CANONICAL[short_name]
            entry = {
                "position": block["sep_choice"][0], "layer": block["sep_choice"][1],
                "selection": block["sep_selection"], "n": block["n"],
                "spearman": block["metrics"]["sep"]["spearman"],
                "auroc_se": block["metrics"]["sep"]["auroc_se"],
            }
            targets.setdefault(model, {}).setdefault("settings", {})[out_setting] = entry

    out = {
        "_meta": {
            "description": "Canonical SEP-vs-true-SE reference (Spearman primary, AUROC_se "
                            "secondary), extracted from results/se_fidelity_proxy_vs_sep.json.",
            "primary_metric": "spearman = Spearman(SEP prediction, continuous SE label)",
            "secondary_metric": "auroc_se = AUROC of SEP prediction vs SE binarized at a "
                                 "train-fit best_split threshold (easier metric, reads higher)",
            "source": os.path.relpath(SRC, os.path.dirname(OUT)),
            "generated_by": "amortized_ue/build_sep_reference.py",
            "regenerate": "python -m amortized_ue.build_sep_reference",
            "verified": "Llama-2/Mistral trivia_qa_fresh_n1000 independently recomputed from "
                        "scratch via compute_sep() on 2026-08-25, matched to 4 dp.",
            "not_covered": "AUROC-vs-incorrect (correctness) SEP numbers -- see "
                           "correctness_eval_e41_fixedlayer.json / correctness_eval_ood.json.",
        },
        "targets": targets,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    return out


def main():
    out = build()
    print(f"wrote {OUT}\n")
    print(f"{'target':28s}{'setting':26s}{'pos':5s}{'layer':6s}{'spearman':>9s}{'auroc_se':>9s}")
    for model, d in out["targets"].items():
        for setting, e in d["settings"].items():
            print(f"{model:28s}{setting:26s}{e['position']:5s}{e['layer']:<6d}"
                  f"{e['spearman']:>9.3f}{e['auroc_se']:>9.3f}")


if __name__ == "__main__":
    main()
