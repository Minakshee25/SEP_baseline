"""Assemble the milestone combined table: each target LLM's OWN uncertainty estimators
(True SE / SEP / Ridge / its own single-model 5-arm proxy) alongside the RAW frozen
cross-LLM transfer of the 5-arm proxy from a source LLM -- ID (fresh TriviaQA n1000) and
OOD (SQuAD n1000) side by side. Assembly only: reads existing result JSONs, computes nothing.

Sources:
  amortized_ue/results/baseline_table_freshn1000.json        (own ID  : SE / SEP / Ridge)
  amortized_ue/results/baseline_table_squad_ood_n1000.json   (own OOD : SE / SEP / Ridge)
  amortized_ue/results/samemodel_5arm_id_ood.json            (own proxy arms, Llama-2 + Mistral)
  amortized_ue/results/cross_llm_5arm_fresh_n1000_correctness.json  (transfer arms, 4 pairs, ID+OOD)

Writes:
  amortized_ue/results/cross_llm_5arm_combined_table.json
  amortized_ue/results/cross_llm_5arm_combined_table.csv

Metric convention (unchanged from the sources):
  spearman      = Spearman(prediction, continuous target SE)
  auroc_inc     = AUROC(prediction, incorrect),  incorrect = accuracy < 0.5
  proxy arm rows: mean / std over the 5 saved seeds (metric computed per seed then averaged).
  SE / SEP / Ridge: single fit, std = null / blank.
"""
from __future__ import annotations

import csv
import json

RES = "amortized_ue/results"
OUT_JSON = f"{RES}/cross_llm_5arm_combined_table.json"
OUT_CSV = f"{RES}/cross_llm_5arm_combined_table.csv"

SHORT = {"Llama-2-7b-chat": "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "DeepSeek"}
TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2",
           "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]
# raw cross-LLM source per target (the pair actually run)
XFER_SOURCE = {"Llama-2-7b-chat": "Mistral-7B-Instruct-v0.2",
               "Mistral-7B-Instruct-v0.2": "Llama-2-7b-chat",
               "Meta-Llama-3-8B-Instruct": "Llama-2-7b-chat",
               "deepseek-llm-7b-chat": "Llama-2-7b-chat"}


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    bid = {r["model"]: r for r in _load(f"{RES}/baseline_table_freshn1000.json")["rows"]}
    bood = {r["model"]: r for r in _load(f"{RES}/baseline_table_squad_ood_n1000.json")["rows"]}
    selfp = _load(f"{RES}/samemodel_5arm_id_ood.json")["results"]
    xf = {}
    for r in _load(f"{RES}/cross_llm_5arm_fresh_n1000_correctness.json")["results"]:
        xf[(r["source_model"], r["target_model"], r["dataset"])] = r["arms"]

    rows = []

    def add(target, method, source, own, xfer):
        """own / xfer: dict with id_spearman(_std), id_auroc(_std), ood_* or None."""
        def g(d, k):
            return None if d is None else d.get(k)
        rows.append({
            "target_model": target, "target_short": SHORT[target],
            "method": method, "transfer_source_model": source,
            "transfer_source_short": SHORT[source] if source else None,
            "own_id_spearman": g(own, "id_sp"), "own_id_spearman_std": g(own, "id_sp_std"),
            "own_id_auroc_incorrect": g(own, "id_au"), "own_id_auroc_incorrect_std": g(own, "id_au_std"),
            "own_ood_spearman": g(own, "ood_sp"), "own_ood_spearman_std": g(own, "ood_sp_std"),
            "own_ood_auroc_incorrect": g(own, "ood_au"), "own_ood_auroc_incorrect_std": g(own, "ood_au_std"),
            "xfer_id_spearman": g(xfer, "id_sp"), "xfer_id_spearman_std": g(xfer, "id_sp_std"),
            "xfer_id_auroc_incorrect": g(xfer, "id_au"), "xfer_id_auroc_incorrect_std": g(xfer, "id_au_std"),
            "xfer_ood_spearman": g(xfer, "ood_sp"), "xfer_ood_spearman_std": g(xfer, "ood_sp_std"),
            "xfer_ood_auroc_incorrect": g(xfer, "ood_au"), "xfer_ood_auroc_incorrect_std": g(xfer, "ood_au_std"),
        })

    for target in TARGETS:
        src = XFER_SOURCE[target]
        # --- baseline methods (own only, single fit) --------------------------------------------
        ri, ro = bid[target], bood[target]
        for meth, pfx in [("True SE", "SE"), ("SEP", "SEP"), ("Ridge", "Ridge")]:
            own = {"id_sp": ri[f"{pfx}_spearman"], "id_au": ri[f"{pfx}_auroc_incorrect"],
                   "ood_sp": ro[f"{pfx}_spearman"], "ood_au": ro[f"{pfx}_auroc_incorrect"]}
            add(target, meth, None, own, None)
        # --- 5-arm proxy: own (if it exists) + raw cross-LLM transfer ---------------------------
        selfarms = selfp.get(target, {})
        for arm in ARMS:
            own = None
            if selfarms:
                si, so = selfarms["ID"][arm], selfarms["OOD"][arm]
                own = {"id_sp": si["spearman_mean"], "id_sp_std": si["spearman_std"],
                       "id_au": si["auroc_incorrect_mean"], "id_au_std": si["auroc_incorrect_std"],
                       "ood_sp": so["spearman_mean"], "ood_sp_std": so["spearman_std"],
                       "ood_au": so["auroc_incorrect_mean"], "ood_au_std": so["auroc_incorrect_std"]}
            xi = xf[(src, target, "trivia_qa")][arm]
            xo = xf[(src, target, "squad")][arm]
            xfer = {"id_sp": xi["spearman_mean"], "id_sp_std": xi["spearman_std"],
                    "id_au": xi["auroc_incorrect_mean"], "id_au_std": xi["auroc_incorrect_std"],
                    "ood_sp": xo["spearman_mean"], "ood_sp_std": xo["spearman_std"],
                    "ood_au": xo["auroc_incorrect_mean"], "ood_au_std": xo["auroc_incorrect_std"]}
            add(target, f"proxy:{arm}", src, own, xfer)

    payload = {
        "_meta": {
            "description": "Milestone combined table -- per-target OWN uncertainty estimators "
                           "(True SE / SEP / Ridge / own single-model 5-arm proxy) vs RAW frozen "
                           "cross-LLM transfer of the 5-arm proxy from a source LLM. ID = fresh "
                           "TriviaQA n1000, OOD = SQuAD n1000. Assembly only.",
            "metrics": {"spearman": "Spearman(pred, continuous target SE)",
                        "auroc_incorrect": "AUROC(pred, incorrect); incorrect = accuracy < 0.5"},
            "proxy_rows": "mean/std over 5 seeds (metric per seed then averaged); "
                          "SE/SEP/Ridge = single fit, std = null",
            "own_proxy_available_for": ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"],
            "transfer_pairs": {SHORT[t]: f"{SHORT[XFER_SOURCE[t]]} -> {SHORT[t]}" for t in TARGETS},
            "sources": ["baseline_table_freshn1000.json", "baseline_table_squad_ood_n1000.json",
                        "samemodel_5arm_id_ood.json", "cross_llm_5arm_fresh_n1000_correctness.json"],
            "generated_by": "amortized_ue/build_cross_llm_combined_table.py",
        },
        "rows": rows,
    }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)

    fields = list(rows[0].keys())
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})

    print(f"wrote {OUT_JSON}\nwrote {OUT_CSV}\n{len(rows)} rows "
          f"({len(TARGETS)} targets x (3 baseline + 5 proxy arms))")
    # quick echo
    for r in rows:
        o = "" if r["own_id_spearman"] is None else f"own ID rho {r['own_id_spearman']:.3f}"
        x = "" if r["xfer_id_spearman"] is None else f"xfer({r['transfer_source_short']}) ID rho {r['xfer_id_spearman']:.3f}"
        print(f"  {r['target_short']:9s} {r['method']:16s} {o:20s} {x}")


if __name__ == "__main__":
    main()
