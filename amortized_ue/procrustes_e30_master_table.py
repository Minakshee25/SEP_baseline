"""E30 — assemble the FULL-POWER four-model master table from the per-step JSONs (read-only).

Per (source -> Llama-2 reference) pair: alignment recovery, the E25 mechanism-A increment
(aligned-control) + bootstrap CI, the CKA (how rotationally-alignable the two hidden spaces are),
and the label-free-ensemble-vs-supervised-SEP delta + CI. DeepSeek & Mistral are evaluated at N=1000
(fit n2000 -> fresh n1000); Llama-3 at N=200 (n2000 test split -- it has no fresh n1000). The question:
do recovery / increment / alignability track FAMILY relatedness (Llama-3 == same family as the
reference; Mistral, DeepSeek == different lineages)? Emits explicit flags for every increment CI that
includes 0 and for the Llama-3 N=200 power limit.
"""
from __future__ import annotations
import json


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def align(path):
    d = load(path)
    if not d or not d.get("evals"):
        return {"status": f"missing {path}"}
    b = max(d["evals"], key=lambda e: e["n"])                     # most-powered eval block
    amc = b["aligned_minus_control"]; lo, hi = amc["lo95"], amc["hi95"]
    return {"source": d["source"], "target": d["target"], "position": d.get("position") or "TBG",
            "N": b["n"], "floor": b["floor"], "control": b["control_mechA"], "aligned": b["aligned"],
            "skyline": b["skyline"], "recovery": b["gap_recovered_frac"], "cka": b["reconstruction"]["cka"],
            "increment": amc["mean"], "ci": [lo, hi], "significant": bool(lo > 0 or hi < 0)}


def ens(path):
    d = load(path)
    if not d:
        return {"status": f"missing {path}"}
    m = d["metrics"]; rk = next(k for k in d["ensemble_minus_sep"] if k.startswith("RANK FUSION"))
    dl = d["ensemble_minus_sep"][rk]
    return {"regime": d["regime"], "N": d["n_eval"], "sep_layer": d["sep_best_layer"], "sep_auroc": d["sep_eval_auroc"],
            "ensemble_auroc": m["RANK FUSION (label-free)"]["auroc"], "ensemble_spearman": m["RANK FUSION (label-free)"]["spearman"],
            "delta_auroc": dl["auroc"], "delta_spearman": dl["spearman"],
            "beats_auroc": bool(dl["auroc"]["lo95"] > 0), "beats_spearman": bool(dl["spearman"]["lo95"] > 0)}


B = "amortized_ue/"
alignment = {
    "Mistral->Llama-2 (N=1000)":  align(B + "procrustes_e25_mistral_to_llama2.json"),
    "DeepSeek->Llama-2 (N=1000)": align(B + "procrustes_e30_deepseek_to_llama2_fullpower.json"),
    "Llama-2->DeepSeek (N=1000)": align(B + "procrustes_e30_llama2_to_deepseek_fullpower.json"),
    "Llama-3->Llama-2 (N=200)":   align(B + "procrustes_e30_llama3_to_llama2.json"),
    "Llama-2->Llama-3 (N=200)":   align(B + "procrustes_e30_llama2_to_llama3.json"),
}
ensemble = {
    "Mistral-7B-Instruct-v0.2 (N=1000)":  ens(B + "procrustes_e30_ensemble_sep_Mistral-7B-Instruct-v0.2.json"),
    "deepseek-llm-7b-chat (N=1000)":      ens(B + "procrustes_e30_ensemble_sep_deepseek-llm-7b-chat.json"),
    "Meta-Llama-3-8B-Instruct (N=200)":   ens(B + "procrustes_e30_ensemble_sep_Meta-Llama-3-8B-Instruct.json"),
}
family = {"Mistral-7B-Instruct-v0.2": "different lineage", "deepseek-llm-7b-chat": "different lineage",
          "Meta-Llama-3-8B-Instruct": "SAME family (Meta Llama)"}

flags = []
for k, v in alignment.items():
    if v.get("significant") is False:
        flags.append(f"E25 increment CI includes 0 for [{k}]: {v['increment']:+.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}] (N={v['N']}).")
flags.append("Llama-3 is limited to N=200 (only n2000 exists, no fresh n1000) -> its increment CI is inherently wider.")
flags.append("KEY: family is at best a WEAK predictor of alignability. CKA ranks Llama-3 (same family) 0.87 > "
             "Mistral (different) 0.80 >> DeepSeek (different) 0.25 -- Llama-3 is highest, but Mistral is close "
             "behind and DeepSeek is a striking low-CKA OUTLIER despite matching 4096 dims. The genuine "
             "model-specific increment tracks CKA (Mistral +0.032 sig; DeepSeek ~0), NOT the ~0.92-0.95 recovery, "
             "which is high for ALL pairs because it is dominated by shared question-difficulty.")

master = {"note": "E30 full-power four-model master table. Reference = Llama-2-7b-chat.",
          "family_vs_reference": family, "alignment_E24_E25": alignment,
          "ensemble_vs_SEP_E30": ensemble, "scope_flags": flags}
with open(B + "procrustes_e30_master_table.json", "w") as f:
    json.dump(master, f, indent=2)

print("\n" + "=" * 104)
print("E30 FULL-POWER MASTER TABLE — cross-LLM alignment + label-free UE, reference = Llama-2")
print("=" * 104)
print("ALIGNMENT (recovery, CKA) + MECHANISM-A INCREMENT (E25):")
print(f"  {'pair':28s}{'N':>6s}{'pos':>5s}{'recov':>8s}{'CKA':>7s}{'increment [95% CI]':>26s}  sig")
for k, v in alignment.items():
    if "recovery" not in v:
        print(f"  {k:28s}  {v.get('status')}"); continue
    sig = "YES" if v["significant"] else "no"
    print(f"  {k:28s}{v['N']:>6d}{v['position']:>5s}{v['recovery']:>7.1%} {v['cka']:>6.3f}"
          f"{v['increment']:>+9.3f} [{v['ci'][0]:+.3f},{v['ci'][1]:+.3f}]   {sig}")
print("\nLABEL-FREE ENSEMBLE vs SUPERVISED SEP (E30):")
print(f"  {'target':34s}{'ens AUROC':>10s}{'ens rho':>9s}{'SEP AUROC':>10s}{'Δ AUROC [CI]':>26s}  beats")
for k, v in ensemble.items():
    if "ensemble_auroc" not in v:
        print(f"  {k:34s}  {v.get('status')}"); continue
    d = v["delta_auroc"]; beats = "AUROC+rho" if v["beats_auroc"] else ("rho only" if v["beats_spearman"] else "on par")
    print(f"  {k:34s}{v['ensemble_auroc']:>10.3f}{v['ensemble_spearman']:>9.3f}{v['sep_auroc']:>10.3f}"
          f"{d['mean']:>+9.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}]   {beats}")
print("\nSCOPE FLAGS:")
for x in flags:
    print("  * " + x)
print("=" * 104)
print("\nwrote amortized_ue/procrustes_e30_master_table.json")
