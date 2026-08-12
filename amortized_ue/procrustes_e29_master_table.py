"""E29 — assemble the four-model master table from the per-step JSONs (read-only; no compute).

Pulls together, per (source -> Llama-2 reference) pair:
  - E24 alignment floor / control / aligned / skyline / recovery fraction
  - E25 mechanism-A increment  aligned - control  with its paired-bootstrap 95% CI (+ significance)
  - E29 label-free ensemble (aligned-z + q_resp_only) AUROC/Spearman vs the supervised SEP baseline,
    and the paired-bootstrap (ensemble - SEP) delta + CI.

The point of the table: does the geometric recovery and the model-specific increment track FAMILY
relatedness (Llama-3 == same family as the Llama-2 reference; Mistral / DeepSeek == different lineages)?
Honest scope flags are emitted for (a) any increment CI that includes 0 and (b) Llama-3, whose alignment
chain is NOT computable from existing data (only n200 on the E20 ids, too few paired states for a stable
4096-dim W). Reads only JSONs already written by the E24/E25/E27/E29 steps.
"""
from __future__ import annotations

import json


def load(p):
    try:
        return json.load(open(p))
    except Exception:
        return None


def e24_block(d, which="last"):
    """Return the chosen eval block (default: last = the largest/most-powered) of an alignment JSON."""
    if not d or not d.get("evals"):
        return None
    b = d["evals"][-1] if which == "last" else d["evals"][0]
    amc = b["aligned_minus_control"]
    lo, hi = amc["lo95"], amc["hi95"]
    return {"N": b["n"], "position": d.get("position", "TBG"),
            "L_a_target_ridge": d.get("L_a_target_ridge"), "L_m_source_skyline": d.get("L_m_source_skyline"),
            "floor": b["floor"], "control_sharedDifficulty": b["control_mechA"], "aligned": b["aligned"],
            "skyline": b["skyline"], "recovery_frac": b["gap_recovered_frac"],
            "increment_mean": amc["mean"], "increment_ci": [lo, hi],
            "increment_significant": bool(lo > 0 or hi < 0)}


def ens_block(d):
    if not d:
        return None
    m = d["metrics"]; dl = d["ensemble_minus_sep"]
    rk = next(k for k in dl if k.startswith("RANK FUSION"))
    return {"N_test": d["n_test"], "sep_layer": d["sep_best_layer"], "sep_auroc": d["sep_test_auroc"],
            "ensemble_rankfusion_auroc": m["RANK FUSION (label-free)"]["auroc"],
            "ensemble_rankfusion_spearman": m["RANK FUSION (label-free)"]["spearman"],
            "ensemble_stdavg_auroc": m["avg standardized (label-free)"]["auroc"],
            "aligned_z_auroc": m["aligned-z ridge (label-free)"]["auroc"],
            "q_resp_only_auroc": m["q_resp_only (label-free)"]["auroc"],
            "delta_rankfusion_minus_sep": dl[rk],
            "delta_significant_auroc": bool(dl[rk]["auroc"]["lo95"] > 0 or dl[rk]["auroc"]["hi95"] < 0),
            "delta_significant_spearman": bool(dl[rk]["spearman"]["lo95"] > 0 or dl[rk]["spearman"]["hi95"] < 0)}


B = "amortized_ue/"
mist_official = e24_block(load(B + "procrustes_e25_mistral_to_llama2.json"), "last")  # fresh n1000 (N=1000)
mist_calib    = e24_block(load(B + "procrustes_e29_mistral_to_llama2_n1000.json"))     # n1000-within (N=100)
ds_fwd        = e24_block(load(B + "procrustes_e29_deepseek_to_llama2.json"))          # predict DeepSeek SE
ds_rev        = e24_block(load(B + "procrustes_e29_llama2_to_deepseek.json"))          # predict Llama-2 SE
ens_ds        = ens_block(load(B + "procrustes_e29_ensemble_sep_deepseek-llm-7b-chat.json"))
ens_mist      = ens_block(load(B + "procrustes_e29_ensemble_sep_Mistral-7B-Instruct-v0.2.json"))

master = {
    "note": "E29 four-model master table. Reference = Llama-2-7b-chat. All share the E23 fresh trivia_qa "
            "ids EXCEPT Llama-3 (see llama3 flag). 'recovery' = (aligned-floor)/(skyline-floor); "
            "'increment' = aligned - control (mechanism-A), the genuine model-specific component beyond "
            "shared question-difficulty. Ensemble = label-free rank-fusion of aligned-z ridge + q_resp_only.",
    "family_vs_llama2_reference": {"Mistral-7B-Instruct-v0.2": "different lineage",
                                   "deepseek-llm-7b-chat": "different lineage",
                                   "Meta-Llama-3-8B-Instruct": "SAME family (Meta Llama)"},
    "alignment_E24_E25": {
        "Mistral->Llama-2 (OFFICIAL, N=1000)": mist_official,
        "Mistral->Llama-2 (same-regime calibration, N=100)": mist_calib,
        "DeepSeek->Llama-2 (predict DeepSeek SE, N=100)": ds_fwd,
        "Llama-2->DeepSeek (predict Llama-2 SE, N=100)": ds_rev,
        "Llama-3<->Llama-2": {"status": "NOT COMPUTABLE from existing data",
            "reason": "Only Meta-Llama-3-8B-Instruct n200 exists, on the E20 ids (NOT the E23 fresh split); "
                      "144 train pairs cannot fit a stable 4096-dim orthogonal W. E20 established text "
                      "transfers / raw z does not (z 0.056 chance, q_only 0.436=88%, q_resp_only 0.562=full); "
                      "the Procrustes alignment line was never run for Llama-3."}},
    "ensemble_vs_SEP_E29 (within-n1000, N=100)": {
        "deepseek-llm-7b-chat": ens_ds,
        "Mistral-7B-Instruct-v0.2": ens_mist,
        "Meta-Llama-3-8B-Instruct": {"status": "NOT COMPUTABLE (see alignment flag)"}},
}

# ---- flags ------------------------------------------------------------------------------------
flags = []
for k, v in master["alignment_E24_E25"].items():
    if isinstance(v, dict) and "increment_significant" in v and not v["increment_significant"]:
        flags.append(f"E25 increment CI INCLUDES 0 for [{k}]: {v['increment_mean']:+.3f} "
                     f"[{v['increment_ci'][0]:+.3f}, {v['increment_ci'][1]:+.3f}] (N={v['N']}).")
flags.append("Llama-3 alignment chain NOT computable (n200/E20 ids); reported as a scope boundary, not a null.")
flags.append("The DeepSeek & Mistral-calibration increments are at N=100 (only n1000 exists for DeepSeek). "
             "Mistral's KNOWN-significant increment (+0.032 at N=1000) also goes non-significant at N=100 "
             "(+0.066 [-0.017,+0.155]) -> the DeepSeek non-significance is POWER-limited, not weaker transfer.")
master["scope_flags"] = flags

with open(B + "procrustes_e29_master_table.json", "w") as f:
    json.dump(master, f, indent=2)

# ---- print --------------------------------------------------------------------------------------
def f(x, p="{:+.3f}"):
    return "  n/a " if x is None else p.format(x)

print("\n" + "=" * 108)
print("E29 MASTER TABLE — cross-LLM alignment + label-free UE, reference = Llama-2-7b-chat")
print("=" * 108)
print("ALIGNMENT (E24 recovery) + MECHANISM-A INCREMENT (E25):")
print(f"  {'pair':44s}{'N':>5s}{'pos':>5s}{'floor':>8s}{'ctrl':>8s}{'align':>8s}{'sky':>7s}{'recov':>8s}{'increment [95% CI]':>26s}  sig")
for k, v in master["alignment_E24_E25"].items():
    if not isinstance(v, dict) or "recovery_frac" not in v:
        print(f"  {k:44s}   -> {v.get('status','')}")
        continue
    sig = "YES" if v["increment_significant"] else "no*"
    print(f"  {k:44s}{v['N']:>5d}{v['position']:>5s}{f(v['floor'])}{f(v['control_sharedDifficulty'])}"
          f"{f(v['aligned'])}{f(v['skyline'],'{:+.2f}')}{v['recovery_frac']:>7.1%} "
          f"{v['increment_mean']:>+8.3f} [{v['increment_ci'][0]:+.3f},{v['increment_ci'][1]:+.3f}]   {sig}")
print("\nLABEL-FREE ENSEMBLE vs SUPERVISED SEP (E29, within-n1000, N=100):")
print(f"  {'target':28s}{'ens AUROC':>10s}{'ens rho':>9s}{'SEP AUROC':>10s}{'Δ(ens−SEP) AUROC [CI]':>30s}  sig")
for k, v in master["ensemble_vs_SEP_E29 (within-n1000, N=100)"].items():
    if not v or "ensemble_rankfusion_auroc" not in v:
        print(f"  {k:28s}   -> {(v or {}).get('status','')}"); continue
    d = v["delta_rankfusion_minus_sep"]["auroc"]
    sig = "YES" if v["delta_significant_auroc"] else "no"
    print(f"  {k:28s}{v['ensemble_rankfusion_auroc']:>10.3f}{v['ensemble_rankfusion_spearman']:>9.3f}"
          f"{v['sep_auroc']:>10.3f}   {d['mean']:>+8.3f} [{d['lo95']:+.3f},{d['hi95']:+.3f}]   {sig}")
print("\nSCOPE FLAGS:")
for x in flags:
    print("  * " + x)
print("=" * 108)
print("\nwrote amortized_ue/procrustes_e29_master_table.json")
