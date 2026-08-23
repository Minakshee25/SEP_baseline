"""Log E51's summary metrics (proxy q_resp_only vs SEP, SE-fidelity) to W&B. Additive: reads the
already-saved `results/se_fidelity_proxy_vs_sep.json`, logs no new dataset/checkpoint (there isn't
one -- E51 only ran inference over existing artifacts), just the comparison metrics for tracking.

    python -m amortized_ue.log_e51_wandb
"""
from __future__ import annotations

import os
import json

import wandb

RESULTS = "amortized_ue/results/se_fidelity_proxy_vs_sep.json"
PROJECT = "amortized_ue_stage2"
RUN_NAME = "E51_proxy_vs_sep_se_fidelity"


def main():
    with open(RESULTS) as f:
        data = json.load(f)
    rows = data["_final_table"]

    run = wandb.init(entity=os.environ.get("WANDB_ENT"), project=PROJECT, name=RUN_NAME,
                     job_type="eval", config={"bootstrap_resamples": 10000,
                                              "n_settings": len(rows),
                                              "source_json": RESULTS})

    table = wandb.Table(columns=["setting", "target", "sep_rho", "proxy_rho", "delta_rho",
                                  "rho_ci_lo", "rho_ci_hi", "rho_ci_excludes_zero",
                                  "sep_auc", "proxy_auc", "delta_auc",
                                  "auc_ci_lo", "auc_ci_hi", "auc_ci_excludes_zero", "verdict"])
    for r in rows:
        rho_ci = r["rho_ci"]
        auc_ci = r["auc_ci"] or [None, None]
        table.add_data(r["setting"], r["target"], r["sep_rho"], r["ens_rho"], r["delta_rho"],
                       rho_ci[0], rho_ci[1], r["rho_ci_excludes_zero"],
                       r["sep_auc"], r["ens_auc"], r["delta_auc"],
                       auc_ci[0], auc_ci[1], r["auc_ci_excludes_zero"], r["verdict"])
    run.log({"e51_summary_table": table})

    n = len(rows)
    n_rho_wins = sum(1 for r in rows if r["rho_ci_excludes_zero"] and r["delta_rho"] > 0)
    n_rho_ties = sum(1 for r in rows if not r["rho_ci_excludes_zero"])
    n_rho_losses = sum(1 for r in rows if r["rho_ci_excludes_zero"] and r["delta_rho"] < 0)
    n_auc_wins = sum(1 for r in rows if r["auc_ci_excludes_zero"] and (r["delta_auc"] or 0) > 0)
    n_auc_ties = sum(1 for r in rows if not r["auc_ci_excludes_zero"])
    n_auc_losses = sum(1 for r in rows if r["auc_ci_excludes_zero"] and (r["delta_auc"] or 0) < 0)
    mean_delta_rho = sum(r["delta_rho"] for r in rows) / n
    mean_delta_auc = sum(r["delta_auc"] for r in rows if r["delta_auc"] is not None) / \
        sum(1 for r in rows if r["delta_auc"] is not None)

    run.summary.update({
        "n_settings": n,
        "rho_wins": n_rho_wins, "rho_ties": n_rho_ties, "rho_losses": n_rho_losses,
        "auc_wins": n_auc_wins, "auc_ties": n_auc_ties, "auc_losses": n_auc_losses,
        "mean_delta_rho": mean_delta_rho, "mean_delta_auc": mean_delta_auc,
    })
    for r in rows:
        key = f"{r['setting']}/{r['target']}"
        run.summary[f"{key}/delta_rho"] = r["delta_rho"]
        run.summary[f"{key}/delta_auc"] = r["delta_auc"]
        run.summary[f"{key}/verdict"] = r["verdict"]

    print(f"rho:  {n_rho_wins}/{n} proxy>SEP, {n_rho_ties} tie, {n_rho_losses} SEP>proxy "
          f"(mean delta {mean_delta_rho:+.3f})")
    print(f"auc:  {n_auc_wins}/{n} proxy>SEP, {n_auc_ties} tie, {n_auc_losses} SEP>proxy "
          f"(mean delta {mean_delta_auc:+.3f})")
    print(f"W&B run: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
