"""Push an already-built Stage-1 dataset to W&B as a versioned artifact (an extra copy;
local disk stays the source of truth). Use for datasets built before push-by-default, or to
re-push. Loads the existing manifest for metrics -- does NOT re-run the target LLM.

  python -m amortized_ue.push_dataset_wandb --model_name Meta-Llama-3-8B-Instruct \
      --dataset trivia_qa --num_samples 200
"""
from __future__ import annotations

import os
import json
import glob
import logging
import argparse

from amortized_ue.config import Stage1Config
from amortized_ue import wandb_io


def main():
    p = argparse.ArgumentParser(description="Push an existing Stage-1 dataset to W&B.")
    p.add_argument("--model_name", required=True)
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, required=True)
    p.add_argument("--artifact_name", default=None, help="override the auto-distinct name")
    a = p.parse_args()

    cfg = Stage1Config(model_name=a.model_name, dataset=a.dataset,
                       num_samples=a.num_samples, wandb_artifact_name=a.artifact_name)
    if not os.path.isdir(cfg.records_dir()):
        raise FileNotFoundError(f"no records dir at {cfg.records_dir()!r} -- build it first")
    n = len(glob.glob(os.path.join(cfg.records_dir(), "*.pt")))

    metrics = {"n_records": n}
    if os.path.exists(cfg.manifest_path()):                       # enrich from manifest if present
        man = json.load(open(cfg.manifest_path()))
        meta = man.get("meta", man)
        for k in ("n_records", "mean_accuracy", "mean_cluster_assignment_entropy"):
            if k in meta:
                metrics[k] = meta[k]

    logging.info("Pushing %s (%d records) as artifact %r -> project %s",
                 cfg.resolved_run_name(), n, cfg.resolved_artifact_name(), cfg.wandb_project)
    wandb_io.sync_to_wandb(cfg, metrics)
    print(f"pushed: {cfg.resolved_artifact_name()}  ({n} records)")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
