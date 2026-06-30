"""Optional W&B mirror: upload the exact local files as a versioned artifact.

This is an *additional* copy, never the only place the data lives. The artifact
is just the same per-prompt .pt files + manifest.json uploaded under one record
schema, so `loaders.load_wandb` returns records identical to `loaders.load_local`.
"""
from __future__ import annotations

import os
import logging

from amortized_ue.config import Stage1Config


def sync_to_wandb(config: Stage1Config, metrics: dict) -> None:
    import wandb

    run = wandb.init(
        entity=config.wandb_entity,
        project=config.wandb_project,
        name=config.resolved_run_name(),
        config=config.as_dict(),
    )
    run.log(metrics)

    artifact = wandb.Artifact(
        name=config.wandb_artifact_name,
        type="dataset",
        metadata={**config.as_dict(), **metrics},
    )
    # Add the manifest and every per-prompt record file (same files, same schema).
    artifact.add_file(config.manifest_path(), name="manifest.json")
    artifact.add_dir(config.records_dir(), name="records")
    run.log_artifact(artifact)
    run.finish()
    logging.info("Pushed Stage 1 artifact %s to W&B project %s",
                 config.wandb_artifact_name, config.wandb_project)
