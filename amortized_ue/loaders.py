"""Load Stage 1 records from either source, returning identical id-keyed dicts.

`load_records(config)` switches on `config.load_source` ("local" | "wandb").
Local disk is the source of truth and works fully offline; W&B is an extra copy
of the same files. Both paths return `{prompt_id: record}` with the same schema.
"""
from __future__ import annotations

import os
import glob

from amortized_ue.config import Stage1Config
from amortized_ue import record as rec


def load_local(run_dir: str) -> dict:
    """Load every record under <run_dir>/records, keyed by record['id']."""
    records_dir = os.path.join(run_dir, "records")
    if not os.path.isdir(records_dir):
        raise FileNotFoundError(f"No records dir at {records_dir!r}")

    out = {}
    for path in sorted(glob.glob(os.path.join(records_dir, "*.pt"))):
        record = rec.load_record(path)
        out[record["id"]] = record

    # If a manifest exists, cross-check counts (positional join is never relied on).
    manifest_path = os.path.join(run_dir, "manifest.json")
    if os.path.exists(manifest_path):
        manifest = rec.read_manifest(manifest_path)
        expected = set(manifest["records"].keys())
        found = set(out.keys())
        if expected != found:
            missing, extra = expected - found, found - expected
            raise RuntimeError(
                f"Manifest/records mismatch in {run_dir}: "
                f"missing={sorted(missing)[:5]}, extra={sorted(extra)[:5]}")
    return out


def load_wandb(config: Stage1Config, download_root: str | None = None) -> dict:
    """Download the W&B artifact (same files) and load it like a local dir."""
    import wandb

    download_root = download_root or os.path.join(config.run_dir(), "_wandb_download")
    entity = config.wandb_entity
    ref = f"{entity + '/' if entity else ''}{config.wandb_project}/{config.wandb_artifact_name}:latest"

    api = wandb.Api()
    artifact = api.artifact(ref, type="dataset")
    artifact_dir = artifact.download(root=download_root)
    return load_local(artifact_dir)


def load_records(config: Stage1Config) -> dict:
    if config.load_source == "local":
        return load_local(config.run_dir())
    elif config.load_source == "wandb":
        return load_wandb(config)
    raise ValueError(f"Unknown load_source {config.load_source!r} (use 'local' or 'wandb').")
