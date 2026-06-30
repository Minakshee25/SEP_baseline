"""Stage 1 record schema: build, save, load, and describe one per-prompt record.

One self-contained record per prompt, keyed by prompt id. The semantic-entropy
label lives inside the same record as the text and hidden states, so everything
is joined by id (never by list position). Tensors are stored at native dtype.
"""
from __future__ import annotations

import os
import re
import json
import hashlib
from typing import Any

import torch


SCHEMA_VERSION = "stage1-v1"


def safe_filename(prompt_id: str) -> str:
    """Filesystem-safe, collision-free filename stem for an arbitrary prompt id."""
    stem = re.sub(r"[^A-Za-z0-9_.-]", "_", str(prompt_id))[:80]
    digest = hashlib.sha1(str(prompt_id).encode("utf-8")).hexdigest()[:8]
    return f"{stem}_{digest}"


def build_record(
    *,
    prompt_id: str,
    question: str,
    context: Any,
    reference: dict,
    canonical_response: str,
    canonical_accuracy: float,
    canonical_token_log_likelihoods: list,
    tbg_embedding: torch.Tensor,
    slt_embedding: torch.Tensor,
    sample_responses: list,
    sample_token_log_likelihoods: list,
    semantic_ids: list,
    cluster_assignment_entropy: float,
    meta: dict,
) -> dict:
    """Assemble the canonical Stage 1 record.

    Positions follow the user's definitions (NOT the SEP repo's inverted keys):
      TBG = token-before-generation = hidden[0]            (all layers)
      SLT = second-last generated token = hidden[n_gen-2]  (all layers)
    See memory note sep-tbg-slt-naming-inversion.
    """
    assert len(sample_responses) == len(sample_token_log_likelihoods) == len(semantic_ids), \
        "high-temp samples and their semantic ids must be aligned 1:1"

    samples = [
        {"response": r, "token_log_likelihoods": ll, "semantic_id": sid}
        for r, ll, sid in zip(sample_responses, sample_token_log_likelihoods, semantic_ids)
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "id": prompt_id,
        "question": question,
        "context": context,
        "reference": reference,
        "canonical": {
            "response": canonical_response,
            "accuracy": float(canonical_accuracy),
            "token_log_likelihoods": canonical_token_log_likelihoods,
            "hidden_states": {
                "TBG": tbg_embedding,   # hidden[0]          (token before generation)
                "SLT": slt_embedding,   # hidden[n_gen - 2]  (second-last token)
            },
        },
        "samples": samples,
        "labels": {
            # primary continuous label (raw float, never binarised in Stage 1)
            "cluster_assignment_entropy": float(cluster_assignment_entropy),
            "semantic_ids": list(semantic_ids),
            "n_clusters": int(len(set(semantic_ids))),
            "n_samples": int(len(semantic_ids)),
        },
        "meta": meta,
    }


def save_record(record: dict, records_dir: str) -> str:
    """Persist one record as <records_dir>/<safe_id>.pt. Returns the filename."""
    os.makedirs(records_dir, exist_ok=True)
    filename = safe_filename(record["id"]) + ".pt"
    torch.save(record, os.path.join(records_dir, filename))
    return filename


def load_record(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def manifest_entry(record: dict, filename: str) -> dict:
    """Light, tensor-free summary of a record for the manifest index."""
    return {
        "id": record["id"],
        "file": filename,
        "question": record["question"],
        "canonical_response": record["canonical"]["response"],
        "accuracy": record["canonical"]["accuracy"],
        "cluster_assignment_entropy": record["labels"]["cluster_assignment_entropy"],
        "n_clusters": record["labels"]["n_clusters"],
        "n_samples": record["labels"]["n_samples"],
    }


def write_manifest(manifest_path: str, config_dict: dict, meta: dict, entries: list) -> None:
    payload = {
        "schema_version": SCHEMA_VERSION,
        "config": config_dict,
        "meta": meta,
        "n_records": len(entries),
        "records": {e["id"]: e for e in entries},
    }
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(payload, f, indent=2)


def read_manifest(manifest_path: str) -> dict:
    with open(manifest_path) as f:
        return json.load(f)


def describe_record(record: dict) -> str:
    """Human-readable structural dump of one record (shapes/types, not tensor data)."""
    def fmt(v, indent=2):
        pad = " " * indent
        if isinstance(v, torch.Tensor):
            return f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype})"
        if isinstance(v, dict):
            lines = ["{"]
            for k, vv in v.items():
                lines.append(f"{pad}{k!r}: {fmt(vv, indent + 2)}")
            lines.append(" " * (indent - 2) + "}")
            return "\n".join(lines)
        if isinstance(v, list):
            n = len(v)
            if n == 0:
                return "[] (len=0)"
            head = v[0]
            if isinstance(head, dict):
                return f"list(len={n}) of dict; first={fmt(head, indent + 2)}"
            if isinstance(head, (int, float)):
                preview = v[:3]
                return f"list(len={n}) of number; first3={preview}"
            return f"list(len={n}); first={head!r}"
        if isinstance(v, str):
            s = v if len(v) <= 80 else v[:77] + "..."
            return f"str(len={len(v)}): {s!r}"
        return f"{type(v).__name__}: {v!r}"

    return fmt(record)
