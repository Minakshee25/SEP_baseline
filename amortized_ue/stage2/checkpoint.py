"""Stage 2 checkpointing.

A checkpoint stores ONLY the trainable parameters (projector, REG token, head, LoRA
adapters) plus the metadata needed to rebuild and evaluate the model — never the frozen
backbone, which is reloaded from HuggingFace by name. This keeps a checkpoint ~40-55 MB
(vs ~6 GB for the backbone) and lets one trained proxy be scored on many datasets and
target models without retraining.

Metadata captured:
  - selected (position, layer, k) — which stored z the proxy consumes;
  - h_in — the *target* LLM's hidden size (the projector input dim; differs per target
    model, so it is read back at load time to rebuild the projector correctly);
  - transform (mean, std) — the TRAINING label transform; predictions on any eval dataset
    are decoded with this, exactly as at train time;
  - provenance — arm, seed, proxy model, training dataset + N, full Stage2Config.
"""
from __future__ import annotations

import dataclasses

import torch

from amortized_ue.stage2.config import Stage2Config
from amortized_ue.stage2.data import TargetTransform
from amortized_ue.stage2.model import ProxyModel

CKPT_FORMAT = "amortized_ue.stage2.checkpoint/v1"


def save_checkpoint(model: ProxyModel, meta: dict, path: str) -> None:
    """Persist the model's trainable params (CPU) + metadata to `path`."""
    state = {n: p.detach().cpu().clone()
             for n, p in model.named_parameters() if p.requires_grad}
    torch.save({"format": CKPT_FORMAT, "trainable": state, "meta": meta}, path)


def _cfg_from_meta(meta: dict) -> Stage2Config:
    """Rebuild a Stage2Config from the stored config dict (ignoring unknown keys so a
    checkpoint stays loadable if the config schema later gains fields)."""
    fields = {f.name for f in dataclasses.fields(Stage2Config)}
    return Stage2Config(**{k: v for k, v in meta["config"].items() if k in fields})


def read_meta(path: str) -> dict:
    """Load only the metadata (no weights) — cheap peek at a checkpoint."""
    return torch.load(path, map_location="cpu")["meta"]


def load_checkpoint(path: str, device: str | None = None, model: ProxyModel | None = None):
    """Load a checkpoint into a ProxyModel and return (model, meta, transform).

    Pass an existing `model` (same proxy backbone) to reuse it across many checkpoints
    without reloading the 3B backbone — only the trainable params are swapped in.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu")
    fmt = ckpt.get("format")
    assert fmt == CKPT_FORMAT, f"unexpected checkpoint format {fmt!r} (want {CKPT_FORMAT!r})"
    meta = ckpt["meta"]

    cfg = _cfg_from_meta(meta)
    if model is None:
        model = ProxyModel(cfg, h_in=meta["h_in"])
    # exp2_run.py checkpoints (incl. deploy_checkpoints) carry this format tag but omit
    # `k` / `transform` / `position` / `layer` in their meta (E39 bug, fixed for future saves)
    # -- fall back to the equivalent stored config fields for those. AUROC/rank metrics are
    # invariant to the transform fallback since it's a fixed linear (identity) rescaling; only
    # `transform.decode()`'s absolute SCALE is affected for exp2-line checkpoints.
    model.set_k(meta.get("k", cfg.k_soft_tokens))
    meta.setdefault("position", cfg.selected_position)
    meta.setdefault("layer", cfg.selected_layer)
    model.to(device)

    own = dict(model.named_parameters())
    missing = [n for n in ckpt["trainable"] if n not in own]
    if missing:
        raise RuntimeError(f"checkpoint has params not in the model: {missing[:5]} ...")
    with torch.no_grad():
        for n, t in ckpt["trainable"].items():
            own[n].copy_(t.to(own[n].device, own[n].dtype))

    tr = meta.get("transform")
    transform = TargetTransform(tr["mean"], tr["std"]) if tr else TargetTransform(0.0, 1.0)
    return model, meta, transform
