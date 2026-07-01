"""Stage 2 entry point: smoke test and (later) the full run.

Smoke mode runs the entire path end to end on a handful of prompts for a couple of
steps, printing tensor shapes and a sample prediction, WITHOUT launching real
training or the (position, layer) sweep. The full run is implemented but must be
launched explicitly (and only after the smoke test is confirmed).
"""
from __future__ import annotations

import json
import logging
import argparse

import numpy as np
import torch

from amortized_ue.stage2.config import Stage2Config
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer


def _default_smoke_pos_layer(data: Stage2Data, cfg: Stage2Config):
    # Smoke does not sweep; use a fixed, valid (position, layer): SLT + last layer.
    pos = cfg.sweep_positions[-1]
    layer = min(data.n_layers - 2, data.n_layers - 1)
    return pos, layer


def run_smoke(cfg: Stage2Config) -> dict:
    cfg.smoke = True
    logging.info("=== STAGE 2 SMOKE TEST ===")
    data = Stage2Data(cfg)
    print("\n" + "=" * 78)
    print("DATA")
    print("=" * 78)
    print(f"  n prompts (smoke)     : {len(data.ids)}")
    print(f"  positions             : {data.positions}  (TBG=last input tok; SLT=2nd-last gen tok)")
    print(f"  n layers / hidden size: {data.n_layers} / {data.hidden_size}")
    print(f"  split train/val/test  : {len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)}")
    print(f"  label mean/std (all)  : {data.labels_raw.mean():.4f} / {data.labels_raw.std():.4f}")
    print(f"  target transform      : {cfg.target_transform} "
          f"(train mean={data.transform.mean:.4f}, std={data.transform.std:.4f})")
    print(f"  bin threshold (AUROC) : {data.bin_threshold:.4f}")

    trainer = Trainer(cfg, data)
    pos, layer = _default_smoke_pos_layer(data, cfg)
    print("\n" + "=" * 78)
    print("MODEL")
    print("=" * 78)
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print(f"  proxy backbone        : {cfg.proxy_model} (frozen), d_model={trainer.model.d_model}")
    print(f"  device                : {trainer.device}")
    print(f"  k soft tokens         : {cfg.k_soft_tokens}")
    print(f"  trainable / total pars: {n_train:,} / {n_total:,}")
    print(f"  smoke (position,layer): ({pos}, {layer})")

    # shapes of one forward's inputs
    rows = data.split_indices("train")[: min(cfg.batch_size, len(data.train_idx))]
    z = data.hidden[pos][layer][rows].unsqueeze(1)
    print(f"  z batch shape         : {tuple(z.shape)} (=[B,1,H])")

    print("\n" + "=" * 78)
    print(f"TRAIN {cfg.smoke_steps} STEPS (arm={cfg.arm})")
    print("=" * 78)
    out = trainer.train_arm(pos, layer, arm=cfg.arm, max_steps=cfg.smoke_steps, verbose=True)
    print(f"  {out}")

    # sample prediction (original label space)
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer._forward_batch(rows, pos, layer, train=False, arm=cfg.arm)
    pred_orig = data.transform.decode(pred.float().cpu()).numpy()
    y_true = data.labels_raw[rows].numpy()
    print("\n" + "=" * 78)
    print("SAMPLE PREDICTIONS (original SE space)")
    print("=" * 78)
    for i in range(min(4, len(rows))):
        print(f"  id={data.ids[rows[i]][:28]:28s}  pred={pred_orig[i]:+.4f}  true={y_true[i]:+.4f}")
    print("\nSMOKE TEST OK\n")
    return {"pos": pos, "layer": layer, "train_out": out}


def build(cfg: Stage2Config) -> dict:
    """Full run: select (pos,layer) via z-only sweep, train the multi-arm model,
    evaluate all three arms + the k ablation on the z-only arm. Not auto-launched."""
    data = Stage2Data(cfg)
    trainer = Trainer(cfg, data)

    if cfg.select_pos_layer:
        logging.info("Selecting (position, layer) via z-only sweep on val ...")
        best, table = trainer.sweep_pos_layer(epochs=cfg.sweep_epochs)
        pos, layer = best["position"], best["layer"]
        logging.info("Selected position=%s layer=%d (val_rmse=%.4f)", pos, layer, best["rmse"])
    else:
        pos, layer = cfg.selected_position, cfg.selected_layer
        table = None

    # Final model: train once with modality dropout (serves all arms), evaluate each.
    trainer.reset_trainable()
    trainer.train_arm(pos, layer, arm="z_q_resp")
    arms = {}
    for arm in ("z", "z_q", "z_q_resp"):
        arms[arm] = {s: trainer.evaluate(pos, layer, arm, s) for s in ("val", "test")}

    result = {"selected": {"position": pos, "layer": layer}, "arms": arms, "sweep": table}
    import os
    os.makedirs(cfg.run_dir(), exist_ok=True)
    with open(os.path.join(cfg.run_dir(), "results.json"), "w") as f:
        json.dump({"config": cfg.as_dict(), **result}, f, indent=2)
    logging.info("Results: %s", json.dumps(arms, indent=2))
    return result


def _parse() -> Stage2Config:
    p = argparse.ArgumentParser(description="Stage 2 amortized-UE proxy.")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--arm", default=Stage2Config.arm, choices=["z", "z_q", "z_q_resp"])
    p.add_argument("--k_soft_tokens", type=int, default=Stage2Config.k_soft_tokens)
    p.add_argument("--proxy_model", default=Stage2Config.proxy_model)
    p.add_argument("--lora_r", type=int, default=Stage2Config.lora_r)
    p.add_argument("--lr", type=float, default=Stage2Config.lr)
    p.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    p.add_argument("--stage1_model_name", default=Stage2Config.stage1_model_name)
    p.add_argument("--stage1_dataset", default=Stage2Config.stage1_dataset)
    p.add_argument("--stage1_num_samples", type=int, default=Stage2Config.stage1_num_samples)
    p.add_argument("--smoke_num_prompts", type=int, default=Stage2Config.smoke_num_prompts)
    p.add_argument("--smoke_steps", type=int, default=Stage2Config.smoke_steps)
    a = p.parse_args()
    return Stage2Config(
        smoke=a.smoke, arm=a.arm, k_soft_tokens=a.k_soft_tokens, proxy_model=a.proxy_model,
        lora_r=a.lora_r, lr=a.lr, epochs=a.epochs, stage1_model_name=a.stage1_model_name,
        stage1_dataset=a.stage1_dataset, stage1_num_samples=a.stage1_num_samples,
        smoke_num_prompts=a.smoke_num_prompts, smoke_steps=a.smoke_steps)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = _parse()
    if cfg.smoke:
        run_smoke(cfg)
    else:
        build(cfg)
