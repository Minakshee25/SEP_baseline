"""Stage 2 entry point: report checks, smoke test, and the full run.

  --report : load the dataset, print the SE label-distribution report and verify the
             sweep subsample is strictly from the train split. No model, no training.
  --smoke  : run the whole path on a few prompts for a couple of steps (shapes + a
             sample prediction). No sweep, no real training.
  (default): full run — subsample (pos,layer) sweep by val Spearman -> k ablation on the
             z-only arm -> train the three arm models separately at the best k -> eval.

The full run must be launched explicitly and only after the pre-launch checks are OK.
"""
from __future__ import annotations

import os
import json
import logging
import argparse

import numpy as np
import torch

from amortized_ue.stage2.config import Stage2Config
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer


def report(cfg: Stage2Config) -> dict:
    """Pre-launch checks: label distribution + strict-train subsample verification."""
    data = Stage2Data(cfg)
    print("\n" + "=" * 78)
    print(f"DATASET: {data.cfg.stage1_num_samples} records "
          f"({cfg.stage1_model_name} / {cfg.stage1_dataset})")
    print("=" * 78)
    print(f"  loaded N={len(data.ids)} | split train/val/test = "
          f"{len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)} (seed {cfg.split_seed})")
    print(data.label_distribution_report())

    print("\n" + "=" * 78)
    print(f"SWEEP SUBSAMPLE ({cfg.sweep_subsample_size}, seed {cfg.sweep_subsample_seed})")
    print("=" * 78)
    sub = data.sweep_subsample(cfg.sweep_subsample_size, cfg.sweep_subsample_seed)
    train_set, val_set, test_set = set(data.train_idx.tolist()), set(data.val_idx.tolist()), set(data.test_idx.tolist())
    ssub = set(sub.tolist())
    print(f"  size                 : {len(sub)}")
    print(f"  subset of TRAIN      : {ssub.issubset(train_set)}   (must be True)")
    print(f"  disjoint from VAL    : {ssub.isdisjoint(val_set)}   (must be True)")
    print(f"  disjoint from TEST   : {ssub.isdisjoint(test_set)}   (must be True)")
    print(f"  first 5 subsample idx: {sub[:5].tolist()}")
    ok = ssub.issubset(train_set) and ssub.isdisjoint(val_set) and ssub.isdisjoint(test_set)
    print(f"\n  SUBSAMPLE STRICTLY FROM TRAIN: {ok}")
    return {"ok": ok, "n": len(data.ids),
            "split": [len(data.train_idx), len(data.val_idx), len(data.test_idx)]}


def _default_smoke_pos_layer(data: Stage2Data, cfg: Stage2Config):
    return cfg.sweep_positions[-1], min(data.n_layers - 2, data.n_layers - 1)


def run_smoke(cfg: Stage2Config) -> dict:
    cfg.smoke = True
    logging.info("=== STAGE 2 SMOKE TEST ===")
    data = Stage2Data(cfg)
    print("\n" + "=" * 78 + "\nDATA\n" + "=" * 78)
    print(f"  n prompts (smoke)     : {len(data.ids)}")
    print(f"  positions             : {data.positions}  (TBG=last input tok; SLT=2nd-last gen tok)")
    print(f"  n layers / hidden size: {data.n_layers} / {data.hidden_size}")
    print(f"  split train/val/test  : {len(data.train_idx)}/{len(data.val_idx)}/{len(data.test_idx)}")
    print(f"  target transform      : {cfg.target_transform} "
          f"(train mean={data.transform.mean:.4f}, std={data.transform.std:.4f})")

    trainer = Trainer(cfg, data)
    pos, layer = _default_smoke_pos_layer(data, cfg)
    n_train = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in trainer.model.parameters())
    print("\n" + "=" * 78 + "\nMODEL\n" + "=" * 78)
    print(f"  proxy backbone        : {cfg.proxy_model} (frozen), d_model={trainer.model.d_model}")
    print(f"  device                : {trainer.device}")
    print(f"  projector             : LN -> {trainer.model.h_in}->{cfg.projector_hidden_dim}->k*{trainer.model.d_model}"
          f" (drop {cfg.projector_dropout}, learnable scale)")
    print(f"  k soft tokens         : {cfg.k_soft_tokens}")
    print(f"  trainable / total pars: {n_train:,} / {n_total:,}")
    print(f"  smoke (position,layer): ({pos}, {layer})")

    for arm in ("z", "z_q", "z_q_resp"):
        out = trainer.train_arm(pos, layer, arm=arm, max_steps=cfg.smoke_steps, verbose=False)
        print(f"  train arm={arm:9s} {cfg.smoke_steps} steps -> {out}")

    rows = data.split_indices("train")[: min(cfg.batch_size, len(data.train_idx))]
    trainer.model.eval()
    with torch.no_grad():
        pred = trainer._forward_batch(rows, pos, layer, "z_q_resp")
    pred_orig = data.transform.decode(pred.float().cpu()).numpy()
    y_true = data.labels_raw[rows].numpy()
    print("\n" + "=" * 78 + "\nSAMPLE PREDICTIONS (z_q_resp, original SE space)\n" + "=" * 78)
    for i in range(min(4, len(rows))):
        print(f"  id={data.ids[rows[i]][:28]:28s}  pred={pred_orig[i]:+.4f}  true={y_true[i]:+.4f}")
    print("\nSMOKE TEST OK\n")
    return {"pos": pos, "layer": layer}


def build(cfg: Stage2Config) -> dict:
    data = Stage2Data(cfg)
    trainer = Trainer(cfg, data)

    # 1) (position, layer) selection: z-only sweep on a strict-train subsample, by Spearman
    sub = data.sweep_subsample(cfg.sweep_subsample_size, cfg.sweep_subsample_seed)
    trainer.set_k(cfg.select_k_soft_tokens)
    logging.info("Sweep: z-only over %d positions x %d layers on %d-example train subsample ...",
                 len(cfg.sweep_positions), len(cfg.sweep_layers), len(sub))
    best, sweep_table = trainer.sweep_pos_layer(train_rows=sub, epochs=cfg.sweep_epochs)
    pos, layer = best["position"], best["layer"]
    logging.info("Selected position=%s layer=%d (val_spearman=%.4f)", pos, layer, best["spearman"])

    # 2) k ablation on the z-only arm (full train), select best k
    kbest, k_table = trainer.k_ablation(pos, layer, epochs=cfg.epochs)
    best_k = kbest["k"]
    logging.info("Selected k=%d (val_spearman=%.4f)", best_k, kbest["spearman"])

    # 3) train the three arms separately at the best k on the full train split
    trainer.set_k(best_k)
    arms = {}
    for arm in cfg.arms:
        trainer.reset_trainable()
        trainer.train_arm(pos, layer, arm=arm, epochs=cfg.epochs)
        arms[arm] = {s: trainer.evaluate(pos, layer, arm, s) for s in ("val", "test")}
        logging.info("arm=%s test spearman=%.4f auroc=%.4f rmse=%.4f",
                     arm, arms[arm]["test"]["spearman"], arms[arm]["test"]["auroc"], arms[arm]["test"]["rmse"])

    result = {"selected": {"position": pos, "layer": layer, "k": best_k},
              "arms": arms, "sweep": sweep_table, "k_ablation": k_table}
    os.makedirs(cfg.run_dir(), exist_ok=True)
    with open(os.path.join(cfg.run_dir(), "results.json"), "w") as f:
        json.dump({"config": cfg.as_dict(), **result}, f, indent=2)
    logging.info("Wrote results -> %s", os.path.join(cfg.run_dir(), "results.json"))
    return result


def _parse() -> tuple[Stage2Config, str]:
    p = argparse.ArgumentParser(description="Stage 2 amortized-UE proxy.")
    p.add_argument("--report", action="store_true", help="pre-launch checks only")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--k_soft_tokens", type=int, default=Stage2Config.k_soft_tokens)
    p.add_argument("--proxy_model", default=Stage2Config.proxy_model)
    p.add_argument("--lr", type=float, default=Stage2Config.lr)
    p.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    p.add_argument("--batch_size", type=int, default=Stage2Config.batch_size)
    p.add_argument("--stage1_num_samples", type=int, default=Stage2Config.stage1_num_samples)
    p.add_argument("--smoke_num_prompts", type=int, default=Stage2Config.smoke_num_prompts)
    p.add_argument("--smoke_steps", type=int, default=Stage2Config.smoke_steps)
    a = p.parse_args()
    mode = "report" if a.report else ("smoke" if a.smoke else "full")
    cfg = Stage2Config(
        k_soft_tokens=a.k_soft_tokens, proxy_model=a.proxy_model, lr=a.lr, epochs=a.epochs,
        batch_size=a.batch_size, stage1_num_samples=a.stage1_num_samples,
        smoke=a.smoke, smoke_num_prompts=a.smoke_num_prompts, smoke_steps=a.smoke_steps)
    return cfg, mode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, mode = _parse()
    if mode == "report":
        report(cfg)
    elif mode == "smoke":
        run_smoke(cfg)
    else:
        build(cfg)
