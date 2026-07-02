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


# ----------------------------- multi-seed aggregation -------------------------
_AGG_KEYS = ("spearman", "auroc", "rmse", "mae", "r2")


def _stats(vals: list) -> dict:
    vals = [float(v) for v in vals if v == v]           # drop NaN
    if not vals:
        return {"mean": float("nan"), "std": float("nan"), "values": []}
    std = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": float(np.mean(vals)), "std": std, "values": vals}


def _summarize(trials: list, arms, split: str) -> dict:
    """Per-arm mean±std over trials for each metric on the given split."""
    return {arm: {k: _stats([t["arms"][arm][split][k] for t in trials]) for k in _AGG_KEYS}
            for arm in arms}


def _paired(trials: list, arms, split: str, ref: str = "z") -> dict:
    """Per-trial paired difference (arm - ref) for spearman/auroc, with frac_positive."""
    out = {}
    for arm in arms:
        if arm == ref:
            continue
        entry = {}
        for k in ("spearman", "auroc"):
            diffs = [t["arms"][arm][split][k] - t["arms"][ref][split][k] for t in trials]
            diffs = [d for d in diffs if d == d]
            s = _stats(diffs)
            s["frac_positive"] = float(np.mean([d > 0 for d in diffs])) if diffs else float("nan")
            entry[k] = s
        out[f"{arm}_minus_{ref}"] = entry
    return out


def build(cfg: Stage2Config) -> dict:
    data = Stage2Data(cfg)
    trainer = Trainer(cfg, data)

    # --- selection: reuse a saved (pos,layer,k) or run the sweep + k-ablation ---
    if cfg.reuse_selection:
        sel = _load_selected(cfg)
        pos, layer, best_k = sel["position"], sel["layer"], sel["k"]
        sweep_table = k_table = None
        logging.info("Reusing selection: position=%s layer=%d k=%d (sweep/k-ablation skipped)",
                     pos, layer, best_k)
    else:
        # 1) (position, layer) selection: z-only sweep on a strict-train subsample, by Spearman
        sub = data.sweep_subsample(cfg.sweep_subsample_size, cfg.sweep_subsample_seed)
        trainer.reseed(cfg.seed)
        trainer.set_k(cfg.select_k_soft_tokens)
        logging.info("Sweep: z-only over %d positions x %d layers on %d-example train subsample ...",
                     len(cfg.sweep_positions), len(cfg.sweep_layers), len(sub))
        best, sweep_table = trainer.sweep_pos_layer(train_rows=sub, epochs=cfg.sweep_epochs)
        pos, layer = best["position"], best["layer"]
        logging.info("Selected position=%s layer=%d (val_spearman=%.4f)", pos, layer, best["spearman"])

        # 2) k ablation on the z-only arm (full train), select best k
        trainer.reseed(cfg.seed)
        kbest, k_table = trainer.k_ablation(pos, layer, epochs=cfg.epochs)
        best_k = kbest["k"]
        logging.info("Selected k=%d (val_spearman=%.4f)", best_k, kbest["spearman"])

    # --- multi-seed arm training: each arm on its own (seed,trial,arm) stream ----
    trials = []
    for s in cfg.arm_trial_seeds:
        logging.info("=== arm trial seed=%d ===", s)
        arm_res = trainer.train_arms_trial(pos, layer, best_k, cfg.arms, trial_seed=s)
        trials.append({"seed": s, "arms": arm_res})
        for arm in cfg.arms:
            m = arm_res[arm]["test"]
            logging.info("seed=%d arm=%-9s test spearman=%.4f auroc=%.4f rmse=%.4f",
                         s, arm, m["spearman"], m["auroc"], m["rmse"])

    summary = _summarize(trials, cfg.arms, "test")
    paired = _paired(trials, cfg.arms, "test", ref="z")
    for arm in cfg.arms:
        sp, au = summary[arm]["spearman"], summary[arm]["auroc"]
        logging.info("arm=%-9s test spearman %.4f±%.4f  auroc %.4f±%.4f",
                     arm, sp["mean"], sp["std"], au["mean"], au["std"])

    result = {"selected": {"position": pos, "layer": layer, "k": best_k},
              "seeds": list(cfg.arm_trial_seeds), "trials": trials,
              "summary": summary, "paired": paired,
              "sweep": sweep_table, "k_ablation": k_table}
    os.makedirs(cfg.run_dir(), exist_ok=True)
    out = os.path.join(cfg.run_dir(), "results_multiseed.json")
    with open(out, "w") as f:
        json.dump({"config": cfg.as_dict(), **result}, f, indent=2)
    logging.info("Wrote results -> %s", out)
    return result


def _load_selected(cfg: Stage2Config) -> dict:
    """Read the selected (position, layer, k) from the in-distribution run's results.json,
    falling back to explicit config overrides."""
    path = os.path.join(cfg.run_dir(), "results.json")
    if os.path.exists(path):
        sel = json.load(open(path)).get("selected", {})
        if sel.get("position") is not None:
            return {"position": sel["position"], "layer": sel["layer"], "k": sel["k"]}
    if cfg.selected_position is None or cfg.selected_layer is None or cfg.selected_k is None:
        raise RuntimeError(
            "No prior results.json and no selected_position/layer/k in config; "
            "run the in-distribution build first or set the overrides.")
    return {"position": cfg.selected_position, "layer": cfg.selected_layer, "k": cfg.selected_k}


def build_ood(cfg: Stage2Config) -> dict:
    """OOD eval: train each arm on the in-distribution dataset, evaluate on the OOD dataset.

    Uses the (position, layer, k) already selected in-distribution (no re-selection). The
    OOD dataset is never used for training or selection.
    """
    import dataclasses
    assert cfg.ood_dataset, "set cfg.ood_dataset (e.g. 'squad')"
    data = Stage2Data(cfg)                                    # in-distribution (train)
    ood_cfg = dataclasses.replace(cfg, stage1_dataset=cfg.ood_dataset,
                                  stage1_num_samples=cfg.ood_num_samples, smoke=False)
    ood_data = Stage2Data(ood_cfg)                            # OOD (eval only)
    logging.info("OOD: train on %s (N=%d) -> eval on %s (N=%d, all rows)",
                 cfg.stage1_dataset, len(data.ids), cfg.ood_dataset, len(ood_data.ids))

    sel = _load_selected(cfg)
    pos, layer, k = sel["position"], sel["layer"], sel["k"]
    logging.info("Using in-distribution selection: position=%s layer=%d k=%d", pos, layer, k)

    trainer = Trainer(cfg, data)
    # multi-seed: each trial reuses the same (seed,trial,arm) streams as build(), so the
    # ID test numbers here match build()'s for the same trial_seed (removes the caveat).
    trials = []
    for s in cfg.arm_trial_seeds:
        logging.info("=== OOD arm trial seed=%d ===", s)
        arm_res = trainer.train_arms_trial(pos, layer, k, cfg.arms, trial_seed=s, ood_data=ood_data)
        trials.append({"seed": s, "arms": arm_res})
        for arm in cfg.arms:
            idt, ood = arm_res[arm]["test"], arm_res[arm]["ood"]
            logging.info("seed=%d arm=%-9s ID test sp=%.4f au=%.4f | OOD(%s) sp=%.4f au=%.4f",
                         s, arm, idt["spearman"], idt["auroc"], cfg.ood_dataset,
                         ood["spearman"], ood["auroc"])

    result = {"selected": sel, "in_distribution": cfg.stage1_dataset, "ood": cfg.ood_dataset,
              "seeds": list(cfg.arm_trial_seeds), "trials": trials,
              "id_summary": _summarize(trials, cfg.arms, "test"),
              "ood_summary": _summarize(trials, cfg.arms, "ood"),
              "id_paired": _paired(trials, cfg.arms, "test", ref="z"),
              "ood_paired": _paired(trials, cfg.arms, "ood", ref="z")}
    for arm in cfg.arms:
        sp = result["ood_summary"][arm]["spearman"]
        au = result["ood_summary"][arm]["auroc"]
        logging.info("arm=%-9s OOD(%s) spearman %.4f±%.4f  auroc %.4f±%.4f",
                     arm, cfg.ood_dataset, sp["mean"], sp["std"], au["mean"], au["std"])

    out = os.path.join(cfg.run_dir(), f"ood_results_{cfg.ood_dataset}_multiseed.json")
    with open(out, "w") as f:
        json.dump({"config": cfg.as_dict(), **result}, f, indent=2)
    logging.info("Wrote OOD results -> %s", out)
    return result


def _parse() -> tuple[Stage2Config, str]:
    p = argparse.ArgumentParser(description="Stage 2 amortized-UE proxy.")
    p.add_argument("--report", action="store_true", help="pre-launch checks only")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--ood", action="store_true", help="train ID, evaluate on --ood_dataset")
    p.add_argument("--ood_dataset", default=None)
    p.add_argument("--ood_num_samples", type=int, default=Stage2Config.ood_num_samples)
    p.add_argument("--k_soft_tokens", type=int, default=Stage2Config.k_soft_tokens)
    p.add_argument("--proxy_model", default=Stage2Config.proxy_model)
    p.add_argument("--lr", type=float, default=Stage2Config.lr)
    p.add_argument("--epochs", type=int, default=Stage2Config.epochs)
    p.add_argument("--batch_size", type=int, default=Stage2Config.batch_size)
    p.add_argument("--stage1_num_samples", type=int, default=Stage2Config.stage1_num_samples)
    p.add_argument("--smoke_num_prompts", type=int, default=Stage2Config.smoke_num_prompts)
    p.add_argument("--smoke_steps", type=int, default=Stage2Config.smoke_steps)
    p.add_argument("--seeds", default=None,
                   help="arm trial seeds: comma list '0,1,2' or an int N -> 0..N-1")
    p.add_argument("--reuse_selection", action="store_true",
                   help="skip the sweep/k-ablation; reuse the saved (pos,layer,k)")
    a = p.parse_args()
    mode = "report" if a.report else ("smoke" if a.smoke else ("ood" if a.ood else "full"))

    if a.seeds is None:
        seeds = Stage2Config.arm_trial_seeds
    elif "," in a.seeds:
        seeds = tuple(int(x) for x in a.seeds.split(","))
    else:
        seeds = tuple(range(int(a.seeds)))

    cfg = Stage2Config(
        k_soft_tokens=a.k_soft_tokens, proxy_model=a.proxy_model, lr=a.lr, epochs=a.epochs,
        batch_size=a.batch_size, stage1_num_samples=a.stage1_num_samples,
        ood_dataset=a.ood_dataset, ood_num_samples=a.ood_num_samples,
        arm_trial_seeds=seeds, reuse_selection=a.reuse_selection,
        smoke=a.smoke, smoke_num_prompts=a.smoke_num_prompts, smoke_steps=a.smoke_steps)
    return cfg, mode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, mode = _parse()
    if mode == "report":
        report(cfg)
    elif mode == "smoke":
        run_smoke(cfg)
    elif mode == "ood":
        build_ood(cfg)
    else:
        build(cfg)
