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

    for arm in cfg.arms:          # default = the 3 z-arms; --arms can add the text-only ones
        out = trainer.train_arm(pos, layer, arm=arm, max_steps=cfg.smoke_steps, verbose=False)
        print(f"  train arm={arm:12s} {cfg.smoke_steps} steps -> {out}")

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
    ckpt_dir = os.path.join(cfg.run_dir(), "checkpoints") if cfg.save_checkpoints else None
    if ckpt_dir:
        logging.info("Saving per-(arm,seed) checkpoints under %s", ckpt_dir)
    trials = []
    for s in cfg.arm_trial_seeds:
        logging.info("=== arm trial seed=%d ===", s)
        arm_res = trainer.train_arms_trial(pos, layer, best_k, cfg.arms, trial_seed=s,
                                           save_dir=ckpt_dir)
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

    if ckpt_dir and cfg.push_wandb:
        _push_checkpoints_wandb(cfg, ckpt_dir,
                                selected={"position": pos, "layer": layer, "k": best_k})
    return result


def _load_selected(cfg: Stage2Config) -> dict:
    """Resolve the (position, layer, k) to train at.

    An explicit config override WINS over a prior results.json: the 3B sweep's own
    selection proved unreliable (it trains on a 600-example subsample for 3 epochs per
    candidate, too noisy to rank layers), and the cheap exact ridge sweep in
    `amortized_ue/linear_ceiling_probe.py` should be used to pick the layer instead.
    Falls back to the saved sweep result when no override is given."""
    if cfg.z_inputs:
        # multi-input z: the forward path reads cfg.z_inputs directly, so (position, layer) is
        # only a nominal label for logging/checkpoint meta. Take it from the first entry.
        pos, layer = Stage2Data.parse_z_inputs(cfg.z_inputs)[0]
        return {"position": pos, "layer": layer,
                "k": cfg.selected_k or cfg.k_soft_tokens}
    if (cfg.selected_position is not None and cfg.selected_layer is not None
            and cfg.selected_k is not None):
        return {"position": cfg.selected_position, "layer": cfg.selected_layer,
                "k": cfg.selected_k}
    path = os.path.join(cfg.run_dir(), "results.json")
    if os.path.exists(path):
        sel = json.load(open(path)).get("selected", {})
        if sel.get("position") is not None:
            return {"position": sel["position"], "layer": sel["layer"], "k": sel["k"]}
    raise RuntimeError(
        "No prior results.json and no selected_position/layer/k in config; "
        "run the in-distribution build first or set the overrides.")


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
    # save checkpoints here too (build() did, build_ood() did NOT — that bug meant --ood runs
    # trained and discarded every model despite save_checkpoints=True).
    ckpt_dir = os.path.join(cfg.run_dir(), "checkpoints") if cfg.save_checkpoints else None
    if ckpt_dir:
        logging.info("Saving per-(arm,seed) checkpoints under %s", ckpt_dir)
    # multi-seed: each trial reuses the same (seed,trial,arm) streams as build(), so the
    # ID test numbers here match build()'s for the same trial_seed (removes the caveat).
    trials = []
    for s in cfg.arm_trial_seeds:
        logging.info("=== OOD arm trial seed=%d ===", s)
        arm_res = trainer.train_arms_trial(pos, layer, k, cfg.arms, trial_seed=s,
                                           ood_data=ood_data, save_dir=ckpt_dir)
        trials.append({"seed": s, "arms": arm_res})
        for arm in cfg.arms:
            tr_, idt, ood = arm_res[arm]["train"], arm_res[arm]["test"], arm_res[arm]["ood"]
            # train-test gap = the direct overfitting diagnostic (ridge ref: 0.856/0.642, gap 0.21)
            logging.info("seed=%d arm=%-11s TRAIN sp=%.4f | ID test sp=%.4f au=%.4f "
                         "(gap %+.3f) | OOD(%s) sp=%.4f au=%.4f",
                         s, arm, tr_["spearman"], idt["spearman"], idt["auroc"],
                         tr_["spearman"] - idt["spearman"], cfg.ood_dataset,
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

    os.makedirs(cfg.run_dir(), exist_ok=True)   # build() creates it; --ood can run standalone
    out = os.path.join(cfg.run_dir(), f"ood_results_{cfg.ood_dataset}_multiseed.json")
    with open(out, "w") as f:
        json.dump({"config": cfg.as_dict(), **result}, f, indent=2)
    logging.info("Wrote OOD results -> %s", out)
    return result


# ------------------------------ checkpoint eval -------------------------------
def _summarize_by_arm(by_arm: dict) -> dict:
    """{arm: [per-seed metric dicts]} -> {arm: {metric: mean±std}}."""
    return {arm: {k: _stats([r[k] for r in rows]) for k in _AGG_KEYS}
            for arm, rows in by_arm.items()}


def _paired_by_arm(by_arm: dict, ref: str = "z") -> dict:
    """Per-seed paired (arm - ref) differences, aligned by seed."""
    ref_by_seed = {r["seed"]: r for r in by_arm.get(ref, [])}
    out = {}
    for arm, rows in by_arm.items():
        if arm == ref:
            continue
        entry = {}
        for k in ("spearman", "auroc"):
            diffs = [r[k] - ref_by_seed[r["seed"]][k] for r in rows if r["seed"] in ref_by_seed]
            s = _stats(diffs)
            s["frac_positive"] = float(np.mean([d > 0 for d in diffs])) if diffs else float("nan")
            entry[k] = s
        out[f"{arm}_minus_{ref}"] = entry
    return out


def run_eval(cfg: Stage2Config) -> dict:
    """Load saved checkpoints and score them on one or more datasets — NO retraining.

    The checkpoints' own training dataset is scored on its held-out TEST split (ID);
    any `--eval_datasets name:N` are scored on ALL rows (OOD). Results are aggregated
    per arm across seeds (mean±std) exactly like the training runs, so this reproduces
    the ID/OOD numbers without recomputing the models.
    """
    import glob
    import dataclasses
    from collections import defaultdict
    from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta

    ckpt_dir = cfg.ckpt_dir or os.path.join(cfg.run_dir(), "checkpoints")
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    assert paths, f"no *.pt checkpoints in {ckpt_dir}"
    logging.info("Eval: %d checkpoints in %s", len(paths), ckpt_dir)

    first = read_meta(paths[0])
    id_ds, id_n = first["train_dataset"], first["train_num_samples"]
    targets = [(id_ds, id_n, "test")]                       # ID: held-out test split
    for spec in cfg.eval_datasets:                          # OOD: all rows
        name, n = spec.split(":")
        targets.append((name, int(n), "all"))

    model, trainer = None, None
    out = {"ckpt_dir": ckpt_dir, "id_dataset": id_ds, "target_model": first["stage1_model_name"],
           "datasets": {}}
    for ds, n, split in targets:
        eval_cfg = dataclasses.replace(cfg, stage1_dataset=ds, stage1_num_samples=n, smoke=False)
        eval_data = Stage2Data(eval_cfg)
        by_arm = defaultdict(list)
        for p in paths:
            model, meta, transform = load_checkpoint(p, model=model)
            if trainer is None:
                trainer = Trainer(cfg, eval_data, model=model)
            trainer.data = eval_data
            m = trainer.evaluate_on(meta["position"], meta["layer"], meta["arm"],
                                    eval_data, transform, split=split)
            by_arm[meta["arm"]].append({"seed": meta["seed"], **m})
        summary = _summarize_by_arm(by_arm)
        out["datasets"][ds] = {"split": split, "n": len(eval_data.ids),
                               "summary": summary, "paired": _paired_by_arm(by_arm, ref="z"),
                               "per_seed": {a: rows for a, rows in by_arm.items()}}
        for arm in sorted(by_arm):
            au, sp = summary[arm]["auroc"], summary[arm]["spearman"]
            logging.info("eval %-9s [%-4s] arm=%-9s auroc %.4f±%.4f  spearman %.4f±%.4f",
                         ds, split, arm, au["mean"], au["std"], sp["mean"], sp["std"])

    dest = os.path.join(ckpt_dir, "eval_summary.json")
    with open(dest, "w") as f:
        json.dump({"config": cfg.as_dict(), **out}, f, indent=2)
    logging.info("Wrote eval summary -> %s", dest)

    if cfg.push_wandb:
        _push_checkpoints_wandb(cfg, ckpt_dir, selected={k: first[k] for k in ("position", "layer", "k")})
    return out


def _push_checkpoints_wandb(cfg: Stage2Config, ckpt_dir: str, selected: dict) -> None:
    """Push the local checkpoint dir as a versioned W&B artifact (type=model)."""
    import wandb
    name = f"stage2_ckpts_{cfg.stage1_model_name}_{cfg.stage1_dataset}_n{cfg.stage1_num_samples}"
    run = wandb.init(project=cfg.wandb_project, entity=os.environ.get("WANDB_ENT"),
                     name=name, job_type="checkpoint", config=cfg.as_dict())
    art = wandb.Artifact(name, type="model",
                         metadata={"selected": selected, "seeds": list(cfg.arm_trial_seeds),
                                   "arms": list(cfg.arms), "proxy_model": cfg.proxy_model})
    art.add_dir(ckpt_dir)
    run.log_artifact(art)
    run.finish()
    logging.info("Pushed checkpoints to W&B artifact %r (project %s)", name, cfg.wandb_project)


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
    # checkpoints save by DEFAULT now (Stage2Config.save_checkpoints=True). --save_checkpoints
    # is kept as a harmless explicit opt-in; --no_save_checkpoints opts out.
    p.add_argument("--save_checkpoints", dest="save_checkpoints", action="store_true",
                   help="save a checkpoint per (arm, seed) under run_dir/checkpoints (default ON)")
    p.add_argument("--no_save_checkpoints", dest="save_checkpoints", action="store_false",
                   help="opt OUT of saving checkpoints")
    p.set_defaults(save_checkpoints=Stage2Config.save_checkpoints)
    p.add_argument("--push_wandb", action="store_true",
                   help="build: also push the checkpoint dir as a W&B artifact")
    p.add_argument("--eval", action="store_true",
                   help="load saved checkpoints and score them (no retraining)")
    p.add_argument("--ckpt_dir", default=None,
                   help="eval: dir of *.pt checkpoints (default run_dir/checkpoints)")
    p.add_argument("--selected_position", default=None, choices=["TBG", "SLT"],
                   help="force the z position (skips/overrides the 3B sweep; use with "
                        "--reuse_selection). Pick it with linear_ceiling_probe.py.")
    p.add_argument("--selected_layer", type=int, default=None,
                   help="force the z layer (see --selected_position)")
    p.add_argument("--selected_k", type=int, default=None,
                   help="force k soft tokens (else read from the prior results.json)")
    p.add_argument("--arms", default=None,
                   help="comma-separated arms to train. Default: z,z_q,z_q_resp. The TEXT-ONLY "
                        "arms q_only / q_resp_only drop z entirely (no target-LLM forward pass) "
                        "-- e.g. 'z,z_q,z_q_resp,q_only,q_resp_only' for the full 5-arm set.")
    # --- regularisation knobs (the proxy is UNDER-regularised: train 0.891 / test 0.590,
    #     vs optimal ridge's 0.856 / 0.642 -- it memorises harder and generalises worse) ---
    p.add_argument("--weight_decay", type=float, default=Stage2Config.weight_decay,
                   help="AdamW weight decay (default 0.01). The analogue of ridge's alpha, which "
                        "is selected at ~1e4 on this data — so 0.01 is very weak. Try 0.1 / 1.0.")
    p.add_argument("--projector_type", default=Stage2Config.projector_type,
                   choices=["mlp", "linear"],
                   help="'linear' drops the GELU + hidden layer. The z->SE relation is LINEAR "
                        "(MLP loses to ridge at every input), so the MLP is capacity to overfit with.")
    p.add_argument("--projector_dropout", type=float, default=Stage2Config.projector_dropout,
                   help="projector dropout (default 0.1)")
    p.add_argument("--lora_r", type=int, default=Stage2Config.lora_r,
                   help="LoRA rank (default 16). 0 DISABLES LoRA entirely — the backbone is frozen "
                        "and the task is linear, so LoRA across 28 layers is pure overfitting capacity.")
    p.add_argument("--projector_hidden_dim", type=int, default=Stage2Config.projector_hidden_dim,
                   help="projector bottleneck width (default 256, the original locked value). "
                        "256 compresses a 4096-dim z ~16x -- and 32x when --z_inputs stacks two "
                        "positions (h_in=8192). Widen to 1024/2048 when stacking.")
    p.add_argument("--z_inputs", default=None,
                   help="comma-separated POSITION:LAYER pairs to feed together, e.g. "
                        "'TBG:22,SLT:15'. Stacked to [B,n,H] and flattened by the projector "
                        "(h_in widens to n*H). Overrides --selected_position/--selected_layer "
                        "for the z input. Default: single selected (position, layer).")
    p.add_argument("--run_name", default=None,
                   help="output dir name; set it when overriding the layer so the run does "
                        "not overwrite the reference TBG-L12 results")
    p.add_argument("--eval_datasets", default=None,
                   help="eval: extra OOD datasets as 'name:N,name:N' (scored on all rows)")
    a = p.parse_args()
    mode = ("report" if a.report else "smoke" if a.smoke else
            "eval" if a.eval else "ood" if a.ood else "full")

    if a.seeds is None:
        seeds = Stage2Config.arm_trial_seeds
    elif "," in a.seeds:
        seeds = tuple(int(x) for x in a.seeds.split(","))
    else:
        seeds = tuple(range(int(a.seeds)))
    eval_datasets = tuple(a.eval_datasets.split(",")) if a.eval_datasets else ()
    z_inputs = tuple(s.strip() for s in a.z_inputs.split(",")) if a.z_inputs else ()
    arms = tuple(s.strip() for s in a.arms.split(",")) if a.arms else Stage2Config.arms

    cfg = Stage2Config(
        k_soft_tokens=a.k_soft_tokens, proxy_model=a.proxy_model, lr=a.lr, epochs=a.epochs,
        batch_size=a.batch_size, stage1_num_samples=a.stage1_num_samples,
        ood_dataset=a.ood_dataset, ood_num_samples=a.ood_num_samples,
        arm_trial_seeds=seeds, reuse_selection=a.reuse_selection,
        save_checkpoints=a.save_checkpoints, push_wandb=a.push_wandb,
        ckpt_dir=a.ckpt_dir, eval_datasets=eval_datasets,
        selected_position=a.selected_position, selected_layer=a.selected_layer,
        selected_k=a.selected_k, run_name=a.run_name, z_inputs=z_inputs,
        projector_hidden_dim=a.projector_hidden_dim, arms=arms,
        weight_decay=a.weight_decay, projector_type=a.projector_type,
        projector_dropout=a.projector_dropout, lora_r=a.lora_r,
        smoke=a.smoke, smoke_num_prompts=a.smoke_num_prompts, smoke_steps=a.smoke_steps)
    return cfg, mode


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg, mode = _parse()
    if mode == "report":
        report(cfg)
    elif mode == "smoke":
        run_smoke(cfg)
    elif mode == "eval":
        run_eval(cfg)
    elif mode == "ood":
        build_ood(cfg)
    else:
        build(cfg)
