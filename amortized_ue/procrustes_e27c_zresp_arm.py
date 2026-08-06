"""E27c — train the z_resp arm (hidden state + response, NO question) and test it directly.

The E27 gate (ensemble) said the response is mildly complementary to aligned-z (+0.03). This trains
the ACTUAL z_resp arm the reference proxy never had, with identical hyperparameters (reconstructed
from a reference checkpoint), then evaluates it cross-LLM on ALIGNED Mistral [TBG:22,SLT:15] just
like E27b. Compares against z (0.545), z_q_resp (0.510), q_resp_only (0.531), ridge (0.580), and the
free ensemble (0.608). Additive; the only code change is a one-line z_resp case in train._arm_text.

    python -m amortized_ue.procrustes_e27c_zresp_arm      # amortized_stage2 env, GPU
"""
from __future__ import annotations

import os
import glob
import json
import dataclasses

import numpy as np
import torch

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta
from amortized_ue.procrustes_e27b_proxy_vs_ridge import run_proxy, fit_align_at, align

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"
CKPT_DIR = "amortized_ue/stage2/runs/E27_zresp_arm/checkpoints"
REFNUMS = {"z": 0.545, "z_q_resp": 0.510, "q_resp_only": 0.531, "ridge": 0.580, "ensemble(z+resp)": 0.608}


def train_zresp():
    """Train z_resp x 5 seeds with the reference config (Llama-2 n2000, TBG:22+SLT:15, k=4)."""
    cfg = _cfg_from_meta(read_meta(os.path.join(REF, "z_seed0.pt")))
    cfg = dataclasses.replace(cfg, run_name="E27_zresp_arm", arms=("z_resp",), save_checkpoints=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = Stage2Data(cfg)                                        # Llama-2 n2000
    trainer = Trainer(cfg, data)
    for s in cfg.arm_trial_seeds:
        print(f"--- training z_resp trial_seed={s} ---", flush=True)
        trainer.train_arms_trial(position="TBG", layer=22, k=cfg.k_soft_tokens,
                                 arms=["z_resp"], trial_seed=s, save_dir=CKPT_DIR)
    return sorted(glob.glob(os.path.join(CKPT_DIR, "z_resp_seed*.pt")))


def evaluate(paths, source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat",
             dataset="trivia_qa", num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15):
    # fit Procrustes TBG:22, SLT:15 (Mistral->Llama-2) on the n2000 1440 train
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, t_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    tr, va, te = splits(len(s_ids))
    mT, lT, WT = fit_align_at(sh["TBG"], th["TBG"], tbg, tr)
    mS, lS, WS = fit_align_at(sh["SLT"], th["SLT"], slt, tr)

    base = _cfg_from_meta(read_meta(paths[0]))
    eval_cfg = dataclasses.replace(base, stage1_model_name=source, stage1_dataset=dataset,
                                   stage1_num_samples=fresh_num_samples, ood_dataset=None, smoke=False)
    data = Stage2Data(eval_cfg)
    raw_TBG = data.hidden["TBG"][tbg].clone().numpy()
    raw_SLT = data.hidden["SLT"][slt].clone().numpy()

    raw = run_proxy(eval_cfg, data, paths)["z_resp"]                        # raw states
    data.hidden["TBG"][tbg] = torch.from_numpy(align(raw_TBG, mT, lT, WT)).float()
    data.hidden["SLT"][slt] = torch.from_numpy(align(raw_SLT, mS, lS, WS)).float()
    aln = run_proxy(eval_cfg, data, paths)["z_resp"]                        # aligned states
    return raw, aln


def run():
    paths = train_zresp()
    print(f"\ntrained {len(paths)} z_resp checkpoints; evaluating on aligned Mistral...\n", flush=True)
    raw, aln = evaluate(paths)
    raw, aln = np.array(raw), np.array(aln)
    print("\n" + "=" * 72)
    print("E27c  z_resp arm (hidden + response, NO question)  Mistral->Llama-2, fresh n1000")
    print("=" * 72)
    print(f"  z_resp  RAW      : {raw.mean():+.3f} ± {raw.std():.3f}")
    print(f"  z_resp  ALIGNED  : {aln.mean():+.3f} ± {aln.std():.3f}   (min {aln.min():+.3f})")
    print("  " + "-" * 68)
    print("  reference (aligned, fresh n1000):")
    for k, v in REFNUMS.items():
        print(f"    {k:16s} {v:+.3f}")
    print("=" * 72 + "\n")
    out = "amortized_ue/procrustes_e27c_zresp_arm.json"
    with open(out, "w") as f:
        json.dump({"z_resp_raw_mean": float(raw.mean()), "z_resp_raw_std": float(raw.std()),
                   "z_resp_aligned_mean": float(aln.mean()), "z_resp_aligned_std": float(aln.std()),
                   "z_resp_aligned_seeds": aln.tolist(), "reference_aligned": REFNUMS}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
