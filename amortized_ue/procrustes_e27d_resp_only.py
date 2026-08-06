"""E27d — train a resp_only arm (RESPONSE text only, NO question, NO z) and test the hypothesis
that dropping the question helps the text pathway.

"q hurts" held in the z-arms (question redundant with z). E27d checks the text-only regime, where
there is no z to make the question redundant -- so the question may actually help. Trains resp_only
(reference config, 5 seeds), evaluates on Mistral fresh n1000 (text -> model-agnostic, no alignment),
and compares to q_resp_only (0.531) and q_only (0.474). Also checks the ensemble ridge-z + resp_only
vs the current best (ridge-z + q_resp_only = 0.608). Additive. GPU. Run in `amortized_stage2`.
"""
from __future__ import annotations

import os
import glob
import json
import dataclasses

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.procrustes_e26_decomposition import boot_diff
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"
CKPT_DIR = "amortized_ue/stage2/runs/E27_resp_only_arm/checkpoints"


def train_resp_only():
    cfg = _cfg_from_meta(read_meta(os.path.join(REF, "z_seed0.pt")))
    cfg = dataclasses.replace(cfg, run_name="E27_resp_only_arm", arms=("resp_only",), save_checkpoints=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = Stage2Data(cfg)                                        # Llama-2 n2000
    trainer = Trainer(cfg, data)
    for s in cfg.arm_trial_seeds:
        print(f"--- training resp_only trial_seed={s} ---", flush=True)
        trainer.train_arms_trial(position="TBG", layer=22, k=cfg.k_soft_tokens,
                                 arms=["resp_only"], trial_seed=s, save_dir=CKPT_DIR)
    return sorted(glob.glob(os.path.join(CKPT_DIR, "resp_only_seed*.pt")))


def arm_preds(ckpt_dir, arm, model_name, dataset, num_samples):
    """Per-id predictions from `arm` (avg over seeds) on `model_name` (text arm: no alignment)."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, f"{arm}_seed*.pt")))
    from amortized_ue.stage2.checkpoint import load_checkpoint
    base = _cfg_from_meta(read_meta(paths[0]))
    cfg = dataclasses.replace(base, stage1_model_name=model_name, stage1_dataset=dataset,
                              stage1_num_samples=num_samples, ood_dataset=None, smoke=False)
    data = Stage2Data(cfg)
    rows = data.split_indices("all"); ids = [data.ids[r] for r in rows]
    model, trainer, acc = None, None, np.zeros(len(rows))
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(cfg, data, model=model)
        trainer.data = data; trainer.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), cfg.batch_size):
                r = rows[i:i + cfg.batch_size]
                preds.append(trainer._forward_batch(r, meta["position"], meta["layer"], arm, data=data).float().cpu())
        acc += transform.decode(torch.cat(preds)).numpy()
    return dict(zip(ids, acc / len(paths)))


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15):
    paths = train_resp_only()
    print(f"\ntrained {len(paths)} resp_only checkpoints; evaluating...\n", flush=True)

    # resp_only text predictions (model-agnostic; no alignment)
    rn = arm_preds(CKPT_DIR, "resp_only", source, dataset, num_samples)
    rf = arm_preds(CKPT_DIR, "resp_only", source, dataset, fresh_num_samples)

    # aligned-z ridge (for the ensemble)
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, _ = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    tr, va, te = splits(len(s_ids))

    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = th[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, th[pos][L][tr] - l); return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R, scR, _, _ = fit_probe(np.concatenate([th["TBG"][tbg], th["SLT"][slt]], 1), t_y, tr, va)

    def zpred(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R.predict(scR.transform(a))
    z_n2000 = zpred(sh["TBG"][tbg], sh["SLT"][slt])
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG", "SLT"])
    z_fresh = zpred(fsh["TBG"][tbg], fsh["SLT"][slt])

    r_n2000 = np.array([rn[i] for i in s_ids]); r_fresh = np.array([rf[i] for i in fs_ids])
    y_f = fs_y
    sp_resp = rho(r_fresh, y_f)

    meta = Ridge(alpha=1.0).fit(np.column_stack([z_n2000[tr], r_n2000[tr]]), s_y[tr])
    ens = meta.predict(np.column_stack([z_fresh, r_fresh]))
    sp_ens = rho(ens, y_f)
    bd = boot_diff(ens, z_fresh, y_f)

    print("\n" + "=" * 70)
    print("E27d  resp_only (response text, NO question)  Mistral->Llama-2, fresh n1000")
    print("=" * 70)
    print(f"  resp_only (NEW)          : {sp_resp:+.3f}")
    print(f"  q_resp_only (q+response) : +0.531   <- does dropping q help?")
    print(f"  q_only (question only)   : +0.474")
    print("  " + "-" * 66)
    print(f"  ensemble ridge-z + resp_only    : {sp_ens:+.3f}")
    print(f"  ensemble ridge-z + q_resp_only  : +0.608   (current best)")
    print(f"  (ensemble - ridge-z alone: {bd[0]:+.3f} [{bd[1]:+.3f}, {bd[2]:+.3f}] P={bd[3]:.2f})")
    print("=" * 70 + "\n")
    out = "amortized_ue/procrustes_e27d_resp_only.json"
    with open(out, "w") as f:
        json.dump({"resp_only_spearman": sp_resp, "q_resp_only": 0.531, "q_only": 0.474,
                   "ensemble_ridgez_resp_only": sp_ens, "ensemble_ridgez_q_resp_only": 0.608,
                   "ensemble_minus_ridgez": {"mean": bd[0], "lo95": bd[1], "hi95": bd[2], "P": bd[3]}}, f, indent=2)
    print(f"wrote {out}")


if __name__ == "__main__":
    run()
