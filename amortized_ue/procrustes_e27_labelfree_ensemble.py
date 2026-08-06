"""E27 correction — LABEL-FREE ensemble of aligned-z + q_resp_only (no Mistral SE labels used to
combine), vs the label-FITTED ridge combiner (which DID use Mistral labels) and the supervised baseline.

The E27 ridge combiner was fit on Mistral SE labels (s_y[tr]) -> the 0.608 ensemble is NOT label-free.
This recomputes honest label-free combinations:
  - raw average           : 0.5*(z + q_resp)                         (both already ~SE scale)
  - standardized average  : 0.5*(zscore(z) + zscore(q_resp))         (per-predictor mean/std from TRAIN
                            predictions -- uses predictions only, NO labels)
and reports them next to: aligned-z alone, q_resp alone, the label-FITTED ridge ensemble (uses labels),
and the Mistral supervised ridge baseline. Same binarisation for all. GPU (q_resp). `amortized_stage2`.
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
from sklearn.metrics import roc_auc_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.stage2.data import best_split, binarize_entropy, Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta, load_checkpoint

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"


def arm_preds(arm, model_name, dataset, num_samples):
    paths = sorted(glob.glob(os.path.join(REF, f"{arm}_seed*.pt")))
    cfg = dataclasses.replace(_cfg_from_meta(read_meta(paths[0])), stage1_model_name=model_name,
                              stage1_dataset=dataset, stage1_num_samples=num_samples, ood_dataset=None, smoke=False)
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
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, _ = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    tr, va, te = splits(len(s_ids))

    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = th[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, th[pos][L][tr] - l); return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R_a, sc_a, _, _ = fit_probe(np.concatenate([th["TBG"][tbg], th["SLT"][slt]], 1), t_y, tr, va)   # Llama-2 (label-free wrt Mistral)
    R_b, sc_b, _, _ = fit_probe(np.concatenate([sh["TBG"][tbg], sh["SLT"][slt]], 1), s_y, tr, va)   # Mistral SUPERVISED baseline

    def zpred(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R_a.predict(sc_a.transform(a))
    z_n2000 = zpred(sh["TBG"][tbg], sh["SLT"][slt])
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG", "SLT"])
    z_f = zpred(fsh["TBG"][tbg], fsh["SLT"][slt])
    base_f = R_b.predict(sc_b.transform(np.concatenate([fsh["TBG"][tbg], fsh["SLT"][slt]], 1)))

    qr_n2000 = arm_preds("q_resp_only", source, dataset, num_samples)
    qr_fresh = arm_preds("q_resp_only", source, dataset, fresh_num_samples)
    qr_tr = np.array([qr_n2000[i] for i in s_ids])[tr]
    qr_f = np.array([qr_fresh[i] for i in fs_ids])
    z_tr = z_n2000[tr]

    # LABEL-FREE combinations (no Mistral SE labels) ------------------------------
    avg_raw = 0.5 * (z_f + qr_f)
    zs = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-8)          # standardise by TRAIN predictions only
    avg_std = 0.5 * (zs(z_f, z_tr) + zs(qr_f, qr_tr))
    # LABEL-FITTED ridge combiner (USES Mistral labels s_y[tr]) -------------------
    meta = Ridge(alpha=1.0).fit(np.column_stack([z_tr, qr_tr]), s_y[tr])
    ens_fitted = meta.predict(np.column_stack([z_f, qr_f]))

    y = fs_y; thr = best_split(torch.tensor(y)); ybin = binarize_entropy(torch.tensor(y), thr).numpy(); valid = ybin >= 0
    def M(p): return rho(p, y), roc_auc_score(ybin[valid], p[valid])

    preds = [
        ("aligned-z ridge",                 z_f,        "label-free"),
        ("q_resp_only (text)",              qr_f,       "label-free"),
        ("avg (raw)  z + q_resp",           avg_raw,    "LABEL-FREE"),
        ("avg (standardized)  z + q_resp",  avg_std,    "LABEL-FREE"),
        ("ridge combiner  z + q_resp",      ens_fitted, "uses Mistral labels"),
        ("Mistral supervised ridge",        base_f,     "uses Mistral labels (BASELINE)"),
    ]
    print("\n" + "=" * 86)
    print(f"E27 LABEL-FREE vs LABEL-FITTED ensembles  {source} SE fresh n{fresh_num_samples} (Nvalid={int(valid.sum())})")
    print("=" * 86)
    print(f"  {'predictor':34s}{'Spearman':>10s}{'AUROC':>9s}   {'labels?':<28s}")
    out = {}
    for name, p, tag in preds:
        sp, au = M(p); out[name] = {"spearman": sp, "auroc": au, "labels": tag}
        print(f"  {name:34s}{sp:>+10.3f}{au:>9.3f}   {tag:<28s}")
    print("=" * 86 + "\n")
    with open("amortized_ue/procrustes_e27_labelfree_ensemble.json", "w") as f:
        json.dump({"threshold": float(thr), "n_valid": int(valid.sum()), "metrics": out}, f, indent=2)
    print("wrote amortized_ue/procrustes_e27_labelfree_ensemble.json")


if __name__ == "__main__":
    run()
