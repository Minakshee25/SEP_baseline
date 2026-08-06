"""E27 — AUROC (+ Spearman) for the key predictors on fresh n1000, incl. the supervised baseline.

Same convention as the pipeline: binarise Mistral's SE with best_split over the eval rows, drop ties,
AUROC = roc_auc_score. All predictors use the SAME binary labels so AUROC is comparable. GPU for the
proxy text arms; CPU for ridges. Run in `amortized_stage2`. Additive.
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
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta, load_checkpoint

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"


def arm_preds(arm, model_name, dataset, num_samples):
    paths = sorted(glob.glob(os.path.join(REF, f"{arm}_seed*.pt")))
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


def metrics(pred, y_raw, ybin, valid):
    return rho(pred, y_raw), roc_auc_score(ybin[valid], pred[valid])


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15):
    # ---- hidden states + Procrustes + ridges (CPU) --------------------------------------
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, _ = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    tr, va, te = splits(len(s_ids))

    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = th[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, th[pos][L][tr] - l); return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R_align, sc_a, _, _ = fit_probe(np.concatenate([th["TBG"][tbg], th["SLT"][slt]], 1), t_y, tr, va)     # Llama-2 ridge
    R_base, sc_b, _, _ = fit_probe(np.concatenate([sh["TBG"][tbg], sh["SLT"][slt]], 1), s_y, tr, va)      # Mistral SUPERVISED baseline

    def alignedz(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R_align.predict(sc_a.transform(a))
    def base_pred(TBG_L, SLT_L):
        return R_base.predict(sc_b.transform(np.concatenate([TBG_L, SLT_L], 1)))
    z_n2000 = alignedz(sh["TBG"][tbg], sh["SLT"][slt])
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG", "SLT"])
    z_fresh = alignedz(fsh["TBG"][tbg], fsh["SLT"][slt])
    base_fresh = base_pred(fsh["TBG"][tbg], fsh["SLT"][slt])

    # ---- proxy text arms (GPU) ----------------------------------------------------------
    qo_n2000 = arm_preds("q_only", source, dataset, num_samples); qo_fresh = arm_preds("q_only", source, dataset, fresh_num_samples)
    qr_n2000 = arm_preds("q_resp_only", source, dataset, num_samples); qr_fresh = arm_preds("q_resp_only", source, dataset, fresh_num_samples)
    qo_f = np.array([qo_fresh[i] for i in fs_ids]); qr_f = np.array([qr_fresh[i] for i in fs_ids])
    qr_tr = np.array([qr_n2000[i] for i in s_ids])[tr]

    # ---- ensemble (aligned-z + q_resp_only) fit on train --------------------------------
    meta = Ridge(alpha=1.0).fit(np.column_stack([z_n2000[tr], qr_tr]), s_y[tr])
    ens_fresh = meta.predict(np.column_stack([z_fresh, qr_f]))

    # ---- binarise Mistral fresh SE (best_split over eval rows) --------------------------
    y = fs_y
    thr = best_split(torch.tensor(y))
    ybin = binarize_entropy(torch.tensor(y), thr).numpy()
    valid = ybin >= 0
    print(f"binarisation: best_split={thr:.3f}  positives={int(ybin[valid].sum())}/{int(valid.sum())}")

    preds = {
        "q_only (text)": qo_f,
        "q_resp_only (text)": qr_f,
        "aligned-z ridge (label-free)": z_fresh,
        "ENSEMBLE z+q_resp (ridge combiner, USES Mistral labels)": ens_fresh,   # NOT label-free -- see procrustes_e27_labelfree_ensemble.py for the label-free average
        "Mistral supervised ridge (BASELINE)": base_fresh,
    }
    print("\n" + "=" * 72)
    print(f"E27 AUROC + Spearman  {source} SE, fresh n{fresh_num_samples} (N valid={int(valid.sum())})")
    print("=" * 72)
    print(f"  {'predictor':38s}{'Spearman':>11s}{'AUROC':>10s}")
    out = {}
    for name, p in preds.items():
        sp, au = metrics(p, y, ybin, valid)
        out[name] = {"spearman": sp, "auroc": au}
        print(f"  {name:38s}{sp:>+11.3f}{au:>10.3f}")
    print("=" * 72 + "\n")
    with open("amortized_ue/procrustes_e27_auroc.json", "w") as f:
        json.dump({"threshold": float(thr), "n_valid": int(valid.sum()), "metrics": out}, f, indent=2)
    print("wrote amortized_ue/procrustes_e27_auroc.json")


if __name__ == "__main__":
    run()
