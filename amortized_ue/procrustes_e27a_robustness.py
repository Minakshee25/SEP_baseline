"""E27a-robust — confirm "aligned hidden state adds over question text" across seeds,
anchor resamples, directions, and eval sets.

E27a (single config, Mistral->Llama-2, fresh n1000) found the aligned hidden-state prediction
carries SE info beyond the q_only TEXT prediction (semi-partial +0.091, CI excludes 0). This
stress-tests that with three axes of variation:

  * MULTIPLE SEEDS   : each of the 5 q_only text-predictor seeds separately (not just their avg).
  * ALIGNMENT RESAMPLE: bootstrap the 1440 anchor pairs, refit W + Llama-2 ridge (n_boot times) ->
                        distribution of the semi-partial, so the aligned side isn't one fixed fit.
  * MULTIPLE EXPERIMENTS: both directions (Mistral->Llama-2, Llama-2->Mistral) x both eval sets
                          (N=200 n2000 test split, N=1000 E23 fresh).

Primary metric per cell: semi-partial = spearman(aligned, SOURCE SE | q_only TEXT removed from SE).
GPU for the q_only forward pass; CPU for Procrustes + stats. Run in `amortized_stage2`. Additive.
"""
from __future__ import annotations

import os
import glob
import json
import dataclasses
import argparse

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from sklearn.linear_model import Ridge

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.procrustes_alignment import best_tbg_layer
from amortized_ue.procrustes_e26_decomposition import semi_partial, boot_semi_partial, boot_diff
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta, _cfg_from_meta

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"


def qonly_per_seed(model_name, dataset, num_samples):
    """q_only (text) SE predictions on `model_name`: (per_seed {id:pred}[5], avg {id:pred})."""
    paths = sorted(glob.glob(os.path.join(REF, "q_only_seed*.pt")))
    base = _cfg_from_meta(read_meta(paths[0]))
    cfg = dataclasses.replace(base, stage1_model_name=model_name, stage1_dataset=dataset,
                              stage1_num_samples=num_samples, ood_dataset=None, smoke=False)
    data = Stage2Data(cfg)
    rows = data.split_indices("all"); ids = [data.ids[r] for r in rows]
    model, trainer, per_seed = None, None, []
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(cfg, data, model=model)
        trainer.data = data; trainer.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), cfg.batch_size):
                r = rows[i:i + cfg.batch_size]
                preds.append(trainer._forward_batch(r, meta["position"], meta["layer"], "q_only", data=data).float().cpu())
        per_seed.append(dict(zip(ids, transform.decode(torch.cat(preds)).numpy())))
    avg = {i: float(np.mean([ps[i] for ps in per_seed])) for i in ids}
    return per_seed, avg


def fit_map(S_La, T_La, t_y, tr, va, idx=None):
    """Fit Llama-2 ridge (target->SE) + Procrustes W (source->target) on train rows `tr` (or a
    resampled subset `idx` of tr). Returns a predictor: source_La -> aligned SE prediction."""
    rows = tr if idx is None else tr[idx]
    R_L, sc_L, _, _ = fit_probe(T_La, t_y, rows, va)      # ridge always alpha-picked on the fixed val
    m_mean, l_mean = S_La[rows].mean(0, keepdims=True), T_La[rows].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(S_La[rows] - m_mean, T_La[rows] - l_mean)
    return lambda X: R_L.predict(sc_L.transform((X - m_mean) @ W + l_mean))


def cell(source, target, eval_kind, qo_src, S, s_y, T, t_y, tr, va, te, L_a, Sf, fs_y, fs_ids, s_ids, n_boot):
    """One (direction, eval-set) cell: aligned vs text, semi-partial across seeds + anchor resamples."""
    aligned_full = fit_map(S[L_a], T[L_a], t_y, tr, va)
    if eval_kind == "fresh":
        src_eval, y = Sf[L_a], fs_y
        text_seed = [np.array([ps[i] for i in fs_ids]) for ps in qo_src["fresh"][0]]
        text_avg = np.array([qo_src["fresh"][1][i] for i in fs_ids])
    else:  # n2000 test split (N=200)
        src_eval, y = S[L_a][te], s_y[te]
        te_ids = [s_ids[i] for i in te]
        text_seed = [np.array([ps[i] for i in te_ids]) for ps in qo_src["n2000"][0]]
        text_avg = np.array([qo_src["n2000"][1][i] for i in te_ids])

    algn = aligned_full(src_eval)
    sp_aligned, sp_text = rho(algn, y), rho(text_avg, y)
    sp = semi_partial(algn, y, text_avg)
    bp = boot_semi_partial(algn, y, text_avg)                          # CI over eval questions
    per_seed = [semi_partial(algn, y, ts) for ts in text_seed]         # 5 seeds
    # anchor resample: refit W+ridge on bootstrapped train, recompute semi-partial vs the avg text
    rng = np.random.default_rng(0); boot_sp = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(tr), len(tr))
        boot_sp.append(semi_partial(fit_map(S[L_a], T[L_a], t_y, tr, va, idx=idx)(src_eval), y, text_avg))
    # ensemble gap vs text (ridge fit on train)
    algn_tr = aligned_full(S[L_a][tr])
    qo_tr = np.array([qo_src["n2000"][1][s_ids[i]] for i in tr])
    meta = Ridge(alpha=1.0).fit(np.column_stack([qo_tr, algn_tr]), s_y[tr])
    ens = meta.predict(np.column_stack([text_avg, algn]))
    bd = boot_diff(ens, text_avg, y)
    return {"source": source, "target": target, "eval": eval_kind, "n": len(y),
            "aligned_spearman": sp_aligned, "text_spearman": sp_text,
            "semi_partial_avg": sp, "semi_partial_ci": [bp[1], bp[2]], "semi_partial_P": bp[3],
            "per_seed_mean": float(np.mean(per_seed)), "per_seed_std": float(np.std(per_seed)),
            "per_seed_min": float(np.min(per_seed)), "per_seed_all_pos": bool(np.all(np.array(per_seed) > 0)),
            "anchor_resample_mean": float(np.mean(boot_sp)), "anchor_resample_std": float(np.std(boot_sp)),
            "anchor_resample_min": float(np.min(boot_sp)),
            "ensemble_minus_text": bd[0], "ensemble_minus_text_ci": [bd[1], bd[2]]}


def run(dataset="trivia_qa", num_samples=2000, fresh_num_samples=1000, n_boot=20,
        out="amortized_ue/procrustes_e27a_robustness.json"):
    L2, MI = "Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"
    # ---- load paired TBG (n2000 + fresh) for both models -----------------------
    def lm(m, n): h, y, ids = load_matrix(Stage1Config(model_name=m, dataset=dataset, num_samples=n), ["TBG"]); return h["TBG"], y, ids
    L2_2k, L2_2k_y, ids2k = lm(L2, num_samples)
    MI_2k, MI_2k_y, ids2k_b = lm(MI, num_samples); assert ids2k == ids2k_b
    L2_1k, L2_1k_y, ids1k = lm(L2, fresh_num_samples)
    MI_1k, MI_1k_y, ids1k_b = lm(MI, fresh_num_samples); assert ids1k == ids1k_b
    tr, va, te = splits(len(ids2k))

    # ---- q_only text preds on BOTH source models (the model whose SE we predict) -
    print("running q_only proxy on Mistral (n2000, n1000) and Llama-2 (n2000, n1000)...")
    qo = {MI: {"n2000": qonly_per_seed(MI, dataset, num_samples), "fresh": qonly_per_seed(MI, dataset, fresh_num_samples)},
          L2: {"n2000": qonly_per_seed(L2, dataset, num_samples), "fresh": qonly_per_seed(L2, dataset, fresh_num_samples)}}

    L_a_fwd, _ = best_tbg_layer(L2_2k, L2_2k_y, tr, va)   # target=Llama-2 ridge layer (forward)
    L_a_rev, _ = best_tbg_layer(MI_2k, MI_2k_y, tr, va)   # target=Mistral ridge layer (reverse)

    cells = []
    for ek in ("fresh", "test"):
        cells.append(cell(MI, L2, ek, qo[MI], MI_2k, MI_2k_y, L2_2k, L2_2k_y, tr, va, te, L_a_fwd,
                          MI_1k, MI_1k_y, ids1k, ids2k, n_boot))
    for ek in ("fresh", "test"):
        cells.append(cell(L2, MI, ek, qo[L2], L2_2k, L2_2k_y, MI_2k, MI_2k_y, tr, va, te, L_a_rev,
                          L2_1k, L2_1k_y, ids1k, ids2k, n_boot))

    # ---- report ---------------------------------------------------------------
    print("\n" + "=" * 104)
    print("E27a ROBUSTNESS: semi-partial spearman(aligned HIDDEN, SE | q_only TEXT removed) -- must be > 0")
    print("=" * 104)
    print(f"  {'direction':22s}{'eval':>7s}{'text':>8s}{'algn':>8s}{'semi-part [95% CI]':>24s}{'5-seed mean±std(min)':>24s}{'anchor-rs mean±std':>20s}")
    for c in cells:
        d = f"{c['source'].split('-')[0]}->{c['target'].split('-')[0]}"
        ci = f"{c['semi_partial_avg']:+.3f}[{c['semi_partial_ci'][0]:+.2f},{c['semi_partial_ci'][1]:+.2f}]"
        ss = f"{c['per_seed_mean']:+.3f}±{c['per_seed_std']:.3f}({c['per_seed_min']:+.2f})"
        rs = f"{c['anchor_resample_mean']:+.3f}±{c['anchor_resample_std']:.3f}"
        print(f"  {d:22s}{c['eval']:>7s}{c['text_spearman']:>8.3f}{c['aligned_spearman']:>8.3f}{ci:>24s}{ss:>24s}{rs:>20s}")
    allpos = all(c["semi_partial_ci"][0] > 0 for c in cells)
    allseed = all(c["per_seed_all_pos"] for c in cells)
    print("=" * 104)
    print(f"  semi-partial CI excludes 0 in ALL {len(cells)} cells: {allpos} | all 5 seeds positive in every cell: {allseed}")
    print("=" * 104 + "\n")

    with open(out, "w") as f:
        json.dump({"dataset": dataset, "n_boot": n_boot, "L_a_forward": int(L_a_fwd),
                   "L_a_reverse": int(L_a_rev), "cells": cells}, f, indent=2)
    print(f"wrote {out}")
    return cells


def _parse():
    p = argparse.ArgumentParser(description="E27a robustness battery (needs GPU for q_only).")
    p.add_argument("--n_boot", type=int, default=20)
    p.add_argument("--out", default="amortized_ue/procrustes_e27a_robustness.json")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(n_boot=a.n_boot, out=a.out)
