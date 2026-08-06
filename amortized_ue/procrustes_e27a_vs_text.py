"""E27a — does the aligned HIDDEN STATE carry SE info BEYOND the question TEXT?

E25/E26 controlled the aligned Procrustes transfer against a hidden-state difficulty reader
(Llama-2's own states). E27a controls it against the model-agnostic TEXT signal: the `q_only`
arm of the reference 3B proxy, which predicts SE from the question text alone. If, after
removing what the question text explains, the aligned hidden-state prediction STILL adds
signal about Mistral's SE, then hidden states + alignment uncover uncertainty the question
alone misses.

On the E23 fresh n1000 batch, vs Mistral SE:
  - control = q_only text prediction (reference proxy, avg over 5 seeds)   [needs NO target model]
  - aligned = Mistral TBG -> Procrustes W -> Llama-2 frozen ridge          [needs target hidden states]
  (1) semi-partial: spearman(aligned, Mistral SE | q_only removed from SE) + bootstrap 95% CI
  (2) ensemble: q_only + aligned (avg + 2-input ridge on train) vs q_only alone

GPU for the q_only text forward pass (3B proxy); CPU for Procrustes + stats. Run in the
`amortized_stage2` env. Additive; reuses E24/E26 + eval_cross_llm helpers read-only.
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
from amortized_ue.procrustes_e26_decomposition import (
    semi_partial, partial_spearman, boot_semi_partial, boot_diff)
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta, _cfg_from_meta


def qonly_preds(ckpt_dir, target_model, dataset, num_samples):
    """Per-id q_only (text-only) SE predictions on `target_model`, averaged over the 5 seeds."""
    paths = sorted(glob.glob(os.path.join(ckpt_dir, "q_only_seed*.pt")))
    assert paths, f"no q_only_seed*.pt in {ckpt_dir}"
    base = _cfg_from_meta(read_meta(paths[0]))
    eval_cfg = dataclasses.replace(base, stage1_model_name=target_model, stage1_dataset=dataset,
                                   stage1_num_samples=num_samples, ood_dataset=None, smoke=False)
    data = Stage2Data(eval_cfg)
    rows = data.split_indices("all")
    ids = [data.ids[r] for r in rows]
    model, trainer, acc = None, None, np.zeros(len(rows))
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(eval_cfg, data, model=model)
        trainer.data = data
        trainer.model.eval()
        preds = []
        with torch.no_grad():                                # inference only -- no autograd graph
            for i in range(0, len(rows), eval_cfg.batch_size):
                r = rows[i:i + eval_cfg.batch_size]
                preds.append(trainer._forward_batch(r, meta["position"], meta["layer"], "q_only",
                                                     data=data).float().cpu())
        acc += transform.decode(torch.cat(preds)).numpy()
    return dict(zip(ids, acc / len(paths)))


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000,
        ref_ckpt="amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints",
        out="amortized_ue/procrustes_e27a_vs_text.json"):
    # ---- CPU: fit Llama-2 ridge + Procrustes W on the n2000 1440-train pairs (no SE labels in W)
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG"])
    th, t_y, t_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG"])
    assert s_ids == t_ids
    S, T = sh["TBG"], th["TBG"]
    tr, va, te = splits(len(s_ids))
    L_a, _ = best_tbg_layer(T, t_y, tr, va)
    R_L, sc_L, _, _ = fit_probe(T[L_a], t_y, tr, va)
    m_mean, l_mean = S[L_a][tr].mean(0, keepdims=True), T[L_a][tr].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(S[L_a][tr] - m_mean, T[L_a][tr] - l_mean)
    aligned_pred = lambda src_La: R_L.predict(sc_L.transform((src_La - m_mean) @ W + l_mean))
    print(f"fit on n{num_samples} train={len(tr)}  target ridge TBG L_a={L_a}")

    # ---- GPU: q_only TEXT predictions on the SOURCE (Mistral) for n2000 (train ensemble) + fresh
    qo_n2000 = qonly_preds(ref_ckpt, source, dataset, num_samples)
    qo_fresh = qonly_preds(ref_ckpt, source, dataset, fresh_num_samples)
    print(f"q_only text preds: n{num_samples}={len(qo_n2000)}  fresh n{fresh_num_samples}={len(qo_fresh)}")

    # ---- assemble train arrays (aligned + q_only + SE), ordered by sorted ids -------
    algn_n2000 = aligned_pred(S[L_a])                        # ordered like s_ids
    qo_n2000_arr = np.array([qo_n2000[i] for i in s_ids])
    algn_tr, qo_tr, y_tr = algn_n2000[tr], qo_n2000_arr[tr], s_y[tr]

    # ---- fresh n1000 eval arrays ----------------------------------------------------
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG"])
    Sf = fsh["TBG"]
    algn_f = aligned_pred(Sf[L_a])
    qo_f = np.array([qo_fresh[i] for i in fs_ids])
    y_f = fs_y
    n = len(y_f)
    sp_qonly, sp_aligned = rho(qo_f, y_f), rho(algn_f, y_f)
    print(f"fresh n{fresh_num_samples}: q_only(text)={sp_qonly:+.3f}  aligned(hidden)={sp_aligned:+.3f}")

    # ---- (1) semi-partial: does aligned add over TEXT? ------------------------------
    sp_partial = semi_partial(algn_f, y_f, qo_f)
    bp = boot_semi_partial(algn_f, y_f, qo_f)
    sp_partial_full = partial_spearman(algn_f, y_f, qo_f)

    # ---- (2) ensemble: text + aligned vs text alone --------------------------------
    ens_avg = 0.5 * (qo_f + algn_f)
    sp_ens_avg = rho(ens_avg, y_f)
    meta = Ridge(alpha=1.0).fit(np.column_stack([qo_tr, algn_tr]), y_tr)
    ens_ridge = meta.predict(np.column_stack([qo_f, algn_f]))
    sp_ens_ridge = rho(ens_ridge, y_f)
    bd_avg = boot_diff(ens_avg, qo_f, y_f)
    bd_ridge = boot_diff(ens_ridge, qo_f, y_f)

    # ---- report --------------------------------------------------------------------
    print("\n" + "=" * 80)
    print(f"E27a  aligned HIDDEN vs QUESTION TEXT: {source} -> {target}, vs {source} SE, fresh n{fresh_num_samples} (N={n})")
    print("=" * 80)
    print("  (1) SEMI-PARTIAL  spearman(aligned, SE | q_only TEXT removed from SE):")
    sep = "ABOVE ZERO (hidden adds over text)" if bp[1] > 0 else "overlaps 0 (redundant with text)"
    print(f"        semi-partial = {sp_partial:+.3f}   95% CI [{bp[1]:+.3f}, {bp[2]:+.3f}]  P(>0)={bp[3]:.2f}  -> {sep}")
    print(f"        (robustness: symmetric rank-based partial Spearman = {sp_partial_full:+.3f})")
    print("  " + "-" * 76)
    print("  (2) ENSEMBLE vs text alone (Spearman vs Mistral SE, fresh n1000):")
    print(f"        q_only TEXT alone        : {sp_qonly:+.3f}")
    print(f"        aligned HIDDEN alone     : {sp_aligned:+.3f}")
    print(f"        ensemble (avg)           : {sp_ens_avg:+.3f}   (avg - text)   {bd_avg[0]:+.3f} [{bd_avg[1]:+.3f}, {bd_avg[2]:+.3f}] P(>0)={bd_avg[3]:.2f}")
    print(f"        ensemble (2-input ridge) : {sp_ens_ridge:+.3f}   (ridge - text) {bd_ridge[0]:+.3f} [{bd_ridge[1]:+.3f}, {bd_ridge[2]:+.3f}] P(>0)={bd_ridge[3]:.2f}")
    print(f"        ridge meta-weights [text, aligned] = [{meta.coef_[0]:+.3f}, {meta.coef_[1]:+.3f}]")
    print("=" * 80 + "\n")

    result = {
        "source": source, "target": target, "dataset": dataset, "fit_num_samples": num_samples,
        "fresh_num_samples": fresh_num_samples, "n_eval": n, "L_a_target_ridge": int(L_a),
        "qonly_text_spearman": sp_qonly, "aligned_hidden_spearman": sp_aligned,
        "semi_partial_over_text": {"value": sp_partial, "lo95": bp[1], "hi95": bp[2],
                                   "frac_positive": bp[3], "symmetric_partial_spearman": sp_partial_full},
        "ensemble": {"avg_spearman": sp_ens_avg, "ridge_spearman": sp_ens_ridge,
                     "avg_minus_text": {"mean": bd_avg[0], "lo95": bd_avg[1], "hi95": bd_avg[2], "frac_positive": bd_avg[3]},
                     "ridge_minus_text": {"mean": bd_ridge[0], "lo95": bd_ridge[1], "hi95": bd_ridge[2], "frac_positive": bd_ridge[3]},
                     "ridge_meta_weights": {"text": float(meta.coef_[0]), "aligned": float(meta.coef_[1])}},
    }
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


def _parse():
    p = argparse.ArgumentParser(description="E27a: aligned hidden state vs question text (needs GPU for q_only).")
    p.add_argument("--source", default="Mistral-7B-Instruct-v0.2")
    p.add_argument("--target", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--fresh_num_samples", type=int, default=1000)
    p.add_argument("--out", default="amortized_ue/procrustes_e27a_vs_text.json")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.source, a.target, a.dataset, a.num_samples, a.fresh_num_samples, out=a.out)
