"""E27b — PROXY (all 5 arms) vs RIDGE, all on the SAME aligned hidden state [TBG:22, SLT:15].

E27a compared the aligned hidden state (ridge, TBG only) against the question TEXT. E27b makes the
comparison model-vs-model on IDENTICAL input: feed the frozen reference SLM proxy ALIGNED Mistral
hidden states at the proxy's OWN layers (TBG:22 + SLT:15, both Procrustes-mapped Mistral->Llama-2,
fit on the 1440 train pairs, NO SE labels), and evaluate every arm x 5 seeds. A ridge fit on the
IDENTICAL Llama-2 [TBG:22,SLT:15] and applied to the same aligned input is the fair linear baseline.

Two passes so we can see the alignment's effect per arm:
  * RAW      : proxy/ridge on Mistral's raw states (z-arms should be ~chance, as E20-E23).
  * ALIGNED  : proxy/ridge on Procrustes-aligned states.
Arms: z, z_q, z_q_resp (use the aligned z), q_only, q_resp_only (text; alignment-invariant).
Layers TBG:22/SLT:15 are validated ridge-optimal for Llama-2 (0.600/0.584). vs Mistral SE, fresh n1000.

GPU for the proxy; CPU for Procrustes + ridge. Run in `amortized_stage2`. Additive; injects aligned
states in-memory (data.hidden[pos][layer]) and touches nothing on disk.
"""
from __future__ import annotations

import os
import glob
import json
import argparse
import dataclasses
from collections import defaultdict

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, rho
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta, _cfg_from_meta

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"
ARMS = ["z", "z_q", "z_q_resp", "q_only", "q_resp_only"]


def fit_align_at(S_pos, T_pos, layer, tr):
    """Procrustes source->target at one (pos,layer): returns (m_mean, l_mean, W)."""
    m_mean = S_pos[layer][tr].mean(0, keepdims=True)
    l_mean = T_pos[layer][tr].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(S_pos[layer][tr] - m_mean, T_pos[layer][tr] - l_mean)
    return m_mean, l_mean, W


def align(x, m_mean, l_mean, W):
    return (x - m_mean) @ W + l_mean


def run_proxy(eval_cfg, eval_data, paths):
    """Return {arm: [spearman per seed]} on the current eval_data state (raw or injected)."""
    model, trainer, by_arm = None, None, defaultdict(list)
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(eval_cfg, eval_data, model=model)
        trainer.data = eval_data
        trainer.model.eval()
        with torch.no_grad():
            m = trainer.evaluate_on(meta["position"], meta["layer"], meta["arm"],
                                    eval_data, transform, split="all")
        by_arm[meta["arm"]].append(float(m["spearman"]))
    return by_arm


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15,
        out="amortized_ue/procrustes_e27b_proxy_vs_ridge.json"):
    # ---- CPU: fit Procrustes (TBG:22, SLT:15) + a ridge on the SAME stacked input --------
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, t_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    assert s_ids == t_ids
    tr, va, te = splits(len(s_ids))
    mT, lT, WT = fit_align_at(sh["TBG"], th["TBG"], tbg, tr)          # Mistral TBG:22 -> Llama-2
    mS, lS, WS = fit_align_at(sh["SLT"], th["SLT"], slt, tr)          # Mistral SLT:15 -> Llama-2
    # ridge on Llama-2 [TBG:22 || SLT:15] -> Llama-2 SE (mirrors how the proxy was trained)
    Xt = np.concatenate([th["TBG"][tbg], th["SLT"][slt]], axis=1)
    R, scR, aR, _ = fit_probe(Xt, t_y, tr, va)
    print(f"fit: Procrustes TBG:{tbg}+SLT:{slt}, ridge on Llama-2 stacked (alpha={aR})")

    # ---- build the fresh-n1000 eval data (proxy reads data.hidden[pos][layer]) -----------
    base = _cfg_from_meta(read_meta(sorted(glob.glob(os.path.join(REF, '*.pt')))[0]))
    eval_cfg = dataclasses.replace(base, stage1_model_name=source, stage1_dataset=dataset,
                                   stage1_num_samples=fresh_num_samples, ood_dataset=None, smoke=False)
    data = Stage2Data(eval_cfg)
    y = data.labels_raw.numpy()                                       # Mistral fresh SE
    raw_TBG = data.hidden["TBG"][tbg].clone().numpy()                 # Mistral fresh raw states
    raw_SLT = data.hidden["SLT"][slt].clone().numpy()
    aln_TBG = align(raw_TBG, mT, lT, WT)
    aln_SLT = align(raw_SLT, mS, lS, WS)
    paths = sorted(glob.glob(os.path.join(REF, "*.pt")))

    # ---- ridge raw vs aligned (same stacked input as the proxy) --------------------------
    ridge_raw = rho(R.predict(scR.transform(np.concatenate([raw_TBG, raw_SLT], 1))), y)
    ridge_aln = rho(R.predict(scR.transform(np.concatenate([aln_TBG, aln_SLT], 1))), y)

    # ---- PROXY: raw pass, then inject aligned states and re-run --------------------------
    print("proxy RAW pass...")
    raw_arm = run_proxy(eval_cfg, data, paths)
    data.hidden["TBG"][tbg] = torch.from_numpy(aln_TBG).float()       # inject aligned z
    data.hidden["SLT"][slt] = torch.from_numpy(aln_SLT).float()
    print("proxy ALIGNED pass...")
    aln_arm = run_proxy(eval_cfg, data, paths)

    def stat(v):
        v = np.array(v); return float(v.mean()), float(v.std()), float(v.min())

    # ---- report --------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"E27b PROXY vs RIDGE on aligned [TBG:{tbg},SLT:{slt}]  {source}->{target}, vs {source} SE (fresh n{fresh_num_samples})")
    print("=" * 78)
    print(f"  {'predictor':16s}{'RAW (mean±std)':>22s}{'ALIGNED (mean±std)':>24s}")
    rows = {}
    for arm in ARMS:
        rm = stat(raw_arm[arm]); am = stat(aln_arm[arm])
        rows[arm] = {"raw": rm, "aligned": am}
        tag = "(text; align-invariant)" if arm in ("q_only", "q_resp_only") else ""
        print(f"  {arm:16s}{rm[0]:+.3f} ± {rm[1]:.3f}      {am[0]:+.3f} ± {am[1]:.3f}   {tag}")
    print(f"  {'ridge [TBG+SLT]':16s}{ridge_raw:+.3f}            {ridge_aln:+.3f}")
    print("=" * 78)
    best_arm = max(ARMS, key=lambda a: rows[a]["aligned"][0])
    print(f"  best proxy arm (aligned): {best_arm} = {rows[best_arm]['aligned'][0]:+.3f}   | ridge = {ridge_aln:+.3f}")
    print("=" * 78 + "\n")

    result = {"source": source, "target": target, "dataset": dataset, "fresh_num_samples": fresh_num_samples,
              "tbg_layer": tbg, "slt_layer": slt, "ridge_raw": ridge_raw, "ridge_aligned": ridge_aln,
              "arms": {a: {"raw_mean": rows[a]["raw"][0], "raw_std": rows[a]["raw"][1],
                           "aligned_mean": rows[a]["aligned"][0], "aligned_std": rows[a]["aligned"][1],
                           "aligned_min": rows[a]["aligned"][2],
                           "raw_seeds": raw_arm[a], "aligned_seeds": aln_arm[a]} for a in ARMS},
              "best_proxy_arm_aligned": best_arm}
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


def _parse():
    p = argparse.ArgumentParser(description="E27b proxy(all arms) vs ridge on aligned hidden state (GPU).")
    p.add_argument("--source", default="Mistral-7B-Instruct-v0.2")
    p.add_argument("--target", default="Llama-2-7b-chat")
    p.add_argument("--fresh_num_samples", type=int, default=1000)
    p.add_argument("--tbg", type=int, default=22)
    p.add_argument("--slt", type=int, default=15)
    p.add_argument("--out", default="amortized_ue/procrustes_e27b_proxy_vs_ridge.json")
    return p.parse_args()


if __name__ == "__main__":
    a = _parse()
    run(a.source, a.target, fresh_num_samples=a.fresh_num_samples, tbg=a.tbg, slt=a.slt, out=a.out)
