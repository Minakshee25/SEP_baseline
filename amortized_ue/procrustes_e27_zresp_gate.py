"""E27 cheap gate — is the RESPONSE text complementary to the aligned hidden state (z)?

Decides whether training a `z_resp` proxy arm is worth it, WITHOUT retraining. Combines the
aligned-z RIDGE prediction (Mistral->Llama-2 Procrustes at TBG:22+SLT:15) with the q_resp_only
proxy prediction (response text), on the fresh n1000 vs Mistral SE:
  - semi-partial(response, SE | aligned-z removed)  -> does the response add OVER z?
  - semi-partial(aligned-z, SE | response removed)  -> does z add over the response?
  - ensemble (2-input ridge on train) vs aligned-z alone and vs response alone.
If the response adds significant signal over aligned-z, a z_resp arm is worth training; else skip.
GPU for q_resp_only; CPU for the rest. Run in `amortized_stage2`. Additive.
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
from amortized_ue.procrustes_e26_decomposition import semi_partial, boot_semi_partial, boot_diff
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.train import Trainer
from amortized_ue.stage2.checkpoint import load_checkpoint, read_meta, _cfg_from_meta

REF = "amortized_ue/stage2/runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints"


def arm_preds(arm, model_name, dataset, num_samples):
    """Per-id SE predictions from `arm` (avg over its 5 seeds) on `model_name`."""
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


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15,
        out="amortized_ue/procrustes_e27_zresp_gate.json"):
    # ---- aligned-z RIDGE (Mistral->Llama-2 at TBG:22+SLT:15), preds on n2000 + fresh -----
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, t_ids = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    assert s_ids == t_ids
    tr, va, te = splits(len(s_ids))

    def fit_at(pos, L):
        m = sh[pos][L][tr].mean(0, keepdims=True); l = th[pos][L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(sh[pos][L][tr] - m, th[pos][L][tr] - l)
        return m, l, W
    mT, lT, WT = fit_at("TBG", tbg); mS, lS, WS = fit_at("SLT", slt)
    R, scR, aR, _ = fit_probe(np.concatenate([th["TBG"][tbg], th["SLT"][slt]], 1), t_y, tr, va)

    def aligned_z(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R.predict(scR.transform(a))
    z_n2000 = aligned_z(sh["TBG"][tbg], sh["SLT"][slt])                    # ordered like s_ids
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG", "SLT"])
    z_fresh = aligned_z(fsh["TBG"][tbg], fsh["SLT"][slt])

    # ---- q_resp_only (RESPONSE text) proxy preds on n2000 + fresh (GPU) ------------------
    resp_n2000 = arm_preds("q_resp_only", source, dataset, num_samples)
    resp_fresh = arm_preds("q_resp_only", source, dataset, fresh_num_samples)
    r_n2000 = np.array([resp_n2000[i] for i in s_ids])
    r_fresh = np.array([resp_fresh[i] for i in fs_ids])

    y_f = fs_y
    sp_z, sp_r = rho(z_fresh, y_f), rho(r_fresh, y_f)

    # ---- partials both ways --------------------------------------------------------------
    resp_over_z = semi_partial(r_fresh, y_f, z_fresh); b_roz = boot_semi_partial(r_fresh, y_f, z_fresh)
    z_over_resp = semi_partial(z_fresh, y_f, r_fresh); b_zor = boot_semi_partial(z_fresh, y_f, r_fresh)

    # ---- ensemble (2-input ridge on train) ----------------------------------------------
    meta = Ridge(alpha=1.0).fit(np.column_stack([z_n2000[tr], r_n2000[tr]]), s_y[tr])
    ens = meta.predict(np.column_stack([z_fresh, r_fresh]))
    sp_ens = rho(ens, y_f)
    bd_z = boot_diff(ens, z_fresh, y_f)     # ensemble - aligned-z
    bd_r = boot_diff(ens, r_fresh, y_f)     # ensemble - response

    print("\n" + "=" * 78)
    print(f"E27 GATE: is RESPONSE complementary to aligned-z?  {source}->{target}, vs {source} SE, fresh n{fresh_num_samples}")
    print("=" * 78)
    print(f"  aligned-z (ridge) alone : {sp_z:+.3f}")
    print(f"  response (q_resp_only)  : {sp_r:+.3f}")
    print(f"  ensemble (z + response) : {sp_ens:+.3f}")
    print("  " + "-" * 74)
    v1 = "ADDS (train z_resp)" if b_roz[1] > 0 else "redundant (skip z_resp)"
    print(f"  response OVER aligned-z  (semi-partial): {resp_over_z:+.3f}  95% CI [{b_roz[1]:+.3f}, {b_roz[2]:+.3f}]  P={b_roz[3]:.2f} -> {v1}")
    print(f"  aligned-z OVER response  (semi-partial): {z_over_resp:+.3f}  95% CI [{b_zor[1]:+.3f}, {b_zor[2]:+.3f}]  P={b_zor[3]:.2f}")
    print(f"  ensemble - aligned-z : {bd_z[0]:+.3f} [{bd_z[1]:+.3f}, {bd_z[2]:+.3f}] P={bd_z[3]:.2f}")
    print(f"  ensemble - response  : {bd_r[0]:+.3f} [{bd_r[1]:+.3f}, {bd_r[2]:+.3f}] P={bd_r[3]:.2f}")
    print("=" * 78 + "\n")

    result = {"source": source, "target": target, "fresh_num_samples": fresh_num_samples,
              "tbg": tbg, "slt": slt, "aligned_z_spearman": sp_z, "response_spearman": sp_r,
              "ensemble_spearman": sp_ens,
              "response_over_z": {"value": resp_over_z, "lo95": b_roz[1], "hi95": b_roz[2], "P": b_roz[3]},
              "z_over_response": {"value": z_over_resp, "lo95": b_zor[1], "hi95": b_zor[2], "P": b_zor[3]},
              "ensemble_minus_z": {"mean": bd_z[0], "lo95": bd_z[1], "hi95": bd_z[2], "P": bd_z[3]},
              "ensemble_minus_response": {"mean": bd_r[0], "lo95": bd_r[1], "hi95": bd_r[2], "P": bd_r[3]}}
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    run()
