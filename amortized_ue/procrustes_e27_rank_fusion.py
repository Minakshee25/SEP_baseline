"""E27 — rank-fusion variant of the label-free ensemble.

Modeled on procrustes_e27_labelfree_ensemble.py, reusing the same predictors (aligned-z ridge +
q_resp_only, Mistral->Llama-2). Rank fusion: map each predictor's outputs to normalized ranks via its
empirical CDF fit on TRAIN predictions only (no labels), average the two, score vs Mistral SE. Reported
next to the raw average and standardized average (all label-free) plus the individual predictors, the
label-fitted ridge combiner, and the Mistral supervised ridge baseline. Same best_split binarisation.

Also runs on the squad OOD eval to test shape-robustness under distribution shift (the ID-fit CDF /
standardiser applied to shifted predictions). GPU (q_resp); CPU otherwise. `amortized_stage2`. Additive.
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


def arm_preds(arm, model_name, dataset, num_samples, ckpt_dir=None, data_dir=None, run_name=None):
    """ckpt_dir defaults to REF (the Llama-2-trained REFERENCE proxy); pass e.g. the E22 Mistral
    proxy's checkpoint dir to score a proxy trained on a DIFFERENT source model (E42). data_dir
    overrides Stage1Config.output_dir (e.g. a node-local /data2 copy) for the target's records.
    run_name overrides Stage1Config.run_name (e.g. a "_nothink" build dir, E55/E65)."""
    # "*" prefix tolerates a filename prefix before "{arm}_seed" (e.g. deploy_checkpoints'
    # "deploy_{arm}_seedN.pt" vs REF/E22's bare "{arm}_seedN.pt") without a false match on a
    # longer arm name (e.g. "z_q_seed" never matches the "z_seed" pattern's substring).
    paths = sorted(glob.glob(os.path.join(ckpt_dir or REF, f"*{arm}_seed*.pt")))
    cfg = dataclasses.replace(_cfg_from_meta(read_meta(paths[0])), stage1_model_name=model_name,
                              stage1_dataset=dataset, stage1_num_samples=num_samples, ood_dataset=None, smoke=False,
                              **({"stage1_output_dir": data_dir} if data_dir else {}),
                              **({"stage1_run_name": run_name} if run_name else {}))
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


def ecdf(train_vals):
    """Empirical-CDF transform fit on train predictions only (no labels): value -> normalized rank."""
    s = np.sort(np.asarray(train_vals, dtype=float))
    return lambda x: np.searchsorted(s, np.asarray(x, dtype=float), side="right") / len(s)


def boot_delta(pa, pb, y_raw, ybin, valid, B=1000, seed=0):
    """Paired bootstrap of metric(pa)-metric(pb) over eval rows (E25/E26 convention): Spearman +
    AUROC, each returned as (mean, lo95, hi95)."""
    rng = np.random.default_rng(seed); n = len(y_raw)
    ds, da = [], []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        ds.append(rho(pa[idx], y_raw[idx]) - rho(pb[idx], y_raw[idx]))
        vi = idx[valid[idx]]                                                      # resampled valid rows
        if len(np.unique(ybin[vi])) == 2:
            da.append(roc_auc_score(ybin[vi], pa[vi]) - roc_auc_score(ybin[vi], pb[vi]))
    ci = lambda d: (float(np.mean(d)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)))
    return {"spearman": ci(ds), "auroc": ci(da)}


def score_block(name, z_eval, qr_eval, y_eval, z_tr, qr_tr, s_y_tr, out, floor_eval=None):
    """All combinations on one eval set, using ID-train normalizers (z_tr/qr_tr) fit label-free.
    floor_eval (optional): raw (unaligned) source states through the target ridge -- the NO-W floor."""
    thr = best_split(torch.tensor(y_eval)); yb = binarize_entropy(torch.tensor(y_eval), thr).numpy(); v = yb >= 0
    def M(p): return rho(p, y_eval), roc_auc_score(yb[v], p[v])
    zs = lambda x, ref: (x - ref.mean()) / (ref.std() + 1e-8)                     # standardise by TRAIN preds
    cdf_z, cdf_r = ecdf(z_tr), ecdf(qr_tr)                                        # CDF fit on TRAIN preds (label-free)
    combos = {}
    if floor_eval is not None:                                                    # floor: raw z, NO Procrustes W
        combos["floor (raw z, NO W)"] = floor_eval
    combos.update({
        "aligned-z ridge": z_eval,
        "q_resp_only (text)": qr_eval,
        "avg (raw)": 0.5 * (z_eval + qr_eval),
        "avg (standardized)": 0.5 * (zs(z_eval, z_tr) + zs(qr_eval, qr_tr)),
        "RANK FUSION (empirical-CDF avg)": 0.5 * (cdf_z(z_eval) + cdf_r(qr_eval)),
    })
    if s_y_tr is not None:                                                        # label-fitted combiner (uses labels)
        meta = Ridge(alpha=1.0).fit(np.column_stack([z_tr, qr_tr]), s_y_tr)
        combos["ridge combiner (USES labels)"] = meta.predict(np.column_stack([z_eval, qr_eval]))
    print("\n" + "=" * 78)
    print(f"E27 RANK FUSION — {name}  (Nvalid={int(v.sum())}, best_split={thr:.3f})")
    print("=" * 78)
    print(f"  {'predictor / fusion':36s}{'Spearman':>10s}{'AUROC':>9s}")
    res = {}
    for k, p in combos.items():
        sp, au = M(p); res[k] = {"spearman": sp, "auroc": au}
        tag = "" if "USES labels" in k else "  (label-free)"
        print(f"  {k:36s}{sp:>+10.3f}{au:>9.3f}{tag}")
    # paired bootstrap: RANK FUSION - avg(standardized), Spearman + AUROC, 95% CI
    bd = boot_delta(combos["RANK FUSION (empirical-CDF avg)"], combos["avg (standardized)"], y_eval, yb, v)
    res["_bootstrap_rankfusion_minus_stdavg"] = {
        "spearman": {"mean": bd["spearman"][0], "lo95": bd["spearman"][1], "hi95": bd["spearman"][2]},
        "auroc": {"mean": bd["auroc"][0], "lo95": bd["auroc"][1], "hi95": bd["auroc"][2]}}
    print("  " + "-" * 74)
    print(f"  Δ(rank fusion − std-avg)  Spearman {bd['spearman'][0]:+.3f} [{bd['spearman'][1]:+.3f}, {bd['spearman'][2]:+.3f}]"
          f"   AUROC {bd['auroc'][0]:+.3f} [{bd['auroc'][1]:+.3f}, {bd['auroc'][2]:+.3f}]")
    beats = bd["spearman"][1] > 0 or bd["auroc"][1] > 0
    print(f"  -> rank fusion {'BEATS' if beats else 'TIES'} std-avg "
          f"(Spearman CI {'excludes' if bd['spearman'][1] > 0 else 'includes'} 0; "
          f"AUROC CI {'excludes' if bd['auroc'][1] > 0 else 'includes'} 0)")
    print("=" * 78)
    out[name] = res
    return res


def run(source="Mistral-7B-Instruct-v0.2", target="Llama-2-7b-chat", dataset="trivia_qa",
        num_samples=2000, fresh_num_samples=1000, tbg=22, slt=15, ood_dataset="squad", ood_num_samples=1000):
    # --- fit W + Llama-2 ridge on trivia n2000 train (ID); baseline ridge on Mistral ----------
    sh, s_y, s_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    th, t_y, _ = load_matrix(Stage1Config(model_name=target, dataset=dataset, num_samples=num_samples), ["TBG", "SLT"])
    tr, va, te = splits(len(s_ids))

    def fit_at(H_src, H_tgt, L):
        m = H_src[L][tr].mean(0, keepdims=True); l = H_tgt[L][tr].mean(0, keepdims=True)
        W, _ = orthogonal_procrustes(H_src[L][tr] - m, H_tgt[L][tr] - l); return m, l, W
    mT, lT, WT = fit_at(sh["TBG"], th["TBG"], tbg)
    mS, lS, WS = fit_at(sh["SLT"], th["SLT"], slt)
    R, sc, _, _ = fit_probe(np.concatenate([th["TBG"][tbg], th["SLT"][slt]], 1), t_y, tr, va)

    def zpred(TBG_L, SLT_L):
        a = np.concatenate([(TBG_L - mT) @ WT + lT, (SLT_L - mS) @ WS + lS], 1)
        return R.predict(sc.transform(a))

    # ID-train predictions (used to fit the CDF / standardiser -- label-free) -----------------
    z_tr = zpred(sh["TBG"][tbg], sh["SLT"][slt])[tr]
    qr_n2000 = arm_preds("q_resp_only", source, dataset, num_samples)
    qr_tr = np.array([qr_n2000[i] for i in s_ids])[tr]

    out = {}

    # --- ID: trivia fresh n1000 -------------------------------------------------------------
    fsh, fs_y, fs_ids = load_matrix(Stage1Config(model_name=source, dataset=dataset, num_samples=fresh_num_samples), ["TBG", "SLT"])
    z_f = zpred(fsh["TBG"][tbg], fsh["SLT"][slt])
    qr_fresh = arm_preds("q_resp_only", source, dataset, fresh_num_samples)
    qr_f = np.array([qr_fresh[i] for i in fs_ids])
    score_block(f"ID trivia_qa fresh n{fresh_num_samples}", z_f, qr_f, fs_y, z_tr, qr_tr, s_y[tr], out)

    # --- OOD: squad (needs source-model squad hidden states + records) ----------------------
    ood_dir = f"amortized_ue/data/stage1/{source}_{ood_dataset}_n{ood_num_samples}_full/records"
    if os.path.isdir(ood_dir) and glob.glob(os.path.join(ood_dir, "*.pt")):
        osh, oy, o_ids = load_matrix(Stage1Config(model_name=source, dataset=ood_dataset, num_samples=ood_num_samples), ["TBG", "SLT"])
        z_o = zpred(osh["TBG"][tbg], osh["SLT"][slt])
        qr_ood = arm_preds("q_resp_only", source, ood_dataset, ood_num_samples)
        qr_o = np.array([qr_ood[i] for i in o_ids])
        # floor control: raw (unaligned) Mistral squad states through the SAME Llama-2 ridge, NO W --
        # isolates whether the trivia-fit Procrustes W is doing the work on the shifted (squad) domain.
        floor_o = R.predict(sc.transform(np.concatenate([osh["TBG"][tbg], osh["SLT"][slt]], 1)))
        # s_y_tr=None -> label-free fusions only OOD (a label-fitted combiner would need OOD labels).
        score_block(f"OOD {ood_dataset} n{ood_num_samples}", z_o, qr_o, oy, z_tr, qr_tr, None, out, floor_eval=floor_o)
    else:
        out["OOD_squad"] = {"status": f"SKIPPED — no {source} {ood_dataset} dataset on disk ({ood_dir}). "
                            "aligned-z is Mistral->Llama-2 so it needs Mistral squad hidden states, which "
                            "must be built (Stage 1) before this OOD test can run."}
        print(f"\n[OOD SKIPPED] {out['OOD_squad']['status']}")

    with open("amortized_ue/procrustes_e27_rank_fusion.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote amortized_ue/procrustes_e27_rank_fusion.json")


if __name__ == "__main__":
    run()
