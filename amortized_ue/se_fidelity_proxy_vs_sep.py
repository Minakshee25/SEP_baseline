"""SE-FIDELITY (not correctness) head-to-head: the final `q_resp_only` proxy vs SEP. Additive;
trains nothing -- reuses saved LOLO predictions (E37/E43) and saved proxy checkpoints (deploy /
Qwen-Gemma deploy) for inference only, and the existing SEP-training recipe
(`sep_single_fixed_layer`/`sep_single_val_selected`, E41-corrected layers where established).

Every predictor is scored against the SAME continuous `cluster_assignment_entropy` (CAE) label on
the SAME held-out rows: Spearman(pred, SE) and AUROC(pred, high-vs-low SE, best_split threshold
fit on the FIT-side TRAIN split only, never on eval). Per-seed proxy scores are reported alongside
the 3-seed ensemble (never just the ensemble). A paired bootstrap (shared resample indices) gives
the 95% CI of (proxy - SEP) on both metrics, for the ensemble and for every individual seed.

Settings (each independently runnable via --only):
  lolo    LOLO trivia_qa, 4 targets (Llama-2/Mistral/Llama-3/DeepSeek). Proxy = E37/E43 leave-one-
          LLM-out checkpoints (per-seed predictions already saved in exp2_lolo_full.json -- CPU
          only, no proxy forward pass needed). SEP = E41 fixed-layer (exp2_run.BEST_TBG), fit on
          the target's OWN n2000 train split, evaluated on the SAME 200 te rows as the LOLO proxy
          (id-mapping audited, matching E38's convention).
  squad   deploy proxy (all-4-pooled, trained on trivia) on squad OOD, Llama-2 + Mistral only (the
          only 2 targets with squad records). SEP fit on the target's trivia n2000, evaluated OOD
          on squad (E41 fixed layer) -- the standard cross-dataset SEP test (mirrors E39).
  fresh   deploy proxy on a FRESH disjoint trivia_qa n1000 (0 id-overlap with the n2000 training
          set, verified per target) for all 4 training models. SEP fit on n2000 train, evaluated
          on the same fresh n1000 (E41 fixed layer).
  qwengemma  deploy proxy zero-shot on 4 Qwen/Gemma small-tier targets (never in the proxy's
          training pool). SEP is target-specific, fit on that model's OWN n2000 training tier and
          evaluated on its disjoint n1000 eval tier (verified 0 id-overlap per E44) -- a FAIR SEP,
          not a re-selected-on-eval one. No E41 CV layer exists yet for these families, so the
          layer is picked by leak-free VAL-selection (sep_single_val_selected, selects on the
          fit-side val split, never on eval) rather than a fixed CV layer -- flagged per-target.

Envs: `lolo` is se_probes/CPU only. `squad`/`fresh` need `amortized_stage2` + a free GPU (deploy
proxy forward pass, checkpoints at amortized_ue/results/deploy_checkpoints). `qwengemma` needs
`amortized_stage2_v5` (the /data2 venv) + a free GPU (checkpoints at
/data2/mn1025/stage2_checkpoints/deploy_checkpoints, data at /data2/mn1025/stage1).

    python -m amortized_ue.se_fidelity_proxy_vs_sep --only lolo
    python -m amortized_ue.se_fidelity_proxy_vs_sep --only squad fresh --data_dir /data2/mn1025/stage1
    /data2/mn1025/conda_envs/amortized_stage2_v5/bin/python -m amortized_ue.se_fidelity_proxy_vs_sep --only qwengemma
    python -m amortized_ue.se_fidelity_proxy_vs_sep --only summary   # assemble the final table (no torch needed)
"""
from __future__ import annotations

import os
import json
import glob
import dataclasses
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits
from amortized_ue.correctness_eval import (
    sep_single_val_selected, sep_single_fixed_layer, paired_bootstrap_auc, ci)
from amortized_ue import exp2_run as E2

OUT = "amortized_ue/results/se_fidelity_proxy_vs_sep.json"
DEPLOY_CKPT_MAIN = "amortized_ue/results/deploy_checkpoints"
DEPLOY_CKPT_QG = "/data2/mn1025/stage2_checkpoints/deploy_checkpoints"
QG_DATA_DIR = "/data2/mn1025/stage1"
QG_TARGETS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
LOLO_JSON = "amortized_ue/results/exp2_lolo_full.json"
LOLO_CKPT_DIR = "amortized_ue/stage2/runs/E37_LOLO_ckpt/checkpoints"
LOLO_SQUAD_TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]   # only 2 targets have squad records
LONG = {v: k for k, v in E2.SHORT.items()}


# ------------------------------------------------------------------------------------------------
# generic paired-bootstrap Spearman (mirrors correctness_eval.paired_bootstrap_auc's convention:
# ONE shared set of resampled indices reused for every predictor, so downstream deltas are paired)
# ------------------------------------------------------------------------------------------------
def _rank_rows(a):
    """Row-wise ranks of a 2D array (no tie-averaging -- fine for continuous scores)."""
    order = np.argsort(a, axis=1, kind="stable")
    ranks = np.empty_like(order, dtype=float)
    rows = np.arange(a.shape[0])[:, None]
    ranks[rows, order] = np.arange(a.shape[1])
    return ranks


def paired_bootstrap_spearman(preds: dict, y, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.integers(0, n, size=(B, n))
    y = np.asarray(y, dtype=float)
    y_r = _rank_rows(y[idx])
    y_c = y_r - y_r.mean(1, keepdims=True)
    y_n = np.sqrt((y_c ** 2).sum(1))
    boot = {}
    for name, s in preds.items():
        s = np.asarray(s, dtype=float)
        s_r = _rank_rows(s[idx])
        s_c = s_r - s_r.mean(1, keepdims=True)
        s_n = np.sqrt((s_c ** 2).sum(1))
        boot[name] = (y_c * s_c).sum(1) / (y_n * s_n)
    return boot


def spearman(a, b):
    from scipy.stats import spearmanr
    r = spearmanr(np.asarray(a, float), np.asarray(b, float)).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


# ------------------------------------------------------------------------------------------------
# per-seed proxy inference (copy of procrustes_e27_rank_fusion.arm_preds that keeps every seed's
# prediction instead of only the mean -- same checkpoints, same forward pass, no retraining)
# ------------------------------------------------------------------------------------------------
def arm_preds_per_seed(arm, model_name, dataset, num_samples, ckpt_dir, data_dir=None):
    import torch
    from amortized_ue.stage2.data import Stage2Data
    from amortized_ue.stage2.train import Trainer
    from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta, load_checkpoint

    paths = sorted(glob.glob(os.path.join(ckpt_dir, f"*{arm}_seed*.pt")))
    if not paths:
        raise FileNotFoundError(f"no '{arm}' checkpoints under {ckpt_dir}")
    cfg = dataclasses.replace(
        _cfg_from_meta(read_meta(paths[0])), stage1_model_name=model_name,
        stage1_dataset=dataset, stage1_num_samples=num_samples, ood_dataset=None, smoke=False,
        **({"stage1_output_dir": data_dir} if data_dir else {}))
    data = Stage2Data(cfg)
    rows = data.split_indices("all")
    ids = [data.ids[r] for r in rows]
    model, trainer = None, None
    per_seed = []
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(cfg, data, model=model)
        trainer.data = data
        trainer.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), cfg.batch_size):
                r = rows[i:i + cfg.batch_size]
                preds.append(trainer._forward_batch(r, meta["position"], meta["layer"], arm, data=data).float().cpu())
        per_seed.append(transform.decode(torch.cat(preds)).numpy())
    return ids, np.stack(per_seed)          # [n_seeds, N], aligned with `ids`


def arm_preds_per_seed_prefixed(arm, ckpt_prefix, model_name, dataset, num_samples, ckpt_dir, data_dir=None):
    """Same as arm_preds_per_seed, but for a checkpoint directory holding MULTIPLE targets'
    checkpoints together (e.g. the LOLO dir: `<HeldOutTarget>_<arm>_seed<N>.pt` for all 4 folds
    in one folder) -- filters the glob to `ckpt_prefix` so only the intended fold's checkpoints
    are loaded. `model_name`/`dataset`/`num_samples` select the EVAL data (may differ from the
    fold's held-out target's training-time dataset -- that's the whole point of an OOD test)."""
    import torch
    from amortized_ue.stage2.data import Stage2Data
    from amortized_ue.stage2.train import Trainer
    from amortized_ue.stage2.checkpoint import read_meta, _cfg_from_meta, load_checkpoint

    paths = sorted(glob.glob(os.path.join(ckpt_dir, f"{ckpt_prefix}_{arm}_seed*.pt")))
    if not paths:
        raise FileNotFoundError(f"no '{ckpt_prefix}_{arm}_seed*' checkpoints under {ckpt_dir}")
    cfg = dataclasses.replace(
        _cfg_from_meta(read_meta(paths[0])), stage1_model_name=model_name,
        stage1_dataset=dataset, stage1_num_samples=num_samples, ood_dataset=None, smoke=False,
        **({"stage1_output_dir": data_dir} if data_dir else {}))
    data = Stage2Data(cfg)
    rows = data.split_indices("all")
    ids = [data.ids[r] for r in rows]
    model, trainer = None, None
    per_seed = []
    for p in paths:
        model, meta, transform = load_checkpoint(p, model=model)
        if trainer is None:
            trainer = Trainer(cfg, data, model=model)
        trainer.data = data
        trainer.model.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(rows), cfg.batch_size):
                r = rows[i:i + cfg.batch_size]
                preds.append(trainer._forward_batch(r, meta["position"], meta["layer"], arm, data=data).float().cpu())
        per_seed.append(transform.decode(torch.cat(preds)).numpy())
    return ids, np.stack(per_seed)          # [n_seeds, N], aligned with `ids`


# ------------------------------------------------------------------------------------------------
# SEP: fit on (fit_dataset, fit_num_samples)'s TRAIN split, evaluate on given eval rows/hidden
# ------------------------------------------------------------------------------------------------
def compute_sep(target, eval_dataset, eval_num_samples, data_dir=None, fit_dataset="trivia_qa",
                fit_num_samples=2000, use_test_split_as_eval=False, layer=None, eval_data_dir="__same__"):
    """layer=int -> E41 fixed-layer SEP (sep_single_fixed_layer, TBG). layer=None -> leak-free
    val-selected SEP (sep_single_val_selected) for targets with no established CV layer.
    `eval_data_dir` defaults to `data_dir` (same override for fit+eval); pass None explicitly to
    force the eval load onto the NFS default path even when `data_dir` points fit elsewhere (squad
    records are NFS-only, see [[use-data2-not-nfs]]).
    Returns dict: pred, y (continuous SE), ids, yb (binarised SE, -1=invalid), choice, auroc_se, thr."""
    if eval_data_dir == "__same__":
        eval_data_dir = data_dir
    fit_cfg = Stage1Config(model_name=target, dataset=fit_dataset, num_samples=fit_num_samples,
                           **({"output_dir": data_dir} if data_dir else {}))
    hid, y, ids = load_matrix(fit_cfg, ["TBG", "SLT"])
    tr, va, te = splits(len(ids))

    if use_test_split_as_eval:
        eval_hidden, eval_y, eval_rows = hid, y, te
        eval_ids = [ids[i] for i in te]
    else:
        eval_cfg = Stage1Config(model_name=target, dataset=eval_dataset, num_samples=eval_num_samples,
                                **({"output_dir": eval_data_dir} if eval_data_dir else {}))
        eval_hidden, eval_y, eval_ids = load_matrix(eval_cfg, ["TBG", "SLT"])
        eval_rows = np.arange(len(eval_ids))

    if layer is not None:
        p_all, au_se, choice, thr, ybe = sep_single_fixed_layer(
            hid, y, tr, va, eval_hidden, eval_y, eval_rows, "TBG", layer)
        selection = f"E41 fixed TBG:{layer}"
    else:
        p_all, au_se, choice, thr, ybe, _ = sep_single_val_selected(
            hid, y, tr, va, eval_hidden, eval_y, eval_rows)
        selection = "leak-free val-selected (no established CV layer for this target)"

    return {"pred": np.asarray(p_all)[eval_rows], "y": np.asarray(eval_y, float)[eval_rows],
            "ids": eval_ids, "yb": np.asarray(ybe)[eval_rows], "choice": list(choice),
            "auroc_se": au_se, "thr": float(thr), "selection": selection}


# ------------------------------------------------------------------------------------------------
# scoring: given SEP + per-seed proxy predictions (id-aligned), produce the full comparison block
# ------------------------------------------------------------------------------------------------
def score_block(sep, proxy_ids, proxy_per_seed, bootstrap=10000, tag=""):
    """proxy_per_seed: [n_seeds, N_proxy] in `proxy_ids` order. Re-indexes proxy onto sep['ids']
    order (join by id, never by position) and asserts every sep id is covered."""
    id_to_col = {i: c for c, i in enumerate(proxy_ids)}
    missing = [i for i in sep["ids"] if i not in id_to_col]
    assert not missing, f"{tag}: {len(missing)} SEP eval ids have no proxy prediction (e.g. {missing[:3]})"
    cols = [id_to_col[i] for i in sep["ids"]]
    P = proxy_per_seed[:, cols]                                  # [n_seeds, N] aligned to sep['ids']/sep['y']
    ens = P.mean(0)

    y = sep["y"]
    yb = sep["yb"]
    v = yb >= 0

    def m(pred):
        au = float("nan")
        if len(np.unique(yb[v])) == 2:
            from sklearn.metrics import roc_auc_score
            au = float(roc_auc_score(yb[v], pred[v]))
        return {"spearman": spearman(pred, y), "auroc_se": au}

    metrics = {"sep": m(sep["pred"]), "proxy_ensemble": m(ens)}
    for s in range(P.shape[0]):
        metrics[f"proxy_seed{s}"] = m(P[s])

    # paired bootstrap: proxy(ensemble + each seed) minus SEP, on BOTH metrics, shared indices
    all_preds_auroc = {"sep": sep["pred"][v], "proxy_ensemble": ens[v]}
    for s in range(P.shape[0]):
        all_preds_auroc[f"proxy_seed{s}"] = P[s][v]
    boot_au = paired_bootstrap_auc(all_preds_auroc, yb[v], B=bootstrap) if len(np.unique(yb[v])) == 2 else {}

    all_preds_sp = {"sep": sep["pred"], "proxy_ensemble": ens}
    for s in range(P.shape[0]):
        all_preds_sp[f"proxy_seed{s}"] = P[s]
    boot_sp = paired_bootstrap_spearman(all_preds_sp, y, B=bootstrap)

    vs_sep = {}
    for name in list(metrics):
        if name == "sep":
            continue
        entry = {"spearman_delta": ci(boot_sp[name] - boot_sp["sep"])}
        if boot_au:
            entry["auroc_se_delta"] = ci(boot_au[name] - boot_au["sep"])
        for k in entry:
            c = entry[k]
            c["ci_excludes_zero"] = bool(c["lo95"] > 0 or c["hi95"] < 0)
        vs_sep[name] = entry

    return {"n": len(y), "n_valid_se_binary": int(v.sum()), "sep_choice": sep["choice"],
            "sep_selection": sep["selection"], "metrics": metrics,
            "bootstrap_vs_sep": vs_sep, "bootstrap_resamples": bootstrap}


def print_block(label, block):
    print(f"\n{'-' * 78}\n{label}\n{'-' * 78}")
    print(f"  SEP {block['sep_choice']} ({block['sep_selection']})")
    print(f"  {'predictor':18s}{'spearman':>10s}{'auroc_se':>10s}   Δspearman vs SEP        Δauroc_se vs SEP")
    m = block["metrics"]
    print(f"  {'sep':18s}{m['sep']['spearman']:>10.3f}{m['sep']['auroc_se']:>10.3f}")
    for name in [k for k in m if k != "sep"]:
        d = block["bootstrap_vs_sep"][name]
        sp_c = d["spearman_delta"]
        au_c = d.get("auroc_se_delta")
        au_str = (f"{au_c['mean']:+.3f} [{au_c['lo95']:+.3f},{au_c['hi95']:+.3f}] "
                  f"({'excl0' if au_c['ci_excludes_zero'] else 'incl0'})") if au_c else "n/a"
        print(f"  {name:18s}{m[name]['spearman']:>10.3f}{m[name]['auroc_se']:>10.3f}   "
              f"{sp_c['mean']:+.3f} [{sp_c['lo95']:+.3f},{sp_c['hi95']:+.3f}] "
              f"({'excl0' if sp_c['ci_excludes_zero'] else 'incl0'})   {au_str}")


# ==================================================================================================
# SETTING 1 — LOLO trivia_qa, 4 targets. CPU only (proxy preds already saved in exp2_lolo_full.json)
# ==================================================================================================
def run_lolo(bootstrap, data_dir, lolo_json=LOLO_JSON):
    with open(lolo_json) as f:
        folds = json.load(f)
    print(f"\n{'#' * 92}\n# SETTING 1: LOLO trivia_qa (4 targets) -- proxy per-seed preds from {lolo_json}\n{'#' * 92}")
    out = {}
    for fold in folds:
        short = fold["info"]["target"]
        target = LONG[short]
        layer = E2.BEST_TBG[target]

        cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000,
                           **({"output_dir": data_dir} if data_dir else {}))
        with open(cfg.manifest_path()) as f:
            all_ids = sorted(json.load(f)["records"].keys())
        tr, va, te = splits(len(all_ids))
        te_ids = [all_ids[i] for i in te]
        y_saved = np.asarray(fold["target_y"], dtype=float)
        assert len(te_ids) == len(y_saved), "te-row count mismatch vs saved LOLO target_y"

        sep = compute_sep(target, eval_dataset="trivia_qa", eval_num_samples=None, data_dir=data_dir,
                          fit_num_samples=2000, use_test_split_as_eval=True, layer=layer)
        max_dev = float(np.max(np.abs(sep["y"] - y_saved)))
        print(f"\n  {short}: id-mapping audit max|SE(te) - LOLO target_y| = {max_dev:.3e} "
              f"({'MATCH' if max_dev < 1e-5 else 'MISMATCH -- STOP'})")
        assert max_dev < 1e-5, f"{short}: te rows do not reproduce the LOLO fold's target_y"

        P = np.asarray(fold["arms"]["q_resp_only"]["te_pred_by_seed"], dtype=float)   # [n_seeds, 200], same te order
        block = score_block(sep, te_ids, P, bootstrap=bootstrap, tag=f"lolo/{short}")
        block["target"] = short
        block["proxy_provenance"] = "E37/E43 LOLO q_resp_only (trained on the OTHER 3 models, never this target's data)"
        out[short] = block
        print_block(f"LOLO -- held out {short} (proxy trained on the other 3)", block)
    return out


# ==================================================================================================
# SETTING 2 — deploy proxy on squad OOD, Llama-2 + Mistral. Needs GPU (amortized_stage2).
# ==================================================================================================
def run_squad(bootstrap, trivia_data_dir):
    targets = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]
    print(f"\n{'#' * 92}\n# SETTING 2: deploy proxy on squad OOD (Llama-2 + Mistral)\n{'#' * 92}")
    out = {}
    for target in targets:
        layer = E2.BEST_TBG[target] if target != E2.ANCHOR else 30
        sep = compute_sep(target, eval_dataset="squad", eval_num_samples=1000, data_dir=trivia_data_dir,
                          eval_data_dir=None, fit_num_samples=2000, use_test_split_as_eval=False, layer=layer)
        ids, P = arm_preds_per_seed("q_resp_only", target, "squad", 1000, ckpt_dir=DEPLOY_CKPT_MAIN)
        block = score_block(sep, ids, P, bootstrap=bootstrap, tag=f"squad/{target}")
        block["target"] = target
        block["proxy_provenance"] = "DEPLOY proxy (all-4-pooled trivia-trained) -> squad; target WAS in the pool, cross-DATASET only"
        out[target] = block
        print_block(f"squad OOD -- {target}", block)
    return out


# ==================================================================================================
# SETTING 2b — LOLO proxy (trained on the OTHER 3 targets, never this one) on squad OOD.
# Llama-2 + Mistral only (the only 2 targets with squad records). Needs GPU.
# Cross-LLM (never saw this target) AND cross-dataset (never saw squad) simultaneously.
# ==================================================================================================
def run_lolo_squad(bootstrap, trivia_data_dir):
    print(f"\n{'#' * 92}\n# SETTING 2b: LOLO proxy on squad OOD (Llama-2 + Mistral) -- "
          f"never saw this target OR squad\n{'#' * 92}")
    out = {}
    for target in LOLO_SQUAD_TARGETS:
        short = E2.SHORT[target]
        layer = E2.BEST_TBG[target] if target != E2.ANCHOR else 30
        sep = compute_sep(target, eval_dataset="squad", eval_num_samples=1000, data_dir=trivia_data_dir,
                          eval_data_dir=None, fit_num_samples=2000, use_test_split_as_eval=False, layer=layer)
        ids, P = arm_preds_per_seed_prefixed("q_resp_only", short, target, "squad", 1000,
                                             ckpt_dir=LOLO_CKPT_DIR)
        block = score_block(sep, ids, P, bootstrap=bootstrap, tag=f"lolo_squad/{short}")
        block["target"] = short
        block["proxy_provenance"] = ("E37/E43 LOLO q_resp_only (trained on the OTHER 3 models' trivia_qa, "
                                     "never this target's data OR squad) -> squad OOD")
        out[short] = block
        print_block(f"LOLO squad OOD -- held out {short} (proxy trained on the other 3, trivia only)", block)
    return out


# ==================================================================================================
# SETTING 3 — deploy proxy on fresh trivia_qa n1000, all 4 training models. Needs GPU.
# ==================================================================================================
def run_fresh(bootstrap, data_dir):
    targets = list(E2.MODELS)
    print(f"\n{'#' * 92}\n# SETTING 3: deploy proxy on FRESH trivia_qa n1000, 4 training models\n{'#' * 92}")
    out = {}
    for target in targets:
        layer = E2.BEST_TBG[target] if target != E2.ANCHOR else 30

        # some targets' n2000/n1000 trivia sets are staged on /data2, others only on the NFS
        # default -- try the preferred dir first, fall back to NFS default per target.
        tdir = data_dir
        fit_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000,
                               **({"output_dir": tdir} if tdir else {}))
        eval_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000,
                                **({"output_dir": tdir} if tdir else {}))
        if not (os.path.exists(fit_cfg.manifest_path()) and os.path.exists(eval_cfg.manifest_path())):
            tdir = None
            fit_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000)
            eval_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000)
        if not (os.path.exists(fit_cfg.manifest_path()) and os.path.exists(eval_cfg.manifest_path())):
            print(f"  [SKIP] {target}: fresh n1000 or n2000 manifest not found under {data_dir} or NFS default")
            continue
        data_dir_for_target = tdir
        with open(fit_cfg.manifest_path()) as f:
            fit_ids = set(json.load(f)["records"].keys())
        with open(eval_cfg.manifest_path()) as f:
            eval_ids_all = set(json.load(f)["records"].keys())
        overlap = fit_ids & eval_ids_all
        if overlap:
            print(f"  [SKIP] {target}: fresh n1000 overlaps n2000 train by {len(overlap)} ids -- not genuinely fresh")
            continue

        sep = compute_sep(target, eval_dataset="trivia_qa", eval_num_samples=1000, data_dir=data_dir_for_target,
                          fit_num_samples=2000, use_test_split_as_eval=False, layer=layer)
        ids, P = arm_preds_per_seed("q_resp_only", target, "trivia_qa", 1000,
                                    ckpt_dir=DEPLOY_CKPT_MAIN, data_dir=data_dir_for_target)
        block = score_block(sep, ids, P, bootstrap=bootstrap, tag=f"fresh/{target}")
        block["target"] = target
        block["proxy_provenance"] = "DEPLOY proxy (all-4-pooled, includes this target's OWN trivia data) -> fresh disjoint n1000"
        out[E2.SHORT[target]] = block
        print_block(f"fresh trivia n1000 -- {target}", block)
    return out


# ==================================================================================================
# SETTING 4 — deploy proxy zero-shot on Qwen/Gemma, vs a FAIR target-specific SEP. Needs GPU.
# ==================================================================================================
def run_qwengemma(bootstrap):
    print(f"\n{'#' * 92}\n# SETTING 4: deploy proxy zero-shot on Qwen/Gemma vs fair target-specific SEP\n{'#' * 92}")
    out = {}
    for target in QG_TARGETS:
        fit_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000, output_dir=QG_DATA_DIR)
        eval_cfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=1000, output_dir=QG_DATA_DIR)
        with open(fit_cfg.manifest_path()) as f:
            fit_ids = set(json.load(f)["records"].keys())
        with open(eval_cfg.manifest_path()) as f:
            eval_ids_all = set(json.load(f)["records"].keys())
        overlap = fit_ids & eval_ids_all
        assert not overlap, f"{target}: n2000 train / n1000 eval overlap by {len(overlap)} -- SEP would leak"
        print(f"  {target}: n2000 train ({len(fit_ids)}) / n1000 eval ({len(eval_ids_all)}) -- "
              f"0 id overlap, fair split confirmed")

        sep = compute_sep(target, eval_dataset="trivia_qa", eval_num_samples=1000, data_dir=QG_DATA_DIR,
                          fit_num_samples=2000, use_test_split_as_eval=False, layer=None)   # no E41 CV layer yet
        ids, P = arm_preds_per_seed("q_resp_only", target, "trivia_qa", 1000,
                                    ckpt_dir=DEPLOY_CKPT_QG, data_dir=QG_DATA_DIR)
        block = score_block(sep, ids, P, bootstrap=bootstrap, tag=f"qwengemma/{target}")
        block["target"] = target
        block["proxy_provenance"] = "DEPLOY proxy (all-4-pooled, Llama-2/Mistral/Llama-3/DeepSeek only) -> zero-shot Qwen/Gemma"
        out[target] = block
        print_block(f"Qwen/Gemma zero-shot -- {target}", block)
    return out


# ==================================================================================================
def load_out():
    if os.path.exists(OUT):
        with open(OUT) as f:
            return json.load(f)
    return {}


def save_out(d):
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w") as f:
        json.dump(d, f, indent=2)
    print(f"\n  -> saved to {OUT}")


def print_final_table(all_out):
    print("\n" + "=" * 100)
    print("FINAL TABLE -- proxy (q_resp_only) vs SEP, SE-fidelity (Spearman / AUROC-vs-SE)")
    print("=" * 100)
    hdr = (f"{'setting':10s}{'target':16s}{'SEP rho':>9s}{'ens rho':>9s}{'Δrho':>8s}{'ci':>10s}"
           f"{'SEP auc':>9s}{'ens auc':>9s}{'Δauc':>8s}{'ci':>10s}{'verdict':>10s}")
    print(hdr)
    rows = []
    for setting in ["lolo", "squad", "lolo_squad", "fresh", "qwengemma"]:
        if setting not in all_out:
            continue
        for target, b in all_out[setting].items():
            m = b["metrics"]
            d = b["bootstrap_vs_sep"]["proxy_ensemble"]
            sp_d, au_d = d["spearman_delta"], d.get("auroc_se_delta")
            sp_excl = sp_d["ci_excludes_zero"]
            au_excl = au_d["ci_excludes_zero"] if au_d else False
            if sp_excl and sp_d["mean"] > 0:
                verdict = "proxy>SEP"
            elif sp_excl and sp_d["mean"] < 0:
                verdict = "SEP>proxy"
            else:
                verdict = "tie"
            row = {"setting": setting, "target": target,
                  "sep_rho": m["sep"]["spearman"], "ens_rho": m["proxy_ensemble"]["spearman"],
                  "delta_rho": sp_d["mean"], "rho_ci": [sp_d["lo95"], sp_d["hi95"]], "rho_ci_excludes_zero": sp_excl,
                  "sep_auc": m["sep"]["auroc_se"], "ens_auc": m["proxy_ensemble"]["auroc_se"],
                  "delta_auc": au_d["mean"] if au_d else None,
                  "auc_ci": [au_d["lo95"], au_d["hi95"]] if au_d else None,
                  "auc_ci_excludes_zero": au_excl, "verdict": verdict}
            rows.append(row)
            rho_ci = f"[{sp_d['lo95']:+.2f},{sp_d['hi95']:+.2f}]"
            auc_ci = f"[{au_d['lo95']:+.2f},{au_d['hi95']:+.2f}]" if au_d else "n/a"
            delta_auc = row["delta_auc"] if row["delta_auc"] is not None else float("nan")
            print(f"{setting:10s}{target:16s}{row['sep_rho']:>9.3f}{row['ens_rho']:>9.3f}{row['delta_rho']:>+8.3f}"
                  f"{rho_ci:>10s}{row['sep_auc']:>9.3f}{row['ens_auc']:>9.3f}"
                  f"{delta_auc:>+8.3f}{auc_ci:>10s}{verdict:>10s}")
    return rows


def main():
    p = argparse.ArgumentParser(description="Proxy (q_resp_only) vs SEP: SE-fidelity head-to-head.")
    p.add_argument("--only", nargs="+", default=["lolo", "squad", "fresh", "qwengemma"],
                   choices=["lolo", "squad", "lolo_squad", "fresh", "qwengemma", "summary"])
    p.add_argument("--data_dir", default=None, help="Stage1Config.output_dir override for the 4 main models (e.g. /data2/mn1025/stage1)")
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--out", default=OUT)
    args = p.parse_args()

    all_out = load_out()          # incremental: keep whatever settings already ran

    if "lolo" in args.only:
        all_out["lolo"] = run_lolo(args.bootstrap, args.data_dir)
        save_out(all_out)
    if "squad" in args.only:
        all_out["squad"] = run_squad(args.bootstrap, args.data_dir)
        save_out(all_out)
    if "lolo_squad" in args.only:
        all_out["lolo_squad"] = run_lolo_squad(args.bootstrap, args.data_dir)
        save_out(all_out)
    if "fresh" in args.only:
        all_out["fresh"] = run_fresh(args.bootstrap, args.data_dir)
        save_out(all_out)
    if "qwengemma" in args.only:
        all_out["qwengemma"] = run_qwengemma(args.bootstrap)
        save_out(all_out)

    if set(args.only) & {"lolo", "squad", "lolo_squad", "fresh", "qwengemma", "summary"}:
        rows = print_final_table(all_out)
        all_out["_final_table"] = rows
        save_out(all_out)


if __name__ == "__main__":
    main()
