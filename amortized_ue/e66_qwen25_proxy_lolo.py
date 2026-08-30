"""E66 — swap the proxy BACKBONE (Llama-3.2-3B -> Qwen2.5-3B) and re-run the thesis experiment.

Question: E37/E38's headline (a text-only `q_resp_only` proxy, trained on 3 target LLMs, predicts
the held-out 4th's semantic entropy and catches its wrong answers, beating that model's own
supervised SEP) was obtained with a Llama-family reader (`meta-llama/Llama-3.2-3B`). Does it
survive when the reader is a DIFFERENT pretraining lineage? If yes, the result is about the
transferable signal in the text, not about Llama-family representational kinship.

Design — ONE leave-one-LLM-out fold, identical recipe to E37/E53/E63/E65, only `proxy_model` changes:
  * held out = Mistral-7B-Instruct-v0.2 ; trained on Llama-2-7b-chat + Meta-Llama-3-8B-Instruct +
    deepseek-llm-7b-chat (trivia_qa n2000 each). Mistral is the canonical clean cross-family fold.
  * arm = q_resp_only (text only: "Question: {q}\nAnswer: {canonical response}"). NO hidden states,
    NO Procrustes alignment -> nothing model-specific from Mistral, and no target LLM layer choice
    enters the proxy at all (layers only matter for the SEP / ridge baselines below, handled
    leak-free).
  * frozen **Qwen/Qwen2.5-3B** + LoRA r16/a32 on q/k/v/o_proj, projector_hidden_dim 1024, k=4,
    10 epochs, batch 8 x grad_accum 4 (eff 32 — grad-accum reproduces the batch-32 gradient
    exactly; no batchnorm in ProxyModel), 3 seeds. SE label z-scored PER TRAIN MODEL (train-only
    mean/std) before pooling.

Evaluation — on Mistral's FRESH shared-ID trivia n1000 (all 1000 rows, disjoint from every training
pool: n2000 train-pool ∩ n1000 = 0, asserted), the E45/E64/E65 5x-power convention:
    proxy_qwen25_q_resp_only        label-free on Mistral  — THIS experiment's predictor
    proxy_llama32_3b_q_resp_only    the SAME LOLO fold with the ORIGINAL backbone (E37/E43
                                    checkpoints, scored on the identical rows) — the head-to-head
    true_semantic_entropy           10-sample CAE label (sampling upper bound)
    sep_fixed_TBG31                 Mistral's OWN supervised SEP at the E41/E51/E62 fixed layer
    sep_val_selected               same, but layer re-picked on val (reference)
    ridge_own_model                 white-box linear ceiling: ridge on Mistral's OWN TBG+SLT
                                    hidden states, layer picked leak-free by val Spearman — context,
                                    NOT a fair opponent
  metrics: Spearman(pred, SE), AUROC_incorrect, AUROC_binarised_SE ; paired bootstrap of
  Δ AUROC_incorrect for the Qwen proxy vs {llama-3.2-3b proxy, SEP-fixed, true SE}.

Envs:
  --stage check       se_probes / CPU  — is the training + eval data on disk?
  --stage train|eval  amortized_stage2_v5 + a free GPU (Qwen2.5-3B needs transformers 4.52)
  --stage all         train then eval
  --stage push_wandb  se_probes_v5 / CPU  — push the 3 checkpoints as a W&B artifact

Run from the repo root:
    python -m amortized_ue.e66_qwen25_proxy_lolo --stage check --data_dir /data2/mn1025/stage1
    HF_HOME=/data2/mn1025/hf_cache python -m amortized_ue.e66_qwen25_proxy_lolo --stage all \
        --data_dir /data2/mn1025/stage1
    python -m amortized_ue.e66_qwen25_proxy_lolo --stage push_wandb
"""
from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, rho

# ------------------------------------------------------------------ config ----
HELD = "Mistral-7B-Instruct-v0.2"
SOURCES = ["Llama-2-7b-chat", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
PROXY_MODEL = "Qwen/Qwen2.5-3B"
ARM = "q_resp_only"
TRAIN_N = 2000
EVAL_N = 1000
SEEDS = [0, 1, 2]
SEP_FIXED = ("TBG", 31)                       # Mistral's E41/E51/E62 fixed SEP layer (== exp2_run.BEST_TBG)

DEFAULT_DATA_DIR = "/data2/mn1025/stage1"
_HERE = os.path.dirname(os.path.abspath(__file__))
CKPT_DIR = os.path.join(_HERE, "stage2", "runs", "E66_qwen25_proxy_lolo", "checkpoints", HELD)
LLAMA32_LOLO_CKPT_DIR = os.path.join(_HERE, "stage2", "runs", "E37_LOLO_ckpt", "checkpoints")
LLAMA32_LOLO_PREFIX = "Mistral"               # E37/E43 saved the Mistral fold as `Mistral_<arm>_seed<N>.pt`
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_MAIN = os.path.join(RESULTS_DIR, "e66_qwen25_proxy_lolo.json")
OUT_CURVES = os.path.join(RESULTS_DIR, "e66_qwen25_proxy_lolo_train_curves.json")
WANDB_ARTIFACT = "stage2_ckpts_E66_qwen25_proxy_lolo"
TAG = f"held-{HELD}"


def run_name(model, n):
    return f"{model}_trivia_qa_n{n}_full"


def s1cfg(model, n, data_dir):
    return Stage1Config(model_name=model, dataset="trivia_qa", num_samples=n,
                        output_dir=data_dir, run_name=run_name(model, n))


# ------------------------------------------------------------------- check ----
def do_check(data_dir, verbose=True):
    """Training data = SOURCES n2000 ; eval data = HELD n2000 (baselines) + HELD n1000 (eval rows)."""
    ok = True
    need = [(m, TRAIN_N) for m in SOURCES] + [(HELD, TRAIN_N), (HELD, EVAL_N)]
    rows = []
    for m, n in need:
        cfg = s1cfg(m, n, data_dir)
        rd = cfg.records_dir()
        n_pt = len(glob.glob(os.path.join(rd, "*.pt"))) if os.path.isdir(rd) else 0
        n_man = None
        if os.path.isfile(cfg.manifest_path()):
            with open(cfg.manifest_path()) as f:
                n_man = len(json.load(f).get("records", {}))
        done = n_pt >= n and (n_man is not None and n_man >= n)
        ok &= done
        rows.append((m, run_name(m, n), n_pt, n_man, done))
    if verbose:
        print(f"{'model':30s}{'run_name':46s}{'n_pt':>7s}{'n_manifest':>12s}   ready")
        for m, rn, n_pt, n_man, done in rows:
            print(f"{m:30s}{rn:46s}{n_pt:>7d}{str(n_man):>12s}   {'YES' if done else 'no'}")
        n_l32 = len(glob.glob(os.path.join(LLAMA32_LOLO_CKPT_DIR, f"{LLAMA32_LOLO_PREFIX}_{ARM}_seed*.pt")))
        print(f"\nLlama-3.2-3B LOLO Mistral-fold checkpoints found: {n_l32}/3 "
              f"({'ok' if n_l32 >= 3 else 'MISSING — head-to-head arm will be skipped'})")
        print(f"ALL DATA READY: {ok}")
    return ok


# ------------------------------------------------------------------- train ----
def load_pool(data_dir):
    """Pool (question, canonical response, per-model TRAIN-z-scored SE) train/val rows from the 3
    source models. Mirrors e63/e65 load_pool exactly: per-model splits() on that model's sorted-id
    order; SE z-scored with TRAIN-ONLY mean/std applied to both tr and va (no val leakage)."""
    ptr = {"q": [], "r": [], "y": []}
    pva = {"q": [], "r": [], "y": []}
    stats = {}
    for m in SOURCES:
        recs = load_records(s1cfg(m, TRAIN_N, data_dir))
        ids = sorted(recs.keys())
        assert len(ids) == TRAIN_N, f"{m}: expected {TRAIN_N} records, got {len(ids)}"
        tr, va, te = splits(len(ids))
        q = [recs[i]["question"] for i in ids]
        r = [recs[i]["canonical"]["response"] for i in ids]
        y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
        mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
        ptr["q"] += [q[i] for i in tr]; ptr["r"] += [r[i] for i in tr]; ptr["y"] += list((y[tr] - mu) / sd)
        pva["q"] += [q[i] for i in va]; pva["r"] += [r[i] for i in va]; pva["y"] += list((y[va] - mu) / sd)
        stats[m] = {"n": len(ids), "n_tr": int(len(tr)), "n_va": int(len(va)),
                    "mean_CAE_train": mu, "std_CAE_train": sd}
        print(f"    {m:30s} n={len(ids)} tr={len(tr)} va={len(va)} mean_CAE(train)={mu:.3f}")
    train = {"y": np.array(ptr["y"], dtype=np.float32), "q": ptr["q"], "r": ptr["r"]}
    val = {"y": np.array(pva["y"], dtype=np.float32), "q": pva["q"], "r": pva["r"]}
    return train, val, stats


def do_train(data_dir, seeds, batch_size, grad_accum):
    import torch
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z
    from amortized_ue.exp2_run import train_arm

    done = sorted(glob.glob(os.path.join(CKPT_DIR, f"*{ARM}_seed*.pt")))
    if len(done) >= len(seeds):
        print(f"[{HELD}] {len(done)} checkpoints already present -> skip training")
        return

    cfg = Stage2Config(proxy_model=PROXY_MODEL, projector_hidden_dim=1024, k_soft_tokens=4,
                       epochs=10, batch_size=batch_size, grad_accum=grad_accum)
    print(f"proxy backbone = {cfg.proxy_model}")
    model = ProxyModel(cfg, h_in=1).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  d_model={model.d_model}  trainable params="
          f"{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    print(f"\n{'=' * 80}\n[fold] held out {HELD}  (train on {' + '.join(SOURCES)})\n{'=' * 80}")
    train, val, stats = load_pool(data_dir)
    print(f"  pooled: train rows={len(train['y'])}  val rows={len(val['y'])}")

    train["z"] = np.zeros((len(train["y"]), 1), dtype=np.float32)   # q_resp_only never reads z
    val["z"] = np.zeros((len(val["y"]), 1), dtype=np.float32)
    tgt = dict(val)                                                 # in-dist val-pool sanity target

    os.makedirs(CKPT_DIR, exist_ok=True)
    res = train_arm(train, val, tgt, ARM, seeds, cfg, model, torch, nn,
                    get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                    ckpt_dir=CKPT_DIR, tag=TAG)
    print(f"  val-pool sanity Spearman per seed: {[round(s, 3) for s in res['te_spearman']]}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    curves = {"held_out": HELD, "sources": SOURCES, "proxy_model": PROXY_MODEL, "arm": ARM,
              "seeds": list(seeds), "per_model_stats": stats, "train_config": cfg.as_dict(),
              "n_train": len(train["y"]), "n_val": len(val["y"]),
              "val_pool_sanity_spearman_by_seed": res["te_spearman"],
              "val_pool_pred_by_seed": [[float(v) for v in p] for p in res["te_pred_by_seed"]],
              "val_pool_y": [float(v) for v in tgt["y"]],
              "curves_by_seed": res["curves_by_seed"]}
    with open(OUT_CURVES, "w") as f:
        json.dump(curves, f, indent=1)
    print(f"  curves -> {OUT_CURVES}")


# -------------------------------------------------------------------- eval ----
def do_eval(data_dir, bootstrap):
    import torch  # noqa: F401  (arm_preds needs it)
    from sklearn.metrics import roc_auc_score
    from amortized_ue.procrustes_e27_rank_fusion import arm_preds
    from amortized_ue.se_fidelity_proxy_vs_sep import arm_preds_per_seed_prefixed
    from amortized_ue.correctness_eval import (
        load_accuracy, sep_single_val_selected, sep_single_fixed_layer, paired_bootstrap_auc, ci)
    from amortized_ue.linear_ceiling_probe import fit_probe

    if len(glob.glob(os.path.join(CKPT_DIR, f"*{ARM}_seed*.pt"))) < len(SEEDS):
        raise SystemExit(f"[{HELD}] Qwen2.5-3B checkpoints missing -> run --stage train first")

    print(f"\n{'#' * 92}\n# E66 — held out {HELD}; Qwen2.5-3B q_resp_only proxy trained on "
          f"{' + '.join(SOURCES)}\n#   eval on {HELD}'s FRESH shared-ID trivia n{EVAL_N} (all rows)\n{'#' * 92}")

    # ---- fit rows (Mistral's OWN n2000 tr/va) for the supervised baselines --------------------
    fit_cfg = s1cfg(HELD, TRAIN_N, data_dir)
    fit_recs = load_records(fit_cfg)
    fit_ids = sorted(fit_recs.keys())
    tr, va, te = splits(len(fit_ids))

    # ---- eval rows (Mistral's fresh n1000, disjoint) -----------------------------------------
    eval_cfg = s1cfg(HELD, EVAL_N, data_dir)
    eval_recs = load_records(eval_cfg)
    eval_ids = sorted(eval_recs.keys())
    assert not (set(fit_ids) & set(eval_ids)), f"{HELD}: n{TRAIN_N} train pool overlaps the n{EVAL_N} eval set"

    se_eval = np.array([eval_recs[i]["labels"]["cluster_assignment_entropy"] for i in eval_ids], dtype=float)
    acc_map = load_accuracy(eval_cfg)
    acc = np.array([acc_map[i] for i in eval_ids], dtype=float)
    incorrect = (acc < 0.5).astype(int)
    pos_rate = float(incorrect.mean())
    print(f"  n_eval={len(eval_ids)}  incorrect_rate={pos_rate:.3f}  mean_acc={acc.mean():.3f}")

    # ---- proxy (Qwen2.5-3B), 3 seeds seed-averaged, on the eval rows -------------------------
    mp = arm_preds(ARM, HELD, "trivia_qa", EVAL_N, ckpt_dir=CKPT_DIR, data_dir=data_dir,
                   run_name=run_name(HELD, EVAL_N))
    proxy_qwen = np.array([mp[i] for i in eval_ids], dtype=float)

    # ---- the SAME LOLO fold with the ORIGINAL backbone (E37/E43), identical rows -------------
    proxy_l32 = None
    if len(glob.glob(os.path.join(LLAMA32_LOLO_CKPT_DIR, f"{LLAMA32_LOLO_PREFIX}_{ARM}_seed*.pt"))) >= 3:
        l32_ids, l32_ps = arm_preds_per_seed_prefixed(
            ARM, LLAMA32_LOLO_PREFIX, HELD, "trivia_qa", EVAL_N,
            ckpt_dir=LLAMA32_LOLO_CKPT_DIR, data_dir=data_dir)
        l32_map = dict(zip(l32_ids, l32_ps.mean(0)))
        proxy_l32 = np.array([l32_map[i] for i in eval_ids], dtype=float)
        print(f"  Llama-3.2-3B LOLO proxy loaded (seed-avg over {l32_ps.shape[0]} seeds)")
    else:
        print("  Llama-3.2-3B LOLO checkpoints NOT found -> skipping the head-to-head arm")

    # ---- supervised SEP (fixed E41 layer + val-selected) + white-box ridge ceiling ----------
    hid_fit, y_fit, ids_fit = load_matrix(fit_cfg, ["TBG", "SLT"])
    assert ids_fit == fit_ids, "load_matrix id order != manifest order (fit)"
    hid_eval, y_eval_mat, ids_eval = load_matrix(eval_cfg, ["TBG", "SLT"])
    assert ids_eval == eval_ids, "load_matrix id order != manifest order (eval)"
    eval_rows = np.arange(len(eval_ids))
    assert float(np.max(np.abs(y_eval_mat[eval_rows].astype(float) - se_eval))) < 1e-5, "eval SE mismatch"

    sepf_p, sepf_au_se, sepf_choice, thr, ybe = sep_single_fixed_layer(
        hid_fit, y_fit, tr, va, hid_eval, y_eval_mat, eval_rows, *SEP_FIXED)
    sepv_p, sepv_au_se, sepv_choice, _, _, _ = sep_single_val_selected(
        hid_fit, y_fit, tr, va, hid_eval, y_eval_mat, eval_rows)
    sep_fixed_te, sep_val_te = sepf_p[eval_rows], sepv_p[eval_rows]

    rbest = (-np.inf, None, None)   # (val_spearman, (pos,layer,alpha), eval_pred)
    for pos in ("TBG", "SLT"):
        for L in range(hid_fit[pos].shape[0]):
            m, sc, alpha, val_s = fit_probe(hid_fit[pos][L], y_fit.astype(float), tr, va)
            if val_s > rbest[0]:
                rbest = (val_s, (pos, int(L), float(alpha)), m.predict(sc.transform(hid_eval[pos][L][eval_rows])))
    ridge_te, ridge_choice, ridge_val = rbest[2], rbest[1], float(rbest[0])
    del hid_fit, hid_eval

    # ---- score ------------------------------------------------------------------------------
    yb_te = ybe[eval_rows]
    v = yb_te >= 0
    preds = {"proxy_qwen25_q_resp_only": proxy_qwen,
             "true_semantic_entropy": se_eval,
             "sep_fixed_TBG31": sep_fixed_te,
             "sep_val_selected": sep_val_te,
             "ridge_own_model": ridge_te}
    if proxy_l32 is not None:
        preds["proxy_llama32_3b_q_resp_only"] = proxy_l32
    label_free = {"proxy_qwen25_q_resp_only": True, "proxy_llama32_3b_q_resp_only": True,
                  "true_semantic_entropy": False, "sep_fixed_TBG31": False,
                  "sep_val_selected": False, "ridge_own_model": False}

    metrics = {}
    print(f"\n  {'predictor':30s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}  label-free")
    for name, s in preds.items():
        au_inc = float(roc_auc_score(incorrect, s)) if len(np.unique(incorrect)) == 2 else float("nan")
        au_se = float(roc_auc_score(yb_te[v], s[v])) if len(np.unique(yb_te[v])) == 2 else float("nan")
        metrics[name] = {"auroc_incorrect": au_inc, "auroc_binarised_se": au_se,
                         "spearman_se": rho(s, se_eval), "label_free_on_target": label_free[name]}
        print(f"  {name:30s}{au_inc:>11.3f}{au_se:>10.3f}{rho(s, se_eval):>9.3f}"
              f"  {'yes' if label_free[name] else 'NO'}")

    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    vs = {}
    for base in ("proxy_llama32_3b_q_resp_only", "sep_fixed_TBG31", "true_semantic_entropy"):
        if base not in boot:
            continue
        c = ci(boot["proxy_qwen25_q_resp_only"] - boot[base])
        vs[base] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}
        print(f"  Δ AUROC_inc (qwen25 proxy − {base}): {c['mean']:+.3f} "
              f"[{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
              f"({'excludes 0' if vs[base]['ci_excludes_zero'] else 'includes 0'})")

    out = {
        "held_out": HELD, "sources": SOURCES, "proxy_model": PROXY_MODEL, "arm": ARM,
        "eval_set": f"{HELD}_trivia_qa_n{EVAL_N}_full (fresh shared-ID, all rows)",
        "n_test": len(eval_ids), "positive_rate_incorrect": pos_rate, "mean_accuracy": float(acc.mean()),
        "best_split": float(thr), "seeds": list(SEEDS),
        "sep_fixed_choice": list(sepf_choice), "sep_val_choice": list(sepv_choice),
        "ridge_choice_pos_layer_alpha": list(ridge_choice), "ridge_val_spearman": ridge_val,
        "bootstrap_resamples": bootstrap, "metrics": metrics,
        "bootstrap_delta_auroc_incorrect_qwen25_vs": vs,
        "te_ids": list(eval_ids),
        "preds": {k: [float(x) for x in preds[k]] for k in preds},
        "true_se_te": [float(x) for x in se_eval],
        "incorrect_te": [int(x) for x in incorrect],
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_MAIN, "w") as f:
        json.dump(out, f, indent=1)
    print(f"\nwrote {OUT_MAIN}")


# ---------------------------------------------------------------- wandb push ---
def do_push_wandb():
    import wandb
    paths = sorted(glob.glob(os.path.join(CKPT_DIR, f"*{ARM}_seed*.pt")))
    assert len(paths) == len(SEEDS), f"expected {len(SEEDS)} checkpoints, found {len(paths)}"
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=WANDB_ARTIFACT, job_type="checkpoint",
                     config={"arm": ARM, "held_out": HELD, "sources": SOURCES,
                             "proxy_model": PROXY_MODEL,
                             "design": "leave-one-LLM-out, 1 fold, backbone swap vs E37",
                             "recipe": "q_resp_only, 3 seeds, batch 8 x grad_accum 4 (eff 32), "
                                       "projector_hidden_dim 1024, k=4, 10 epochs"})
    art = wandb.Artifact(WANDB_ARTIFACT, type="model",
                         metadata={"held_out": HELD, "sources": SOURCES, "arm": ARM,
                                   "n_seeds": len(SEEDS), "proxy_model": PROXY_MODEL})
    art.add_dir(CKPT_DIR)
    run.log_artifact(art)
    run.finish()
    api = wandb.Api()
    a = api.artifact(f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{WANDB_ARTIFACT}:latest")
    print(f"pushed + verified {WANDB_ARTIFACT}:{a.version}  size={a.size} bytes  n_files={len(list(a.files()))}")


# -------------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["check", "train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    args = p.parse_args()

    if args.stage == "check":
        raise SystemExit(0 if do_check(args.data_dir) else 1)
    if args.stage == "push_wandb":
        do_push_wandb()
        return
    if args.stage in ("train", "all"):
        if not do_check(args.data_dir):
            raise SystemExit("STOP: training/eval data not all on disk (see table above).")
        do_train(args.data_dir, args.seeds, args.batch_size, args.grad_accum)
    if args.stage in ("eval", "all"):
        do_eval(args.data_dir, args.bootstrap)


if __name__ == "__main__":
    main()
