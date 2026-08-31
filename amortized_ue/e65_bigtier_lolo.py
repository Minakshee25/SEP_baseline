"""E65 — leave-one-LLM-out (LOLO) q_resp_only proxy for the 5 BIG-TIER target LLMs.

Question (the thesis, E37 framing, scaled up to 27B targets): train ONE text-only proxy on
4 big-tier models' SE data and ask whether it predicts the 5th (held-out) model's semantic
entropy — and detects its wrong answers — with NO exposure to that model in any form.

Design — identical recipe to E37 / E53 / E63, only the target set changes:
  * 5 folds. Fold h: train on BIGTIER \ {h}, evaluate on h.
  * arm = q_resp_only (text only: "Question: {q}\nAnswer: {canonical response}"). No hidden
    states, no Procrustes alignment — the deployable primitive (E33), and the only arm that
    needs nothing model-specific from a 27B target we never aligned.
  * frozen Llama-3.2-3B + LoRA r16/a32, projector_hidden_dim 1024, k=4, 10 epochs,
    batch_size 8 x grad_accum 4 = effective batch 32 (grad-accum reproduces the batch-32
    gradient exactly — no batchnorm in ProxyModel; needed because long generations hit
    max_seq_len=256 and OOM a true batch-32 forward). 3 seeds.
  * SE label z-scored PER TRAIN MODEL (train-only mean/std) before pooling — there is no
    single absolute SE scale across models (E35/E47).

Evaluation (per held-out model h, on h's OWN n2000 test split = 200 rows, E37-consistent):
    proxy (q_resp_only)          label-free on h  — this experiment's predictor
    true_semantic_entropy        the 10-sample CAE label (sampling upper bound)
    sep_single_val_selected      supervised in-model SEP, leak-free val-selected (h has no
                                 E41 CV layer yet, so val-selection — same as the E51/E53
                                 Qwen/Gemma convention)
    ridge_own_model              supervised linear ceiling: ridge on h's OWN TBG+SLT hidden
                                 states (fit tr, eval te) — context, NOT a fair opponent
  metrics: Spearman(pred, SE), AUROC_incorrect, AUROC_binarised_SE; paired bootstrap of
  Δ AUROC_incorrect vs SEP and vs true SE (shared resample indices).

Big-tier Stage-1 dirs use mixed suffixes: Qwen 27B -> "_nothink" (E55), gemma 27B -> "_full".
Handled by SUFFIX below; nothing else in the pipeline needs to know.

Envs:
  --stage check          se_probes / CPU  — is all 5 models' n2000 training data on disk?
  --stage train | eval   amortized_stage2 + a free GPU (the proxy backbone needs transformers 4.52)
  --stage all            train then eval
  --stage push_wandb     se_probes / CPU  — push the 15 checkpoints as a W&B artifact

Run from the repo root:
    python -m amortized_ue.e65_bigtier_lolo --stage check
    python -m amortized_ue.e65_bigtier_lolo --stage all --data_dir /data2/mn1025/stage1
    python -m amortized_ue.e65_bigtier_lolo --stage push_wandb
"""
from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np
from scipy.stats import spearmanr

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import load_matrix, splits, rho

# ------------------------------------------------------------------ config ----
BIGTIER = ["Qwen3.5-27B", "Qwen3.6-27B", "Qwen3.8-27B", "gemma-2-27b-it", "gemma-3-27b-it"]
SUFFIX = {"Qwen3.5-27B": "nothink", "Qwen3.6-27B": "nothink", "Qwen3.8-27B": "nothink",
          "gemma-2-27b-it": "full", "gemma-3-27b-it": "full"}
ARM = "q_resp_only"
TRAIN_N = 2000
SEEDS = [0, 1, 2]

DEFAULT_DATA_DIR = "/data2/mn1025/stage1"
_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT_ROOT = os.path.join(_HERE, "stage2", "runs", "E65_bigtier_lolo_qresp", "checkpoints")
RESULTS_DIR = os.path.join(_HERE, "results")
OUT_MAIN = os.path.join(RESULTS_DIR, "e65_bigtier_lolo.json")
OUT_MAIN_N1000 = os.path.join(RESULTS_DIR, "e65_bigtier_lolo_n1000.json")   # the correct eval (shared-ID n1000)
OUT_CURVES = os.path.join(RESULTS_DIR, "e65_bigtier_lolo_train_curves.json")
WANDB_ARTIFACT = "stage2_ckpts_E65_bigtier_lolo_qresp"


def run_name(model, n):
    return f"{model}_trivia_qa_n{n}_{SUFFIX[model]}"


def s1cfg(model, n, data_dir):
    return Stage1Config(model_name=model, dataset="trivia_qa", num_samples=n,
                        output_dir=data_dir, run_name=run_name(model, n))


def fold_ckpt_dir(held):
    return os.path.join(DEFAULT_CKPT_ROOT, held)


# ------------------------------------------------------------------- check ----
def do_check(data_dir, n=TRAIN_N, verbose=True, require_manifest=False):
    """True iff every big-tier model has a complete n{n} trivia record set on disk.
    n=TRAIN_N (2000): the training data.  n=1000: the shared-ID eval set (the correct E65).
    require_manifest: also demand the write-once manifest.json (the eval path reads it via
    load_accuracy, so a .pt-count-only gate could fire in the gap before it is written)."""
    ok = True
    rows = []
    for m in BIGTIER:
        cfg = s1cfg(m, n, data_dir)
        rd = cfg.records_dir()
        n_pt = len(glob.glob(os.path.join(rd, "*.pt"))) if os.path.isdir(rd) else 0
        man = cfg.manifest_path()
        n_man = None
        if os.path.isfile(man):
            with open(man) as f:
                d = json.load(f)
            n_man = len(d.get("records", {}))
        done = n_pt >= n and (not require_manifest or (n_man is not None and n_man >= n))
        ok &= done
        rows.append((m, run_name(m, n), n_pt, n_man, done))
    if verbose:
        print(f"{'model':16s}{'run_name':40s}{'n_pt':>7s}{'n_manifest':>12s}   ready")
        for m, rn, n_pt, n_man, done in rows:
            print(f"{m:16s}{rn:40s}{n_pt:>7d}{str(n_man):>12s}   {'YES' if done else 'no'}")
        print(f"\nALL 5 READY (n{n}): {ok}")
    return ok


# ------------------------------------------------------------------- train ----
def load_pool(held, data_dir):
    """Pool (question, canonical response, per-model TRAIN-z-scored SE) train/val rows from the
    4 non-held-out big-tier models. Mirrors e63_lto_deepseek_qwen3_8b.load_pool exactly:
    per-model splits() on that model's sorted-id order; SE z-scored with TRAIN-ONLY mean/std
    applied to both tr and va (no val leakage into the normaliser)."""
    sources = [m for m in BIGTIER if m != held]
    ptr = {"q": [], "r": [], "y": []}
    pva = {"q": [], "r": [], "y": []}
    stats = {}
    for m in sources:
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
        print(f"    {m:16s} n={len(ids)} tr={len(tr)} va={len(va)} mean_CAE(train)={mu:.3f}")
    train = {"y": np.array(ptr["y"], dtype=np.float32), "q": ptr["q"], "r": ptr["r"]}
    val = {"y": np.array(pva["y"], dtype=np.float32), "q": pva["q"], "r": pva["r"]}
    return sources, train, val, stats


def do_train(data_dir, seeds, batch_size, grad_accum):
    import torch
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z, _arm_text
    from amortized_ue.exp2_run import train_arm

    cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4, epochs=10,
                       batch_size=batch_size, grad_accum=grad_accum)
    model = ProxyModel(cfg, h_in=1).to("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_curves = {}
    if os.path.isfile(OUT_CURVES):
        with open(OUT_CURVES) as f:
            all_curves = json.load(f).get("folds", {})

    for held in BIGTIER:
        ck = fold_ckpt_dir(held)
        done = sorted(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt")))
        if len(done) >= len(seeds):
            print(f"\n[{held}] {len(done)} checkpoints already present -> skip")
            continue
        print(f"\n{'=' * 80}\n[fold] held out {held}  (train on the other 4)\n{'=' * 80}")
        sources, train, val, stats = load_pool(held, data_dir)
        print(f"  pooled: train rows={len(train['y'])}  val rows={len(val['y'])}")

        train["z"] = np.zeros((len(train["y"]), 1), dtype=np.float32)   # q_resp_only never reads z
        val["z"] = np.zeros((len(val["y"]), 1), dtype=np.float32)
        tgt = dict(val)                                                 # in-dist val-pool sanity target

        os.makedirs(ck, exist_ok=True)
        res = train_arm(train, val, tgt, ARM, seeds, cfg, model, torch, nn,
                        get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                        ckpt_dir=ck, tag=f"held-{held}")
        print(f"  val-pool sanity Spearman per seed: {[round(s, 3) for s in res['te_spearman']]}")
        all_curves[held] = {"sources": sources, "seeds": list(seeds),
                            "per_model_stats": stats, "train_config": cfg.as_dict(),
                            "n_train": len(train["y"]), "n_val": len(val["y"]),
                            "val_pool_sanity_spearman_by_seed": res["te_spearman"],
                            "val_pool_pred_by_seed": [[float(v) for v in p] for p in res["te_pred_by_seed"]],
                            "val_pool_y": [float(v) for v in tgt["y"]],
                            "curves_by_seed": res["curves_by_seed"]}
        with open(OUT_CURVES, "w") as f:
            json.dump({"arm": ARM, "bigtier": BIGTIER, "data_dir": data_dir, "folds": all_curves}, f)
        print(f"  curves -> {OUT_CURVES}")


# -------------------------------------------------------------------- eval ----
def boot_ci(fn, n, B=10000, seed=0):
    rng = np.random.default_rng(seed)
    v = np.array([fn(rng.integers(0, n, n)) for _ in range(B)])
    return {"mean": float(v.mean()), "lo95": float(np.percentile(v, 2.5)),
            "hi95": float(np.percentile(v, 97.5))}


def do_eval(data_dir, bootstrap, eval_n=TRAIN_N):
    """eval_n == TRAIN_N (2000, default): the preliminary E65 — each held-out model scored on its
    OWN n2000 test split (200 rows), written to OUT_MAIN. This reproduces the committed result.

    eval_n != TRAIN_N (use 1000): THE CORRECT E65 — the proxy is scored on the held-out model's
    FULL shared-ID n{eval_n} trivia set (all rows, disjoint from every training pool: n2000 ∩
    n1000 = 0, verified), 5x the power + an exact cross-model comparison. The supervised SEP /
    ridge baselines are FIT on the held-out model's own n2000 train/val and PREDICTED onto the
    n{eval_n} rows (leak-free by the same disjointness). Written to OUT_MAIN_N1000."""
    import torch  # noqa: F401  (arm_preds needs it)
    from sklearn.metrics import roc_auc_score
    from amortized_ue.procrustes_e27_rank_fusion import arm_preds
    from amortized_ue.stage2.data import best_split, binarize_entropy
    from amortized_ue.correctness_eval import (
        load_accuracy, sep_single_val_selected, paired_bootstrap_auc, ci)
    from amortized_ue.linear_ceiling_probe import fit_probe

    out_path = OUT_MAIN if eval_n == TRAIN_N else OUT_MAIN_N1000
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = {}
    if os.path.isfile(out_path):
        with open(out_path) as f:
            out = json.load(f)

    for held in BIGTIER:
        ck = fold_ckpt_dir(held)
        if len(glob.glob(os.path.join(ck, f"*{ARM}_seed*.pt"))) < len(SEEDS):
            print(f"[{held}] checkpoints missing -> skip (run --stage train first)")
            continue
        print(f"\n{'#' * 88}\n# E65 LOLO — held out {held}  (proxy trained on the other 4 big-tier)"
              f"  [eval on own n{eval_n}{' te-split' if eval_n == TRAIN_N else ' FULL shared-ID set'}]\n{'#' * 88}")

        # baselines always FIT on the held-out model's own n2000 train/val
        fit_cfg = s1cfg(held, TRAIN_N, data_dir)
        fit_recs = load_records(fit_cfg)
        fit_ids = sorted(fit_recs.keys())
        tr, va, te = splits(len(fit_ids))

        # rows the proxy + every baseline are SCORED on
        if eval_n == TRAIN_N:
            eval_cfg, eval_recs, eval_ids = fit_cfg, fit_recs, [fit_ids[i] for i in te]
        else:
            eval_cfg = s1cfg(held, eval_n, data_dir)
            eval_recs = load_records(eval_cfg)
            eval_ids = sorted(eval_recs.keys())
            assert not (set(fit_ids) & set(eval_ids)), \
                f"{held}: n{TRAIN_N} train pool overlaps the n{eval_n} eval set — not leak-free"

        se_eval = np.array([eval_recs[i]["labels"]["cluster_assignment_entropy"] for i in eval_ids], dtype=float)
        acc_map = load_accuracy(eval_cfg)
        acc = np.array([acc_map[i] for i in eval_ids], dtype=float)
        incorrect = (acc < 0.5).astype(int)
        pos_rate = float(incorrect.mean())

        # ---- proxy (q_resp_only), 3 seeds seed-averaged, on the scored rows -------------------
        mp = arm_preds(ARM, held, "trivia_qa", eval_n, ckpt_dir=ck, data_dir=data_dir,
                       run_name=run_name(held, eval_n))
        proxy_te = np.array([mp[i] for i in eval_ids], dtype=float)

        # ---- supervised in-model SEP (leak-free val-selected) + ridge ceiling on own states ---
        # fit hidden = the held-out model's n2000 (tr/va); eval hidden = the scored rows.
        hid_fit, y_fit, ids_fit = load_matrix(fit_cfg, ["TBG", "SLT"])
        assert ids_fit == fit_ids, "load_matrix id order != manifest order (fit)"
        if eval_n == TRAIN_N:
            hid_eval, y_eval_mat, eval_rows = hid_fit, y_fit, te
        else:
            hid_eval, y_eval_mat, ids_eval = load_matrix(eval_cfg, ["TBG", "SLT"])
            assert ids_eval == eval_ids, "load_matrix id order != manifest order (eval)"
            eval_rows = np.arange(len(eval_ids))
        assert float(np.max(np.abs(y_eval_mat[eval_rows].astype(float) - se_eval))) < 1e-5, "eval SE mismatch"

        sep_p, sep_au_se, sep_choice, thr, ybe, _ = sep_single_val_selected(
            hid_fit, y_fit, tr, va, hid_eval, y_eval_mat, eval_rows)
        sep_te = sep_p[eval_rows]

        # supervised linear ceiling (context, NOT a fair opponent): per-(pos,layer) ridge on
        # the held-out model's OWN hidden state (canonical linear_ceiling_probe method:
        # StandardScaler + Ridge, alpha on va), the layer picked LEAK-FREE by val Spearman,
        # prediction reported on te.
        rbest = (-np.inf, None, None)   # (val_spearman, (pos,layer,alpha), eval_pred)
        for pos in ("TBG", "SLT"):
            for L in range(hid_fit[pos].shape[0]):
                m, sc, alpha, val_s = fit_probe(hid_fit[pos][L], y_fit.astype(float), tr, va)
                if val_s > rbest[0]:
                    rbest = (val_s, (pos, int(L), float(alpha)),
                             m.predict(sc.transform(hid_eval[pos][L][eval_rows])))
        ridge_te = rbest[2]
        ridge_choice, ridge_val = rbest[1], float(rbest[0])
        del hid_fit
        if eval_n != TRAIN_N:
            del hid_eval

        # ---- score ---------------------------------------------------------------------------
        yb_te = ybe[eval_rows]
        v = yb_te >= 0
        preds = {"proxy_q_resp_only": proxy_te, "true_semantic_entropy": se_eval,
                 "sep_single_val_selected": sep_te, "ridge_own_model_TBG": ridge_te}
        label_free = {"proxy_q_resp_only": True, "true_semantic_entropy": False,
                      "sep_single_val_selected": False, "ridge_own_model_TBG": False}
        metrics = {}
        print(f"  {'predictor':26s}{'AUROC_inc':>11s}{'AUROC_SE':>10s}{'rho_SE':>9s}  label-free")
        for name, s in preds.items():
            au_inc = float(roc_auc_score(incorrect, s)) if len(np.unique(incorrect)) == 2 else float("nan")
            au_se = float(roc_auc_score(yb_te[v], s[v])) if len(np.unique(yb_te[v])) == 2 else float("nan")
            metrics[name] = {"auroc_incorrect": au_inc, "auroc_binarised_se": au_se,
                             "spearman_se": rho(s, se_eval), "label_free_on_target": label_free[name]}
            print(f"  {name:26s}{au_inc:>11.3f}{au_se:>10.3f}{rho(s, se_eval):>9.3f}"
                  f"  {'yes' if label_free[name] else 'NO'}")

        boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
        vs = {}
        for base in ("sep_single_val_selected", "true_semantic_entropy"):
            c = ci(boot["proxy_q_resp_only"] - boot[base])
            vs[base] = {**c, "ci_excludes_zero": bool(c["lo95"] > 0 or c["hi95"] < 0)}
            print(f"  Δ AUROC_inc (proxy − {base}): {c['mean']:+.3f} "
                  f"[{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
                  f"({'excludes 0' if vs[base]['ci_excludes_zero'] else 'includes 0'})")

        out[held] = {
            "held_out": held, "sources": [m for m in BIGTIER if m != held],
            "eval_n": eval_n, "eval_set": f"n{eval_n}" + ("_te_split" if eval_n == TRAIN_N else "_full_shared_id"),
            "n_test": len(eval_ids), "positive_rate_incorrect": pos_rate,
            "mean_accuracy": float(acc.mean()), "best_split": float(thr),
            "sep_choice": list(sep_choice), "sep_auroc_vs_se": float(sep_au_se),
            "ridge_choice_pos_layer_alpha": list(ridge_choice), "ridge_val_spearman": ridge_val,
            "bootstrap_resamples": bootstrap, "metrics": metrics,
            "bootstrap_delta_auroc_incorrect_vs": vs,
            "proxy_te_preds": [float(x) for x in proxy_te],
            "true_se_te": [float(x) for x in se_eval],
            "sep_te_preds": [float(x) for x in sep_te],
            "incorrect_te": [int(x) for x in incorrect],
            "te_ids": list(eval_ids),
        }
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"  -> saved {len([k for k in out if not k.startswith('_')])} fold(s) to {out_path}")

    # ---- cross-fold summary ----------------------------------------------------------------
    folds = [k for k in out if not k.startswith("_")]
    if folds:
        names = list(out[folds[0]]["metrics"])
        print("\n" + "=" * 88)
        print("SUMMARY — AUROC_incorrect per held-out big-tier model")
        print("=" * 88)
        print(f"{'predictor':26s}" + "".join(f"{h[:11]:>12s}" for h in folds) + f"{'MEAN':>9s}")
        summary = {}
        for n in names:
            vals = [out[h]["metrics"][n]["auroc_incorrect"] for h in folds]
            summary[n] = {"per_fold": {h: out[h]["metrics"][n]["auroc_incorrect"] for h in folds},
                          "mean_auroc_incorrect": float(np.nanmean(vals)),
                          "mean_auroc_se": float(np.nanmean([out[h]["metrics"][n]["auroc_binarised_se"] for h in folds])),
                          "mean_spearman_se": float(np.nanmean([out[h]["metrics"][n]["spearman_se"] for h in folds])),
                          "label_free_on_target": all(out[h]["metrics"][n]["label_free_on_target"] for h in folds)}
            print(f"{n:26s}" + "".join(f"{x:>12.3f}" for x in vals) + f"{np.nanmean(vals):>9.3f}")
        out["_summary"] = summary
        with open(out_path, "w") as f:
            json.dump(out, f, indent=1)
        print(f"\nwrote {out_path}")


# ---------------------------------------------------------------- wandb push ---
def do_push_wandb():
    import wandb
    paths = sorted(glob.glob(os.path.join(DEFAULT_CKPT_ROOT, "*", f"*{ARM}_seed*.pt")))
    assert len(paths) == len(BIGTIER) * len(SEEDS), \
        f"expected {len(BIGTIER) * len(SEEDS)} checkpoints, found {len(paths)}"
    run = wandb.init(project="amortized_ue_stage2", entity=os.environ.get("WANDB_ENT"),
                     name=WANDB_ARTIFACT, job_type="checkpoint",
                     config={"arm": ARM, "bigtier": BIGTIER, "design": "leave-one-LLM-out, 5 folds",
                             "recipe": "q_resp_only, 3 seeds, batch 8 x grad_accum 4 (eff 32), "
                                       "projector_hidden_dim 1024, k=4, 10 epochs"})
    art = wandb.Artifact(WANDB_ARTIFACT, type="model",
                         metadata={"bigtier": BIGTIER, "arm": ARM, "n_folds": len(BIGTIER),
                                   "n_seeds": len(SEEDS), "proxy_model": "meta-llama/Llama-3.2-3B"})
    art.add_dir(DEFAULT_CKPT_ROOT)
    run.log_artifact(art)
    run.finish()
    api = wandb.Api()
    a = api.artifact(f"{os.environ['WANDB_ENT']}/amortized_ue_stage2/{WANDB_ARTIFACT}:latest")
    print(f"pushed + verified {WANDB_ARTIFACT}:{a.version}  size={a.size} bytes  n_files={len(list(a.files()))}")


# -------------------------------------------------------------------- main ----
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["check", "check_eval", "train", "eval", "all", "push_wandb"], default="all")
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=4)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--eval_n", type=int, default=TRAIN_N,
                   help="rows to score the proxy on: TRAIN_N (2000, default) = own n2000 te-split "
                        "(the preliminary E65); 1000 = full shared-ID n1000 set (the correct E65).")
    args = p.parse_args()

    if args.stage == "check":
        raise SystemExit(0 if do_check(args.data_dir) else 1)
    if args.stage == "check_eval":                       # are the n{eval_n} shared-ID eval sets on disk?
        raise SystemExit(0 if do_check(args.data_dir, n=args.eval_n, require_manifest=True) else 1)
    if args.stage == "push_wandb":
        do_push_wandb()
        return
    if args.stage in ("train", "all"):
        if not do_check(args.data_dir):
            raise SystemExit("STOP: not all 5 big-tier n2000 datasets are ready (see table above).")
        do_train(args.data_dir, args.seeds, args.batch_size, args.grad_accum)
    if args.stage in ("eval", "all"):
        if args.eval_n != TRAIN_N and not do_check(args.data_dir, n=args.eval_n, require_manifest=True):
            raise SystemExit(f"STOP: not all 5 big-tier n{args.eval_n} eval sets are ready (see table above).")
        do_eval(args.data_dir, args.bootstrap, eval_n=args.eval_n)


if __name__ == "__main__":
    main()
