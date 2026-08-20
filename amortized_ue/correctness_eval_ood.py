"""OOD (cross-DATASET) correctness eval: trivia-fit predictors -> squad. (E39 — strictly additive.)

E38 showed the E37 proxy's text arm is on par with 10-sample sampling at detecting wrong answers --
but only IN-DISTRIBUTION (trivia_qa, the dataset everything was fit on). This asks the harder
question: **under a dataset shift, how well does the proxy detect wrong answers vs SE and SEP?**

EVERYTHING is fit/trained on trivia_qa and evaluated on squad n1000 (all 1000 rows). squad is a real
shift: mean accuracy 0.24 vs trivia 0.65 (incorrect rate 0.77 vs 0.35) and mean CAE 1.50 vs 0.59.

⚠️ WHICH PROXY. E37's LOLO run saved NO checkpoints (`--ckpt_dir` was never passed), so the
leave-one-LLM-out proxy cannot be run on new data without retraining. The two proxies that DO exist
answer complementary questions, and both are reported:
  * DEPLOY   (`results/deploy_checkpoints/`, 3 seeds) -- trained on ALL 4 models' trivia. On squad this
    is a pure **cross-DATASET** test; the target model WAS in the training pool, so it is NOT
    cross-LLM and NOT label-free w.r.t. the target. This is the deployable artifact, and it is the
    fair peer of SEP (both saw the target's own trivia labels; both are tested on squad).
  * REFERENCE (`runs/REFERENCE_multipos_p1024_5arm_ckpt/`, 5 seeds) -- Llama-2-only proxy. For the
    **Mistral** target this is **cross-LLM AND cross-dataset, fully label-free** -- the strict thesis
    test. Only its TEXT arms are run: its z arm is in Llama-2's native basis, which E20-E23 showed is
    at chance on a model swap (it needs Procrustes W, which the text arms do not).

Baselines, all trivia-fit -> squad:
  true_semantic_entropy   squad's own 10-sample CAE (sampling upper bound; needs N generations)
  sep_single/sep_5layer   supervised in-model SEP, fit on the target's trivia n2000 (layer chosen on
                          trivia val), applied to squad -- the standard cross-dataset SEP test
  ridge_z                 E37's pooled 3-source aligned-z ridge, trivia-fit, applied to squad z
                          aligned with the target's trivia-fit W (E27: the trivia-fit W transfers
                          cross-domain)

squad exists for Llama-2 and Mistral ONLY (not Llama-3 / DeepSeek), so this is a 2-target study.

Env: `amortized_stage2` + a free GPU (the proxy forward passes). Run from the repo root:
    python -m amortized_ue.correctness_eval_ood
"""
from __future__ import annotations

import os
import glob
import json
import argparse

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score, average_precision_score

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits
from amortized_ue.stage2.data import best_split, binarize_entropy
from amortized_ue.correctness_eval import (
    load_accuracy, sep_single_val_selected, sep_single_fixed_layer, sep_5layer_concat,
    accuracy_coverage, prediction_rejection_ratio, paired_bootstrap_auc, ci, COVERAGES)
from amortized_ue import exp2_run as E2

OOD_TARGETS = ["Llama-2-7b-chat", "Mistral-7B-Instruct-v0.2"]      # the only two with squad records
DEPLOY_CKPTS = "amortized_ue/results/deploy_checkpoints"
# sep_single_e36_layer is PRIMARY (E41): sep_single_best_layer re-selects the layer on trivia val,
# which is high-variance on Llama-2's near-tied late-TBG band (picked TBG:21, matching E38's bug).
# Both reported so the correction is auditable.
BASELINES = ["sep_single_e36_layer", "sep_single_best_layer", "true_semantic_entropy"]


# ---- squad z in the trivia-fitted aligned frame (label-free) ---------------------------------
def squad_aligned_z(target, al, fsc, layer):
    """Load squad hidden states at the target's best TBG layer and push them through the SAME
    trivia-fitted Procrustes W + per-model scaler the proxy/ridge were trained with."""
    from amortized_ue.loaders import load_records
    recs = load_records(Stage1Config(model_name=target, dataset="squad", num_samples=1000))
    ids = sorted(recs.keys())
    H = np.stack([recs[i]["canonical"]["hidden_states"]["TBG"][layer].squeeze().float().numpy()
                  for i in ids]).astype(np.float32)
    q = [recs[i]["question"] for i in ids]
    r = [recs[i]["canonical"]["response"] for i in ids]
    y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
    mean_m, W = al[target]
    z = fsc[target].transform((H - mean_m) @ W).astype(np.float32)
    return {"z": z, "q": q, "r": r, "y": y, "ids": ids}


# ---- compat loader for exp2_run checkpoints ---------------------------------------------------
def _load_exp2_ckpt(path, model=None, device=None):
    """Load an `exp2_run.py` checkpoint. Those files carry the checkpoint/v1 format tag but omit the
    `k` / `transform` meta keys, so stage2.checkpoint.load_checkpoint KeyErrors on them (fixed in
    exp2_run for future runs; this reads the ones already on disk). `k` is recovered from the stored
    config. No transform is applied: exp2 targets are z-scored per model upstream, and every metric
    here is rank-based, so the standardized output is used directly."""
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    import dataclasses
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(path, map_location="cpu")
    meta = ckpt["meta"]
    fields = {f.name for f in dataclasses.fields(Stage2Config)}
    cfg = Stage2Config(**{k: v for k, v in meta["config"].items() if k in fields})
    if model is None:
        model = ProxyModel(cfg, h_in=meta["h_in"])
    model.set_k(meta.get("k", cfg.k_soft_tokens))
    model.to(device)
    own = dict(model.named_parameters())
    missing = [n for n in ckpt["trainable"] if n not in own]
    if missing:
        raise RuntimeError(f"checkpoint has params not in the model: {missing[:5]} ...")
    with torch.no_grad():
        for n, t in ckpt["trainable"].items():
            own[n].copy_(t.to(own[n].device, own[n].dtype))
    return model


# ---- run a saved proxy checkpoint set over a dataset ------------------------------------------
def proxy_preds(ckpt_paths, arm, d, model_holder, batch_size=32, max_seq_len=256):
    """Mean prediction over the checkpoint set (seed ensemble). Ranking metrics only, so the
    per-model label z-scoring is left undecoded (a monotone transform)."""
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z
    per_seed = []
    for p in ckpt_paths:
        model = _load_exp2_ckpt(p, model_holder.get("m"))
        model_holder["m"] = model
        model.eval()
        dev = next(model.parameters()).device
        zt = torch.from_numpy(d["z"]).float().unsqueeze(1) if _arm_uses_z(arm) else None
        out = []
        with torch.no_grad():
            for i in range(0, len(d["y"]), batch_size):
                rows = list(range(i, min(i + batch_size, len(d["y"]))))
                z = zt[rows].to(dev) if zt is not None else None
                ids, attn = _tokenize_arm(model.tokenizer, [d["q"][j] for j in rows],
                                          [d["r"][j] for j in rows], arm, max_seq_len)
                if ids is not None:
                    ids, attn = ids.to(dev), attn.to(dev)
                out.append(model(z, ids, attn).float().cpu())
        per_seed.append(torch.cat(out).numpy())
    return np.mean(per_seed, 0), np.stack(per_seed)


def evaluate_target(target, data, sp, al, fsc, anchor_layer, bootstrap=10000,
                    trivia_dir=None, skip_proxy=False):
    print("\n" + "#" * 92)
    print(f"# OOD CORRECTNESS — target {target}:  fit on trivia_qa n2000  ->  eval on squad n1000")
    print("#" * 92)

    layer = anchor_layer if target == E2.ANCHOR else E2.BEST_TBG[target]
    sq = squad_aligned_z(target, al, fsc, layer)
    n = len(sq["ids"])

    acc_cfg = Stage1Config(model_name=target, dataset="squad", num_samples=1000)
    acc_map = load_accuracy(acc_cfg)
    assert set(sq["ids"]).issubset(acc_map), "squad ids missing from the accuracy manifest"
    acc = np.array([acc_map[i] for i in sq["ids"]], dtype=float)
    correct = (acc >= 0.5).astype(int)
    incorrect = 1 - correct
    pos_rate = float(incorrect.mean())
    print(f"  squad N={n}  mean_acc={acc.mean():.3f}  incorrect_rate={pos_rate:.3f}  "
          f"(trivia mean_acc ~0.65 -- this is a real shift)")

    preds, provenance = {}, {}
    preds["true_semantic_entropy"] = sq["y"].astype(float)
    provenance["true_semantic_entropy"] = "squad 10-sample CAE (sampling upper bound)"

    # ---- SEP: fit on the target's trivia n2000 (layer picked on trivia val) -> apply to squad ----
    tcfg = Stage1Config(model_name=target, dataset="trivia_qa", num_samples=2000,
                        **({"output_dir": trivia_dir} if trivia_dir else {}))
    thid, ty, tids = load_matrix(tcfg, ["TBG", "SLT"])
    tr, va, te = splits(len(tids))
    scfg = Stage1Config(model_name=target, dataset="squad", num_samples=1000)
    shid, sy, sids = load_matrix(scfg, ["TBG", "SLT"])
    assert sids == sq["ids"], "squad id order differs between load_matrix and the record loader"
    assert float(np.max(np.abs(sy.astype(float) - sq["y"].astype(float)))) < 1e-5, \
        "squad SE labels differ between the two load paths"
    eval_rows = np.arange(len(sids))
    sep1_p, sep1_au_se, sep1_choice, thr, ybe, _ = sep_single_val_selected(
        thid, ty, tr, va, shid, sy, eval_rows)
    # E41: same probe at E36's leak-free CV-selected layer (never re-selected on trivia val here)
    e36_layer = layer                                # == E2.BEST_TBG[target] / anchor_layer above
    sepf_p, sepf_au_se, sepf_choice, _, _ = sep_single_fixed_layer(
        thid, ty, tr, va, shid, sy, eval_rows, "TBG", e36_layer)
    sep5_p, sep5_au_se, sep5_choice = sep_5layer_concat(thid, ty, tr, va, shid, sy, eval_rows)
    preds["sep_single_best_layer"] = sep1_p
    preds["sep_single_e36_layer"] = sepf_p
    preds["sep_5layer_concat"] = sep5_p
    provenance["sep_single_best_layer"] = f"trivia-fit {sep1_choice} -> squad (supervised, in-model)"
    provenance["sep_single_e36_layer"] = f"trivia-fit {sepf_choice} -> squad (supervised, in-model, E36 layer)"
    provenance["sep_5layer_concat"] = f"trivia-fit {sep5_choice[0]} -> squad (supervised, in-model)"
    yb_eval = ybe                                   # best_split refit convention: see note below
    del thid, shid
    print(f"  SEP single val-selected {sep1_choice} squad SE-AUROC {sep1_au_se:.3f} | "
          f"E36-fixed {sepf_choice} squad SE-AUROC {sepf_au_se:.3f} | "
          f"5-layer {sep5_choice[0]} L{sep5_choice[1][0]}-{sep5_choice[1][-1]} {sep5_au_se:.3f}")

    # ---- ridge_z: E37's pooled 3-source aligned ridge, trivia-fit -> squad ----------------------
    sources = [m for m in E2.MODELS if m != target]
    train, val, _, _ = E2.build_fold(data, sp, target, sources, al, fsc)
    best = None
    for a in E2.ALPHAS:
        r = Ridge(alpha=a).fit(train["z"], train["y"])
        s = E2.rho(r.predict(val["z"]), val["y"])
        if best is None or s > best[0]:
            best = (s, a, r)
    preds["ridge_z"] = best[2].predict(sq["z"])
    provenance["ridge_z"] = f"pooled 3-source aligned ridge (alpha={best[1]:g}), trivia-fit -> squad"
    print(f"  ridge_z: pooled sources {[E2.SHORT[m] for m in sources]}, alpha={best[1]:g}")

    per_seed_store = {}
    if not skip_proxy:
        holder = {}
        # DEPLOY proxy (all-4 trivia-trained): cross-DATASET, target WAS in the pool
        for arm in ["z", "z_q_resp", "q_only", "q_resp_only"]:
            paths = sorted(glob.glob(os.path.join(DEPLOY_CKPTS, f"deploy_{arm}_seed*.pt")))
            if not paths:
                print(f"  [WARN] no deploy checkpoints for arm {arm}")
                continue
            m, ps = proxy_preds(paths, arm, sq, holder)
            preds[f"deploy_{arm}"] = m
            per_seed_store[f"deploy_{arm}"] = ps
            provenance[f"deploy_{arm}"] = f"DEPLOY proxy ({len(paths)} seeds), all-4 trivia-trained -> squad"
            print(f"    deploy/{arm}: {len(paths)} seeds", flush=True)
        if "deploy_z" in preds and "deploy_q_resp_only" in preds:
            F = np.stack([E2.rank_fuse(per_seed_store["deploy_z"][s],
                                       per_seed_store["deploy_q_resp_only"][s])
                          for s in range(per_seed_store["deploy_z"].shape[0])])
            preds["deploy_fuse_z_qresp"] = F.mean(0)
            per_seed_store["deploy_fuse_z_qresp"] = F
            provenance["deploy_fuse_z_qresp"] = "rank fusion of deploy z + q_resp_only"

        # REFERENCE proxy (Llama-2-only): TEXT arms only. On Mistral = cross-LLM AND cross-dataset.
        try:
            from amortized_ue.procrustes_e27_rank_fusion import arm_preds
            for arm in ["q_only", "q_resp_only"]:
                mp = arm_preds(arm, target, "squad", 1000)
                preds[f"reference_{arm}"] = np.array([mp[i] for i in sq["ids"]], dtype=float)
                tag = "cross-LLM + cross-dataset, label-free" if target != E2.ANCHOR \
                    else "in-model, cross-dataset"
                provenance[f"reference_{arm}"] = f"REFERENCE Llama-2 proxy -> squad ({tag})"
                print(f"    reference/{arm}: done", flush=True)
        except Exception as e:
            print(f"  [WARN] REFERENCE arms skipped: {type(e).__name__}: {e}")

    preds["random_control"] = np.random.default_rng(0).random(n)
    provenance["random_control"] = "chance"

    # ---- score ---------------------------------------------------------------------------------
    # NOTE on AUROC_SE: the SE label scale shifts under the dataset swap (trivia mean CAE 0.59 vs
    # squad 1.50), so `best_split` is refit on the squad rows -- the documented OOD convention. The
    # CORRECTNESS target needs no such choice; it is the primary column here.
    metrics = {}
    print(f"\n  {'predictor':26s}{'AUROC_inc':>10s}{'AUPRC':>8s}{'PRR':>7s}{'acc@.90':>8s}"
          f"{'acc@.50':>8s}{'AUROC_SE':>10s}{'rho_SE':>8s}")
    for name, s in preds.items():
        s = np.asarray(s, dtype=float)
        au_inc = float(roc_auc_score(incorrect, s))
        ap_inc = float(average_precision_score(incorrect, s))
        pr = prediction_rejection_ratio(s, incorrect)
        cov = accuracy_coverage(s, correct)
        v = yb_eval >= 0
        au_se = float(roc_auc_score(yb_eval[v], s[v])) if len(np.unique(yb_eval[v])) == 2 else float("nan")
        m = {"auroc_incorrect": au_inc, "auprc_incorrect": ap_inc, "prr": pr,
             "accuracy_coverage": {str(c): cov[c] for c in COVERAGES},
             "auroc_binarised_se": au_se, "spearman_se": E2.rho(s, sq["y"]),
             "provenance": provenance[name]}
        if name in per_seed_store:
            aus = [float(roc_auc_score(incorrect, p)) for p in per_seed_store[name]]
            m["auroc_incorrect_per_seed"] = aus
            m["auroc_incorrect_seed_mean"] = float(np.mean(aus))
            m["auroc_incorrect_seed_std"] = float(np.std(aus))
        metrics[name] = m
        print(f"  {name:26s}{au_inc:>10.3f}{ap_inc:>8.3f}{pr:>7.3f}{cov[0.9]:>8.3f}"
              f"{cov[0.5]:>8.3f}{au_se:>10.3f}{m['spearman_se']:>8.3f}")

    boot = paired_bootstrap_auc(preds, incorrect, B=bootstrap)
    vs = {}
    print(f"\n  paired bootstrap (B={bootstrap}, shared indices) — Δ AUROC_incorrect  [N={n}]")
    for base in BASELINES:
        vs[base] = {}
        print(f"    vs {base}:")
        for name in preds:
            if name == base:
                continue
            c = ci(boot[name] - boot[base])
            excl = c["lo95"] > 0 or c["hi95"] < 0
            vs[base][name] = {**c, "ci_excludes_zero": bool(excl)}
            print(f"      Δ({name:24s}) {c['mean']:+.3f} [{c['lo95']:+.3f}, {c['hi95']:+.3f}] "
                  f"({'excludes 0' if excl else 'includes 0'})")

    return {"target": target, "eval_dataset": "squad", "fit_dataset": "trivia_qa", "n_test": n,
            "mean_accuracy": float(acc.mean()), "positive_rate_incorrect": pos_rate,
            "sep_single_choice": list(sep1_choice), "sep_single_auroc_vs_se": sep1_au_se,
            "sep_e36_layer_choice": list(sepf_choice), "sep_e36_layer_auroc_vs_se": sepf_au_se,
            "sep_5layer_auroc_vs_se": sep5_au_se, "ridge_alpha": float(best[1]),
            "best_split_refit_on_squad": float(thr), "bootstrap_resamples": bootstrap,
            "metrics": metrics, "bootstrap_auroc_incorrect": vs}


def main():
    p = argparse.ArgumentParser(description="OOD (trivia->squad) correctness eval. Additive.")
    p.add_argument("--targets", nargs="+", default=OOD_TARGETS, choices=OOD_TARGETS)
    p.add_argument("--anchor_layer", type=int, default=30)
    p.add_argument("--bootstrap", type=int, default=10000)
    p.add_argument("--trivia_dir", default=None,
                   help="output_dir override for the trivia n2000 loads only (e.g. /data2/mn1025/stage1); squad always reads the default path")
    p.add_argument("--skip_proxy", action="store_true", help="baselines only (CPU, no GPU needed)")
    p.add_argument("--out", default="amortized_ue/results/correctness_eval_ood.json")
    args = p.parse_args()

    print("loading 4 models' trivia states (for alignment + the pooled ridge) ...")
    data, sp = E2.load_all(anchor_layer=args.anchor_layer, data_dir=args.trivia_dir)
    al, fsc = E2.fit_alignment(data, sp[0], args.anchor_layer)

    out = {}
    for t in args.targets:
        out[t] = evaluate_target(t, data, sp, al, fsc, args.anchor_layer,
                                 bootstrap=args.bootstrap, trivia_dir=args.trivia_dir,
                                 skip_proxy=args.skip_proxy)
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"    -> saved {len(out)} target(s) to {args.out}")

    print("\n" + "=" * 92)
    print("SUMMARY — OOD (trivia-fit -> squad) AUROC_incorrect")
    print("=" * 92)
    names = list(next(iter(out.values()))["metrics"])
    print(f"{'predictor':26s}" + "".join(f"{t.split('-7')[0][:9]:>11s}" for t in out) + f"{'MEAN':>9s}")
    for n in names:
        vals = [out[t]["metrics"][n]["auroc_incorrect"] for t in out if n in out[t]["metrics"]]
        print(f"{n:26s}" + "".join(f"{v:>11.3f}" for v in vals) + f"{np.mean(vals):>9.3f}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
