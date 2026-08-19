"""Exp-2 FULL driver — multi-fold LOO cross-LLM proxy: arms x source-count x targets, + late fusion.

Generalizes the VERIFIED exp2_step1_zarm.py (identical alignment / per-model normalization / SAME-
questions pooling — each question seen through all source models -> model-invariance signal). Adds:
  * multi-fold LOO   : hold out each of the 4 models in turn.
  * arms (all 5)     : z (aligned hidden state only), z_q (z+question), z_q_resp (z+question+response),
                       q_only (question text, no z), q_resp_only (question+response text, no z). Reuses
                       the reference arm/tokenizer VERBATIM (stage2.train._arm_text / _tokenize_arm /
                       _arm_uses_z). Text-only arms (q_only/q_resp_only) carry no z (model-agnostic;
                       the thesis pathway); z_q/z_q_resp are the EARLY-fusion comparison to late fusion.
  * source-count     : train on 1 / 2 / 3 source models (same-questions), to see if transfer climbs.
  * late fusion      : rank-fuse z + q_resp_only predictions (E27's winner; label-free empirical-CDF).

Leak-free by construction (same self-audit as step-1): sources train on their `tr`; early-stop on
pooled sources' `va`; the held-out target's `te` is used ONLY for the final Spearman. Target aligned
with its OWN label-free W_T; its SE labels are never used. Records id-joined.

Env: amortized_stage2 (GPU) for the real run; se_probes for --data_only. Reuses ProxyModel (Llama-3.2-3B).
    python -m amortized_ue.exp2_run --data_only                 # CPU: build + audit + ridge, all folds
    python -m amortized_ue.exp2_run --smoke                     # tiny end-to-end
    python -m amortized_ue.exp2_run --arms z q_only q_resp_only --seeds 0 1 2   # full
"""
from __future__ import annotations

import os
import time
import json
import argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr, rankdata
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import splits

ANCHOR = "Llama-2-7b-chat"
MODELS = [ANCHOR, "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "DeepSeek"}
BEST_TBG = {ANCHOR: 30, "Mistral-7B-Instruct-v0.2": 31, "Meta-Llama-3-8B-Instruct": 31,
            "deepseek-llm-7b-chat": 28}
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def rho(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


# ---- data: load once, keyed by model -----------------------------------------
def load_all(anchor_layer=30, smoke=False, data_dir=None):
    """Load all 4 models' best-TBG hidden state + question/response text + SE label, id-joined.

    Returns {model: dict(H[N,4096], q[list], r[list], y[N])}, and (tr, va, te). Uses the FULL records
    (load_records) to get text; the hidden state is the model's best TBG layer (anchor uses anchor_layer).
    data_dir overrides Stage1Config.output_dir (e.g. a node-local /data2 copy) to dodge NFS stalls."""
    data, ids0 = {}, None
    for m in MODELS:
        _kw = {"output_dir": data_dir} if data_dir else {}
        recs = load_records(Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000, **_kw))
        ids = sorted(recs.keys())
        assert ids0 is None or ids == ids0, f"id order differs for {m} (join-by-id broken)"
        ids0 = ids
        layer = anchor_layer if m == ANCHOR else BEST_TBG[m]
        H = np.stack([recs[i]["canonical"]["hidden_states"]["TBG"][layer].squeeze().float().numpy()
                      for i in ids]).astype(np.float32)                        # [N, 4096]
        q = [recs[i]["question"] for i in ids]
        r = [recs[i]["canonical"]["response"] for i in ids]
        y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
        data[m] = {"H": H, "q": q, "r": r, "y": y}
    tr, va, te = splits(len(ids0))
    if smoke:
        tr, va = tr[:120], va[:40]
    return data, (tr, va, te)


def fit_alignment(data, tr, anchor_layer):
    """Label-free Procrustes each model best-TBG -> anchor, + per-model feature scaler; fit on tr."""
    al, fsc = {}, {}
    Ac = data[ANCHOR]["H"][tr] - data[ANCHOR]["H"][tr].mean(0, keepdims=True)
    for m in MODELS:
        mean_m = data[m]["H"][tr].mean(0, keepdims=True)
        W = np.eye(data[m]["H"].shape[1]) if m == ANCHOR \
            else orthogonal_procrustes(data[m]["H"][tr] - mean_m, Ac)[0]
        al[m] = (mean_m, W)
        fsc[m] = StandardScaler().fit((data[m]["H"][tr] - mean_m) @ W)
    return al, fsc


def build_fold(data, splits_tuple, target, sources, al, fsc, n_questions=None, subset_seed=0):
    """Pooled train/val (SAME questions to all sources) + held-out target te. z + text + labels.

    Rows are (z_aligned, question, response, se). Per-model SE z-score on the routed rows (TRAIN).
    Target aligned with its own W_T + scaler (label-free); RAW te labels for Spearman only."""
    tr, va, te = splits_tuple

    def feat(m, idx):
        mean_m, W = al[m]
        return fsc[m].transform((data[m]["H"][idx] - mean_m) @ W).astype(np.float32)

    if n_questions is not None and n_questions < len(tr):
        rows = np.sort(np.random.default_rng(subset_seed).choice(tr, size=n_questions, replace=False))
    else:
        rows = tr
    src_rows = {m: rows for m in sources}                     # SAME questions to every source

    Ztr, ytr, Qtr, Rtr, Zva, yva, Qva, Rva = [], [], [], [], [], [], [], []
    for m in sources:
        rm = src_rows[m]
        mu, sd = data[m]["y"][rm].mean(), data[m]["y"][rm].std() + 1e-12
        Ztr.append(feat(m, rm)); ytr.append((data[m]["y"][rm] - mu) / sd)
        Qtr += [data[m]["q"][i] for i in rm]; Rtr += [data[m]["r"][i] for i in rm]
        Zva.append(feat(m, va)); yva.append((data[m]["y"][va] - mu) / sd)
        Qva += [data[m]["q"][i] for i in va]; Rva += [data[m]["r"][i] for i in va]
    train = {"z": np.vstack(Ztr), "y": np.concatenate(ytr).astype(np.float32), "q": Qtr, "r": Rtr}
    val = {"z": np.vstack(Zva), "y": np.concatenate(yva).astype(np.float32), "q": Qva, "r": Rva}
    tgt = {"z": feat(target, te), "y": data[target]["y"][te].astype(np.float32),
           "q": [data[target]["q"][i] for i in te], "r": [data[target]["r"][i] for i in te]}
    info = {"target": SHORT[target], "sources": [SHORT[m] for m in sources],
            "n_unique_q": len(rows), "n_train": len(train["y"]), "n_val": len(val["y"]),
            "n_te": len(tgt["y"])}
    return train, val, tgt, info


def build_deploy(data, splits_tuple, al, fsc, n_questions=None):
    """DEPLOYABLE build: pool ALL 4 models (no held-out) — the reusable proxy. Train on all tr,
    val on all va; the nominal eval target is the val pool (in-dist sanity — deploy's real output
    is the CHECKPOINT). Same alignment / per-model normalization as build_fold."""
    tr, va, te = splits_tuple

    def feat(m, idx):
        mean_m, W = al[m]
        return fsc[m].transform((data[m]["H"][idx] - mean_m) @ W).astype(np.float32)

    rows = tr if (n_questions is None or n_questions >= len(tr)) \
        else np.sort(np.random.default_rng(0).choice(tr, size=n_questions, replace=False))
    Ztr, ytr, Qtr, Rtr, Zva, yva, Qva, Rva = [], [], [], [], [], [], [], []
    for m in MODELS:
        mu, sd = data[m]["y"][rows].mean(), data[m]["y"][rows].std() + 1e-12
        Ztr.append(feat(m, rows)); ytr.append((data[m]["y"][rows] - mu) / sd)
        Qtr += [data[m]["q"][i] for i in rows]; Rtr += [data[m]["r"][i] for i in rows]
        Zva.append(feat(m, va)); yva.append((data[m]["y"][va] - mu) / sd)
        Qva += [data[m]["q"][i] for i in va]; Rva += [data[m]["r"][i] for i in va]
    train = {"z": np.vstack(Ztr), "y": np.concatenate(ytr).astype(np.float32), "q": Qtr, "r": Rtr}
    val = {"z": np.vstack(Zva), "y": np.concatenate(yva).astype(np.float32), "q": Qva, "r": Rva}
    tgt = dict(val)                                          # nominal eval = val pool (in-dist sanity)
    info = {"target": "DEPLOY-all4", "sources": [SHORT[m] for m in MODELS],
            "n_unique_q": len(rows), "n_train": len(train["y"]), "n_val": len(val["y"]),
            "n_te": len(tgt["y"])}
    return train, val, tgt, info


def ridge_on_z(train, val, tgt):
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(train["z"], train["y"])
        s = rho(r.predict(val["z"]), val["y"])
        if best is None or s > best[0]:
            best = (s, a, r)
    return rho(best[2].predict(tgt["z"]), tgt["y"]), best[1]


# ---- proxy training (arms) ----------------------------------------------------
def train_arm(train, val, tgt, arm, seeds, cfg, model, torch, nn, sched_fn,
              tokenize, uses_z, ckpt_dir=None, tag=""):
    """Train ProxyModel on one arm; return {'te_pred_by_seed': [...], 'te_spearman': [...]}."""
    dev = next(model.parameters()).device

    def zt(d):
        return torch.from_numpy(d["z"]).float().unsqueeze(1)          # [n,1,H]

    def batch_forward(rows, d, ztensor):
        z = ztensor[rows].to(dev) if uses_z(arm) else None
        ids, attn = tokenize(model.tokenizer, [d["q"][i] for i in rows], [d["r"][i] for i in rows],
                             arm, cfg.max_seq_len)
        if ids is not None:
            ids, attn = ids.to(dev), attn.to(dev)
        return model(z, ids, attn)

    ztr, zva, zte = zt(train), zt(val), zt(tgt)
    ytr = torch.from_numpy(train["y"]).float()
    yva = torch.from_numpy(val["y"]).float()

    @torch.no_grad()
    def predict(d, ztensor):
        model.eval(); out = []
        for i in range(0, len(d["y"]), cfg.batch_size):
            rows = list(range(i, min(i + cfg.batch_size, len(d["y"]))))
            out.append(batch_forward(rows, d, ztensor).float().cpu())
        model.train(); return torch.cat(out)

    te_preds, te_spear, curves_by_seed = [], [], []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.reinit_trainable()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        spe = max(1, int(np.ceil(len(ytr) / cfg.batch_size)))
        sched = sched_fn(opt, int(cfg.warmup_ratio * spe * cfg.epochs), spe * cfg.epochs)
        loss_fn = nn.MSELoss()
        best_val, best_state, best_epoch, patience = float("inf"), None, 0, 0
        rng = np.random.default_rng(seed)
        # Rich training log for the thesis: per-step loss/lr/grad-norm; per-epoch train-loss/val-mse/
        # val-spearman; plus meta (best epoch, wall-time, trainable params).
        steps_log, epochs_log = [], []
        gstep, t0 = 0, time.time()
        for ep in range(cfg.epochs):
            model.train()
            order = rng.permutation(len(ytr))
            ep_losses = []
            for i in range(0, len(order), cfg.batch_size):
                rows = order[i:i + cfg.batch_size].tolist()
                pred = batch_forward(rows, train, ztr)
                loss = loss_fn(pred, ytr[rows].to(dev))
                opt.zero_grad(); loss.backward()
                gn = torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)   # returns pre-clip total norm
                opt.step(); sched.step()
                lr = float(sched.get_last_lr()[0])
                steps_log.append({"step": gstep, "loss": float(loss.item()), "lr": lr, "grad_norm": float(gn)})
                ep_losses.append(float(loss.item())); gstep += 1
            val_pred = predict(val, zva)                       # reused for both val_mse and val_spearman
            vm = float(((val_pred - yva) ** 2).mean())
            vsp = rho(val_pred.numpy(), val["y"])
            epochs_log.append({"epoch": ep, "train_loss": sum(ep_losses) / len(ep_losses),
                               "val_mse": vm, "val_spearman": vsp, "lr": lr})
            if vm < best_val:
                best_val, best_epoch = vm, ep
                best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break
        train_log = {"steps": steps_log, "epochs": epochs_log, "best_epoch": best_epoch,
                     "n_epochs_run": len(epochs_log), "train_seconds": round(time.time() - t0, 1),
                     "n_train": len(ytr), "n_val": len(yva),
                     "trainable_params": int(sum(p.numel() for p in params))}
        if best_state is not None:
            with torch.no_grad():
                for k, v in model.named_parameters():
                    if v.requires_grad and k in best_state:
                        v.copy_(best_state[k])
        # SAVE CHECKPOINT (trainable params + meta) — enforces the always-save-checkpoints rule.
        if ckpt_dir is not None:
            from amortized_ue.stage2.checkpoint import save_checkpoint as _save
            os.makedirs(ckpt_dir, exist_ok=True)
            # `k` and `transform` are part of the checkpoint/v1 metadata contract -- without them
            # stage2.checkpoint.load_checkpoint KeyErrors on our own files (hit in E39). Targets here
            # are z-scored PER MODEL upstream in build_fold, so there is no single decode: predictions
            # are already in standardized space and the identity transform is the honest record.
            meta = {"arm": arm, "seed": int(seed), "tag": tag, "h_in": int(model.h_in),
                    "k": int(cfg.k_soft_tokens), "transform": {"mean": 0.0, "std": 1.0},
                    "proxy_model": cfg.proxy_model, "config": cfg.as_dict()}
            _save(model, meta, os.path.join(ckpt_dir, f"{tag}_{arm}_seed{seed}.pt"))
        p = predict(tgt, zte).numpy()
        te_preds.append(p); te_spear.append(rho(p, tgt["y"]))
        curves_by_seed.append(train_log)
    return {"te_pred_by_seed": te_preds, "te_spearman": te_spear, "curves_by_seed": curves_by_seed}


def rank_fuse(pred_a, pred_b):
    """Label-free empirical-CDF rank average of two prediction vectors (E27 late fusion)."""
    ra = rankdata(pred_a) / len(pred_a)
    rb = rankdata(pred_b) / len(pred_b)
    return (ra + rb) / 2.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", default=[m for m in MODELS if m != ANCHOR] + [ANCHOR])
    p.add_argument("--arms", nargs="+",
                   default=["z", "z_q", "z_q_resp", "q_only", "q_resp_only"])   # all 5; z_q/z_q_resp = early-fusion
    p.add_argument("--source_counts", type=int, nargs="+", default=[3])
    p.add_argument("--anchor_layer", type=int, default=30)
    p.add_argument("--n_questions", type=int, default=None)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data_only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--data_dir", default=None,
                   help="Stage-1 output_dir override, e.g. /data2/mn1025/stage1 (fast node-local copy)")
    p.add_argument("--out", default=None, help="results JSON path (default results/exp2_lolo_full.json)")
    p.add_argument("--ckpt_dir", default=None, help="save trained checkpoints here (always-save rule)")
    p.add_argument("--deploy", action="store_true",
                   help="train the DEPLOYABLE proxy on ALL 4 models (no held-out); saves checkpoints + curves")
    args = p.parse_args()

    print(f"loading 4 models (hidden + text + labels) from {args.data_dir or 'default (NFS)'} ...")
    data, sp = load_all(anchor_layer=args.anchor_layer, smoke=args.smoke, data_dir=args.data_dir)
    al, fsc = fit_alignment(data, sp[0], args.anchor_layer)

    if not args.data_only:
        import torch
        import torch.nn as nn
        from transformers import get_cosine_schedule_with_warmup
        from amortized_ue.stage2.config import Stage2Config
        from amortized_ue.stage2.model import ProxyModel
        from amortized_ue.stage2.train import _arm_text, _tokenize_arm, _arm_uses_z  # reuse verbatim
        cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4,
                           epochs=(2 if args.smoke else 10), batch_size=32)
        model = ProxyModel(cfg, h_in=data[ANCHOR]["H"].shape[1]).to("cuda" if torch.cuda.is_available() else "cpu")
    else:
        _arm_text = _tokenize_arm = _arm_uses_z = None

    print("\n" + "=" * 92)
    print("LEAKAGE AUDIT: sources train on tr; early-stop on pooled va; target te -> final Spearman only;")
    print("  W/scaler/label-stats on tr; target W_T label-free; same questions to all sources; id-joined.")
    print("=" * 92)

    res_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    out_path = args.out or os.path.join(res_dir, "exp2_lolo_full.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # ---- DEPLOYABLE run: train the reusable proxy on ALL 4 models; ALWAYS checkpoints + curves ------
    if args.deploy and not args.data_only:
        ckpt = args.ckpt_dir or os.path.join(res_dir, "deploy_checkpoints")
        train, val, tgt, info = build_deploy(data, sp, al, fsc, n_questions=args.n_questions)
        print(f"[DEPLOY] all-4 pool: rows={info['n_train']} val={info['n_val']} arms={args.arms} ckpt->{ckpt}",
              flush=True)
        deploy = {"info": info, "train_config": cfg.as_dict(), "seeds": list(args.seeds), "arms": {}}
        dpath = os.path.join(res_dir, "deploy_curves.json")
        for arm in args.arms:
            r = train_arm(train, val, tgt, arm, args.seeds, cfg, model, torch, nn,
                          get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                          ckpt_dir=ckpt, tag="deploy")
            deploy["arms"][arm] = {"val_spearman": r["te_spearman"], "curves_by_seed": r["curves_by_seed"]}
            with open(dpath, "w") as f:                          # save curves INCREMENTALLY per arm
                json.dump(deploy, f)
            print(f"    [DEPLOY] arm {arm}: val_spearman {np.mean(r['te_spearman']):.3f}  "
                  f"(ckpt + curves saved)", flush=True)
        print(f"[DEPLOY] DONE — {len(args.arms)} arms x {len(args.seeds)} seeds checkpointed to {ckpt}; "
              f"training curves in {dpath}", flush=True)
        return

    rows_out = []
    for target in args.targets:
        pool_all = [m for m in MODELS if m != target]
        for nsrc in args.source_counts:
            sources = pool_all[:nsrc]
            train, val, tgt, info = build_fold(data, sp, target, sources, al, fsc,
                                               n_questions=args.n_questions)
            r_ridge, alpha = ridge_on_z(train, val, tgt)
            line = (f"[target={info['target']:8s} sources={'+'.join(info['sources']):22s} "
                    f"nq={info['n_unique_q']} rows={info['n_train']}] ridge_z={r_ridge:.3f}")
            arm_out, fused = {}, None
            if not args.data_only:
                for arm in args.arms:
                    res = train_arm(train, val, tgt, arm, args.seeds, cfg, model, torch, nn,
                                    get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                                    ckpt_dir=args.ckpt_dir, tag=info['target'])
                    arm_out[arm] = res
                    line += f"  {arm}={np.mean(res['te_spearman']):.3f}"
                    print(f"    [{info['target']}] arm {arm} done: {np.mean(res['te_spearman']):.3f} "
                          f"(seeds {[round(s,3) for s in res['te_spearman']]})", flush=True)  # progress
                if "z" in arm_out and "q_resp_only" in arm_out:
                    fused = [rho(rank_fuse(arm_out["z"]["te_pred_by_seed"][s],
                                           arm_out["q_resp_only"]["te_pred_by_seed"][s]), tgt["y"])
                             for s in range(len(args.seeds))]
                    line += f"  fuse(z+qresp)={np.mean(fused):.3f}"
            print(line, flush=True)
            # PERSIST per-seed spearmans + per-example predictions + target labels — INCREMENTALLY per
            # fold, so a mid-run crash still keeps every completed fold. (This is the save I forgot.)
            rows_out.append({
                "info": info, "ridge_z": r_ridge, "alpha": alpha, "seeds": list(args.seeds),
                "target_y": [float(v) for v in tgt["y"]],
                "arms": {a: {"te_spearman": arm_out[a]["te_spearman"],
                             "te_pred_by_seed": [[float(v) for v in p] for p in arm_out[a]["te_pred_by_seed"]],
                             "curves_by_seed": arm_out[a].get("curves_by_seed")}   # per-epoch train/val -> plots
                         for a in arm_out},
                "fuse_z_qresp": fused,
            })
            with open(out_path, "w") as f:
                json.dump(rows_out, f)
            print(f"    -> saved {len(rows_out)} fold(s) to {out_path}", flush=True)

    if args.data_only:
        print("\n[data_only] all folds built + audited + ridge baseline; no 3B. Review, then run for real.")
    else:
        print(f"\nDONE — full per-seed results + predictions saved to {out_path}")


if __name__ == "__main__":
    main()
