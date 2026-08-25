"""E53a — pooled q_resp_only-only proxy TRAINING on the 4 Qwen/Gemma small-tier models.

Deploy-style (no held-out target): pools TEXT-ONLY (question, canonical response) + per-model
z-scored SE label from Qwen3-8B / Qwen3.5-9B / gemma-7b-it / gemma-2-9b-it's n2000 trivia_qa
train/val splits (`splits(2000)`, the same fixed-seed convention as every other Stage-2 split
in this repo — see exp2_run.py's build_deploy). Trains ONE q_resp_only arm (frozen
Llama-3.2-3B + LoRA + REG head). No hidden states are loaded and no cross-model Procrustes
alignment is needed: `q_resp_only` never touches z (`_arm_uses_z("q_resp_only") == False`),
so `ProxyModel.forward` skips the projector entirely (stage2/model.py) — text arms are
model-agnostic by construction, which is exactly why this arm can be trained by pooling four
target LLMs the proxy has never produced hidden states for in any aligned/shared basis.

Reuses `amortized_ue.exp2_run.train_arm` VERBATIM (it is generic over arm/data already; only
the z array passed in here is an unused (n,1) filler).

This is the reverse direction of E45 (which trained on Llama-2/Mistral/Llama-3/DeepSeek and
tested zero-shot on these same 4 Qwen/Gemma models) — see `e53_eval_on_llama2_mistral.py` for
the zero-shot evaluation on Llama-2 and Mistral this training feeds into.

Env: amortized_stage2 (GPU) for the real run; se_probes for --data_only (CPU audit).
    python -m amortized_ue.e53_train_qwengemma_deploy --data_only     # CPU: pool + audit only
    python -m amortized_ue.e53_train_qwengemma_deploy                # full: 3 seeds, GPU
"""
from __future__ import annotations

import os
import json
import argparse

import numpy as np

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.linear_ceiling_probe import splits
from amortized_ue.exp2_run import train_arm

NEW_MODELS = ["Qwen3-8B", "Qwen3.5-9B", "gemma-7b-it", "gemma-2-9b-it"]
ARM = "q_resp_only"
DEFAULT_DATA_DIR = "/data2/mn1025/stage1"
DEFAULT_CKPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "stage2", "runs", "E53_qwengemma_deploy_qresp", "checkpoints")


def load_pool(data_dir, smoke=False):
    """Pool (question, canonical response, per-model z-scored SE label) train/val rows from
    all 4 NEW_MODELS' n2000 trivia_qa records. No hidden states loaded (q_resp_only never
    reads z) — each model is loaded and split independently (`splits(2000)` on that model's
    own sorted-id order); the questions need not be shared across models for this arm, unlike
    the z-alignment pathway in exp2_run.py."""
    pool_tr = {"q": [], "r": [], "y": []}
    pool_va = {"q": [], "r": [], "y": []}
    stats = {}
    for m in NEW_MODELS:
        recs = load_records(Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000,
                                          output_dir=data_dir))
        ids = sorted(recs.keys())
        assert len(ids) == 2000, f"{m}: expected 2000 records, got {len(ids)}"
        tr, va, te = splits(len(ids))
        if smoke:
            tr, va = tr[:60], va[:20]
        q = [recs[i]["question"] for i in ids]
        r = [recs[i]["canonical"]["response"] for i in ids]
        y = np.array([recs[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
        mu, sd = float(y[tr].mean()), float(y[tr].std() + 1e-12)
        pool_tr["q"] += [q[i] for i in tr]; pool_tr["r"] += [r[i] for i in tr]
        pool_tr["y"] += list((y[tr] - mu) / sd)
        pool_va["q"] += [q[i] for i in va]; pool_va["r"] += [r[i] for i in va]
        pool_va["y"] += list((y[va] - mu) / sd)
        stats[m] = {"n": len(ids), "n_tr": len(tr), "n_va": len(va),
                     "mean_y_train": mu, "std_y_train": sd}
        print(f"  {m:16s} n={len(ids)} tr={len(tr)} va={len(va)} mean_CAE(train)={mu:.3f}")
    train = {"y": np.array(pool_tr["y"], dtype=np.float32), "q": pool_tr["q"], "r": pool_tr["r"]}
    val = {"y": np.array(pool_va["y"], dtype=np.float32), "q": pool_va["q"], "r": pool_va["r"]}
    return train, val, stats


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    p.add_argument("--ckpt_dir", default=DEFAULT_CKPT_DIR)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data_only", action="store_true")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--batch_size", type=int, default=8,
                    help="Lower than the usual 32 (E37/E45 convention) -- a full 256-token batch "
                         "(from Qwen3.5-9B's long <think> traces) OOM'd the 46GB card at batch_size=32 "
                         "on the very first forward pass; see the comment above where cfg is built.")
    p.add_argument("--grad_accum", type=int, default=4,
                    help="Micro-batches of --batch_size accumulated per optimizer step. Default 4 "
                         "(with batch_size=8) reproduces the established effective batch_size=32.")
    p.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "results", "e53_qwengemma_deploy_train_curves.json"))
    args = p.parse_args()

    print(f"Pooling {ARM} training data from {NEW_MODELS} (data_dir={args.data_dir}) ...")
    train, val, stats = load_pool(args.data_dir, smoke=args.smoke)
    print(f"pooled: train rows={len(train['y'])}  val rows={len(val['y'])}")

    print("\n" + "=" * 88)
    print("AUDIT: per-model tr/va/te from splits(2000) on that model's OWN sorted-id order (te "
          "reserved, unused); SE labels z-scored PER MODEL using TRAIN-ONLY mean/std, applied to "
          "both tr and va (no val leakage into normalization stats); zero-shot eval targets "
          "(Llama-2-7b-chat, Mistral-7B-Instruct-v0.2) are NOT in NEW_MODELS -- disjoint models, "
          "so there is no train/eval overlap at the model level regardless of question overlap.")
    print("=" * 88)

    # Unused z filler: q_resp_only never reads z (see module docstring) — h_in=1 keeps the
    # (unused, never-trained) projector tiny; its weights are exported to the checkpoint but
    # are never touched by forward() for this arm, so their content is irrelevant.
    train["z"] = np.zeros((len(train["y"]), 1), dtype=np.float32)
    val["z"] = np.zeros((len(val["y"]), 1), dtype=np.float32)
    tgt = dict(val)   # in-dist sanity target (build_deploy convention, exp2_run.py); the REAL
                       # zero-shot eval is e53_eval_on_llama2_mistral.py, not this script.

    if args.data_only:
        print("\n[data_only] pooled + audited; no 3B. Review, then run for real.")
        return

    import torch
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel
    from amortized_ue.stage2.train import _tokenize_arm, _arm_uses_z, _arm_text

    # batch_size=32 (the value every prior q_resp_only run in this project used, e.g. E37/E45's
    # deploy proxies) OOM'd here at 46GB: those runs trained on Llama-2/Mistral/Llama-3/DeepSeek,
    # whose trivia_qa answers are short, so a max_seq_len=256 batch was never actually reached.
    # Qwen3.5-9B leaves <think>...</think> reasoning traces in `canonical.response` (E44/E49) that
    # can run to hundreds of characters, hitting the 256-token cap -- a single such batch at
    # batch_size=32 needs far more activation memory than any batch this arm has trained on
    # before. (batch_size itself was NOT one of the knobs E16/E17 swept -- those tested
    # weight_decay and projector width/type -- so its effect on final quality here is untested,
    # not "confirmed inert"; that overclaim in an earlier version of this comment is corrected.)
    # Rather than train at a smaller EFFECTIVE batch and hope it's equivalent, --grad_accum
    # accumulates gradients over that many micro-batches of --batch_size before each optimizer
    # step, reproducing the established batch_size=32 recipe's gradient EXACTLY (no batchnorm
    # anywhere in ProxyModel, only LayerNorm, so micro-batch grad-accum == one true batch=32 step
    # -- see the comment in exp2_run.train_arm). Default batch_size=8 x grad_accum=4 = effective 32.
    cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4,
                        epochs=(2 if args.smoke else 10), batch_size=args.batch_size,
                        grad_accum=args.grad_accum)
    model = ProxyModel(cfg, h_in=1).to("cuda" if torch.cuda.is_available() else "cpu")

    lens = [len(model.tokenizer(_arm_text(ARM, q, r), add_special_tokens=False)["input_ids"])
            for q, r in zip(train["q"], train["r"])]
    n_at_cap = sum(1 for l in lens if l >= cfg.max_seq_len)
    print(f"tokenized {ARM} length over the pooled train set: max={max(lens)}  "
          f"p99={int(np.percentile(lens, 99))}  at/over cap({cfg.max_seq_len})={n_at_cap}/{len(lens)}")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    print(f"\nTraining {ARM} on {len(train['y'])} pooled rows "
          f"({'+'.join(NEW_MODELS)}), seeds={args.seeds}, batch_size={args.batch_size} x "
          f"grad_accum={args.grad_accum} (effective batch={args.batch_size * args.grad_accum}), "
          f"ckpt -> {args.ckpt_dir}", flush=True)
    res = train_arm(train, val, tgt, ARM, args.seeds, cfg, model, torch, nn,
                     get_cosine_schedule_with_warmup, _tokenize_arm, _arm_uses_z,
                     ckpt_dir=args.ckpt_dir, tag="qwengemma_deploy")
    print(f"\nDONE. in-dist sanity (val pool) Spearman per seed: "
          f"{[round(s, 3) for s in res['te_spearman']]}  mean={np.mean(res['te_spearman']):.3f}")
    print(f"checkpoints saved to {args.ckpt_dir}")

    # PERSIST training curves + per-seed sanity Spearman + per-model pooling stats -- the exact
    # save exp2_run.py's own --deploy path does (deploy_curves.json), and the thing that was
    # missing once before (E37: a forgotten json.dump lost ~2hr of per-seed data, see
    # [[persist-results-before-done]]). `res["te_pred_by_seed"]` here is the val-pool in-dist
    # sanity prediction, NOT the real zero-shot eval -- that lives in the eval script's own
    # output JSON, scored on Llama-2/Mistral.
    out = {
        "arm": ARM, "train_models": NEW_MODELS, "seeds": list(args.seeds),
        "data_dir": args.data_dir, "ckpt_dir": args.ckpt_dir,
        "per_model_stats": stats, "train_config": cfg.as_dict(),
        "n_train": len(train["y"]), "n_val": len(val["y"]),
        "val_pool_sanity_spearman_by_seed": res["te_spearman"],
        "curves_by_seed": res["curves_by_seed"],
        "val_pool_pred_by_seed": [[float(v) for v in p] for p in res["te_pred_by_seed"]],
        "val_pool_y": [float(v) for v in tgt["y"]],
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f)
    print(f"training curves + per-seed sanity results saved to {args.out}")


if __name__ == "__main__":
    main()
