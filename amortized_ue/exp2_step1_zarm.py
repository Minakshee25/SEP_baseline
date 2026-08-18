"""Exp-2 STEP 1 — the minimal vertical slice: pooled multi-source proxy, z-arm, LOO on ONE target.

Purpose: exercise the WHOLE new pipeline end-to-end on the simplest arm, to verify it is correct
before scaling to all folds / source-count sweep / text arms / stacking. Answers the core question in
miniature: does the proxy trained on POOLED aligned z from 3 source models predict SE for an UNSEEN 4th
model (Llama-3) — above chance (E20 z-transfer 0.056) and vs the E35 best-layer ridge (0.603)?

Reuses the AUDITED alignment/normalization from e35_pooling_matched_partition_bestlayer.py verbatim
(label-free Procrustes each source best-TBG -> Llama-2 anchor, per-model feature scaler, per-model SE
label z-score, all fit on TRAIN only). The held-out target is aligned with its OWN label-free W_T +
scaler; its SE labels are NEVER used (label-free transfer). The proxy is the reference ProxyModel
(Llama-3.2-3B, LoRA), z fed as one soft-token vector [B,1,H]; z-only sequence [k soft][REG].

Leak-free by construction (printed self-audit): sources train on their `tr`; early-stop on pooled
sources' `va`; the target's `te` is touched ONLY for the final Spearman. `tr`/`va`/`te` are disjoint
question sets shared across models (id-joined).

Modes:
  --data_only : CPU, no 3B. Build the pipeline, print shapes + leakage audit + the RIDGE baseline on the
                identical pooled z (proxy-vs-ridge sanity). RUN THIS FIRST to verify correctness.
  --smoke     : tiny end-to-end 3B run (few rows, 1 seed, 2 epochs) to confirm the model path runs.
  (default)   : full run, --seeds seeds.

Env: amortized_stage2 (GPU) for the real run; se_probes is fine for --data_only.
"""
from __future__ import annotations

import argparse
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits

ANCHOR = "Llama-2-7b-chat"
MODELS = [ANCHOR, "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "DeepSeek"}
# leak-free best TBG layer per model (reconfirm_layers.py, val/5-fold CV)
BEST_TBG = {ANCHOR: 30, "Mistral-7B-Instruct-v0.2": 31, "Meta-Llama-3-8B-Instruct": 31,
            "deepseek-llm-7b-chat": 28}
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
# reference anchors for interpretation
CHANCE_LLAMA3 = 0.056      # E20 raw z-transfer Llama-2 proxy -> Llama-3
E35_RIDGE_LLAMA3 = 0.603   # E35 best-layer pooled ridge @1440, anchor 30, held-out Llama-3


def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)


def build_pooled(target=("Meta-Llama-3-8B-Instruct"), anchor_layer=30, smoke=False,
                 n_questions=None, subset_seed=0):
    """Load 4 models at best-TBG, align sources->anchor (label-free, tr), return pooled train/val +
    the held-out target's te. All alignment/scaling/label-stats fit on TRAIN only.

    SAME questions to all 3 sources (each question -> 3 model-views), so the proxy gets the model-
    INVARIANCE signal (same Q, 3 different model-z's -> same SE) — the point of a model-agnostic proxy.
    We deliberately do NOT use a disjoint partition: that confounds model-identity with the question set
    and denies the invariance signal, so a disjoint-pooled proxy can learn a non-transferable shortcut.
      n_questions=None : all `tr` (1440) to every source  -> 3x1440 = 4320 rows (primary).
      n_questions=N    : the SAME N questions to every source -> 3xN rows (volume control that KEEPS
                         invariance; probes whether the proxy needs more unique questions)."""
    mats, ys, ids0 = {}, {}, None
    for m in MODELS:
        cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
        hidden, y, ids = load_matrix(cfg, ["TBG"])
        layer = anchor_layer if m == ANCHOR else BEST_TBG[m]
        mats[m], ys[m] = hidden["TBG"][layer], y
        assert ids0 is None or ids == ids0, "id order differs across models (join-by-id broken)"
        ids0 = ids
    tr, va, te = splits(len(ids0))
    if smoke:
        tr, va = tr[:120], va[:40]

    # label-free alignment: each model best-TBG -> anchor best-TBG, fit on tr; per-model feature scaler
    al, fsc = {}, {}
    Ac = mats[ANCHOR][tr] - mats[ANCHOR][tr].mean(0, keepdims=True)
    for m in MODELS:
        mean_m = mats[m][tr].mean(0, keepdims=True)
        W = np.eye(mats[m].shape[1]) if m == ANCHOR else orthogonal_procrustes(mats[m][tr] - mean_m, Ac)[0]
        al[m] = (mean_m, W)
        fsc[m] = StandardScaler().fit((mats[m][tr] - mean_m) @ W)

    def feat(m, idx):
        mean_m, W = al[m]
        return fsc[m].transform((mats[m][idx] - mean_m) @ W)

    sources = [m for m in MODELS if m != target]
    if n_questions is not None and n_questions < len(tr):        # SAME N questions to every source
        q = np.sort(np.random.default_rng(subset_seed).choice(tr, size=n_questions, replace=False))
        src_rows = {m: q for m in sources}
    else:                                                        # all tr to every source
        src_rows = {m: tr for m in sources}
    Xtr, ytr, Xva, yva = [], [], [], []
    for m in sources:
        rows_m = src_rows[m]
        mu, sd = ys[m][rows_m].mean(), ys[m][rows_m].std() + 1e-12   # per-model label z-score on routed rows (E35)
        Xtr.append(feat(m, rows_m)); ytr.append((ys[m][rows_m] - mu) / sd)
        Xva.append(feat(m, va)); yva.append((ys[m][va] - mu) / sd)
    Xtr = np.vstack(Xtr).astype(np.float32); ytr = np.concatenate(ytr).astype(np.float32)
    Xva = np.vstack(Xva).astype(np.float32); yva = np.concatenate(yva).astype(np.float32)

    # held-out target: its OWN label-free W_T + scaler; RAW te labels (used only for Spearman)
    Xte = feat(target, te).astype(np.float32)
    yte_raw = ys[target][te].astype(np.float32)

    n_uniq = len(next(iter(src_rows.values())))
    info = {"sources": [SHORT[m] for m in sources], "target": SHORT[target],
            "regime": f"same-{n_uniq}Q-to-all3 ({len(sources)}x{n_uniq}={len(ytr)} rows)",
            "n_unique_questions": n_uniq, "n_pooled_tr": len(ytr), "n_pooled_va": len(yva),
            "n_te_target": len(yte_raw), "H": Xtr.shape[1],
            "anchor_layer": anchor_layer, "src_layers": {SHORT[m]: BEST_TBG[m] for m in sources}}
    return (Xtr, ytr, Xva, yva, Xte, yte_raw), info


def audit(info):
    print("=" * 84)
    print("LEAKAGE SELF-AUDIT (verify before trusting numbers)")
    print("  [1] alignment W + feature scaler + label mu/sd fit on TRAIN (`tr`) only")
    print("  [2] early-stop / selection uses pooled SOURCES' `va`; target `te` -> final Spearman ONLY")
    print("  [3] target is NEVER in training (LOO); its W_T + scaler are label-free (hidden states only)")
    print("  [4] id-joined (assert ids identical); continuous SE; Spearman (rank -> transform-invariant)")
    print(f"  target={info['target']}  sources={info['sources']}  H={info['H']}  anchor=TBG:{info['anchor_layer']}")
    print(f"  REGIME={info['regime']}")
    print(f"  src layers={info['src_layers']}  pooled_tr={info['n_pooled_tr']} va={info['n_pooled_va']} te={info['n_te_target']}")
    print("=" * 84)


def ridge_baseline(data):
    Xtr, ytr, Xva, yva, Xte, yte = data
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        s = rho(r.predict(Xva), yva)                # alpha on pooled VAL
        if best is None or s > best[0]:
            best = (s, a, r)
    return rho(best[2].predict(Xte), yte), best[1]   # target te Spearman, chosen alpha


def train_proxy(data, seeds, smoke=False):
    import torch
    import torch.nn as nn
    from transformers import get_cosine_schedule_with_warmup
    from amortized_ue.stage2.config import Stage2Config
    from amortized_ue.stage2.model import ProxyModel

    Xtr, ytr, Xva, yva, Xte, yte = data
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = Stage2Config(projector_hidden_dim=1024, k_soft_tokens=4,
                       epochs=(2 if smoke else 10), batch_size=32)
    H = Xtr.shape[1]
    model = ProxyModel(cfg, h_in=H).to(dev)

    def to_z(X):
        return torch.from_numpy(X).float().unsqueeze(1)            # [n,1,H]
    ztr, ytr_t = to_z(Xtr), torch.from_numpy(ytr).float()
    zva, yva_t = to_z(Xva), torch.from_numpy(yva).float()
    zte = to_z(Xte)

    @torch.no_grad()
    def predict(z):
        model.eval(); out = []
        for i in range(0, len(z), cfg.batch_size):
            out.append(model(z[i:i + cfg.batch_size].to(dev), None, None).float().cpu())
        model.train(); return torch.cat(out)

    results = []
    for seed in seeds:
        torch.manual_seed(seed); np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        model.reinit_trainable()
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
        spe = max(1, int(np.ceil(len(ztr) / cfg.batch_size)))
        total = spe * cfg.epochs
        sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup_ratio * total), total)
        loss_fn = nn.MSELoss()
        best_val, best_state, patience = float("inf"), None, 0
        rng = np.random.default_rng(seed)
        for ep in range(cfg.epochs):
            model.train()
            order = rng.permutation(len(ztr))
            for i in range(0, len(order), cfg.batch_size):
                idx = order[i:i + cfg.batch_size]
                pred = model(ztr[idx].to(dev), None, None)
                loss = loss_fn(pred, ytr_t[idx].to(dev))
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
                opt.step(); sched.step()
            val_mse = float(((predict(zva) - yva_t) ** 2).mean())   # early stop on pooled VAL
            if val_mse < best_val:
                best_val = val_mse
                best_state = {k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad}
                patience = 0
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break
        if best_state is not None:
            with torch.no_grad():
                for k, v in model.named_parameters():
                    if v.requires_grad and k in best_state:
                        v.copy_(best_state[k])
        s = rho(predict(zte).numpy(), yte)                          # target te Spearman
        results.append(s)
        print(f"  seed {seed}: proxy z-arm LOO Spearman = {s:.4f}")
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--target", default="Meta-Llama-3-8B-Instruct")
    p.add_argument("--anchor_layer", type=int, default=30)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--data_only", action="store_true", help="CPU: pipeline + audit + ridge baseline; no 3B")
    p.add_argument("--smoke", action="store_true", help="tiny end-to-end 3B run")
    p.add_argument("--n_questions", type=int, default=None,
                   help="SAME N questions to all 3 sources (3xN rows). Omit = all tr (1440 -> 4320).")
    args = p.parse_args()

    data, info = build_pooled(target=args.target, anchor_layer=args.anchor_layer, smoke=args.smoke,
                              n_questions=args.n_questions)
    audit(info)
    r_ridge, alpha = ridge_baseline(data)
    print(f"\nRIDGE on the identical pooled z  -> Llama-3 te Spearman = {r_ridge:.4f}  (alpha={alpha})")
    print(f"reference anchors: chance {CHANCE_LLAMA3:.3f}  |  E35 best-layer pooled ridge {E35_RIDGE_LLAMA3:.3f}")

    if args.data_only:
        print("\n[data_only] pipeline built, audited, ridge sanity done — no 3B run. Review, then run for real.")
        return

    print(f"\ntraining proxy (seeds={args.seeds}, smoke={args.smoke}) ...")
    res = train_proxy(data, args.seeds, smoke=args.smoke)
    print("\n" + "=" * 84)
    print(f"PROXY z-arm LOO on {info['target']}: {np.mean(res):.4f} ± {np.std(res):.4f}   (n_seeds={len(res)})")
    print(f"  vs chance {CHANCE_LLAMA3:.3f} | vs ridge-on-same-z {r_ridge:.3f} | vs E35 ridge {E35_RIDGE_LLAMA3:.3f}")
    print("=" * 84)


if __name__ == "__main__":
    main()
