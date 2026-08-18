"""Matched-total control, BEST-LAYER variant of e35_pooling_matched_partition.py.

Identical audited logic to the original (same matched-partition design, same per-model
normalization, same label-free W + feature scalers on train, same alpha-on-val, same
id-join) — the ONLY change is the LAYER each model is read at:

  original : L=22 for EVERY model (source AND anchor)  -> suboptimal (each model's best is NOT 22)
  here     : each SOURCE at its OWN leak-free best TBG layer, aligned into the ANCHOR's best
             TBG layer (--anchor_layer, default 30). Layers reconfirmed leak-free 2026-08-17
             (reconfirm_layers.py, val / 5-fold CV): Llama-2 30, Mistral 31, Llama-3 31, DeepSeek 28.

Motivation: the published E35 ran Mistral/Llama-3 ~0.05-0.07 Spearman below their best layer, so its
pooling magnitudes are provisional. This re-baselines with proper layers. Run --anchor_layer 30 and 22
to test the depth-matching hypothesis (sources peak ~31, so late->late may align cleaner than ->22).

For each held-out target T, at each `total` budget:
  single : all `total` questions routed through Llama-2 (anchor)  -> test T
  pooled : those same questions PARTITIONED across the 3 non-T sources (each aligned) -> test T
pooled>single => genuine model-diversity, not rows/questions. CPU, se_probes env.
"""
import sys
sys.path.insert(0, "/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes")
import json
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
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral", "Meta-Llama-3-8B-Instruct": "Llama-3",
         "deepseek-llm-7b-chat": "deepseek"}
# leak-free best TBG layer per SOURCE model (reconfirm_layers.py); anchor layer is --anchor_layer.
SRC_LAYER = {"Mistral-7B-Instruct-v0.2": 31, "Meta-Llama-3-8B-Instruct": 31, "deepseek-llm-7b-chat": 28}
TARGETS = MODELS[1:]
TOTALS = [150, 300, 600, 1200, 1440]                       # divisible by 3
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = 4


def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anchor_layer", type=int, default=30, help="Llama-2 frame TBG layer (test 30 vs 22)")
    p.add_argument("--out", default=None)
    args = p.parse_args()
    A = args.anchor_layer
    LAYER = {m: (A if m == ANCHOR else SRC_LAYER[m]) for m in MODELS}

    print("=" * 84)
    print("E35 BEST-LAYER matched partition — LEAKAGE SELF-AUDIT")
    print("  [1] alpha selected on VAL, target scored on TE only (never selected on test)")
    print("  [2] Procrustes W + feature scaler + per-model label mu/sd all fit on TRAIN only")
    print("  [3] all 4 models id-joined (assert ids identical); continuous SE; Spearman")
    print(f"  layers: anchor {ANCHOR}=TBG:{A}; sources " +
          ", ".join(f"{SHORT[m]}=TBG:{LAYER[m]}" for m in TARGETS))
    print("=" * 84)

    # ---- load each model at its chosen layer (records warm in cache from reconfirm) --------
    mats, ys, ids0 = {}, {}, None
    for m in MODELS:
        cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
        hidden, y, ids = load_matrix(cfg, ["TBG"])
        mats[m], ys[m] = hidden["TBG"][LAYER[m]], y      # <-- per-model best layer (only change vs orig)
        assert ids0 is None or ids == ids0, "id order differs across models"
        ids0 = ids
    tr, va, te = splits(len(ids0))
    print(f"train pool={len(tr)}; SAME questions both arms, only model-routing differs\n")

    # ---- label-free alignment: each source best-layer -> anchor best-layer, on tr ----------
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

    def fit_transfer(src_subs, T):                         # {model: idx-into-tr}
        Xtr, ytr, Xva, yva = [], [], [], []
        for m, s in src_subs.items():
            mu, sd = ys[m][tr][s].mean(), ys[m][tr][s].std() + 1e-12   # per-model label z-score, TRAIN
            Xtr.append(feat(m, tr[s])); ytr.append((ys[m][tr][s] - mu) / sd)
            Xva.append(feat(m, va));    yva.append((ys[m][va] - mu) / sd)
        Xtr, ytr, Xva, yva = np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xva), np.concatenate(yva)
        best = None
        for a in ALPHAS:
            r = Ridge(alpha=a).fit(Xtr, ytr)
            sc = rho(r.predict(Xva), yva)                  # alpha selected on VAL
            if best is None or sc > best[0]:
                best = (sc, r)
        return rho(best[1].predict(feat(T, te)), ys[T][te])   # scored on TE only

    # ---- matched-partition sweep -----------------------------------------------------------
    print(f"{'total':>6} | " + "".join(f"{SHORT[T]+' s/p':>18s}" for T in TARGETS) + "   (single / pooled)")
    print("-" * 78)
    results = {"anchor_layer": A, "src_layer": SRC_LAYER, "totals": {}}
    for total in TOTALS:
        per = total // 3
        row = {T: {"s": [], "p": []} for T in TARGETS}
        for seed in range(SEEDS):
            rng = np.random.default_rng(seed)
            q = rng.choice(len(tr), size=total, replace=False)     # ONE question set, both arms
            g = [q[0:per], q[per:2 * per], q[2 * per:3 * per]]
            for T in TARGETS:
                srcs = [m for m in MODELS if m != T]
                row[T]["s"].append(fit_transfer({ANCHOR: q}, T))
                row[T]["p"].append(fit_transfer({srcs[i]: g[i] for i in range(3)}, T))
        line = f"{total:>6} | "
        results["totals"][total] = {}
        for T in TARGETS:
            s, pl = float(np.mean(row[T]["s"])), float(np.mean(row[T]["p"]))
            results["totals"][total][SHORT[T]] = {"single": s, "pooled": pl,
                                                  "single_seeds": row[T]["s"], "pooled_seeds": row[T]["p"]}
            line += f"   {s:.3f}/{pl:.3f}"
        print(line)
    print("\nread: SAME questions + rows, only 1-model vs 3-model routing differs.")
    print("  pooled(p) > single(s) => diversity genuinely helps; p ~= s => routing adds nothing.")

    out = args.out or f"scratch_xllm/e35_bestlayer_matched_anchor{A}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=1)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
