"""LOO pilot v2 — FIX: per-model normalization before pooling.

Bug in v1: pooled labels were raw SE across models with different SE scales/means
(Llama-3 ~0.47, Llama-2 ~0.59, DeepSeek ~0.80) and a single shared feature scaler ->
per-model label offsets + scale that the centered states can't explain, handicapping
pooling. Fix: z-score each source's SE labels (own train stats) and standardize its
features with its OWN scaler; standardize the held-out target with its OWN scaler.
Spearman is rank-based so per-model z-scoring is eval-neutral. Individual baseline is
essentially unchanged (single model already uses its own scaler) -> only pooling changes.
"""
import sys
sys.path.insert(0, "/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes")
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits

L = 22
ANCHOR = "Llama-2-7b-chat"
MODELS = [ANCHOR, "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral", "Meta-Llama-3-8B-Instruct": "Llama-3",
         "deepseek-llm-7b-chat": "deepseek"}
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]

mats, ys, ids0 = {}, {}, None
for m in MODELS:
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
    hidden, y, ids = load_matrix(cfg, ["TBG"])
    mats[m], ys[m] = hidden["TBG"][L], y
    assert ids0 is None or ids == ids0
    ids0 = ids
tr, va, te = splits(len(ids0))
print(f"L{L}: N={len(ids0)}, train {len(tr)}/val {len(va)}/test {len(te)}; frame={SHORT[ANCHOR]}")
print("mean SE by model:", {SHORT[m]: round(float(ys[m][tr].mean()), 3) for m in MODELS}, "\n")

def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)

# label-free alignment into the Llama-2 frame; then PER-MODEL feature scaler + label z-stats (on tr)
Ac = mats[ANCHOR][tr] - mats[ANCHOR][tr].mean(0, keepdims=True)
al, fsc, lmu, lsd = {}, {}, {}, {}
for m in MODELS:
    mean_m = mats[m][tr].mean(0, keepdims=True)
    W = np.eye(mats[m].shape[1]) if m == ANCHOR else orthogonal_procrustes(mats[m][tr] - mean_m, Ac)[0]
    al[m] = (mean_m, W)
    tr_feat = (mats[m][tr] - mean_m) @ W
    fsc[m] = StandardScaler().fit(tr_feat)                       # per-model feature scaler
    lmu[m], lsd[m] = float(ys[m][tr].mean()), float(ys[m][tr].std() + 1e-12)  # per-model label stats
def feat(m, idx):
    mean_m, W = al[m]
    return fsc[m].transform((mats[m][idx] - mean_m) @ W)        # aligned + per-model standardized
def zlabel(m, idx):
    return (ys[m][idx] - lmu[m]) / lsd[m]                        # per-model z-scored SE

def fit_ridge(sources):
    Xtr = np.vstack([feat(m, tr) for m in sources]); ytr = np.concatenate([zlabel(m, tr) for m in sources])
    Xva = np.vstack([feat(m, va) for m in sources]); yva = np.concatenate([zlabel(m, va) for m in sources])
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        s = rho(r.predict(Xva), yva)
        if best is None or s > best[0]: best = (s, r)
    return best[1]
def pred_on(r, T):
    return r.predict(feat(T, te))
def boot(pa, pb, y, B=2000, seed=0):
    rng = np.random.default_rng(seed); n = len(y); d = np.empty(B)
    for b in range(B):
        i = rng.integers(0, n, n); d[b] = rho(pa[i], y[i]) - rho(pb[i], y[i])
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))

print(f"{'held-out T':>10} | {'POOLED-3':>9} | singles (train 1 -> test T)                 | {'best1':>6} {'mean1':>6}")
print("-" * 100)
res = {}
for T in MODELS:
    sources = [m for m in MODELS if m != T]
    rP = fit_ridge(sources); predP = pred_on(rP, T); sP = rho(predP, ys[T][te])
    single = {m: (lambda p: (rho(p, ys[T][te]), p))(pred_on(fit_ridge([m]), T)) for m in sources}
    best_m = max(single, key=lambda k: single[k][0]); best1 = single[best_m][0]
    mean1 = float(np.mean([single[m][0] for m in sources]))
    sstr = "  ".join(f"{SHORT[m]}:{single[m][0]:.3f}" for m in sources)
    print(f"{SHORT[T]:>10} | {sP:>9.3f} | {sstr:<44} | {best1:>6.3f} {mean1:>6.3f}")
    db = boot(predP, single[best_m][1], ys[T][te])
    res[T] = (sP, best1, mean1, db)

print("\nPOOLED - BEST-SINGLE (paired bootstrap, 95% CI):")
for T in MODELS:
    _, _, _, db = res[T]
    tag = "sig>0" if db[1] > 0 else ("sig<0" if db[2] < 0 else "incl 0")
    print(f"  {SHORT[T]:>10}: {db[0]:+.3f} [{db[1]:+.3f}, {db[2]:+.3f}] ({tag})")
