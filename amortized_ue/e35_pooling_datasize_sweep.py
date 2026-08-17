"""Data-size sweep, corrected pipeline (per-model normalization) — single vs pooled.

For each held-out target T (the 3 non-anchor targets), at each labeled-data budget n_sub:
  - single : Llama-2 ridge on n_sub of its train rows -> transfer to T
  - pooled : the 3 non-T models, each n_sub rows, per-model normalized -> transfer to T
Alignment W, per-model feature scalers = label-free, fit on FULL train (NOT swept — we
sweep the RIDGE's labeled rows only). Labels z-scored per model on the subsample. Test on
T's te. 3 random subsamples/size. Answers: does less data help/hurt, and does the fixed
pooling ever beat single across data sizes?
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
TARGETS = MODELS[1:]                                  # single source = Llama-2, so T != Llama-2
SIZES = [50, 100, 200, 400, 800, 1440]
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = 3

mats, ys, ids0 = {}, {}, None
for m in MODELS:
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
    hidden, y, ids = load_matrix(cfg, ["TBG"])
    mats[m], ys[m] = hidden["TBG"][L], y
    assert ids0 is None or ids == ids0
    ids0 = ids
tr, va, te = splits(len(ids0))
print(f"L{L}: train pool={len(tr)}, val={len(va)}, test={len(te)}; per-model normalization\n")

def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)

# label-free preproc on FULL train (not swept): mean, W, per-model feature scaler
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

def fit_transfer(sources, sub_by_src, T):
    Xtr, ytr, Xva, yva = [], [], [], []
    for m in sources:
        s = sub_by_src[m]
        mu, sd = ys[m][tr][s].mean(), ys[m][tr][s].std() + 1e-12    # per-model label z (on subsample)
        Xtr.append(feat(m, tr[s])); ytr.append((ys[m][tr][s] - mu) / sd)
        Xva.append(feat(m, va));    yva.append((ys[m][va] - mu) / sd)
    Xtr, ytr, Xva, yva = np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xva), np.concatenate(yva)
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        sc = rho(r.predict(Xva), yva)
        if best is None or sc > best[0]: best = (sc, r)
    return rho(best[1].predict(feat(T, te)), ys[T][te])

print(f"{'n_sub':>6} | " + "".join(f"{SHORT[T]+'(s/p)':>18s}" for T in TARGETS) + "   (single / pooled-3)")
print("-" * 78)
for n in SIZES:
    row = {T: {"s": [], "p": []} for T in TARGETS}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        subs = {m: rng.choice(len(tr), size=n, replace=False) for m in MODELS}
        for T in TARGETS:
            row[T]["s"].append(fit_transfer([ANCHOR], subs, T))                    # Llama-2 only
            row[T]["p"].append(fit_transfer([m for m in MODELS if m != T], subs, T))  # pooled 3
    line = f"{n:>6} | "
    for T in TARGETS:
        line += f"   {np.mean(row[T]['s']):.3f}/{np.mean(row[T]['p']):.3f}"
    print(line)
print("\nread: single = Llama-2->T ; pooled = (3 non-T)->T, both per-model normalized.")
print("less data hurts if columns fall at small n; pooled>single if p>s consistently.")
