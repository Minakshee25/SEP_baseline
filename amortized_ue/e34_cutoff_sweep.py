"""Cutoff sweep: do the aligned models share the same uncertainty DIRECTION once we look
only inside the well-determined subspace (drop the noisy low-variance directions)?

Space: standardized anchor (Llama-2) TBG L22 -- same space as the ~218 effective-rank
figure and the earlier 0.45 full-vector cosines. Label-free PCA on the anchor's own
standardized training states ranks directions by variance (how well-determined they are).
For a cutoff k we keep the top-k PCs, project each model's readout direction onto them,
and take cosine. Cross-model (vs anchor) is read against the same-model split-half ceiling
computed the identical way at each k. Sweep k so no single arbitrary cutoff drives it.
"""
import sys
sys.path.insert(0, "/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes")
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import train_test_split
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, fit_probe, SEED

L = 22
ANCHOR = "Llama-2-7b-chat"
MODELS = [ANCHOR, "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "deepseek"}
KS = [10, 25, 50, 100, 200, 500, 1000, 1440]

# ---- load all four models' L22 states + SE labels (same ids) ----
mats, ys, ids0 = {}, {}, None
for m in MODELS:
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
    hidden, y, ids = load_matrix(cfg, ["TBG"])
    mats[m], ys[m] = hidden["TBG"][L], y
    assert ids0 is None or ids == ids0, f"{m} ids differ"
    ids0 = ids
tr, va, te = splits(len(ids0))
anchor_mean = mats[ANCHOR][tr].mean(0, keepdims=True)
anchor_std = mats[ANCHOR][tr].std(0)
print(f"L{L}: N={len(ids0)}, D={mats[ANCHOR].shape[1]}, |train|={len(tr)} (half={len(tr)//2})")

# ---- label-free PCA of the anchor's STANDARDIZED training states ----
Zt = (mats[ANCHOR][tr] - anchor_mean) / anchor_std            # standardized anchor train
_, S, Vt = np.linalg.svd(Zt, full_matrices=False)            # Vt rows = PCs, variance-ordered
ev = S**2
eff_rank = (ev.sum()**2) / (ev**2).sum()
cumvar = np.cumsum(ev) / ev.sum()
print(f"effective rank ~= {eff_rank:.0f};  variance captured by top-k:"
      + "".join(f"  k{k}:{cumvar[min(k, len(cumvar))-1]:.2f}" for k in [50, 100, 200, 500]) + "\n")

def dir_std(coef, scale, W):
    return ((coef / scale) @ W) * anchor_std                  # readout dir in standardized anchor space
def cos_topk(ca, cb, k):
    a, b = ca[:k], cb[:k]
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

# ---- per model: fit readout + W, get PC coords of the direction ----
coords, W_of, mean_of = {}, {}, {}
for m in MODELS:
    X = mats[m]
    f, sc, _, _ = fit_probe(X, ys[m], tr, va)
    if m == ANCHOR:
        W = np.eye(X.shape[1])
    else:
        A = X[tr] - X[tr].mean(0, keepdims=True)
        B = mats[ANCHOR][tr] - anchor_mean
        W, _ = orthogonal_procrustes(A, B)
    W_of[m], mean_of[m] = W, X[tr].mean(0, keepdims=True)
    coords[m] = Vt @ dir_std(f.coef_, sc.scale_, W)           # direction in PC coordinates

# ---- same-model split-half ceiling directions (carried through the same W) ----
ceil_coords = {}
for m in MODELS:
    X = mats[m]
    hA, hB = train_test_split(tr, test_size=0.5, random_state=SEED)
    fA, scA, _, _ = fit_probe(X, ys[m], np.sort(hA), va)
    fB, scB, _, _ = fit_probe(X, ys[m], np.sort(hB), va)
    W = W_of[m]
    ceil_coords[m] = (Vt @ dir_std(fA.coef_, scA.scale_, W),
                      Vt @ dir_std(fB.coef_, scB.scale_, W))

# ---- sweep ----
nonanchor = [m for m in MODELS if m != ANCHOR]
print("CROSS-MODEL cosine vs anchor  (uncertainty direction, top-k PC subspace):")
hdr = "  " + f"{'k':>6}" + "".join(f"{SHORT[m]:>10s}" for m in nonanchor) + f"{'anchorCeil':>12s}"
print(hdr)
rows = {}
for k in KS:
    line = "  " + f"{k:>6}"
    rec = {}
    for m in nonanchor:
        c = cos_topk(coords[ANCHOR], coords[m], k)
        rec[m] = c
        line += f"{c:>10.3f}"
    ac = cos_topk(*ceil_coords[ANCHOR], k)
    rec["anchor_ceiling"] = ac
    line += f"{ac:>12.3f}"
    rows[k] = rec
    print(line)

print("\nSAME-MODEL split-half ceiling  (top-k PC subspace) -- the yardstick per model:")
print("  " + f"{'k':>6}" + "".join(f"{SHORT[m]:>10s}" for m in MODELS))
for k in KS:
    line = "  " + f"{k:>6}"
    for m in MODELS:
        line += f"{cos_topk(*ceil_coords[m], k):>10.3f}"
    print(line)

print("\nread: cross ~ its column's ceiling AND stable across k  => same direction (trustworthy);")
print("      cross swings with k or sits far below ceiling      => not established / subspace-only")
