"""Valid ceiling for 'same uncertainty direction?' via a matched same-vs-different design.

The flaw before: the ceiling changed TWO things vs the cross-model number (same model but
DIFFERENT questions), so it wasn't a fair reference. Fix: make BOTH comparisons span the
same two disjoint question halves, so they carry identical sampling+estimation noise; the
ONLY difference is same-model vs different-model.

Standardized anchor (Llama-2) TBG L22 space, directions projected onto the anchor's top-k
PCs (label-free), swept over k. W fit label-free on the full train (uses no SE labels).

  CEILING_A(k)  = cos_topk( dir_anchor(half1),  dir_anchor(half2) )     same model, cross-split
  CEILING_B(k)  = cos_topk( dir_B(half1),       dir_B(half2) )          each model with itself
  CROSS_B(k)    = cos_topk( dir_anchor(half1),  dir_B(half2) )          diff model, cross-split
                  (averaged with the half-swapped version for symmetry)

Read: CROSS_B ~ the CEILINGs and stable across k  => model swap adds nothing beyond noise
      => same uncertainty direction, validated against a fair ceiling.
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
KS = [10, 25, 50, 100, 200, 1440]

mats, ys, ids0 = {}, {}, None
for m in MODELS:
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
    hidden, y, ids = load_matrix(cfg, ["TBG"])
    mats[m], ys[m] = hidden["TBG"][L], y
    assert ids0 is None or ids == ids0
    ids0 = ids
tr, va, te = splits(len(ids0))
anchor_mean = mats[ANCHOR][tr].mean(0, keepdims=True)
anchor_std = mats[ANCHOR][tr].std(0)
Zt = (mats[ANCHOR][tr] - anchor_mean) / anchor_std
_, _, Vt = np.linalg.svd(Zt, full_matrices=False)                 # anchor PCs (label-free, full train)
h1, h2 = np.sort(train_test_split(tr, test_size=0.5, random_state=SEED)[0]), \
         np.sort(train_test_split(tr, test_size=0.5, random_state=SEED)[1])
print(f"L{L}: |train|={len(tr)}, halves {len(h1)}/{len(h2)} (disjoint)\n")

def dir_std(coef, scale, W):
    return ((coef / scale) @ W) * anchor_std
def cos_topk(a, b, k):
    ca, cb = (Vt @ a)[:k], (Vt @ b)[:k]
    return float(ca @ cb / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-12))

# W per model (label-free, full train); two half-data readout directions per model
W_of, d1, d2 = {}, {}, {}
Ac = mats[ANCHOR][tr] - anchor_mean
for m in MODELS:
    X = mats[m]
    if m == ANCHOR:
        W = np.eye(X.shape[1])
    else:
        W, _ = orthogonal_procrustes(X[tr] - X[tr].mean(0, keepdims=True), Ac)
    W_of[m] = W
    fA, scA, _, _ = fit_probe(X, ys[m], h1, va)
    fB, scB, _, _ = fit_probe(X, ys[m], h2, va)
    d1[m] = dir_std(fA.coef_, scA.scale_, W)
    d2[m] = dir_std(fB.coef_, scB.scale_, W)

nonanchor = [m for m in MODELS if m != ANCHOR]
print("MATCHED same-vs-different, direction cosine in top-k PC subspace:")
print(f"  {'k':>5}  " + "".join(f"{SHORT[m]+'(cross)':>16s}" for m in nonanchor)
      + f"{'A-ceiling':>11s}")
for k in KS:
    line = f"  {k:>5}  "
    for m in nonanchor:
        cross = 0.5 * (cos_topk(d1[ANCHOR], d2[m], k) + cos_topk(d2[ANCHOR], d1[m], k))
        selfc = 0.5 * (cos_topk(d1[m], d2[m], k) + cos_topk(d2[m], d1[m], k))  # =symmetric anyway
        line += f"{cross:>10.3f}/{selfc:<5.3f}"   # cross / that model's own ceiling
    aceil = cos_topk(d1[ANCHOR], d2[ANCHOR], k)
    line += f"{aceil:>11.3f}"
    print(line)
print("\n  format:  <cross-model> / <that model's own same-model ceiling>   ;  last col = anchor ceiling")
print("  same direction confirmed if  cross  ~=  the ceilings, across k.")
