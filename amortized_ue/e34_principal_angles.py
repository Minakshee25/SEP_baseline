"""Principal-angle / subspace-overlap test.

SE is a scalar -> one readout direction (already tested via the cosine sweep; it aligns).
So the meaningful 'whole subspace' question is about the REPRESENTATION subspace that
direction lives in: after the exact Procrustes alignment (raw-centered states, orthogonal
W, as in procrustes_alignment.py), do the models' top-k STATE subspaces coincide
dimension-by-dimension, or does only the dominant axis match?

For each aligned model vs the anchor (Llama-2, TBG L22):
  - top-k PCA subspace of the anchor's raw-centered train states  vs  top-k PCA subspace
    of the model's ALIGNED (Xc @ W) raw-centered train states.
  - principal angles between them: singular values of P_aᵀ P_x = cos(angles) in [0,1].
    Report mean cos (overall overlap) and #dims with cos>0.7 (how many truly coincide).
  - same-model split-half ceiling: two halves of the anchor, subspace from each, principal
    angles between them = the noise floor of subspace estimation at that k.
Swept over k so the read doesn't hinge on one cutoff.
"""
import sys
sys.path.insert(0, "/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes")
import numpy as np
from scipy.linalg import orthogonal_procrustes
from sklearn.model_selection import train_test_split
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits, SEED

L = 22
ANCHOR = "Llama-2-7b-chat"
MODELS = [ANCHOR, "Mistral-7B-Instruct-v0.2", "Meta-Llama-3-8B-Instruct", "deepseek-llm-7b-chat"]
SHORT = {ANCHOR: "Llama-2", "Mistral-7B-Instruct-v0.2": "Mistral",
         "Meta-Llama-3-8B-Instruct": "Llama-3", "deepseek-llm-7b-chat": "deepseek"}
KS = [5, 10, 25, 50, 100, 200]

mats, ids0 = {}, None
for m in MODELS:
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000)
    hidden, _, ids = load_matrix(cfg, ["TBG"])
    mats[m] = hidden["TBG"][L]
    assert ids0 is None or ids == ids0
    ids0 = ids
tr, va, te = splits(len(ids0))
print(f"L{L}: N={len(ids0)}, D={mats[ANCHOR].shape[1]}, |train|={len(tr)}\n")

def pcs(states):
    """Orthonormal PCs (columns), variance-ordered, of raw-centered states."""
    C = states - states.mean(0, keepdims=True)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    return Vt.T                                            # [D, r]
def principal_cos(Pa, Pb, k):
    """cos of principal angles between the two top-k subspaces (descending)."""
    M = Pa[:, :k].T @ Pb[:, :k]
    return np.linalg.svd(M, compute_uv=False)             # length-k, each in [0,1]

# MATCHED sample size: every subspace estimated from the SAME 720 rows, and every W fit on
# 720 rows, so the same-model ceiling and the cross-model number are a fair comparison.
hA, hB = train_test_split(tr, test_size=0.5, random_state=SEED)
hA, hB = np.sort(hA), np.sort(hB)                          # 720 each
print(f"matched n per subspace = {len(hA)} (anchor ref = half A; ceiling partner = half B)\n")

# reference = anchor on half A (720); ceiling partner = anchor on half B (720)
P_anchor = pcs(mats[ANCHOR][hA])
P_ceiling = pcs(mats[ANCHOR][hB])

# each model's ALIGNED subspace, also from half A (720), W fit on half A
aligned_P = {ANCHOR: P_anchor}
AcA = mats[ANCHOR][hA] - mats[ANCHOR][hA].mean(0, keepdims=True)
for m in [x for x in MODELS if x != ANCHOR]:
    XcA = mats[m][hA] - mats[m][hA].mean(0, keepdims=True)
    W, _ = orthogonal_procrustes(XcA, AcA)                # 720-row Procrustes fit
    aligned_P[m] = pcs(XcA @ W)
P_hA, P_hB = P_anchor, P_ceiling                          # ceiling: anchor half A vs half B (both 720)

nonanchor = [m for m in MODELS if m != ANCHOR]
print("MEAN cos(principal angle) between top-k STATE subspaces (1.0 = subspaces coincide):")
print("  " + f"{'k':>5}" + "".join(f"{SHORT[m]:>10s}" for m in nonanchor) + f"{'anchorCeil':>12s}")
for k in KS:
    line = "  " + f"{k:>5}"
    for m in nonanchor:
        line += f"{principal_cos(P_anchor, aligned_P[m], k).mean():>10.3f}"
    line += f"{principal_cos(P_hA, P_hB, k).mean():>12.3f}"
    print(line)

print("\n#DIMS that truly coincide (cos(principal angle) > 0.7), out of k:")
print("  " + f"{'k':>5}" + "".join(f"{SHORT[m]:>10s}" for m in nonanchor) + f"{'anchorCeil':>12s}")
for k in KS:
    line = "  " + f"{k:>5}"
    for m in nonanchor:
        line += f"{int((principal_cos(P_anchor, aligned_P[m], k) > 0.7).sum()):>10d}"
    line += f"{int((principal_cos(P_hA, P_hB, k) > 0.7).sum()):>12d}"
    print(line)

print("\nread: mean cos ~ anchor ceiling AND high  => the whole top-k subspace coincides,")
print("      not just the dominant axis;  dims>0.7 counts how many directions genuinely match.")
