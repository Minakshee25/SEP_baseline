"""Throwaway diagnostic: is the ~0.4 split-half readout-direction cosine real, and why?

Anchor only (Llama-2, TBG L22), in the model's OWN standardized space (no W, no anchor_std
rescaling -- the cleanest possible question). Over many RANDOM half-splits and swept over
ridge alpha, report:
  - cosine(coef_A, coef_B)  for two ridges on disjoint random halves
  - Spearman(predA, predB)  on the held-out test states  (prediction stability)
so we can see directly whether coefficients are unstable while predictions are stable,
and how it depends on regularization + data size.
"""
import sys, os
sys.path.insert(0, "/vol/bitbucket/mn1025/individual_project/semantic-entropy-probes")
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from amortized_ue.config import Stage1Config
from amortized_ue.linear_ceiling_probe import load_matrix, splits

L = 22
cfg = Stage1Config(model_name="Llama-2-7b-chat", dataset="trivia_qa", num_samples=2000)
hidden, y, ids = load_matrix(cfg, ["TBG"])
X = hidden["TBG"][L]                       # [N, 4096]
tr, va, te = splits(len(ids))
N, D = X.shape
print(f"anchor Llama-2 TBG L{L}: N={N}, D={D}, |train|={len(tr)} (half={len(tr)//2}), |test|={len(te)}")

# effective rank of the (standardized) training gram -- how ill-conditioned is the fit?
Xtr = StandardScaler().fit_transform(X[tr])
s = np.linalg.svd(Xtr, compute_uv=False)
ev = s**2
eff_rank = (ev.sum()**2) / (ev**2).sum()
print(f"features D={D}, effective rank of train covariance ~= {eff_rank:.1f} "
      f"(=> {D/eff_rank:.0f}x more dims than effective directions)\n")

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
def rho(a, b):
    return float(spearmanr(a, b).correlation)

rng = np.random.default_rng(0)
K = 15
print(f"{'alpha':>8}{'coef cosine':>14}{'pred Spearman':>16}   (mean +- std over {K} random half-splits)")
for alpha in [1.0, 10.0, 100.0, 1000.0, 10000.0, 100000.0]:
    coscs, preds = [], []
    for _ in range(K):
        perm = rng.permutation(tr)
        h = len(perm) // 2
        hA, hB = perm[:h], perm[h:]
        scA = StandardScaler().fit(X[hA]); scB = StandardScaler().fit(X[hB])
        mA = Ridge(alpha=alpha).fit(scA.transform(X[hA]), y[hA])
        mB = Ridge(alpha=alpha).fit(scB.transform(X[hB]), y[hB])
        coscs.append(cos(mA.coef_, mB.coef_))
        preds.append(rho(mA.predict(scA.transform(X[te])), mB.predict(scB.transform(X[te]))))
    print(f"{alpha:>8.0f}{np.mean(coscs):>9.3f} +-{np.std(coscs):<4.3f}"
          f"{np.mean(preds):>11.3f} +-{np.std(preds):<4.3f}")

# control: full-overlap sanity -- same half twice must give cosine 1.000
sc = StandardScaler().fit(X[tr[:len(tr)//2]])
m1 = Ridge(alpha=10000).fit(sc.transform(X[tr[:len(tr)//2]]), y[tr[:len(tr)//2]])
m2 = Ridge(alpha=10000).fit(sc.transform(X[tr[:len(tr)//2]]), y[tr[:len(tr)//2]])
print(f"\nsanity (identical data twice) cosine = {cos(m1.coef_, m2.coef_):.3f}  (must be 1.000)")
