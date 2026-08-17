"""Matched-total control, CLEAN partitioned design.

Fixes the weaknesses of matched_total.py: single and pooled now use the SAME question set
and the SAME unique-question count; the ONLY difference is model-routing.

For each held-out target T, at each `total` budget:
  draw ONE set of `total` questions from tr (seeded).
  - single : all `total` questions routed through Llama-2            -> test T
  - pooled : PARTITION those same `total` questions into 3 equal groups, one per source,
             each group routed through its source model (aligned)     -> test T
Same questions, same rows, same unique-question count. pooled>single => genuine diversity
(model-routing), not row count / question coverage. Per-model normalization; W + feature
scalers label-free on full train. 4 seeds (random question set + partition).
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
TARGETS = MODELS[1:]
TOTALS = [150, 300, 600, 1200, 1440]                       # divisible by 3
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]
SEEDS = 4

DATA_DIR = None   # set to a node-local copy (e.g. "/data2/<user>/stage1") to dodge NFS stalls; None => repo default
mats, ys, ids0 = {}, {}, None
for m in MODELS:
    _kw = {"output_dir": DATA_DIR} if DATA_DIR else {}
    cfg = Stage1Config(model_name=m, dataset="trivia_qa", num_samples=2000, **_kw)
    hidden, y, ids = load_matrix(cfg, ["TBG"])
    mats[m], ys[m] = hidden["TBG"][L], y
    assert ids0 is None or ids == ids0
    ids0 = ids
tr, va, te = splits(len(ids0))
print(f"L{L}: train pool={len(tr)}; SAME questions both arms, only model-routing differs\n")

def rho(a, b):
    if np.std(a) < 1e-12 or np.std(b) < 1e-12: return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if r is None or np.isnan(r) else float(r)

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

def fit_transfer(src_subs, T):                             # {model: idx-into-tr}
    Xtr, ytr, Xva, yva = [], [], [], []
    for m, s in src_subs.items():
        mu, sd = ys[m][tr][s].mean(), ys[m][tr][s].std() + 1e-12
        Xtr.append(feat(m, tr[s])); ytr.append((ys[m][tr][s] - mu) / sd)
        Xva.append(feat(m, va));    yva.append((ys[m][va] - mu) / sd)
    Xtr, ytr, Xva, yva = np.vstack(Xtr), np.concatenate(ytr), np.vstack(Xva), np.concatenate(yva)
    best = None
    for a in ALPHAS:
        r = Ridge(alpha=a).fit(Xtr, ytr)
        sc = rho(r.predict(Xva), yva)
        if best is None or sc > best[0]: best = (sc, r)
    return rho(best[1].predict(feat(T, te)), ys[T][te])

print(f"{'total':>6} | " + "".join(f"{SHORT[T]+' s/p':>18s}" for T in TARGETS) + "   (single / pooled-PARTITIONED)")
print("-" * 78)
for total in TOTALS:
    per = total // 3
    row = {T: {"s": [], "p": []} for T in TARGETS}
    for seed in range(SEEDS):
        rng = np.random.default_rng(seed)
        q = rng.choice(len(tr), size=total, replace=False)     # ONE question set (shared by both arms)
        g = [q[0:per], q[per:2 * per], q[2 * per:3 * per]]      # 3 equal partitions
        for T in TARGETS:
            srcs = [m for m in MODELS if m != T]
            row[T]["s"].append(fit_transfer({ANCHOR: q}, T))                       # all q -> Llama-2
            row[T]["p"].append(fit_transfer({srcs[i]: g[i] for i in range(3)}, T)) # partition -> 3 srcs
    line = f"{total:>6} | "
    for T in TARGETS:
        line += f"   {np.mean(row[T]['s']):.3f}/{np.mean(row[T]['p']):.3f}"
    print(line)
print("\nread: SAME questions + SAME rows, only 1-model vs 3-model routing differs.")
print("  pooled(p) > single(s) => diversity genuinely helps; p ~= s => routing across models adds nothing.")
