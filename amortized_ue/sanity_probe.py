"""THROWAWAY diagnostic: SEP-style probe on the amortized-UE Stage-1 records.

Quick sanity check before building the SLM proxy: do the stored hidden states
carry learnable signal about the semantic-entropy label? For each layer we fit a
plain logistic regression on that single layer's hidden state and predict a
BINARISED version of the SE label, then report per-layer test AUROC for the TBG
and SLT positions.

This is diagnostic code only. It does NOT modify Stage-1 code or data, saves no
models, and touches no W&B. Binarisation happens here only; the stored continuous
`cluster_assignment_entropy` labels are never changed. The probe logic
(best_split, binarize_entropy, per-layer LogisticRegression, the 0.2/0.1 split at
seed 42) mirrors semantic_entropy_probes/run_llama2_probe.py so the check is
faithful to the SEP baseline.

Run from the repo root with the se_probes env active:
    python -m amortized_ue.sanity_probe
"""
from __future__ import annotations

import os
import warnings

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records

warnings.filterwarnings("ignore")  # match SEP: plain default LR may not converge on 4096-dim

SEED = 42
VAL_SIZE = 0.2   # of the train+val remainder, as in SEP create_Xs_and_ys
TEST_SIZE = 0.1  # of all data, as in SEP create_Xs_and_ys
PLOT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sanity_probe_auroc.png")


# ---- verbatim from run_llama2_probe.py (SEP notebook cells 16) ----
def best_split(entropy: torch.Tensor) -> float:
    ents = entropy.numpy()
    splits = np.linspace(1e-10, ents.max(), 100)
    split_mses = []
    for split in splits:
        low_idxs, high_idxs = ents < split, ents >= split
        low_mean = np.mean(ents[low_idxs])
        high_mean = np.mean(ents[high_idxs])
        mse = np.sum((ents[low_idxs] - low_mean) ** 2) + np.sum((ents[high_idxs] - high_mean) ** 2)
        split_mses.append(np.sum(mse))
    return float(splits[int(np.argmin(np.array(split_mses)))])


def binarize_entropy(entropy: torch.Tensor, thres: float = 0.0) -> torch.Tensor:
    binary_entropy = torch.full_like(entropy, -1, dtype=torch.float)
    binary_entropy[entropy < thres] = 0
    binary_entropy[entropy > thres] = 1
    return binary_entropy


def load_matrices(config: Stage1Config):
    """Return (TBG [L,N,H], SLT [L,N,H], entropy [N]) in a fixed id-sorted order."""
    records = load_records(config)
    ids = sorted(records.keys())  # deterministic order; join is by id, not position
    tbg = torch.stack([records[i]["canonical"]["hidden_states"]["TBG"] for i in ids])
    slt = torch.stack([records[i]["canonical"]["hidden_states"]["SLT"] for i in ids])
    # [N, L, 1, H] -> [L, N, H]
    tbg = tbg.squeeze(-2).transpose(0, 1).to(torch.float32)
    slt = slt.squeeze(-2).transpose(0, 1).to(torch.float32)
    entropy = torch.tensor(
        [records[i]["labels"]["cluster_assignment_entropy"] for i in ids]
    ).to(torch.float32)
    return tbg, slt, entropy, len(ids)


def report_label_distribution(entropy: torch.Tensor) -> None:
    e = entropy.numpy()
    print("\n--- SE label (cluster_assignment_entropy) distribution ---")
    print(f"  N={e.size}  min={e.min():.4f}  max={e.max():.4f}  "
          f"mean={e.mean():.4f}  std={e.std():.4f}  median={np.median(e):.4f}")
    frac_zero = float(np.mean(e <= 1e-9))
    print(f"  fraction exactly 0 (single-cluster prompts): {frac_zero:.3f}")
    # coarse histogram across the observed range
    counts, edges = np.histogram(e, bins=8)
    for c, lo, hi in zip(counts, edges[:-1], edges[1:]):
        bar = "#" * int(40 * c / max(counts.max(), 1))
        print(f"  [{lo:5.2f},{hi:5.2f})  {c:4d} {bar}")
    spread = "healthy spread" if e.std() > 0.05 and frac_zero < 0.95 else "WARNING: little spread"
    print(f"  -> {spread}")


def check_hidden_states(name: str, mat: torch.Tensor, n: int) -> None:
    print(f"\n--- hidden-state check: {name} ---")
    print(f"  shape={tuple(mat.shape)} (expected [L+1, N={n}, H]), dtype={mat.dtype}")
    n_nan = int(torch.isnan(mat).sum())
    n_inf = int(torch.isinf(mat).sum())
    # an all-zero vector for any (layer, sample) is a red flag
    zero_vecs = int((mat.abs().sum(dim=-1) == 0).sum())
    print(f"  NaNs={n_nan}  Infs={n_inf}  all-zero vectors={zero_vecs}")
    status = "OK" if (n_nan == 0 and n_inf == 0 and zero_vecs == 0) else "WARNING"
    print(f"  -> {status}")


def probe_per_layer(mat: torch.Tensor, y_bin: np.ndarray) -> np.ndarray:
    """Per-layer test AUROC. SEP-style 3-way split (test 0.1, then val 0.2), seed 42."""
    X = mat.numpy()
    n_layers = X.shape[0]
    aurocs = np.full(n_layers, np.nan)
    for layer in range(n_layers):
        Xl = X[layer]
        X_tv, X_test, y_tv, y_test = train_test_split(
            Xl, y_bin, test_size=TEST_SIZE, random_state=SEED)
        X_train, _X_val, y_train, _y_val = train_test_split(
            X_tv, y_tv, test_size=VAL_SIZE, random_state=SEED)
        if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
            continue  # degenerate split -> leave nan
        model = LogisticRegression()
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        aurocs[layer] = roc_auc_score(y_test, probs)
    return aurocs


def summarize(name: str, aurocs: np.ndarray) -> tuple[int, float]:
    best = int(np.nanargmax(aurocs))
    print(f"\n  {name}: mean test AUROC={np.nanmean(aurocs):.3f} | "
          f"best layer {best} AUROC={aurocs[best]:.3f}")
    per_layer = "  ".join(f"L{l}={a:.3f}" for l, a in enumerate(aurocs))
    print(f"    per-layer: {per_layer}")
    return best, float(aurocs[best])


def save_plot(tbg_aurocs: np.ndarray, slt_aurocs: np.ndarray) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    layers = np.arange(len(tbg_aurocs))
    plt.figure(figsize=(8, 5))
    plt.plot(layers, tbg_aurocs, marker="o", label="TBG")
    plt.plot(layers, slt_aurocs, marker="s", label="SLT")
    plt.axhline(0.5, color="grey", ls="--", lw=1, label="chance (0.5)")
    plt.xlabel("layer")
    plt.ylabel("test AUROC")
    plt.title("SEP-style probe on Stage-1 records (binarised SE)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=120)
    print(f"\nSaved AUROC-vs-layer plot -> {PLOT_PATH}")


def main():
    config = Stage1Config(num_samples=400)
    print(f"Loading Stage-1 records from: {config.run_dir()}")
    tbg, slt, entropy, n = load_matrices(config)

    # --- sanity facts ---
    report_label_distribution(entropy)
    check_hidden_states("TBG", tbg, n)
    check_hidden_states("SLT", slt, n)

    # --- binarise (diagnostic only; stored continuous labels untouched) ---
    split = best_split(entropy)
    y_bin = binarize_entropy(entropy, split)
    y = y_bin.numpy().astype(int)
    pos_rate = float(np.mean(y))
    dummy = max(pos_rate, 1 - pos_rate)
    print(f"\n--- binarisation (SEP best_split) ---")
    print(f"  threshold={split:.4f}  positive_rate={pos_rate:.3f}  "
          f"majority/dummy accuracy={dummy:.3f}")

    # --- per-layer probes ---
    print("\n" + "=" * 64)
    print(f"PER-LAYER TEST AUROC  (N={n}, split test={TEST_SIZE}/val={VAL_SIZE}, seed={SEED})")
    print("=" * 64)
    tbg_aurocs = probe_per_layer(tbg, y)
    slt_aurocs = probe_per_layer(slt, y)
    tbg_best, tbg_best_auc = summarize("TBG", tbg_aurocs)
    slt_best, slt_best_auc = summarize("SLT", slt_aurocs)

    save_plot(tbg_aurocs, slt_aurocs)

    # --- verdict ---
    overall_best = max(tbg_best_auc, slt_best_auc)
    print("\n" + "=" * 64)
    print("VERDICT")
    print("=" * 64)
    print(f"  TBG best: layer {tbg_best} AUROC={tbg_best_auc:.3f}")
    print(f"  SLT best: layer {slt_best} AUROC={slt_best_auc:.3f}")
    if overall_best >= 0.6:
        print(f"  -> Best AUROC {overall_best:.3f} is clearly above chance (0.5). "
              f"The hidden states carry learnable SE signal; OK to proceed to the SLM proxy.")
    elif overall_best >= 0.55:
        print(f"  -> Best AUROC {overall_best:.3f} is only modestly above chance. "
              f"Weak but present signal; proceed with caution.")
    else:
        print(f"  -> Best AUROC {overall_best:.3f} hovers near chance (0.5) across layers. "
              f"Investigate the Stage-1 data before building anything further.")


if __name__ == "__main__":
    main()
