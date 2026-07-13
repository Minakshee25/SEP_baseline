"""Diagnostic: how much SE signal is linearly readable from ONE hidden state?

Answers the question the Stage-2 numbers raise: we recover ~51% of the achievable
signal (Spearman 0.467 vs a label-noise ceiling of 0.914) -- is the shortfall the
PROXY's fault, or is the hidden state simply not that informative about SE?

This fits a plain ridge regression from a single (position, layer) hidden state to the
CONTINUOUS `cluster_assignment_entropy` label, on exactly the split Stage-2 uses, and
reports test Spearman. That is a direct, architecture-free measure of the linear
information content of the hidden state, comparable like-for-like with the proxy's
Spearman.

Interpretation:
  linear ~= proxy   -> the hidden state is the bottleneck; the 3B backbone adds little,
                       and the headroom is in WHAT we feed it, not how we model it.
  linear <<  proxy  -> the backbone is genuinely extracting nonlinear signal.
  linear >>  proxy  -> the proxy is underperforming and the shortfall is ours to fix.

Also fits the same probe train-on-trivia / eval-on-squad to get the linear OOD number.

Diagnostic only: reads Stage-1 records, writes no models, touches no W&B, and modifies
nothing under semantic_uncertainty/.

Run from the repo root in the `se_probes` env:
    python -m amortized_ue.linear_ceiling_probe
"""
from __future__ import annotations

import json
import argparse
import warnings

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records

warnings.filterwarnings("ignore")

SEED = 42
TEST_SIZE = 0.1   # matches Stage2Config
VAL_SIZE = 0.2    # matches Stage2Config
ALPHAS = [1.0, 10.0, 100.0, 1000.0, 10000.0]


def load_matrix(cfg: Stage1Config, positions: list[str]):
    """Return (hidden{pos: [L+1, N, H] float32}, labels [N], ids) ordered like Stage2Data."""
    records = load_records(cfg)
    ids = sorted(records.keys())                      # same ordering as Stage2Data
    hidden = {}
    for pos in positions:
        stacked = torch.stack([records[i]["canonical"]["hidden_states"][pos] for i in ids])
        hidden[pos] = stacked.squeeze(-2).transpose(0, 1).to(torch.float32).numpy()
    labels = np.array(
        [records[i]["labels"]["cluster_assignment_entropy"] for i in ids], dtype=np.float32)
    return hidden, labels, ids


def splits(n: int):
    idx = np.arange(n)
    tv, test = train_test_split(idx, test_size=TEST_SIZE, random_state=SEED)
    train, val = train_test_split(tv, test_size=VAL_SIZE, random_state=SEED)
    return np.sort(train), np.sort(val), np.sort(test)


def rho(a, b) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    r = spearmanr(a, b).correlation
    return 0.0 if (r is None or np.isnan(r)) else float(r)


def fit_probe(X, y, tr, va, alphas=ALPHAS):
    """Ridge with alpha chosen on val Spearman. Returns (model, scaler, best_alpha)."""
    scaler = StandardScaler().fit(X[tr])
    Xtr, Xva = scaler.transform(X[tr]), scaler.transform(X[va])
    best, best_a, best_m = -np.inf, None, None
    for a in alphas:
        m = Ridge(alpha=a).fit(Xtr, y[tr])
        s = rho(m.predict(Xva), y[va])
        if s > best:
            best, best_a, best_m = s, a, m
    return best_m, scaler, best_a, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--ood_dataset", default="squad")
    p.add_argument("--ood_num_samples", type=int, default=1000)
    p.add_argument("--positions", nargs="+", default=["TBG", "SLT"])
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cfg = Stage1Config(model_name=args.model_name, dataset=args.dataset,
                       num_samples=args.num_samples)
    hidden, y, ids = load_matrix(cfg, args.positions)
    tr, va, te = splits(len(ids))
    n_layers = hidden[args.positions[0]].shape[0]
    print(f"\nID {args.dataset} n={len(ids)}  split {len(tr)}/{len(va)}/{len(te)}  "
          f"layers={n_layers}")

    ood_hidden = ood_y = None
    if args.ood_dataset:
        ocfg = Stage1Config(model_name=args.model_name, dataset=args.ood_dataset,
                            num_samples=args.ood_num_samples)
        ood_hidden, ood_y, ood_ids = load_matrix(ocfg, args.positions)
        print(f"OOD {args.ood_dataset} n={len(ood_ids)} (all rows, eval-only)")

    print("\nridge: hidden state -> continuous SE, test Spearman (ID) / all-rows (OOD)")
    print(f"{'pos':<5}{'layer':>6}{'alpha':>8}{'val':>8}{'ID test':>9}{'OOD':>8}")
    results, best = [], {"id": (-np.inf, None), "ood": (-np.inf, None)}
    for pos in args.positions:
        for layer in range(n_layers):
            X = hidden[pos][layer]                       # [N, H]
            m, sc, alpha, val_s = fit_probe(X, y, tr, va)
            id_s = rho(m.predict(sc.transform(X[te])), y[te])
            ood_s = float("nan")
            if ood_hidden is not None:
                ood_s = rho(m.predict(sc.transform(ood_hidden[pos][layer])), ood_y)
            results.append({"position": pos, "layer": layer, "alpha": alpha,
                            "val_spearman": val_s, "id_test_spearman": id_s,
                            "ood_spearman": ood_s})
            if id_s > best["id"][0]:
                best["id"] = (id_s, (pos, layer))
            if not np.isnan(ood_s) and ood_s > best["ood"][0]:
                best["ood"] = (ood_s, (pos, layer))
            print(f"{pos:<5}{layer:>6}{alpha:>8.0f}{val_s:>8.3f}{id_s:>9.3f}{ood_s:>8.3f}")

    print(f"\nbest ID  : {best['id'][0]:.4f}  at {best['id'][1]}")
    print(f"best OOD : {best['ood'][0]:.4f}  at {best['ood'][1]}")
    sel = [r for r in results if r["position"] == "TBG" and r["layer"] == 12]
    if sel:
        r = sel[0]
        print(f"at the proxy's selected TBG L12: ID {r['id_test_spearman']:.4f}  "
              f"OOD {r['ood_spearman']:.4f}")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"results": results,
                       "best_id": {"spearman": best["id"][0], "at": best["id"][1]},
                       "best_ood": {"spearman": best["ood"][0], "at": best["ood"][1]}},
                      f, indent=1)
        print(f"\nwrote {args.out}")
    print()


if __name__ == "__main__":
    main()
