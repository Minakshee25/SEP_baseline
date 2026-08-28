"""Estimate the label-noise ceiling on the SE target, read-only over Stage-1 records.

The `cluster_assignment_entropy` label is an *estimate* of SE from a finite number of
high-temperature samples (N=10). Re-sampling would give a different value for the same
prompt, so the label carries measurement noise. No model -- however good -- can rank
against a noisy target better than the target ranks against itself. That self-agreement
is the ceiling, and it is what turns "Spearman 0.47" into "we recover X% of the
achievable signal".

Method (split-half reliability):
  1. Split each prompt's stored `semantic_ids` into two disjoint halves of n//2.
  2. Recompute `cluster_assignment_entropy` on each half (SEP's own function, reused
     unchanged) -> two independent SE estimates per prompt.
  3. Spearman across prompts between half-A and half-B = reliability of an (n//2)-sample
     estimate. Averaged over `--repeats` random splits.
  4. Spearman-Brown up-corrects that to the reliability of the actual n-sample label:
         rel_n = 2*r_half / (1 + r_half)
  5. Ceiling on the observable correlation = sqrt(rel_n)   (classical attenuation)
  6. recovered% = observed Spearman / ceiling

Reuses the stored `semantic_id`s, so it needs no LLM and no entailment re-run: this
measures the finite-sample noise in the entropy estimate, holding the DeBERTa clustering
fixed. Total label noise is therefore somewhat higher than measured, making this ceiling
mildly optimistic -- i.e. the true recovered% is a little HIGHER than reported here.

Run from the repo root in the `se_probes` env:
    python -m amortized_ue.label_noise_ceiling
    python -m amortized_ue.label_noise_ceiling --dataset squad --num_samples 1000
"""
from __future__ import annotations

import os
import json
import glob
import argparse

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.sep_bridge import cluster_assignment_entropy


def _se_of_subset(semantic_ids: list) -> float:
    """SE of a subset of samples, via SEP's cluster_assignment_entropy (unmodified).

    That function calls np.bincount, which assumes cluster ids are contiguous from 0.
    A subset generally omits some clusters, leaving zero counts -> 0*log(0) = nan. We
    therefore relabel the subset's ids to 0..k-1 first. Relabelling is a pure renaming:
    the entropy depends only on the multiset of cluster counts, so the value is exactly
    what SEP would compute for those samples.
    """
    _, contiguous = np.unique(np.asarray(semantic_ids), return_inverse=True)
    return float(cluster_assignment_entropy(contiguous))


def load_semantic_ids(cfg: Stage1Config) -> tuple[list, list]:
    """Return (ids, semantic_ids_per_prompt), ordered exactly as Stage2Data orders them.

    Stage2Data uses `sorted(records.keys())`, so we match that to keep the split
    replication below row-for-row identical to the training/eval code.
    """
    records_dir = os.path.join(cfg.run_dir(), "records")
    if not os.path.isdir(records_dir):
        raise FileNotFoundError(f"No records dir at {records_dir!r}")

    by_id = {}
    for path in sorted(glob.glob(os.path.join(records_dir, "*.pt"))):
        # Load only what we need; the hidden states are irrelevant here.
        rec = torch.load(path, map_location="cpu")
        by_id[rec["id"]] = list(rec["labels"]["semantic_ids"])

    ids = sorted(by_id.keys())
    return ids, [by_id[i] for i in ids]


def split_half_reliability(sem_ids: list, repeats: int, seed: int) -> tuple[float, float]:
    """Mean +/- std Spearman between SE computed on two disjoint halves of the samples."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(repeats):
        a, b = [], []
        for ids_i in sem_ids:
            n = len(ids_i)
            half = n // 2
            if half == 0:
                continue
            perm = rng.permutation(n)
            arr = np.asarray(ids_i)
            a.append(_se_of_subset(arr[perm[:half]]))
            b.append(_se_of_subset(arr[perm[half : 2 * half]]))
        a, b = np.asarray(a), np.asarray(b)
        if a.std() < 1e-12 or b.std() < 1e-12:
            continue
        rho = spearmanr(a, b).correlation
        if rho is not None and not np.isnan(rho):
            rs.append(float(rho))
    if not rs:
        raise RuntimeError("no valid split-half draws (labels degenerate?)")
    return float(np.mean(rs)), float(np.std(rs))


def spearman_brown(r_half: float) -> float:
    """Reliability of the full n-sample label, given the reliability of an n/2 one."""
    return 2.0 * r_half / (1.0 + r_half)


def test_rows(n: int, test_size: float, split_seed: int) -> np.ndarray:
    """Replicate Stage2Data's held-out test split (same call, same seeds)."""
    idx = np.arange(n)
    _, test_idx = train_test_split(idx, test_size=test_size, random_state=split_seed)
    return np.sort(test_idx)


def ceiling_report(name: str, sem_ids: list, repeats: int, seed: int) -> dict:
    r_half, r_std = split_half_reliability(sem_ids, repeats, seed)
    rel_n = spearman_brown(r_half)
    ceiling = float(np.sqrt(max(rel_n, 0.0)))
    n_samples = int(np.median([len(s) for s in sem_ids]))
    out = {
        "rows": len(sem_ids),
        "samples_per_prompt": n_samples,
        "r_half": r_half,
        "r_half_std": r_std,
        "reliability_n": rel_n,
        "ceiling": ceiling,
    }
    print(
        f"  {name:<28s} rows={out['rows']:<5d} n={n_samples}  "
        f"r_half={r_half:.4f}±{r_std:.4f}  rel_n={rel_n:.4f}  ceiling={ceiling:.4f}"
    )
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", default="Llama-2-7b-chat")
    p.add_argument("--dataset", default="trivia_qa")
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--repeats", type=int, default=200,
                   help="random split-half draws to average over")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test_size", type=float, default=0.1, help="matches Stage2Config")
    p.add_argument("--split_seed", type=int, default=42, help="matches Stage2Config")
    p.add_argument("--observed", type=str, default=None,
                   help="path to a *_multiseed.json to convert ceilings into recovered%")
    p.add_argument("--observed_block", default="summary",
                   help="'summary' (ID test) or 'ood_summary' (OOD all-rows)")
    p.add_argument("--out", default=None, help="optional path to write JSON")
    p.add_argument("--data_dir", default=None,
                   help="Stage-1 records root (default: amortized_ue/data/stage1); "
                        "point at /data2/mn1025/stage1 to read the fast off-NFS copy")
    args = p.parse_args()

    cfg_kw = dict(
        model_name=args.model_name, dataset=args.dataset, num_samples=args.num_samples)
    if args.data_dir:
        cfg_kw["output_dir"] = args.data_dir
    cfg = Stage1Config(**cfg_kw)
    ids, sem_ids = load_semantic_ids(cfg)

    print(f"\n{args.dataset} n={args.num_samples}  ({len(ids)} records)")
    print(f"label-noise ceiling  ({args.repeats} split-half draws, Spearman-Brown corrected)")

    res = {"dataset": args.dataset, "num_samples": args.num_samples, "repeats": args.repeats}

    # All rows: the ceiling that applies to the OOD evaluation (which scores every row).
    res["all_rows"] = ceiling_report("all rows (OOD eval basis)", sem_ids, args.repeats, args.seed)

    # Held-out test rows: the ceiling that applies to the reported ID test Spearman.
    t_idx = test_rows(len(ids), args.test_size, args.split_seed)
    res["test_rows"] = ceiling_report(
        "test rows (ID eval basis)", [sem_ids[i] for i in t_idx], args.repeats, args.seed)

    # Convert observed Spearman -> % of achievable signal recovered.
    if args.observed:
        with open(args.observed) as f:
            obs = json.load(f)
        block = obs[args.observed_block]
        basis = "all_rows" if args.observed_block == "ood_summary" else "test_rows"
        ceiling = res[basis]["ceiling"]
        print(f"\n  recovered signal (observed / ceiling), basis={basis}, ceiling={ceiling:.4f}")
        res["recovered"] = {}
        for arm, m in block.items():
            observed = m["spearman"]["mean"]
            frac = observed / ceiling if ceiling > 0 else float("nan")
            res["recovered"][arm] = {
                "observed_spearman": observed, "ceiling": ceiling, "recovered_frac": frac}
            print(f"    {arm:<10s} observed={observed:.4f}  ->  {frac:6.1%} of achievable")

    if args.out:
        with open(args.out, "w") as f:
            json.dump(res, f, indent=1)
        print(f"\n  wrote {args.out}")
    print()


if __name__ == "__main__":
    main()
