"""Stage 2 data: load Stage-1 records, build z/text/label tensors, split, transform.

Read-only over Stage-1: records are loaded via the existing loaders and joined by
id. The continuous label `cluster_assignment_entropy` is the regression target.
The split reproduces the Stage-1 diagnostic exactly (test 0.1, then val 0.2 of the
remainder, seed 42, over the id-sorted order) so the held-out test set matches.

Hidden states are referred to by physical position only:
  TBG = last input token before generation; SLT = second-last generated token.
Both are stored as [L+1, 1, H] (L+1 = 33 = embedding + 32 layers). z for a given
(position, layer) is hidden[position][layer] squeezed to [H].
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
from amortized_ue.stage2.config import Stage2Config


# ---- diagnostic binarisation (verbatim from sanity_probe.py / SEP) -------------
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


def binarize_entropy(entropy: torch.Tensor, thres: float) -> torch.Tensor:
    out = torch.full_like(entropy, -1, dtype=torch.float)
    out[entropy < thres] = 0
    out[entropy > thres] = 1
    return out


@dataclass
class TargetTransform:
    """Standardise the continuous target on train stats; invert for reporting."""
    mean: float
    std: float

    def encode(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def decode(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean


class Stage2Data:
    """Holds all records in memory and serves z/text/labels by split and (pos, layer)."""

    def __init__(self, cfg: Stage2Config):
        self.cfg = cfg
        s1 = Stage1Config(
            model_name=cfg.stage1_model_name,
            dataset=cfg.stage1_dataset,
            num_samples=cfg.stage1_num_samples,
            load_source=cfg.stage1_load_source,
        )
        records = load_records(s1)
        self.ids = sorted(records.keys())               # deterministic; join by id
        if cfg.smoke:
            self.ids = self.ids[: cfg.smoke_num_prompts]

        self.questions = [records[i]["question"] for i in self.ids]
        self.responses = [records[i]["canonical"]["response"] for i in self.ids]

        # hidden[pos] -> [L+1, N, H]; positions referred to by physical meaning.
        self.positions = list(cfg.sweep_positions)
        self.hidden = {}
        for pos in self.positions:
            stacked = torch.stack(
                [records[i]["canonical"]["hidden_states"][pos] for i in self.ids]
            )  # [N, L+1, 1, H]
            self.hidden[pos] = stacked.squeeze(-2).transpose(0, 1).to(torch.float32)  # [L+1, N, H]

        self.n_layers = self.hidden[self.positions[0]].shape[0]
        self.hidden_size = self.hidden[self.positions[0]].shape[-1]

        self.labels_raw = torch.tensor(
            [records[i]["labels"]["cluster_assignment_entropy"] for i in self.ids],
            dtype=torch.float32,
        )

        # --- split: reproduce the Stage-1 diagnostic membership exactly ----------
        idx = np.arange(len(self.ids))
        tv_idx, test_idx = train_test_split(
            idx, test_size=cfg.test_size, random_state=cfg.split_seed)
        train_idx, val_idx = train_test_split(
            tv_idx, test_size=cfg.val_size, random_state=cfg.split_seed)
        self.train_idx = np.sort(train_idx)
        self.val_idx = np.sort(val_idx)
        self.test_idx = np.sort(test_idx)

        # --- target transform fit on train only ----------------------------------
        train_y = self.labels_raw[self.train_idx]
        if cfg.target_transform == "standardize":
            self.transform = TargetTransform(float(train_y.mean()), float(train_y.std() + 1e-8))
        else:
            raise ValueError(f"unknown target_transform {cfg.target_transform!r}")

        # --- diagnostic binarisation for AUROC (threshold from train labels) -----
        self.bin_threshold = best_split(train_y)
        self.labels_bin = binarize_entropy(self.labels_raw, self.bin_threshold)

    def split_indices(self, split: str) -> np.ndarray:
        return {"train": self.train_idx, "val": self.val_idx, "test": self.test_idx}[split]

    def z(self, position: str, layer: int, rows: np.ndarray) -> torch.Tensor:
        """z vectors [len(rows), H] at a physical position and layer."""
        return self.hidden[position][layer][rows]

    def example_view(self, position: str, layer: int, split: str) -> dict:
        """Everything a training/eval loop needs for one split and one (pos, layer)."""
        rows = self.split_indices(split)
        return {
            "rows": rows,
            "ids": [self.ids[r] for r in rows],
            "z": self.z(position, layer, rows),                    # [n, H] float32
            "questions": [self.questions[r] for r in rows],
            "responses": [self.responses[r] for r in rows],
            "y_raw": self.labels_raw[rows],                        # original space
            "y_std": self.transform.encode(self.labels_raw[rows]),  # training target
            "y_bin": self.labels_bin[rows],                        # for AUROC
        }
