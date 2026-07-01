"""Stage 2 training / evaluation.

Separate model per arm. Each arm is trained on its own fixed, null-free sequence
(no modality dropout):
  - z         : [k soft][REG]
  - z_q       : [k soft][ "Question: {q}\nAnswer:" tokens ][REG]
  - z_q_resp  : [k soft][ "Question: {q}\nAnswer: {response}" tokens ][REG]
MSE on the standardised target; metrics reported in the original label space.

(position, layer) selection is validation-driven: sweep both physical positions and
all stored layers with the z-only arm, trained on a fixed TRAIN-only subsample, and
pick the highest validation Spearman. k is then ablated (z-only) and the best k used
for all three arm models.
"""
from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score
from transformers import get_cosine_schedule_with_warmup

from amortized_ue.stage2.config import Stage2Config
from amortized_ue.stage2.data import Stage2Data
from amortized_ue.stage2.model import ProxyModel


# ------------------------------- text batching --------------------------------
def _arm_text(arm: str, question: str, response: str):
    """The text string an arm reveals (None for the z-only arm)."""
    if arm == "z":
        return None
    if arm == "z_q":
        return f"Question: {question}\nAnswer:"
    if arm == "z_q_resp":
        return f"Question: {question}\nAnswer: {response}"
    raise ValueError(f"unknown arm {arm!r}")


def _tokenize_arm(tok, questions, responses, arm, max_len):
    """Left-padded (input_ids, attention_mask) for an arm's revealed text, or (None,None)."""
    if arm == "z":
        return None, None
    seqs = []
    for q, r in zip(questions, responses):
        ids = tok(_arm_text(arm, q, r), add_special_tokens=False)["input_ids"][:max_len]
        seqs.append(ids)
    T = max(len(s) for s in seqs)
    B = len(seqs)
    pad_id = tok.pad_token_id
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    for b, ids in enumerate(seqs):
        L = len(ids)
        if L == 0:
            continue
        input_ids[b, T - L:] = torch.tensor(ids, dtype=torch.long)   # left pad
        attn[b, T - L:] = 1
    return input_ids, attn


# ------------------------------- metrics --------------------------------------
def regression_and_ranking(pred_orig: np.ndarray, y_raw: np.ndarray, y_bin: np.ndarray) -> dict:
    err = pred_orig - y_raw
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_raw - y_raw.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    # guard Spearman against constant predictions (scipy returns NaN)
    if np.std(pred_orig) < 1e-12 or np.std(y_raw) < 1e-12:
        rho = 0.0
    else:
        rho = spearmanr(pred_orig, y_raw).correlation
        rho = 0.0 if (rho is None or np.isnan(rho)) else float(rho)
    m = {"rmse": rmse, "mae": mae, "r2": r2, "spearman": rho, "auroc": float("nan")}
    valid = y_bin >= 0                                  # drop ambiguous (==threshold)
    if len(np.unique(y_bin[valid])) == 2:
        m["auroc"] = float(roc_auc_score(y_bin[valid], pred_orig[valid]))
    return m


class Trainer:
    """Holds one proxy (backbone loaded once) and trains/evaluates it per (pos, layer, arm)."""

    def __init__(self, cfg: Stage2Config, data: Stage2Data, device: str | None = None):
        self.cfg = cfg
        self.data = data
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.rng = np.random.default_rng(cfg.seed)
        torch.manual_seed(cfg.seed)
        self.model = ProxyModel(cfg, h_in=data.hidden_size).to(self.device)
        self._fresh_state = self._snapshot_trainable()

    # -- snapshot/restore every trainable param so candidates reuse the loaded backbone --
    def _snapshot_trainable(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def _restore(self, snap):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in snap:
                    p.copy_(snap[n])

    def reset_trainable(self):
        self._restore(self._fresh_state)

    def set_k(self, k: int):
        """Swap the projector for a new k (k ablation) and refresh the fresh-state snapshot."""
        self.model.set_k(k)
        self.model.projector.to(self.device)
        self._fresh_state = self._snapshot_trainable()

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _forward_batch(self, rows, position, layer, arm):
        d = self.data
        z = d.hidden[position][layer][rows].unsqueeze(1).to(self.device)   # [B,1,H]
        ids, attn = _tokenize_arm(
            self.model.tokenizer, [d.questions[r] for r in rows], [d.responses[r] for r in rows],
            arm, self.cfg.max_seq_len)
        if ids is not None:
            ids, attn = ids.to(self.device), attn.to(self.device)
        return self.model(z, ids, attn)

    def train_arm(self, position, layer, arm, train_rows=None, max_steps=None,
                  epochs=None, verbose=False):
        cfg = self.cfg
        self.model.train()
        train_rows = self.data.split_indices("train") if train_rows is None else np.asarray(train_rows)
        y_std = self.data.transform.encode(self.data.labels_raw).to(self.device)
        opt = torch.optim.AdamW(self._trainable_params(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        epochs = epochs if epochs is not None else cfg.epochs
        steps_per_epoch = max(1, int(np.ceil(len(train_rows) / cfg.batch_size)))
        total_steps = max_steps if max_steps is not None else steps_per_epoch * epochs
        sched = get_cosine_schedule_with_warmup(
            opt, int(cfg.warmup_ratio * total_steps), total_steps)
        loss_fn = nn.MSELoss()

        best_val, best_state, patience = float("inf"), None, 0
        step = 0
        for ep in range(epochs):
            order = self.rng.permutation(train_rows)
            for i in range(0, len(order), cfg.batch_size):
                rows = order[i:i + cfg.batch_size]
                pred = self._forward_batch(rows, position, layer, arm)
                loss = loss_fn(pred, y_std[rows])
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._trainable_params(), cfg.grad_clip)
                opt.step()
                sched.step()
                step += 1
                if verbose:
                    logging.info("  step %d loss=%.4f", step, loss.item())
                if max_steps is not None and step >= max_steps:
                    return {"stopped": "max_steps", "step": step, "last_loss": float(loss.item())}
            val = self.evaluate(position, layer, arm, "val")
            if val["rmse"] ** 2 < best_val:
                best_val, patience = val["rmse"] ** 2, 0
                best_state = self._snapshot_trainable()
            else:
                patience += 1
                if patience >= cfg.early_stop_patience:
                    break
        if best_state is not None:
            self._restore(best_state)
        return {"val_mse": best_val}

    @torch.no_grad()
    def evaluate(self, position, layer, arm, split):
        self.model.eval()
        rows_all = self.data.split_indices(split)
        preds = []
        for i in range(0, len(rows_all), self.cfg.batch_size):
            rows = rows_all[i:i + self.cfg.batch_size]
            preds.append(self._forward_batch(rows, position, layer, arm).float().cpu())
        pred_orig = self.data.transform.decode(torch.cat(preds)).numpy()
        y_raw = self.data.labels_raw[rows_all].numpy()
        y_bin = self.data.labels_bin[rows_all].numpy()
        self.model.train()
        return regression_and_ranking(pred_orig, y_raw, y_bin)

    # -- selection helper: which of two candidate metric-dicts is better ----------
    def _is_better(self, cand, best):
        if best is None:
            return True
        m = self.cfg.select_metric
        if m == "val_spearman":
            return cand["spearman"] > best["spearman"]
        if m == "val_auroc":
            return cand["auroc"] > best["auroc"]
        return cand["rmse"] < best["rmse"]              # val_mse

    def sweep_pos_layer(self, train_rows, epochs=None):
        """Train the z-only arm for every (position, layer) on `train_rows`; select by
        the configured validation metric (default val Spearman, higher = better)."""
        results, best = [], None
        for pos in self.cfg.sweep_positions:
            for layer in self.cfg.sweep_layers:
                self.reset_trainable()
                self.train_arm(pos, layer, arm=self.cfg.select_arm,
                               train_rows=train_rows, epochs=epochs)
                val = self.evaluate(pos, layer, self.cfg.select_arm, "val")
                cand = {"position": pos, "layer": layer, **val}
                results.append(cand)
                if self._is_better(cand, best):
                    best = cand
                logging.info("sweep pos=%s layer=%2d val_spearman=%.4f val_auroc=%.4f val_rmse=%.4f",
                             pos, layer, val["spearman"], val["auroc"], val["rmse"])
        return best, results

    def k_ablation(self, position, layer, epochs=None):
        """Ablate k on the z-only arm (full train); select best k by the configured metric."""
        results, best = [], None
        for k in self.cfg.k_ablation_values:
            self.set_k(k)
            self.reset_trainable()
            self.train_arm(position, layer, arm="z", epochs=epochs)
            val = self.evaluate(position, layer, "z", "val")
            cand = {"k": k, **val}
            results.append(cand)
            if self._is_better(cand, best):
                best = cand
            logging.info("k-ablation k=%d val_spearman=%.4f val_auroc=%.4f", k, val["spearman"], val["auroc"])
        return best, results
