"""Stage 2 training / evaluation.

One frozen-backbone proxy, served three ways (z / z+question / z+question+response)
via modality dropout. Trains projector + LoRA + null/REG embeddings + head with MSE
on the standardised target; reports metrics in the original label space.

Modality dropout (training), per example and independent of z-dropout:
  - with p_drop_text : null ALL text            -> z-only
  - else, 50/50      : null response only       -> z+question
                       keep question+response   -> z+question+response
  - independently, with p_drop_z : null the soft tokens
Eval serves a fixed arm deterministically (no dropout).

(position, layer) selection is validation-driven: sweep both physical positions and
all stored layers with the z-only arm and pick the lowest val MSE, then reuse.
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
def _tokenize_segments(tok, questions, responses, max_len):
    """Return left-padded ids + attention + question/response span masks.

    Template (physical placement locked in the spec): the text segment is
    'Question: {q}\nAnswer:' then ' {response}'. Question and response spans are
    tracked so the middle arm can null the response only.
    """
    q_texts = [f"Question: {q}\nAnswer:" for q in questions]
    r_texts = [f" {r}" for r in responses]
    seqs, qlens = [], []
    for qt, rt in zip(q_texts, r_texts):
        q_ids = tok(qt, add_special_tokens=False)["input_ids"]
        r_ids = tok(rt, add_special_tokens=False)["input_ids"]
        ids = (q_ids + r_ids)[:max_len]
        seqs.append(ids)
        qlens.append(min(len(q_ids), len(ids)))
    T = max(len(s) for s in seqs)
    B = len(seqs)
    pad_id = tok.pad_token_id
    input_ids = torch.full((B, T), pad_id, dtype=torch.long)
    attn = torch.zeros((B, T), dtype=torch.long)
    q_mask = torch.zeros((B, T), dtype=torch.long)
    r_mask = torch.zeros((B, T), dtype=torch.long)
    for b, (ids, ql) in enumerate(zip(seqs, qlens)):
        L = len(ids)
        input_ids[b, T - L:] = torch.tensor(ids, dtype=torch.long)  # left pad
        attn[b, T - L:] = 1
        q_mask[b, T - L: T - L + ql] = 1
        r_mask[b, T - L + ql:] = 1
    return input_ids, attn, q_mask, r_mask


def _keep_and_dropz(rng, train, arm, attn, q_mask, r_mask, p_text, p_z):
    """Build text_keep_mask [B,T] and drop_z [B] for training or a fixed eval arm."""
    B, T = attn.shape
    keep = torch.zeros((B, T), dtype=torch.long)
    drop_z = torch.zeros(B, dtype=torch.bool)
    if train:
        for b in range(B):
            if rng.random() < p_text:
                pass                                   # all text nulled -> z-only
            elif rng.random() < 0.5:
                keep[b] = q_mask[b]                    # response nulled -> z+Q
            else:
                keep[b] = attn[b]                      # full text -> z+Q+R
            drop_z[b] = rng.random() < p_z
    else:
        if arm == "z":
            pass
        elif arm == "z_q":
            keep = q_mask.clone()
        elif arm == "z_q_resp":
            keep = attn.clone()
        else:
            raise ValueError(f"unknown arm {arm!r}")
    return keep, drop_z


# ------------------------------- metrics --------------------------------------
def regression_and_ranking(pred_orig: np.ndarray, y_raw: np.ndarray, y_bin: np.ndarray) -> dict:
    err = pred_orig - y_raw
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_raw - y_raw.mean()) ** 2)) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    rho = float(spearmanr(pred_orig, y_raw).correlation)
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
        # snapshot of ALL trainable params at init (projector, head, null/reg embeddings,
        # LoRA — peft inits lora_B=0 so this is the identity-backbone start state).
        self._fresh_state = self._snapshot_trainable()

    # -- snapshot/restore every trainable param so the sweep reuses the loaded backbone --
    def _snapshot_trainable(self):
        return {n: p.detach().clone() for n, p in self.model.named_parameters() if p.requires_grad}

    def _restore(self, snap):
        with torch.no_grad():
            for n, p in self.model.named_parameters():
                if p.requires_grad and n in snap:
                    p.copy_(snap[n])

    def reset_trainable(self):
        self._restore(self._fresh_state)

    def _trainable_params(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def _forward_batch(self, rows, position, layer, train, arm):
        d = self.data
        z = d.hidden[position][layer][rows].unsqueeze(1).to(self.device)   # [B,1,H]
        qs = [d.questions[r] for r in rows]
        rs = [d.responses[r] for r in rows]
        ids, attn, q_mask, r_mask = _tokenize_segments(
            self.model.tokenizer, qs, rs, self.cfg.max_seq_len)
        keep, drop_z = _keep_and_dropz(
            self.rng, train, arm, attn, q_mask, r_mask, self.cfg.p_drop_text, self.cfg.p_drop_z)
        ids, attn, keep = ids.to(self.device), attn.to(self.device), keep.to(self.device)
        drop_z = drop_z.to(self.device)
        return self.model(z, ids, attn, keep, drop_z)

    def train_arm(self, position, layer, arm, max_steps=None, epochs=None, verbose=False):
        cfg = self.cfg
        self.model.train()
        train_rows = self.data.split_indices("train")
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
                pred = self._forward_batch(rows, position, layer, train=True, arm=arm)
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
            pred = self._forward_batch(rows, position, layer, train=False, arm=arm)
            preds.append(pred.float().cpu())
        pred_std = torch.cat(preds)
        pred_orig = self.data.transform.decode(pred_std).numpy()
        y_raw = self.data.labels_raw[rows_all].numpy()
        y_bin = self.data.labels_bin[rows_all].numpy()
        self.model.train()
        return regression_and_ranking(pred_orig, y_raw, y_bin)

    def sweep_pos_layer(self, epochs=None):
        """Train the z-only arm for every (position, layer); pick the lowest val MSE."""
        results, best = [], None
        for pos in self.cfg.sweep_positions:
            for layer in self.cfg.sweep_layers:
                self.reset_trainable()
                out = self.train_arm(pos, layer, arm=self.cfg.select_arm, epochs=epochs)
                val = self.evaluate(pos, layer, self.cfg.select_arm, "val")
                results.append({"position": pos, "layer": layer, **val})
                key = val["rmse"]
                if best is None or key < best["rmse"]:
                    best = {"position": pos, "layer": layer, **val}
                logging.info("sweep pos=%s layer=%d val_rmse=%.4f", pos, layer, val["rmse"])
        return best, results
