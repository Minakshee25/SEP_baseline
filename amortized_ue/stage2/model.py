"""Stage 2 proxy model.

A frozen decoder-only backbone (Llama-3.2-3B) reads, in one forward pass:
  [k soft tokens]  (+ [text tokens])  +  [REG readout token]
and a linear head on the REG token's final hidden state regresses the (standardised)
semantic-entropy target.

  - The stored Stage-1 hidden vector z (from the *target* LLM, dim H_in) is mapped by a
    learned projector into k soft tokens in the proxy's embedding space:
        LayerNorm(H_in) -> Linear(H_in, hidden) -> GELU -> Dropout
        -> Linear(hidden, k*d_model) -> reshape [B,k,d_model]
        -> per-token unit-normalise -> * learnable scalar (init = emb_norm)
    The learnable scalar keeps soft tokens in the embedding norm range WITHOUT discarding
    z's magnitude (a fixed norm-match would).
  - Only the projector, LoRA adapters, the REG embedding, and the head are trained; the
    backbone is frozen.
  - Separate model per arm: each arm is trained on its own fixed, null-free sequence, so
    the z-only / z+question arms simply drop the absent text tokens (no learned nulls, no
    modality dropout).

The projector takes a [B, n_layers_in, H] input so a future multi-layer ablation can
feed several layers without an interface change; this build passes n_layers_in = 1.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

from amortized_ue.stage2.config import Stage2Config


_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


class Projector(nn.Module):
    """Map a stored hidden vector (or a small band of layers) -> k soft tokens.

    Interface accepts [B, n_layers_in, H_in]; this build uses n_layers_in = 1 but the
    shape leaves room for a later multi-layer blend without a rewrite. Output soft tokens
    are per-token unit-normalised then scaled by a single learnable scalar (init emb_norm).
    """

    def __init__(self, h_in: int, d_model: int, k: int, hidden_dim: int,
                 dropout: float, init_scale: float, kind: str = "mlp"):
        super().__init__()
        self.k = k
        self.d_model = d_model
        in_dim = h_in  # n_layers_in == 1 -> flatten is a no-op; generalises later
        out_dim = k * d_model
        self.norm_in = nn.LayerNorm(in_dim)
        if kind == "mlp":
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden_dim, out_dim),
            )
        elif kind == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"unknown projector_type {kind!r}")
        # learnable soft-token norm (in embedding range, but not information-destroying)
        self.scale = nn.Parameter(torch.tensor(float(init_scale)))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, n_layers_in, H_in]; collapse the single layer for this build.
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        h = self.norm_in(z)
        out = self.net(h).view(z.shape[0], self.k, self.d_model)   # [B, k, d_model]
        unit = out / (out.norm(dim=-1, keepdim=True) + 1e-6)
        return unit * self.scale


class ProxyModel(nn.Module):
    def __init__(self, cfg: Stage2Config, h_in: int):
        super().__init__()
        self.cfg = cfg
        self.h_in = h_in
        dtype = _DTYPES[cfg.backbone_dtype]
        self.model_dtype = dtype

        self.tokenizer = AutoTokenizer.from_pretrained(cfg.proxy_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"   # REG token stays the final position

        backbone = AutoModelForCausalLM.from_pretrained(cfg.proxy_model, torch_dtype=dtype)
        backbone.config.use_cache = False
        for p in backbone.parameters():
            p.requires_grad_(False)

        lora = LoraConfig(
            r=cfg.lora_r, lora_alpha=cfg.lora_alpha, lora_dropout=cfg.lora_dropout,
            target_modules=list(cfg.lora_target_modules), bias="none", task_type="CAUSAL_LM",
        )
        self.backbone = get_peft_model(backbone, lora)

        d_model = backbone.config.hidden_size
        self.d_model = d_model
        self.embed_tokens = self.backbone.get_input_embeddings()

        # backbone mean embedding-row norm (init for the learnable soft-token scale)
        with torch.no_grad():
            emb_norm = self.embed_tokens.weight.float().norm(dim=-1).mean()
        self.register_buffer("emb_norm", emb_norm)

        self._build_projector(cfg.k_soft_tokens)

        # learned readout token, initialised near the embedding scale
        self.reg_token = nn.Parameter(torch.randn(d_model) * (float(emb_norm) / (d_model ** 0.5)))

        if cfg.head_hidden_mult and cfg.head_hidden_mult > 0:
            hh = cfg.head_hidden_mult * d_model
            self.head = nn.Sequential(nn.Linear(d_model, hh), nn.GELU(), nn.Linear(hh, 1))
        else:
            self.head = nn.Linear(d_model, 1)

        # keep the small trainable modules in fp32 for stable optimisation
        self.head.float()

    def _build_projector(self, k: int):
        """(Re)create the projector for a given number of soft tokens (used by the k ablation)."""
        self.projector = Projector(
            h_in=self.h_in, d_model=self.d_model, k=k,
            hidden_dim=self.cfg.projector_hidden_dim, dropout=self.cfg.projector_dropout,
            init_scale=float(self.emb_norm), kind=self.cfg.projector_type,
        ).float().to(self.emb_norm.device)
        self.cfg.k_soft_tokens = k

    def set_k(self, k: int):
        """Swap in a fresh projector for a new k without reloading the backbone."""
        self._build_projector(k)

    def reinit_trainable(self):
        """Re-initialise every trainable parameter in place, under the active RNG.

        Used by the multi-seed arm study so each trial's init varies with its seed
        without reloading the frozen backbone. Reinitialises the projector, the REG
        readout, the head and the LoRA adapters; the frozen backbone is untouched.
        """
        d = self.d_model
        self._build_projector(self.cfg.k_soft_tokens)                 # fresh projector at current k
        with torch.no_grad():
            self.reg_token.copy_(torch.randn(d, device=self.reg_token.device)
                                 * (float(self.emb_norm) / (d ** 0.5)))
        for m in self.head.modules():                                  # head Linear(s)
            if isinstance(m, nn.Linear):
                m.reset_parameters()
        self.head.float()
        for module in self.backbone.modules():                         # LoRA adapters
            if hasattr(module, "reset_lora_parameters"):
                module.reset_lora_parameters("default", init_lora_weights=True)

    # --- forward ----------------------------------------------------------------
    def forward(self, z, text_input_ids=None, text_attention_mask=None):
        """z: [B, n_layers_in, H_in]; text_* optional (None => z-only arm).

        Builds [k soft] (+ [text]) + [REG]; regresses SE from the REG token's final
        hidden state. Returns pred [B] in standardised target space.
        """
        B = z.shape[0]
        d = self.d_model
        md = self.model_dtype
        dev = self.emb_norm.device

        # soft tokens in fp32 (stable projector) -> cast to backbone dtype at the boundary
        soft = self.projector(z.to(dev)).to(md)                       # [B, k, d]
        parts = [soft]
        attn_parts = [torch.ones(B, soft.shape[1], device=dev, dtype=torch.long)]

        if text_input_ids is not None and text_input_ids.shape[1] > 0:
            text_emb = self.embed_tokens(text_input_ids.to(dev)).to(md)   # [B, T, d]
            parts.append(text_emb)
            attn_parts.append(text_attention_mask.to(dev).long())

        reg = self.reg_token.to(md).view(1, 1, d).expand(B, 1, d)
        parts.append(reg)
        attn_parts.append(torch.ones(B, 1, device=dev, dtype=torch.long))

        inputs_embeds = torch.cat(parts, dim=1)                       # [B, k(+T)+1, d]
        attn = torch.cat(attn_parts, dim=1)
        position_ids = (attn.cumsum(-1) - 1).clamp(min=0)

        out = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            position_ids=position_ids, output_hidden_states=True,
        )
        reg_hidden = out.hidden_states[-1][:, -1, :]                  # [B, d] REG token
        return self.head(reg_hidden.float()).squeeze(-1)             # [B] standardised space
