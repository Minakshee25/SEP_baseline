"""Stage 2 proxy model.

A frozen decoder-only backbone (Llama-3.2-3B) reads, in one forward pass:
  [k soft tokens]  +  [text tokens]  +  [REG readout token]
and a linear head on the REG token's final hidden state regresses the (standardised)
semantic-entropy target.

  - The stored Stage-1 hidden vector z (from the *target* LLM, dim H_stage1) is mapped
    by a learned projector into k soft tokens in the proxy's embedding space, then
    norm-matched to the backbone's mean embedding-row norm.
  - Only the projector, LoRA adapters, learned null embeddings, the REG embedding,
    and the head are trained; the backbone is frozen.
  - Modality dropout is realised by replacing a modality's embeddings with a learned
    null vector (null_z for the soft tokens, null_text for text positions), keeping the
    sequence structure fixed so the REG slot is always last and the three input arms
    (z / z+question / z+question+response) are served by the same graph.

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
    shape leaves room for a later multi-layer blend without a rewrite.
    """

    def __init__(self, h_in: int, d_model: int, k: int, hidden_dim: int,
                 dropout: float, kind: str = "mlp"):
        super().__init__()
        self.k = k
        self.d_model = d_model
        in_dim = h_in  # n_layers_in == 1 -> flatten is a no-op; generalises later
        out_dim = k * d_model
        if kind == "mlp":
            hidden = hidden_dim  # bottleneck: keeps params small for low-N training
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(hidden, out_dim),
            )
        elif kind == "linear":
            self.net = nn.Linear(in_dim, out_dim)
        else:
            raise ValueError(f"unknown projector_type {kind!r}")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, n_layers_in, H_in]; collapse the single layer for this build.
        if z.dim() == 3:
            z = z.reshape(z.shape[0], -1)
        out = self.net(z)                       # [B, k*d_model]
        return out.view(z.shape[0], self.k, self.d_model)


class ProxyModel(nn.Module):
    def __init__(self, cfg: Stage2Config, h_in: int):
        super().__init__()
        self.cfg = cfg
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

        # backbone mean embedding-row norm (target for norm-matching soft tokens)
        with torch.no_grad():
            emb_norm = self.embed_tokens.weight.float().norm(dim=-1).mean()
        self.register_buffer("emb_norm", emb_norm)

        self.projector = Projector(
            h_in=h_in, d_model=d_model, k=cfg.k_soft_tokens,
            hidden_dim=cfg.projector_hidden_dim, dropout=cfg.projector_dropout,
            kind=cfg.projector_type,
        )

        # learned null embeddings + readout token, initialised near the embedding scale
        def _init_vec():
            v = torch.randn(d_model) * (float(emb_norm) / (d_model ** 0.5))
            return nn.Parameter(v)
        self.null_z = _init_vec()
        self.null_text = _init_vec()
        self.reg_token = _init_vec()

        head_in = d_model
        if cfg.head_hidden_mult and cfg.head_hidden_mult > 0:
            hh = cfg.head_hidden_mult * d_model
            self.head = nn.Sequential(nn.Linear(head_in, hh), nn.GELU(), nn.Linear(hh, 1))
        else:
            self.head = nn.Linear(head_in, 1)

        # keep the small trainable modules in fp32 for stable optimisation
        self.projector.float()
        self.head.float()

    # --- soft tokens from z -----------------------------------------------------
    def soft_tokens(self, z: torch.Tensor, drop_z: torch.Tensor) -> torch.Tensor:
        """z: [B, n_layers_in, H_in] -> [B, k, d_model], norm-matched, with z-dropout."""
        soft = self.projector(z)                                  # [B, k, d_model] fp32
        if self.cfg.norm_match:
            unit = soft / (soft.norm(dim=-1, keepdim=True) + 1e-6)
            soft = unit * self.emb_norm.to(soft.dtype)
        if drop_z is not None and drop_z.any():
            soft = soft.clone()
            soft[drop_z] = self.null_z.to(soft.dtype)
        return soft

    def forward(self, z, text_input_ids, text_attention_mask, text_keep_mask, drop_z):
        """All tensors batched.

        z                 : [B, n_layers_in, H_in]   stored hidden vector(s)
        text_input_ids    : [B, T]                    left-padded question(+response)
        text_attention_mask:[B, T]                    1 for real tokens, 0 for pad
        text_keep_mask    : [B, T]                    1 keep real text embed, 0 -> null_text
                                                      (pad positions are 0 here too)
        drop_z            : [B] bool                  replace soft tokens with null_z
        returns pred [B] in standardised target space.
        """
        B = text_input_ids.shape[0]
        d = self.d_model
        md = self.model_dtype                                      # backbone compute dtype

        # soft tokens are produced in fp32 (stable projector), then cast to the backbone
        # dtype at the boundary; gradients still flow to the fp32 projector through the cast.
        soft = self.soft_tokens(z, drop_z).to(md)                 # [B, k, d]

        text_emb = self.embed_tokens(text_input_ids).to(md)          # [B, T, d]
        keep = text_keep_mask.unsqueeze(-1).to(md)                   # [B, T, 1]
        null_t = self.null_text.to(md).view(1, 1, d)
        text_emb = keep * text_emb + (1.0 - keep) * null_t           # null where not kept

        reg = self.reg_token.to(md).view(1, 1, d).expand(B, 1, d)

        inputs_embeds = torch.cat([soft, text_emb, reg], dim=1)      # [B, k+T+1, d]

        k = soft.shape[1]
        ones_soft = torch.ones(B, k, device=text_emb.device, dtype=text_attention_mask.dtype)
        ones_reg = torch.ones(B, 1, device=text_emb.device, dtype=text_attention_mask.dtype)
        attn = torch.cat([ones_soft, text_attention_mask, ones_reg], dim=1)  # [B, k+T+1]
        position_ids = (attn.long().cumsum(-1) - 1).clamp(min=0)

        out = self.backbone(
            inputs_embeds=inputs_embeds, attention_mask=attn,
            position_ids=position_ids, output_hidden_states=True,
        )
        reg_hidden = out.hidden_states[-1][:, -1, :]                 # [B, d] REG token
        pred = self.head(reg_hidden.float()).squeeze(-1)            # [B] standardised space
        return pred
