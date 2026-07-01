"""Stage 2 configuration — every knob, with the locked defaults.

Defaults encode the decisions agreed for this build. Nothing here is chosen
silently: each field corresponds to an explicit decision in the Stage-2 spec.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, asdict, field

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OUT = os.path.join(_THIS_DIR, "runs")


@dataclass
class Stage2Config:
    # --- Stage-1 source records (read-only) -------------------------------------
    # Identifies which Stage-1 run to consume. Resolved through Stage1Config so the
    # local data dir / run-name convention is reused verbatim.
    stage1_model_name: str = "Llama-2-7b-chat"
    stage1_dataset: str = "trivia_qa"
    stage1_num_samples: int = 400
    stage1_load_source: str = "local"          # "local" | "wandb"

    # --- proxy backbone ---------------------------------------------------------
    proxy_model: str = "meta-llama/Llama-3.2-3B"   # official, gated access cleared
    backbone_dtype: str = "bfloat16"
    max_seq_len: int = 256

    # --- which stored (position, layer) feeds z ---------------------------------
    # Both positions are referred to by their physical meaning, never hard-coded
    # to one. "TBG" = last input token before generation; "SLT" = second-last
    # generated token (verified from the records). Selection is validation-driven:
    # when select_pos_layer is True these are ignored and chosen by the sweep.
    select_pos_layer: bool = True
    sweep_positions: tuple = ("TBG", "SLT")    # both physical positions
    sweep_layers: tuple = tuple(range(33))     # all 33 (embedding + 32 layers)
    selected_position: str | None = None       # filled in after selection
    selected_layer: int | None = None
    select_metric: str = "val_mse"             # criterion for the sweep
    select_arm: str = "z"                      # selection uses the z-only arm
    select_k_soft_tokens: int = 4              # selection done at default k
    sweep_epochs: int = 3                      # cheap epochs per candidate during the sweep

    # --- projector (hidden vector -> soft tokens) -------------------------------
    k_soft_tokens: int = 4                      # configurable; ablate {1,4,8}
    projector_type: str = "mlp"                 # 2-layer MLP, GELU
    projector_hidden_dim: int = 512             # bottleneck width (4096 -> hidden -> k*d_model);
                                                # small on purpose for N=400 (redefined for big-data run)
    projector_dropout: float = 0.0
    norm_match: bool = True                     # scale soft tokens to backbone emb norm

    # --- readout / head ---------------------------------------------------------
    readout: str = "reg_token"                  # dedicated appended [REG] token
    head_hidden_mult: int = 0                   # 0 => linear head

    # --- input arms + modality dropout ------------------------------------------
    # One model served three ways. arm controls evaluation/serving; during training
    # modality dropout randomises the present modalities.
    arm: str = "z_q_resp"                       # "z" | "z_q" | "z_q_resp"
    p_drop_text: float = 0.5
    p_drop_z: float = 0.1

    # --- LoRA -------------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # --- optimisation -----------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    batch_size: int = 16
    grad_accum: int = 1
    epochs: int = 10
    early_stop_metric: str = "val_mse"
    early_stop_patience: int = 3
    grad_clip: float = 1.0
    seed: int = 42

    # --- target transform -------------------------------------------------------
    target_transform: str = "standardize"       # z-score on train; report orig space

    # --- split (consistent with the Stage-1 diagnostic) -------------------------
    test_size: float = 0.1                       # of all data
    val_size: float = 0.2                        # of the train+val remainder
    split_seed: int = 42

    # --- output -----------------------------------------------------------------
    output_dir: str = _DEFAULT_OUT
    run_name: str | None = None

    # --- smoke ------------------------------------------------------------------
    smoke: bool = False
    smoke_num_prompts: int = 8
    smoke_steps: int = 2

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        tag = "smoke" if self.smoke else "full"
        return f"stage2_{self.stage1_model_name}_{self.stage1_dataset}_{self.arm}_k{self.k_soft_tokens}_{tag}"

    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.resolved_run_name())

    def as_dict(self) -> dict:
        return asdict(self)
