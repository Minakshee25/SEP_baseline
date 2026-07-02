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
    stage1_num_samples: int = 2000             # big-data run (n2000_full)
    stage1_load_source: str = "local"          # "local" | "wandb"

    # --- OOD evaluation (train on stage1_dataset, eval on a 2nd dataset) ---------
    ood_dataset: str | None = None             # e.g. "squad"; None disables OOD
    ood_num_samples: int = 1000                # size of the OOD Stage-1 dataset

    # --- proxy backbone (frozen; not to be changed) -----------------------------
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
    selected_position: str | None = None       # filled in after selection (or OOD override)
    selected_layer: int | None = None
    selected_k: int | None = None              # fixed k for OOD (else read from prior results.json)
    select_metric: str = "val_spearman"        # sweep selection: val Spearman (higher=better)
    select_arm: str = "z"                      # selection uses the z-only arm
    select_k_soft_tokens: int = 4              # (pos,layer) selection done at k=4
    sweep_epochs: int = 3                      # cheap epochs per candidate during the sweep
    sweep_subsample_size: int = 600            # sweep trains on a fixed 600-example TRAIN subsample
    sweep_subsample_seed: int = 42             # seed for that subsample (train split only)

    # --- projector (hidden vector -> soft tokens) -------------------------------
    #   LayerNorm(H_in) -> Linear(H_in, hidden) -> GELU -> Dropout
    #   -> Linear(hidden, k*d_model) -> reshape [B,k,d_model]
    #   -> per-token unit-normalise -> * learnable scalar (init = emb_norm)
    k_soft_tokens: int = 4                      # configurable; ablate {1,4,8}
    projector_type: str = "mlp"                 # "mlp" (bottleneck) | "linear"
    projector_hidden_dim: int = 256             # bottleneck width
    projector_dropout: float = 0.1
    k_ablation_values: tuple = (1, 4, 8)        # k sweep on the z-only arm

    # --- readout / head (not to be changed) -------------------------------------
    readout: str = "reg_token"                  # dedicated appended [REG] token
    head_hidden_mult: int = 0                   # 0 => linear head

    # --- input arms -------------------------------------------------------------
    # Separate model per arm: each arm is trained on its own fixed, null-free
    # sequence (no modality dropout). z-only = [k soft][REG]; z+q drops the
    # response tokens; z+q+resp keeps both.
    arm: str = "z_q_resp"                       # "z" | "z_q" | "z_q_resp" (smoke/eval default)
    arms: tuple = ("z", "z_q", "z_q_resp")      # the three arms trained separately

    # --- LoRA (not to be changed) -----------------------------------------------
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # --- optimisation -----------------------------------------------------------
    lr: float = 1e-4
    weight_decay: float = 0.01
    scheduler: str = "cosine"
    warmup_ratio: float = 0.03
    batch_size: int = 32                        # bumped from 16 (more data; fits L40)
    grad_accum: int = 1
    epochs: int = 10
    early_stop_metric: str = "val_mse"          # training early-stop (MSE objective)
    early_stop_patience: int = 3
    grad_clip: float = 1.0
    seed: int = 42

    # --- multi-seed arm training (variance study) -------------------------------
    # Each arm is trained under its own deterministic (seed, trial_seed, arm) RNG
    # stream — model re-init, batch-shuffle order and dropout — independent of the
    # sweep/k-ablation consumption. This makes build and build_ood agree for a given
    # seed and gives a mean±std per arm across trials, so the text-arm advantage can
    # be tested against run-to-run noise (Stage-2 to-do #1).
    arm_trial_seeds: tuple = (0, 1, 2)           # trial seeds for the arm variance study
    reuse_selection: bool = False                # skip sweep/k-ablation; reuse saved/override (pos,layer,k)

    # --- target transform -------------------------------------------------------
    target_transform: str = "standardize"       # z-score on train; report orig space

    # --- split (test 0.1 -> val 0.2 of remainder, seed 42; id-sorted) -----------
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
        return f"stage2_{self.stage1_model_name}_{self.stage1_dataset}_n{self.stage1_num_samples}_{tag}"

    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.resolved_run_name())

    def as_dict(self) -> dict:
        return asdict(self)
