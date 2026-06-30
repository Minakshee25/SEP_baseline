"""Stage 1 configuration.

A single dataclass holds every knob: target model, dataset, sampling settings,
clustering settings, output location, and the load source. Defaults mirror the
SEP baseline (arXiv:2406.15927) so Stage 1 reproduces the same generation /
semantic-entropy procedure faithfully.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, asdict, field


# Repo root = parent of this file's directory (amortized_ue/ -> repo).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEFAULT_OUTPUT = os.path.join(_REPO_ROOT, "amortized_ue", "data", "stage1")


@dataclass
class Stage1Config:
    # --- target model + dataset -------------------------------------------------
    model_name: str = "Llama-2-7b-chat"      # loaded via NousResearch mirror (see huggingface_models.py)
    dataset: str = "trivia_qa"

    # --- prompt set -------------------------------------------------------------
    num_samples: int = 400                    # number of validation prompts to build records for
    num_few_shot: int = 5                     # few-shot examples drawn from the train split
    random_seed: int = 10                     # matches SEP default for reproducible sampling

    # --- generation / sampling (mirrors SEP baseline) ---------------------------
    num_generations: int = 10                 # high-temperature samples per prompt
    temperature: float = 1.0                  # high-temperature value
    low_temperature: float = 0.1              # canonical "most likely" answer temperature
    model_max_new_tokens: int = 50            # short-form
    metric: str = "squad"                     # F1-based accuracy for the canonical answer
    brief_prompt: str = "default"
    brief_always: bool = False
    enable_brief: bool = True
    use_context: bool = False
    use_mc_options: bool = True               # passed to load_ds (no-op for trivia_qa)

    # --- semantic-entropy / clustering (mirrors SEP baseline) -------------------
    entailment_model: str = "deberta"         # only DeBERTa supported in this Stage 1 build
    condition_on_question: bool = True        # prefix "{question} {response}" before clustering
    strict_entailment: bool = True            # both-direction strict entailment for equivalence

    # --- output / storage -------------------------------------------------------
    output_dir: str = _DEFAULT_OUTPUT         # single configurable output location
    run_name: str | None = None               # subdir under output_dir; auto-derived if None
    overwrite: bool = False                   # re-generate prompts whose record file already exists

    # --- load source ------------------------------------------------------------
    load_source: str = "local"                # "local" | "wandb" (default local; fully offline-capable)

    # --- W&B (optional extra copy; never the only place data lives) -------------
    push_to_wandb: bool = False               # upload the local files as a versioned artifact
    wandb_project: str = "amortized_ue_stage1"
    wandb_entity: str | None = field(default_factory=lambda: os.getenv("WANDB_ENT"))
    wandb_artifact_name: str = "stage1_records"

    # --- smoke test -------------------------------------------------------------
    smoke: bool = False
    smoke_num_samples: int = 3

    def resolved_run_name(self) -> str:
        if self.run_name:
            return self.run_name
        n = self.smoke_num_samples if self.smoke else self.num_samples
        tag = "smoke" if self.smoke else "full"
        return f"{self.model_name}_{self.dataset}_n{n}_{tag}"

    def run_dir(self) -> str:
        return os.path.join(self.output_dir, self.resolved_run_name())

    def records_dir(self) -> str:
        return os.path.join(self.run_dir(), "records")

    def manifest_path(self) -> str:
        return os.path.join(self.run_dir(), "manifest.json")

    def effective_num_samples(self) -> int:
        return self.smoke_num_samples if self.smoke else self.num_samples

    def as_dict(self) -> dict:
        return asdict(self)
