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

    # --- targeted subset build (cross-LLM: build only specific held-out ids) ----
    # When `only_ids` is set, the seed-driven selection still runs at
    # `selection_num_samples` (so the SAME questions are drawn as the full run this
    # subset comes from), and only the prompts whose id is in `only_ids` are built.
    # This lets a second target LLM be run on exactly another run's held-out ids.
    only_ids: tuple = ()                      # empty -> build the full selection as usual
    selection_num_samples: int = 0            # 0 -> use num_samples; else the pool to select from

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

    # --- W&B (an extra copy; local disk stays the source of truth) --------------
    # Default ON so every real build is mirrored off the shared NFS (which periodically
    # degrades and has no offsite backup). Smoke builds never push (guarded in stage1.build).
    push_to_wandb: bool = True                 # upload the local files as a versioned artifact
    wandb_project: str = "amortized_ue_stage1"
    wandb_entity: str | None = field(default_factory=lambda: os.getenv("WANDB_ENT"))
    # None -> auto-distinct name per (model, dataset, N) so each dataset is its OWN artifact
    # (reusable independently) instead of stacking as versions of one shared name. Set a
    # string to override.
    wandb_artifact_name: str | None = None

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

    def resolved_artifact_name(self) -> str:
        """W&B artifact name: the explicit override, else a name unique to this dataset so
        each (model, dataset, N) is its own reloadable artifact (not a version of a shared
        name). W&B artifact names allow [A-Za-z0-9._-] only, so sanitize like filenames."""
        if self.wandb_artifact_name:
            return self.wandb_artifact_name
        import re
        stem = f"stage1_records_{self.model_name}_{self.dataset}_n{self.effective_num_samples()}"
        return re.sub(r"[^A-Za-z0-9._-]", "-", stem)

    def as_dict(self) -> dict:
        return asdict(self)
