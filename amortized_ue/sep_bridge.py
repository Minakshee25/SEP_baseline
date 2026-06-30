"""Bridge to the SEP repo's working logic (read-only reuse).

We add ../semantic_uncertainty to sys.path and import the original sampling,
prompt-construction, semantic-entropy, and hidden-state code unchanged. Nothing
in the SEP repo is edited; this module only re-exports the pieces Stage 1 calls
and builds the argparse `args` object those functions expect.
"""
from __future__ import annotations

import os
import sys
import argparse

# --- locate and register the SEP package root -------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
SEP_ROOT = os.path.join(_REPO_ROOT, "semantic_uncertainty")
if not os.path.isdir(os.path.join(SEP_ROOT, "uncertainty")):
    raise RuntimeError(f"SEP package not found at {SEP_ROOT!r}")
if SEP_ROOT not in sys.path:
    sys.path.insert(0, SEP_ROOT)

# --- re-export the SEP logic we reuse, unchanged ----------------------------
from uncertainty.utils import utils as sep_utils                       # noqa: E402
from uncertainty.data.data_utils import load_ds                       # noqa: E402
from uncertainty.uncertainty_measures.semantic_entropy import (        # noqa: E402
    EntailmentDeberta,
    get_semantic_ids,
    cluster_assignment_entropy,
    logsumexp_by_id,
    predictive_entropy,
)

__all__ = [
    "SEP_ROOT",
    "sep_utils",
    "load_ds",
    "EntailmentDeberta",
    "get_semantic_ids",
    "cluster_assignment_entropy",
    "logsumexp_by_id",
    "predictive_entropy",
    "build_sep_args",
]


def build_sep_args(config) -> argparse.Namespace:
    """Construct the argparse Namespace the SEP functions expect.

    Start from SEP's own parser defaults (so every field the SEP code reads is
    present and baseline-faithful), then override only the knobs Stage 1 exposes.
    """
    parser = sep_utils.get_parser(stages=["generate", "compute"])
    args = parser.parse_args([])  # all defaults

    args.model_name = config.model_name
    args.dataset = config.dataset
    args.num_samples = config.effective_num_samples()
    args.num_few_shot = config.num_few_shot
    args.random_seed = config.random_seed
    args.num_generations = config.num_generations
    args.temperature = config.temperature
    args.model_max_new_tokens = config.model_max_new_tokens
    args.metric = config.metric
    args.brief_prompt = config.brief_prompt
    args.brief_always = config.brief_always
    args.enable_brief = config.enable_brief
    args.use_context = config.use_context
    args.use_mc_options = config.use_mc_options
    args.entailment_model = config.entailment_model
    args.condition_on_question = config.condition_on_question
    args.strict_entailment = config.strict_entailment

    # Stage 1 builds its own records; disable SEP's wandb/p_true side effects.
    args.compute_p_true = False
    args.compute_uncertainties = False
    return args
