# CLAUDE.md — `amortized_ue/` (amortized UE, Stage 1)

> **Scope: amortized-UE Stage 1.** This file governs the `amortized_ue/` module only.
> The repo-root `../CLAUDE.md` is also in effect (it owns environment setup, the
> Imperial-DoC machine quirks, wandb auth, model compatibility, and the SEP
> baseline-reproduction rules). Read both; this file does **not** repeat the env setup.

## What this module is

MSc project: **amortized uncertainty estimation** — eventually, train a small model
to predict a large LLM's semantic entropy in a single forward pass, avoiding the
multi-sample cost at inference. **This module is Stage 1 only: offline dataset
construction.** Building the proxy model or its training is explicitly **out of
scope here** — do not start it unless asked.

Stage 1 produces, for one target LLM and a QA dataset, one **self-contained record
per prompt** that a later training stage can consume **without ever re-running the
target LLM**.

## Relationship to the SEP repo (read-only reuse)

`amortized_ue/` is a sibling folder inside this repo, not a separate project. It
**imports SEP's working logic read-only** via `sys.path` (`sep_bridge.py` adds
`../semantic_uncertainty`). **Nothing under `semantic_uncertainty/` or
`semantic_entropy_probes/` is edited.** The SEP baseline rules in the root
CLAUDE.md still apply to anything reused: do not modify `get_semantic_ids`,
`cluster_assignment_entropy`, `logsumexp_by_id`, the entailment model, the
TBG/SLT extraction, or the sampling. Stage 1 only *calls* them.

Reused unchanged: `HuggingfaceModel.predict(return_latent=True)`, `load_ds`,
prompt construction (`get_make_prompt`, `construct_fewshot_prompt_from_indices`,
`BRIEF_PROMPTS`), `get_metric`, `get_reference`, `split_dataset`,
`EntailmentDeberta`, `get_semantic_ids`, `cluster_assignment_entropy`.

## Files

- `config.py`     — `Stage1Config` dataclass; every knob. Defaults mirror the SEP baseline.
- `sep_bridge.py` — registers `../semantic_uncertainty` on `sys.path`, re-exports the
  reused SEP functions, and builds the SEP argparse `args` (from SEP's own parser
  defaults, then overrides) so reuse stays baseline-faithful.
- `record.py`     — record schema (`stage1-v1`), `save_record`/`load_record`, manifest
  helpers, `describe_record`, filesystem-safe filenames.
- `stage1.py`     — the builder: `build()`, `run_smoke()`, and a CLI.
- `loaders.py`    — `load_local` / `load_wandb` / `load_records` (single source switch).
- `wandb_io.py`   — optional: upload the same local files as a versioned W&B artifact.
- `data/stage1/`  — outputs (gitignored; tensors are GB-scale).

## Record schema (`stage1-v1`, one `.pt` per prompt, keyed by `id`)

```
id, question, context, reference
canonical:                       # the low-temperature (0.1) "most likely" answer
  response, accuracy, token_log_likelihoods
  hidden_states: { TBG: [L+1,1,H], SLT: [L+1,1,H] }   # all layers, native dtype
samples: [ {response, token_log_likelihoods, semantic_id}, ... ]   # N high-temp
labels:
  cluster_assignment_entropy     # PRIMARY label, stored CONTINUOUS (raw float)
  semantic_ids, n_clusters, n_samples
meta: { model, dataset, temperatures, entailment settings, git_commit, positions... }
```

The SE label lives in the same record as the text and hidden states, joined **by id**
— never by list position (this deliberately fixes SEP's positional-join fragility,
see root `SEP_TECHNICAL_REPORT.md` §7).

### Hidden-state positions — IMPORTANT (true-position labelling)

We label by the real token position, per the project spec:

| record key | position                                   | HF index          |
|------------|--------------------------------------------|-------------------|
| `TBG`      | token before generation (last input token) | `hidden[0]`       |
| `SLT`      | second-last generated token                | `hidden[n_gen-2]` |

`predict()` returns `(scalar, sec_last=SLT, last_tok_before_gen=TBG)`; `stage1.py`
unpacks it as `(embedding, slt_emb, tbg_emb)`, matching that order — so our keys are
correct. **SEP's own stored keys are inverted** relative to position: amortized `TBG`
== SEP key `emb_tok_before_eos` == SEP probe `slt_dataset`; amortized `SLT` == SEP key
`emb_last_tok_before_gen` == SEP probe `tbg_dataset`. Keep this in mind when comparing
to SEP/the paper.

## Commands

Run from the repo root with the `se_probes` env active (see root CLAUDE.md for env).

```bash
# smoke test: a few prompts end to end, prints one record's structure
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3

# full run (defaults mirror SEP Llama-2-7b-chat / trivia_qa)
python -m amortized_ue.stage1 --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 400

# optional: also push the same files to W&B as a versioned artifact
python -m amortized_ue.stage1 --num_samples 400 --push_to_wandb
```

Loading (identical records from either source; default local, fully offline):
```python
from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records
records = load_records(Stage1Config(num_samples=400))                       # local
records = load_records(Stage1Config(num_samples=400, load_source="wandb"))  # W&B copy
```

Shared GPUs here are often full; launch via a poll-and-retry wrapper that pins
`CUDA_VISIBLE_DEVICES` to a GPU with ≥~16 GB free. The build is **resumable**
(`overwrite=False` skips existing records), so an OOM mid-run just continues on
relaunch.

## Locked design decisions (do not change without asking)

- SE stored **continuous** (raw float), never binarised in Stage 1; primary label is
  `cluster_assignment_entropy`. Also keep `semantic_ids` + per-sample log-probs so the
  label is recomputable without re-sampling.
- Hidden states at **TBG and SLT, all layers**, for the **low-temp canonical** answer
  only (high-temp samples store text + log-probs, no hidden states).
- Everything joined **by id** inside one self-contained record.
- Single target LLM, raw hidden states (native dtype), no cross-model alignment yet.
- Local disk is the source of truth and must work fully offline; **W&B is an additional
  copy** (same files uploaded), never the only place the data lives. Load source is a
  single config switch defaulting to `local`.

## Current state (updated 2026-06-30)

**Stage 1 COMPLETE for Llama-2-7b-chat / trivia_qa, N=400.**
- Local: `amortized_ue/data/stage1/Llama-2-7b-chat_trivia_qa_n400_full/`
  (`records/` 400 `.pt` + `manifest.json`, ~0.43 GB).
- Metrics: mean_accuracy 0.5775, mean_cluster_assignment_entropy 0.6138, ~26 min/1 GPU.
- W&B copy verified byte-identical (spot-checked SHA-256): project `amortized_ue_stage1`,
  artifact `stage1_records:v0` (401 files), run `4d2lvwzc`.
- Smoke test passes.

**Not started (future stages):** proxy model + its training. Out of scope until asked.
