# Amortized UE — Stage 1: offline dataset construction

Stage 1 builds, for one target LLM and a QA dataset, a self-contained record per
prompt that a later training stage can consume **without ever re-running the
target LLM**. It reuses the SEP repo's sampling, semantic-entropy, and
hidden-state logic read-only (imported from `../semantic_uncertainty`); nothing
in the SEP repo is modified.

## What a record contains (per prompt, keyed by `id`)

```
id, question, context, reference
canonical:                      # the low-temperature (0.1) "most likely" answer
  response, accuracy, token_log_likelihoods
  hidden_states: { TBG: [L+1,1,H], SLT: [L+1,1,H] }   # all layers, native dtype
samples: [ {response, token_log_likelihoods, semantic_id}, ... ]   # N high-temp
labels:
  cluster_assignment_entropy    # primary continuous label (raw float)
  semantic_ids, n_clusters, n_samples
meta: { model, dataset, temperatures, entailment settings, git_commit, ... }
```

The SE label lives in the same record as the text and hidden states, so
everything is joined by `id` — never by list position.

### Hidden-state positions (important)

Positions are labelled by their true meaning, per the project spec:

| record key | position                                   | HF index          |
|------------|--------------------------------------------|-------------------|
| `TBG`      | token before generation (last input token) | `hidden[0]`       |
| `SLT`      | second-last generated token                | `hidden[n_gen-2]` |

> The SEP repo's stored keys are **inverted** relative to these positions
> (`emb_last_tok_before_gen` actually holds the second-last token; `emb_tok_before_eos`
> holds the token-before-generation). For cross-comparison: amortized `TBG` ==
> SEP key `emb_tok_before_eos` == SEP probe `slt_dataset`; amortized `SLT` ==
> SEP key `emb_last_tok_before_gen` == SEP probe `tbg_dataset`.

## Usage

Run from the repo root with the `se_probes` conda env active.

Smoke test (a few prompts end to end, prints one record's structure):

```bash
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3
```

Full run (defaults mirror the SEP Llama-2-7b-chat / trivia_qa baseline):

```bash
python -m amortized_ue.stage1 --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 400
```

Optional W&B mirror (extra copy of the same files as a versioned artifact):

```bash
python -m amortized_ue.stage1 --num_samples 400 --push_to_wandb
```

## Loading records

```python
from amortized_ue.config import Stage1Config
from amortized_ue.loaders import load_records

cfg = Stage1Config(num_samples=400)            # load_source="local" by default
records = load_records(cfg)                     # {id: record}, fully offline

cfg_wandb = Stage1Config(num_samples=400, load_source="wandb")
records_wandb = load_records(cfg_wandb)         # identical records from W&B
```

## Output layout

```
amortized_ue/data/stage1/<run_name>/
  manifest.json          # config + meta + tensor-free per-record index
  records/<id>.pt        # one self-contained record per prompt
```

## Files

- `config.py`     — `Stage1Config` (all knobs; defaults mirror SEP baseline)
- `sep_bridge.py` — read-only import of SEP logic + SEP `args` construction
- `record.py`     — record schema, save/load, manifest, `describe_record`
- `stage1.py`     — the builder (`build`, `run_smoke`, CLI)
- `loaders.py`    — `load_local` / `load_wandb` / `load_records`
- `wandb_io.py`   — optional artifact upload
