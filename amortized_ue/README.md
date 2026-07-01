# Amortized UE — Stage 1 (dataset) + Stage 2 (SLM proxy)

Predict a large LLM's semantic entropy in a **single forward pass**, avoiding the
multi-sample cost at inference. Two stages:

- **Stage 1 (below):** build a self-contained record per prompt (offline SE dataset).
- **[Stage 2](#stage-2--slm-proxy):** train a frozen Llama-3.2-3B to regress the SE
  label from the stored hidden state (soft tokens) plus optional text.

## Stage 1 — offline dataset construction

Stage 1 builds, for one target LLM and a QA dataset, a self-contained record per
prompt that Stage 2 consumes **without ever re-running the target LLM**. It reuses
the SEP repo's sampling, semantic-entropy, and hidden-state logic read-only
(imported from `../semantic_uncertainty`); nothing in the SEP repo is modified.

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

## Stage 1 files

- `config.py`     — `Stage1Config` (all knobs; defaults mirror SEP baseline)
- `sep_bridge.py` — read-only import of SEP logic + SEP `args` construction
- `record.py`     — record schema, save/load, manifest, `describe_record`
- `stage1.py`     — the builder (`build`, `run_smoke`, CLI)
- `loaders.py`    — `load_local` / `load_wandb` / `load_records`
- `wandb_io.py`   — optional artifact upload

---

## Stage 2 — SLM proxy

A frozen **Llama-3.2-3B** reads `[k soft tokens] (+ [text]) + [REG readout]` in one
forward pass; a linear head on the REG token's final hidden state regresses the
standardised SE label. Only the projector, LoRA adapters, REG embedding, and head train.
The stored hidden vector `z` is mapped to `k` soft tokens by a learned projector:
`LayerNorm → Linear(H→256) → GELU → Dropout(0.1) → Linear(256→k·d_model) →
per-token unit-normalise × learnable scalar` (the learnable scale keeps soft tokens in
embedding-norm range without discarding `z`'s magnitude).

**Separate model per arm** (`z` / `z_q` / `z_q_resp`), each trained on its own fixed,
null-free sequence. The `(position, layer)` for `z` is selected by **validation
Spearman** via a z-only sweep on a fixed 600-example train-only subsample; `k∈{1,4,8}`
is ablated on the z-only arm. Metrics per arm: Spearman (primary), RMSE, MAE, R², AUROC.

### Separate environment

Stage 2 needs a newer stack than the pinned `se_probes` (which can't load Llama-3.2).
Use `amortized_stage2` (a clone of `se_probes` upgraded to `transformers==4.52.4` +
`peft` + `accelerate`; torch stays 2.1.1). `se_probes` is left untouched.

### Usage (repo root, `amortized_stage2` env, pin a free GPU)

```bash
python -m amortized_ue.stage2.run --report   # label distribution + subsample checks (no GPU)
python -m amortized_ue.stage2.run --smoke     # full path, a few prompts, 2 steps
python -m amortized_ue.stage2.run             # full run -> stage2/runs/<name>/results.json
```

### Results (Llama-2-7b-chat / trivia_qa, N=2000; selected TBG layer 12, k=4)

| arm (test split) | Spearman | AUROC | RMSE | R² |
|------------------|---------:|------:|-----:|---:|
| z (hidden only)  | 0.459 | 0.758 | 0.574 | 0.176 |
| z + question     | 0.414 | 0.733 | 0.591 | 0.129 |
| **z + question + response** | **0.575** | **0.795** | **0.497** | **0.384** |

z-only ≈ the single-layer linear-probe reference (0.805 AUROC), i.e. the soft token is
used; adding the **canonical response** is what lifts performance (the question alone
does not). Reference: a plain logistic probe on the same hidden state reaches ~0.805 AUROC.

### Stage 2 files

- `config.py` — `Stage2Config` (every knob)
- `data.py`   — id-keyed load, split, target standardise, AUROC binarisation, sweep subsample
- `model.py`  — `Projector` + `ProxyModel` (frozen backbone + LoRA + soft tokens + REG head)
- `train.py`  — `Trainer`: per-arm train/eval, (pos,layer) sweep, k-ablation
- `run.py`    — `--report` / `--smoke` / full-run CLI
