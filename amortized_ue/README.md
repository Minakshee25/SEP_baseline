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
python -m amortized_ue.stage2.run --reuse_selection --seeds 5   # multi-seed -> results_multiseed.json
# OOD: train each arm on the ID dataset, evaluate on a 2nd Stage-1 dataset (eval-only)
python -m amortized_ue.stage2.run --ood --ood_dataset squad --reuse_selection --seeds 5
#   -> stage2/runs/<name>/ood_results_<dataset>_multiseed.json
```

`--seeds N` runs N trial seeds; each arm trains on its own deterministic `(seed, trial, arm)`
RNG stream (init + shuffle + dropout), so the arms are decoupled from the sweep/k-ablation and
`build`/`build_ood` agree per seed. `--reuse_selection` skips the sweep and reuses the saved
(position, layer, k). Single-seed runs (`results.json` / `ood_results_<ds>.json`) still exist
but the text-arm figures there are noise-dominated — **use the multi-seed numbers below.**

### Results — MULTI-SEED (5 seeds; Llama-2-7b-chat, TBG layer 12, k=4). Reference result.

Test AUROC, mean ± std over 5 seeds (ID = trivia_qa N=2000; OOD = train trivia → eval squad N=1000):

| arm | ID (trivia) AUROC | OOD (squad) AUROC |
|-----|:---:|:---:|
| z (hidden only)          | **0.763 ± 0.010** (best) | 0.622 ± 0.016 |
| z + question             | 0.744 ± 0.032 | 0.586 ± 0.045 (worst) |
| z + question + response  | 0.722 ± 0.017 | **0.650 ± 0.005** (best) |

Paired per-seed differences (arm − z), sign consistent across **all 5 seeds**:

- **In-distribution: text HURTS.** z+q+resp − z = −0.041 AUROC (negative 5/5). z-only is the
  strongest and most stable arm (≈ the 0.805 single-layer probe reference — the soft token is used).
- **Out-of-distribution: the response HELPS.** z+q+resp − z = +0.027 AUROC / +0.045 Spearman
  (positive 5/5). Under a real shift z degrades (0.763→0.622) and the canonical response supplies
  transferable signal.
- **The question alone (z+q) hurts in both regimes** — the *response*, not the question, carries signal.

**Headline:** in-distribution the hidden state `z` alone is best and added text is a distractor;
under domain shift `z` degrades and the canonical **response** recovers signal. This **supersedes**
the earlier single-run claims (ID "text helps 0.795>0.758" and OOD "text doesn't transfer"), both
of which were lucky/unlucky seeds. squad is a genuine shift (mean acc 0.24 / mean CAE 1.50 vs
trivia's 0.59 / 0.59). OOD RMSE/R² are miscalibrated by design (label-scale shift).

### Where results are saved

- Stage-1 records: `amortized_ue/data/stage1/<run_name>/records/<id>.pt` + `manifest.json` (gitignored).
- Stage-2 ID multi-seed: `.../stage2_<model>_<dataset>_n<N>_full/results_multiseed.json` (gitignored);
  single-seed `results.json` retained for provenance.
- Stage-2 OOD multi-seed: `.../ood_results_<ood_dataset>_multiseed.json` in the same run dir.
- Logs: `amortized_ue/stage2/logs/multiseed_{id,ood}.log`.
- W&B artifacts (project `amortized_ue_stage1`): `stage1_records:v0` (n400), `stage1_records_n2000`.

### To-do

1. **(DONE 2026-07-02)** Per-arm reseeding + multi-seed run — implemented; 5-seed ID + OOD above.
2. **Multi-layer projector ablation** — feed a band of layers (`n_layers_in > 1` already supported).
3. **Full 2×2 OOD matrix** — also train on squad, eval on trivia_qa.
4. **Hyperparameter pass** — lr, LoRA rank, epochs (winning arm is regime-dependent: z-only ID,
   z+q+resp OOD).

### Stage 2 files

- `config.py` — `Stage2Config` (every knob)
- `data.py`   — id-keyed load, split, target standardise, AUROC binarisation, sweep subsample
- `model.py`  — `Projector` + `ProxyModel` (frozen backbone + LoRA + soft tokens + REG head)
- `train.py`  — `Trainer`: per-arm train/eval, (pos,layer) sweep, k-ablation
- `run.py`    — `--report` / `--smoke` / full-run CLI
