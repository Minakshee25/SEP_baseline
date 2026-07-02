# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A detailed read-only walkthrough of the data-generation and hidden-state-extraction internals lives in `SEP_TECHNICAL_REPORT.md`.

> **New work lives in `amortized_ue/` and has its own `amortized_ue/CLAUDE.md`.** That
> module (amortized uncertainty estimation) reuses the SEP logic read-only and is governed
> by its own scoped CLAUDE.md, which auto-loads when working under `amortized_ue/`. It now
> has **Stage 1 (offline SE dataset)** and **Stage 2 (SLM proxy that predicts SE in one
> forward pass)** — both built. **Stage 2 runs in a separate conda env `amortized_stage2`**
> (cloned from `se_probes`, upgraded to transformers 4.52.4 + peft, for Llama-3.2-3B); this
> root file's `se_probes` env stays pinned for the SEP baseline. This root file stays focused
> on the SEP baseline + shared env/machine setup that `amortized_ue/` inherits.

## Environment Setup

```bash
conda-env update -f sep_enviroment.yaml
conda activate se_probes
```

Required environment variables:
- `USER` — your username (used to create scratch directories)
- `WANDB_ENT` — Weights & Biases entity for logging
- `HUGGING_FACE_HUB_TOKEN` — HuggingFace token (required for Llama models; apply for access at huggingface.co/meta-llama)
- `OPENAI_API_KEY` — only needed for long-form generation with GPT entailment/metric
- `SCRATCH_DIR` — (optional) base directory for wandb output; defaults to `.`

## Machine-specific setup (this host — Imperial DoC)

The home dir (`/homes/<user>`) has a ~12GB quota that silently breaks downloads, so everything large must live on `/vol/bitbucket`.

- **conda env**: `se_probes` lives at `/vol/bitbucket/<user>/conda_envs/se_probes` (first writable entry in `envs_dirs`). conda is not init'd for non-login shells — activate with `source /data/sv/miniconda3/etc/profile.d/conda.sh && conda activate se_probes`. Package cache is redirected via `conda config --prepend pkgs_dirs /vol/bitbucket/<user>/conda_pkgs` (default fell back to home and hit quota).
- **HF cache**: set `export HF_HOME=/vol/bitbucket/<user>/hf_cache` (model weights are GB-scale and overflow the home quota otherwise). `~/.cache/huggingface` is symlinked to it.
- **`~/.bashrc` guard**: the file has `[ -z "$PS1" ] && return` near the top, which stops non-interactive shells (incl. SLURM). All required `export`s (`WANDB_ENT`, `WANDB_API_KEY`, `HUGGING_FACE_HUB_TOKEN`, `OPENAI_API_KEY`, `HF_HOME`) must sit ABOVE that line or SLURM jobs won't see them.
- **`OPENAI_API_KEY` is required even when unused**: `uncertainty/utils/openai.py` builds the client at import time. A placeholder value is fine for the default squad-metric + DeBERTa-entailment runs (no real OpenAI call is made).
- **wandb auth**: this account's API key is 86 chars (Imperial SSO). The `wandb login` CLI in wandb 0.16.0 wrongly rejects keys != 40 chars, but `wandb.init()` accepts the real key — store it in `WANDB_API_KEY` and/or `~/.netrc` (wandb reads `~/.netrc` in all contexts incl. SLURM). Entity is the long-form `<user>-imperial-college-london`, which is valid despite its length.

**Model compatibility under the pinned env** (transformers 4.35.2 / tokenizers 0.15.0 — do not upgrade, see Working rules):
- `Llama-2-7b-chat` works (loaded via the `NousResearch` ungated mirror — see Current state).
- `falcon-7b` works natively (used for the end-to-end pipeline smoke test).
- Mistral-7B-(Instruct-)v0.1 fails to load (newer `tokenizer.json` format → `PyPreTokenizerTypeWrapper` error; fix needs tokenizers too new for the transformers cap).
- Phi-3-mini-128k-instruct: tokenizer loads but `Phi3ForCausalLM` arch isn't in transformers 4.35.2.

`*_UNANSWERABLE` AUROC metrics come out `nan` on trivia_qa (no unanswerable questions) — expected, not a bug.

## Running the Pipeline

All scripts must be run from the `semantic_uncertainty/` directory (imports are relative to that working directory).

**Stage 1 — Generate answers and hidden states:**
```bash
cd semantic_uncertainty
python generate_answers.py --model_name=Llama-2-7b-chat --dataset=trivia_qa
```

**Stage 2 — Compute uncertainty measures** (auto-triggered by Stage 1 if `--compute_uncertainties` is set, which is the default):
```bash
python compute_uncertainty_measures.py --eval_wandb_runid=<WANDB_ID>
```

**Stage 3 — Analyze results** (auto-triggered by Stage 2 if `--analyze_run` is set, which is the default):
```bash
python analyze_results.py --wandb_runids <WANDB_ID>
```

**Stage 4 — Train SEPs:** either the notebook `semantic_entropy_probes/train-latent-probe.ipynb` (full 4-dataset paper experiment, ID + OOD) or the standalone single-dataset scripts `run_llama2_probe.py` / `run_falcon_probe.py` (in-distribution only). Repoint `ds_paths` to the target run's `run-*/files` dir.

**SLURM batch runs:**
```bash
cd slurm && bash run.sh
```

Key `generate_answers.py` flags:
- `--model_name`: Llama-2-7b, Llama-2-13b, Llama-2-70b, Llama-2-7b-chat, Llama-2-13b-chat, Llama-2-70b-chat, falcon-7b, falcon-40b, Mistral-7B-v0.1, Mistral-7B-Instruct-v0.1, Phi-3-mini-128k-instruct
- `--dataset`: trivia_qa, squad, med_qa, bioasq, nq, svamp
- `--num_samples`: number of validation examples (default 400)
- `--num_generations`: number of high-temperature samples per question (default 10)
- `--metric`: squad (default, F1-based), llm, llm_gpt-3.5, llm_gpt-4

Long-form generation config: `--num_few_shot=0 --model_max_new_tokens=100 --brief_prompt=chat --metric=llm_gpt-4 --entailment_model=gpt-3.5`

## Architecture

The repo has three top-level modules (the first two are the SEP baseline; the third
is new work):

### `semantic_uncertainty/` — SE generation pipeline (adapted from [jlko/semantic_uncertainty](https://github.com/jlko/semantic_uncertainty))

Three sequential scripts that share state via **wandb artifacts** (pickle files stored in `wandb.run.dir`):

```
generate_answers.py
  → train_generations.pkl       (hidden states + accuracy for few-shot examples)
  → validation_generations.pkl  (hidden states + accuracy + sampled responses)
  → uncertainty_measures.pkl    (p_true if computed at generate stage)
  → experiment_details.pkl

compute_uncertainty_measures.py
  → uncertainty_measures.pkl    (adds semantic_entropy, p_ik, cluster entropies, etc.)

analyze_results.py
  → logs AUROC / accuracy metrics to wandb
```

`generate_answers.py` can chain directly into `compute_uncertainty_measures.py` (controlled by `--compute_uncertainties` flag, on by default).

**Key internal packages:**
- `uncertainty/models/huggingface_models.py` — `HuggingfaceModel`: wraps HF `generate()`, returns `(answer, log_likelihoods, hidden_states)`. Captures hidden states at three token positions per generation:
  - Last generated token before EOS (scalar embedding, used by p_ik)
  - Second-to-last generated token (all layers stacked → `emb_tok_before_eos`, used by SEPs as SLT position)
  - Last input token before generation starts (all layers stacked → `emb_last_tok_before_gen`, used by SEPs as TBG position)
- `uncertainty/uncertainty_measures/semantic_entropy.py` — `get_semantic_ids()` clusters responses using an entailment model (DeBERTa by default, or GPT-4/3.5/Llama); `logsumexp_by_id()` and `predictive_entropy()` compute SE from clusters
- `uncertainty/uncertainty_measures/p_ik.py` — logistic regression baseline trained on hidden states to predict correctness
- `uncertainty/utils/utils.py` — arg parser (`get_parser()`), model init, prompt construction, metric wrappers, `save()` helper that pickles and syncs to wandb

### `semantic_entropy_probes/` — SEP training

`train-latent-probe.ipynb` trains linear probes on the hidden states collected by Stage 1. Requires a completed `wandb_id` to download `validation_generations.pkl`. Trains two probe types:
- **SEP** (semantic entropy probe): predicts binarized semantic entropy from hidden states
- **Acc. Pr.** (accuracy probe): predicts correctness directly from hidden states

The notebook is wired for the 4-dataset experiment (bioasq/trivia-qa/nq/squad) with OOD cross-dataset tests and multi-panel plots; its plotting crashes on a single dataset. `run_llama2_probe.py` and `run_falcon_probe.py` reproduce ONLY the in-distribution core verbatim (load_dataset → best universal split → binarize_entropy → per-layer LogisticRegression → AUROC) for single-dataset runs. Trained probes are saved as `.pkl` to `semantic_entropy_probes/models/`.

### `amortized_ue/` — amortized UE (new work; see `amortized_ue/CLAUDE.md`)

Sibling module for the amortized-uncertainty MSc project. **Stage 1** builds one
self-contained, **id-keyed** record per prompt (canonical low-temp answer + TBG/SLT
hidden states all layers, N high-temp samples, and a **continuous**
`cluster_assignment_entropy` label) so Stage 2 can train a proxy without re-running the
LLM. **Stage 2** (`amortized_ue/stage2/`) trains a frozen Llama-3.2-3B to regress that SE
label from the stored hidden state (as soft tokens) plus optional text, in one forward
pass — in its **own env** `amortized_stage2`. It **imports the SEP logic read-only** via
`sys.path` and edits nothing under `semantic_uncertainty/`. Full details — schemas,
commands, the TBG/SLT true-position labelling (SEP's keys are inverted), the Stage-2
design, the N=2000 results, and next steps — are in the scoped **`amortized_ue/CLAUDE.md`**,
which auto-loads when working in that folder.

## Key Design Decisions

- All inter-stage data flows through wandb: each stage restores `.pkl` files from a prior run's directory by calling `wandb.run.file(filename).download(...)`. Stages are linked by `--eval_wandb_runid`.
- Records are joined across the saved pickles by **position / dict-iteration order, not by id** — entropy/embedding/accuracy arrays are aligned by index, so ordering must stay stable (see `SEP_TECHNICAL_REPORT.md` §7).
- Hidden states are extracted with `output_hidden_states=True` in `model.generate()`, then stacked across all transformer layers for the probe positions. The scalar last-token embedding (single last layer) is used for p_ik; the full stacked-layer embeddings are used for SEPs.
- bioasq dataset requires manual download from participants-area.bioasq.org.

## Working rules (baseline reproduction)

This is a baseline I must reproduce faithfully, not code to improve.

- First run: `Llama-2-7b-chat` on `trivia_qa`, short-form. Goal is to match the
  SEP paper numbers (arXiv:2406.15927) and confirm the pipeline is correct.
- Do NOT modify SE or probe logic: get_semantic_ids, logsumexp_by_id,
  cluster_assignment_entropy, the entailment model, the TBG/SLT positions, or the
  probe objective. These define the baseline.
- Change only what blocks execution: dependency versions, deprecated API calls,
  paths, device/dtype. Pin every change and explain in one line why the original failed.
- Before editing anything under semantic_uncertainty/uncertainty/, stop and ask.
- Do NOT add new models (Gemma included). New targets are a separate task.
- Never print or echo environment variables.

## Current state (updated 2026-06-29)

**Pipeline proven end-to-end. Real Llama-2-7b-chat N=400 / trivia_qa baseline COMPLETE (Stages 1–4).**

- **Environment: fully working.** conda env `se_probes`, wandb auth (86-char key in `~/.bashrc` + `~/.netrc`), HF cache on `/vol/bitbucket`, all exports above the bashrc guard. See Machine-specific setup.
- **Llama-2 access via ungated mirror:** `meta-llama/Llama-2-7b-chat-hf` is gated/"awaiting review" for HF acct `Minakshee25`. Fix: `NousResearch/Llama-2-7b-chat-hf`, a byte-identical ungated mirror (`LlamaForCausalLM`, 32 layers, 4096 hidden, 32000 vocab, `LlamaTokenizerFast` — same weights/tokenizer/config, baseline stays faithful). Code change (blocks-execution path only): `huggingface_models.py` ~line 109 redirects `Llama-2` → `base='NousResearch'`; original `base='meta-llama'` mapping kept as comments; Llama-3 still meta-llama. No SE/probe logic touched.
- **Llama-2-7b-chat baseline run COMPLETE:** N=400, trivia_qa, Stages 1→2 auto-chained. wandb run id `095l3ou2` (`celestial-night-5`), artifacts at `semantic_uncertainty/mn1025/uncertainty/wandb/run-20260624_170438-095l3ou2/files/`.
- **Llama-2 probe training COMPLETE:** `run_llama2_probe.py` (pointed at run `095l3ou2`) trained SEP + Acc. Pr. at TBG/SLT, 33 layers, SE split 0.814. Per-layer test AUROC — **SEP TBG** mean 0.623 / best layer 18 = 0.695; **SEP SLT** mean 0.608 / best layer 22 = 0.726; **AccPr TBG** mean 0.665 / best layer 11 = 0.795; **AccPr SLT** mean 0.642 / best layer 20 = 0.731. Saved to `semantic_entropy_probes/models/Llama-2-7b-chat_probe_inference.pkl`. Still to do: compare against the SEP paper (arXiv:2406.15927) and reconcile (paper expects SEP highly probeable, often > direct Acc. probe).
- **Falcon-7b (pipeline sanity, NOT the baseline):** N=400 run `9ddn5y2k` (`spring-planet-4`) + probe training validated the full pipeline; per-layer probes in `models/falcon-7b_smoke_inference.pkl`. N<400 is too few to train probes — `test_size=0.1` can leave a single-class test split and `roc_auc_score`/`log_loss` raise `ValueError: y_true contains only one label`.
- **`amortized_ue/` (new work, separate from the baseline — Stages 1 & 2 built):** Stage 1 offline SE datasets for Llama-2-7b-chat: trivia_qa N=400 (`stage1_records:v0`) + N=2000 (`stage1_records_n2000`), and squad N=1000 (OOD, local). Stage 2 SLM proxy (Llama-3.2-3B, separate `amortized_stage2` env) COMPLETE: **in-distribution** (trivia N=2000, TBG L12/k4) z-only test AUROC 0.758, z+q+resp 0.795; **OOD** (trivia→squad) z 0.622 / z+q+resp 0.618 — the hidden-state signal transfers but the text advantage does not. Full results, save locations, and the to-do list are in `amortized_ue/CLAUDE.md` / `amortized_ue/README.md`.

## Outstanding tasks

1. **(Done)** Falcon-7b pipeline validation (generation + Stage 4 probe training).
2. **(Done)** Unblock Llama-2-7b-chat without Meta gated access — `NousResearch` ungated mirror via one-line path change in `huggingface_models.py`.
3. **(Done)** Llama-2-7b-chat N=400 / trivia_qa baseline generation + probe training (run `095l3ou2`).
4. **(Open)** Compare the Llama-2 baseline AUROCs to the SEP paper (arXiv:2406.15927) and document any gap.
5. **(Pending Meta access — provenance only)** When Meta grants gated access for `meta-llama/Llama-2-7b-chat-hf` (acct `Minakshee25`), re-enable the commented-out `base='meta-llama'` mapping in `huggingface_models.py` ~line 109 and disable the `NousResearch` branch. NousResearch is byte-identical, so this is for canonical reproduction; optionally re-run to confirm parity.
