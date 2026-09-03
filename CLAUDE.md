# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

A detailed read-only walkthrough of the data-generation and hidden-state-extraction internals lives in `SEP_TECHNICAL_REPORT.md`.

> 📓 **`EXPERIMENTS.md` (repo root) is the chronological log of every experiment** — what was
> changed each time, what came out, and which conclusions were **retracted**. Read it end-to-end to
> understand how the project got here; it is the narrative document the write-up should be based on.
> (This file and `amortized_ue/CLAUDE.md` describe *current state and how to run things*.)

> **New work lives in `amortized_ue/` and has its own `amortized_ue/CLAUDE.md`.** That
> module (amortized uncertainty estimation) reuses the SEP logic read-only and is governed
> by its own scoped CLAUDE.md, which auto-loads when working under `amortized_ue/`. It now
> has **Stage 1 (offline SE dataset)** and **Stage 2 (SLM proxy that predicts SE in one
> forward pass)** — both built. **The long-term goal of that module — the actual thesis —
> is CROSS-LLM TRANSFER:** train the proxy on one target LLM's data, then test it on a
> *different* target LLM, motivated by the Platonic Representation Hypothesis. See
> `amortized_ue/CLAUDE.md` § "Long-term goal". **Stage 2 runs in a separate conda env
> `amortized_stage2`** (cloned from `se_probes`, upgraded to transformers 4.52.4 + peft,
> for Llama-3.2-3B); this root file's `se_probes` env stays pinned for the SEP baseline.
> This root file stays focused on the SEP baseline + shared env/machine setup that
> `amortized_ue/` inherits.

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

Sibling module for the amortized-uncertainty MSc project. **The end goal is a proxy that
transfers ACROSS target LLMs** (train on LLM A's data, test on LLM B — a test of the
Platonic Representation Hypothesis in the uncertainty domain); that is why the proxy is an
SLM taking text alongside hidden states, not a per-model probe. **Stage 1** builds one
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

## Current state (updated 2026-08-12)

**Pipeline proven end-to-end. Real Llama-2-7b-chat N=400 / trivia_qa baseline COMPLETE (Stages 1–4).**

- **Environment: fully working.** conda env `se_probes`, wandb auth (86-char key in `~/.bashrc` + `~/.netrc`), HF cache on `/vol/bitbucket`, all exports above the bashrc guard. See Machine-specific setup.
- **Llama-2 access via ungated mirror:** `meta-llama/Llama-2-7b-chat-hf` is gated/"awaiting review" for HF acct `Minakshee25`. Fix: `NousResearch/Llama-2-7b-chat-hf`, a byte-identical ungated mirror (`LlamaForCausalLM`, 32 layers, 4096 hidden, 32000 vocab, `LlamaTokenizerFast` — same weights/tokenizer/config, baseline stays faithful). Code change (blocks-execution path only): `huggingface_models.py` ~line 109 redirects `Llama-2` → `base='NousResearch'`; original `base='meta-llama'` mapping kept as comments; Llama-3 still meta-llama. No SE/probe logic touched.
- **Llama-2-7b-chat baseline run COMPLETE:** N=400, trivia_qa, Stages 1→2 auto-chained. wandb run id `095l3ou2` (`celestial-night-5`), artifacts at `semantic_uncertainty/mn1025/uncertainty/wandb/run-20260624_170438-095l3ou2/files/`.
- **Llama-2 probe training COMPLETE:** `run_llama2_probe.py` (pointed at run `095l3ou2`) trained SEP + Acc. Pr. at TBG/SLT, 33 layers, SE split 0.814. Per-layer test AUROC — **SEP TBG** mean 0.623 / best layer 18 = 0.695; **SEP SLT** mean 0.608 / best layer 22 = 0.726; **AccPr TBG** mean 0.665 / best layer 11 = 0.795; **AccPr SLT** mean 0.642 / best layer 20 = 0.731. Saved to `semantic_entropy_probes/models/Llama-2-7b-chat_probe_inference.pkl`. Still to do: compare against the SEP paper (arXiv:2406.15927) and reconcile (paper expects SEP highly probeable, often > direct Acc. probe).
- **Falcon-7b (pipeline sanity, NOT the baseline):** N=400 run `9ddn5y2k` (`spring-planet-4`) + probe training validated the full pipeline; per-layer probes in `models/falcon-7b_smoke_inference.pkl`. N<400 is too few to train probes — `test_size=0.1` can leave a single-class test split and `roc_auc_score`/`log_loss` raise `ValueError: y_true contains only one label`.
- **`amortized_ue/` (new work — Stages 1 & 2 built; cross-LLM transfer characterised end-to-end. Work has continued well past the bullets below — E43–E70 cover the true LOLO proxy, Qwen/Gemma small- + big-tier transfer, squad OOD, latency/efficiency, and the big-tier aligned-ridge. The bullets here stop at E42; `amortized_ue/CLAUDE.md` "Current state" and `EXPERIMENTS.md` are the live record. Most recent: E71 — the E70 aligned-ridge/proxy comparison for the small-tier Qwen/Gemma "set 2"; set 2 reproduces set 1 (proxy > aligned_ridge, fuse doesn't help), opposite of E70's big tier, confirming E70's convergence was a near-identical-Qwen-27B-sibling artefact.). See `amortized_ue/CLAUDE.md` (current state + to-do) and `EXPERIMENTS.md` (full history E0→E71).**
  - **E42 (Mistral-trained proxy → Mistral's squad, filling a gap E39 left open):** the direct Mistral
    mirror of Reference-on-Llama-2 (single-source, own-model, dataset-shift-only) had never been run —
    checked first, confirmed absent. Built from the existing E22 role-swap checkpoints (never
    previously pointed at squad). **Result: 0.748 AUROC_incorrect** — beats Mistral's own SEP (0.669,
    +0.079) and beats the cross-model Reference-on-Mistral case (0.713, +0.035, dataset shift held
    constant, only the training source model differs), losing to true SE by only −0.026 (the smallest
    dataset-shift penalty seen anywhere in the OOD line). **Completes an orderly 4-point ladder:**
    Llama-2-on-itself 0.692 < Llama-2-on-Mistral 0.713 < **Mistral-on-itself 0.748** <
    Deploy/all-4-on-Mistral 0.763 < true SE 0.774 — more/better-matched training data monotonically
    shrinks the dataset-shift penalty. One-line additive change (`arm_preds` gains `ckpt_dir=None`).
  - **⚠️ E41 (user-caught: fix Llama-2's SEP layer-selection variance, rerun E38+E39):** Llama-2's SEP
    was an outlier (0.611 vs 0.72–0.74 for the other 3 targets) because `sep_single_val_selected` landed
    on TBG:21 on a near-tied 360-row val split (TBG:21/23 tie at val 0.7763 to 4dp; the whole L18–32 band
    spans 0.036) — selection noise, not a model property (leaky test-oracle over all 66 layers is only
    0.687). **Fixed additively**: new `sep_single_fixed_layer()` scores SEP at E36's leak-free CV-picked
    layer (Llama-2 TBG:30) instead of re-selecting; `sep_single_val_selected` untouched (E31 stays
    reproducible). **E38 (ID) corrected: SEP mean 0.698→0.717; proxy edge over SEP narrows 0.103→0.084,
    significant on 2/4 not 3/4** (Mistral's gap loses significance; DeepSeek/Llama-2 stay significant).
    **E39 (OOD) barely moves** (SEP mean 0.635→0.645) — proxy still loses to true SE, still beats
    SEP-fixed significantly on both targets, strict thesis test (reference proxy on Mistral vs its SEP)
    still passes at +0.046\*. **No headline overturned; the fix narrows rather than reverses.** New files
    `results/correctness_eval_e41_{fixedlayer,ood_fixedlayer}.json` — originals untouched.
  - **⚠️ E40 (model-specificity of the pooled multi-model RIDGE):** on questions where the targets
    genuinely disagree (SE 1.8 vs 1.2), does the shared probe reproduce the disagreement, or is it only a
    "hard question" detector? **Genuinely model-specific but THIN: pooled r = +0.110 [+0.027, +0.192],
    p=0.0002 = 12.6% of the attainable ceiling (0.870)** — and only on LARGE gaps (unweighted pair accuracy
    0.515, n.s.) and only for well-aligned pairs (Mistral↔Llama-3 +0.262; low-CKA Llama-2↔DeepSeek
    **+0.001**). **Response text is far more model-specific than the aligned hidden state.**
    **⚠️ METHODOLOGICAL, reuse it: the leave-ONE-out null is NEGATIVE, not 0** (a perfect pure-difficulty
    LOO predictor scores **−1.0000**, because the probe trains on the other 3 models and residuals sum to
    zero) — caught by the `q_only` control landing at −0.097 instead of ~0; the fix is a **leave-TWO-out**
    design where one probe scores both members of a pair. Also **the pooled ridge is now SAVED** — the whole
    E35/E36/E37 line had never persisted one (`--ckpt_dir` covered the proxy, not `ridge_on_z`).
  - **⚠️ E39 (OOD, trivia→squad, Llama-2+Mistral):** **E38's parity with sampling is IN-DISTRIBUTION ONLY.** Under a dataset shift **true 10-sample SE is flat (0.773→0.779)** while every amortized predictor loses 0.03–0.07 (`q_resp_only` 0.797→0.739) ⇒ **restores E31's "sampling beats amortization" out of distribution.** But **the proxy still beats supervised SEP OOD on both targets** (+0.113\*/+0.096\* pre-E41; **+0.094\*/+0.096\* vs the E41-fixed SEP**, essentially unchanged), and the **strict thesis test passes**: the Llama-2-trained proxy, which never saw Mistral *or* squad, beats Mistral's own supervised SEP (0.713 vs 0.669 fixed, Δ +0.046\*). **`q_only` collapses OOD** → the *response* text carries the transferable signal.
  - **⭐ E37–E38 (the thesis experiment + its correctness check):** ONE proxy trained on 3 target LLMs' aligned states + text, **leave-one-LLM-out** to the 4th. On SE (E37, Spearman): fuse 0.664 / **`q_resp_only` 0.648** / ridge 0.591 — label-free fusion ≥ supervised-on-sources ridge on all 4. Re-scored vs **actual wrong answers** (E38, AUROC_incorrect, same 200 rows/fold): **`q_resp_only` 0.801 mean — statistically ON PAR with the true 10-sample SE (0.785; all 4 paired CIs include 0) and significantly above supervised SEP on 2/4 after the E41 layer fix** (SEP 0.698→0.717 corrected; DeepSeek/Llama-2 still significant, Mistral no longer on its own), with **no sampling, no target hidden states, no target labels**. This **updates E31's "sampling beats amortization"**, which held for the closed-form predictors but not the trained multi-target proxy. **SE-fidelity over-ranks `fuse` and under-ranks `q_resp_only` ⇒ for wrong-answer detection pick `q_resp_only`.** Caveats: N=200/fold (wide CIs), Llama-2 fold is the anchor (not a clean cross-model test). See E41 above for the SEP-layer correction.
  - **E28–E33 (four targets: + DeepSeek):** full-power four-model alignment table (E29–E30 — **label-free ensemble ≥ supervised SEP on all 3 targets**; alignability tracks **CKA not family**, DeepSeek a low-CKA outlier); correctness-based eval (E31 — SE-fidelity ≠ correctness; true 10-sample SE is the best wrong-answer detector; label-free ensemble ≥ SEP on correctness too) + qualitative follow-ups (E32 — ~10% label noise, weak model-specific signal). **E33 — the decisive cost/benefit: given the model-agnostic text proxy `q_resp_only` (no target fitting/sampling), the aligned hidden-state arm `z_aligned` is NOT worth its per-target W-fitting cost** — a small +0.012–0.018 SE-only top-up that is **flat across CKA**, and **no statistically significant gain on correctness** (all N=1000 paired-CI deltas include 0; the earlier N=200 Llama-3 "−0.034" was a small-sample artifact). **`q_resp_only` is the right primitive for a deployable proxy.**
  - **🎯 THE THESIS: CROSS-LLM TRANSFER.** Train the proxy on target-LLM A's data, test whether it predicts SE for a *different* LLM B — motivated by the **Platonic Representation Hypothesis**. The proxy is an SLM taking question/response **text** alongside the hidden state (text is model-agnostic).
  - **Status: E20–E27 DONE (Llama-2 ↔ Mistral, + Llama-3).** (1) **Text transfers directly**, raw **hidden states do NOT** (z ≈ chance on a model swap), replicated two-family + both directions + fresh 1000-Q batch (E20–E23). (2) The raw z-failure is a **basis mismatch**: a **label-free orthogonal Procrustes** map makes the target's frozen ridge read the source's SE (E24), but it's **weakly PRH-positive** — mostly shared question-difficulty, small genuine increment (+0.03, E25/E26). (3) **Alignment DOES help UE (E27):** best **label-free** estimator = standardized/rank-fusion average of aligned-z ridge + `q_resp_only` → **AUROC 0.867, on par with the matched Mistral SEP baseline (0.857), no target labels**; ridge > 3B proxy on aligned z; late fusion > early fusion; the trivia-fit **W transfers cross-domain** (trivia→squad). Build with `stage1.py --only_ids`; align/score with `amortized_ue/procrustes_e27*.py`.
  - **Reference model — SAVED, 25 checkpoints** (`runs/REFERENCE_multipos_p1024_5arm_ckpt/` + **W&B `stage2_ckpts_REFERENCE_multipos_p1024_5arm_ckpt`**; TBG L22 + SLT L15, projector 1024, k=4, 5 seeds). **z arm ID Spearman 0.602 / AUROC 0.807, OOD 0.368**; **q_only 0.494 / 0.758**. The foundation of every E27 eval.
  - **Proxy is neither over- nor under-fitting** (E15–E17). Its −0.04 gap to ridge is *structural*, not tuning: `weight_decay` is a dead knob, projector-form linear=mlp, capacity flat past width 1024, more data won't help. *(An earlier single-seed "overfits" claim was RETRACTED at 5 seeds.)*
  - **Negative result (single LLM):** a plain **ridge beats the proxy** (0.642 vs 0.602 ID), MLP loses to ridge → z→SE is linear; a linear probe on hidden states ≈ **SEP**. Diagnostics: `linear_ceiling_probe.py` (ridge baseline + layer picker), `label_noise_ceiling.py` (ceiling ≈0.914 ID/0.901 squad; proxy recovers 66%).
  - **⭐ Positive result / the thesis (E12/E13):** **`q_only`** predicts SE **from the question alone, no target-LLM forward pass** (ID 0.494, 54% of ceiling) — a hidden-state probe cannot do this. Controlled vs TF-IDF→ridge: bag-of-words collapses to 0.037 (chance) OOD vs the 3B's 0.259 (7× gap) → not a surface shortcut.
  - ⚠️ **RETRACTED:** the TBG-L12 text-arm claims (*"text hurts ID"*, *"response helps OOD"*) — artefacts of a poorly chosen layer; text effects collapse to noise at the corrected input. Do not cite TBG-L12 numbers.

## Outstanding tasks

1. **(Done)** Falcon-7b pipeline validation (generation + Stage 4 probe training).
2. **(Done)** Unblock Llama-2-7b-chat without Meta gated access — `NousResearch` ungated mirror via one-line path change in `huggingface_models.py`.
3. **(Done)** Llama-2-7b-chat N=400 / trivia_qa baseline generation + probe training (run `095l3ou2`).
4. **(Partly done — E27)** Ran the actual SEP method (single-layer logistic, `best_split`) on the E27 data (n2000→fresh n1000): Mistral SEP 0.857 / Llama-2 SEP 0.795 AUROC; our label-free ensemble (0.867) is on par with the matched Mistral SEP. Saved-SEP 0.726 was N=400-underpowered. **Still open:** reconcile against the SEP *paper's* published numbers (arXiv:2406.15927).
5. **(Pending Meta access — provenance only)** When Meta grants gated access for `meta-llama/Llama-2-7b-chat-hf` (acct `Minakshee25`), re-enable the commented-out `base='meta-llama'` mapping in `huggingface_models.py` ~line 109 and disable the `NousResearch` branch. NousResearch is byte-identical, so this is for canonical reproduction; optionally re-run to confirm parity.
