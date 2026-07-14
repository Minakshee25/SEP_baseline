# CLAUDE.md — `amortized_ue/` (amortized UE: Stage 1 dataset + Stage 2 proxy)

> **Scope: amortized-UE Stage 1 (offline dataset) and Stage 2 (SLM proxy).** This file
> governs the `amortized_ue/` module only. The repo-root `../CLAUDE.md` is also in effect
> (it owns the SEP baseline, the Imperial-DoC machine quirks, wandb auth, model
> compatibility, and the `se_probes` env). Read both. Stage 2 runs in its **own separate
> conda env** (`amortized_stage2`, see the Stage 2 section) — `se_probes` stays pinned.

> 📓 **The chronological experiment log is `../EXPERIMENTS.md`** (repo root) — every experiment E0–E10,
> what changed, what came out, and what was **retracted**, with the reasoning. Read it for *how we got
> here*. This file is *current state + how to run things*. Keep both in sync: **when you run a new
> experiment, add an entry to `EXPERIMENTS.md`.**

## What this module is

MSc project: **amortized uncertainty estimation** — train a small model to predict a
large LLM's semantic entropy in a **single forward pass**, avoiding the multi-sample
cost at inference. Two stages, both now built:

- **Stage 1 (dataset):** for one target LLM + QA dataset, produce one **self-contained,
  id-keyed record per prompt** (canonical answer + TBG/SLT hidden states all layers, N
  high-temp samples, continuous `cluster_assignment_entropy` label) so Stage 2 never
  re-runs the target LLM.
- **Stage 2 (proxy):** train a frozen decoder-only SLM (Llama-3.2-3B) to regress that
  continuous SE label from the stored hidden state (injected as soft tokens) plus optional
  text. Consumes Stage-1 records read-only. See the **Stage 2** section below.

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
- `sanity_probe.py`        — throwaway SEP-style *classification* probe (binarised SE, per-layer AUROC).
- **`linear_ceiling_probe.py`** — ridge from one hidden state → *continuous* SE. **Use this to pick
  the (position, layer)**, and as the baseline every Stage-2 result must be reported against.
- **`label_noise_ceiling.py`**  — split-half reliability of the SE label → the achievable ceiling,
  which turns a raw Spearman into "% of achievable signal recovered".

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

## Stage 2 — SLM proxy (`amortized_ue/stage2/`)

Frozen **Llama-3.2-3B** backbone reads, in one forward pass,
`[k soft tokens] (+ [text]) + [REG readout]` and a linear head on the REG token's final
hidden state regresses the standardised SE label. Only the projector, LoRA adapters, REG
embedding, and head train.

**Files:** `config.py` (`Stage2Config`, every knob), `data.py` (id-keyed load, split,
target standardise, `best_split` binarisation for AUROC, strict-train sweep subsample,
label report), `model.py` (`Projector` + `ProxyModel`), `train.py` (`Trainer`: per-arm
train/eval, sweep, k-ablation), `run.py` (`--report` / `--smoke` / full run).

**Separate env (do not use `se_probes`).** `se_probes` (transformers 4.35.2) rejects
Llama-3.2's `rope_type:"llama3"`. Stage 2 runs in `amortized_stage2` at
`/vol/bitbucket/<user>/conda_envs/amortized_stage2`, made by **cloning `se_probes`**
(hardlinks; avoids a 5 GB torch re-download) then upgrading in the clone to
`transformers==4.52.4` + `peft` + `accelerate` (torch stays 2.1.1). `meta-llama/Llama-3.2-3B`
gated access is cleared for acct Minakshee25 (official weights, no mirror).

**Commands** (repo root, `amortized_stage2` env, pin a free GPU):
```bash
python -m amortized_ue.stage2.run --report   # label distribution + subsample checks, no GPU work
python -m amortized_ue.stage2.run --smoke     # full path, few prompts, 2 steps

# THE REFERENCE COMMAND (2026-07-13). Reproduces the current best result.
# Do NOT use the built-in 3B (pos,layer) sweep — it is unreliable (it picked TBG L12,
# which costs ~0.12 Spearman). Pick the layer with linear_ceiling_probe.py instead.
python -m amortized_ue.stage2.run \
  --ood --ood_dataset squad --ood_num_samples 1000 \
  --seeds 5 --reuse_selection \
  --z_inputs TBG:22,SLT:15 --selected_k 4 \
  --projector_hidden_dim 1024 \
  --run_name stage2_Llama-2-7b-chat_trivia_qa_n2000_multipos_p1024
#   -> ID Spearman 0.602±0.019, OOD 0.368±0.033 (z arm)

# Diagnostics (se_probes env, no GPU) — run these BEFORE any Stage-2 training:
python -m amortized_ue.linear_ceiling_probe    # exact ridge layer sweep + the baseline to beat
python -m amortized_ue.label_noise_ceiling     # achievable ceiling; converts Spearman -> % recovered
```

**New CLI flags (2026-07-13):** `--selected_position/--selected_layer/--selected_k` force the z
input (an explicit override now WINS over a saved `results.json`); `--z_inputs TBG:22,SLT:15`
stacks several positions (h_in widens to n·H automatically); `--projector_hidden_dim` sets the
bottleneck width (default stays 256 so old runs reproduce — pass 1024 when stacking);
`--run_name` keeps a new run from overwriting the reference results.

**Where results are saved** (all gitignored — tensors/JSON are large / run-specific):
- Stage-1 records: `amortized_ue/data/stage1/<run_name>/records/<id>.pt` + `manifest.json`.
- **Reference Stage-2 result:** `stage2/runs/..._n2000_multipos_p1024/ood_results_squad_multiseed.json`
  (log `stage2/logs/multipos_p1024.log`) — TBG22+SLT15, projector 1024, 5 seeds, ID + OOD.
- **Superseded (provenance only, DO NOT CITE):** `..._n2000_full/results_multiseed.json` and
  `ood_results_squad_multiseed.json` (TBG L12 — the retracted text-arm claims), and
  `..._n2000_TBG_L22/` (layer-only fix; its JSON was lost to the `build_ood` mkdir bug, now fixed
  — the numbers survive in `stage2/logs/tbg_L22_multiseed.log`).
- W&B: Stage-1 datasets pushed as artifacts (`stage1_records:v0` for n400,
  `stage1_records_n2000` for n2000) in project `amortized_ue_stage1`.

**Locked Stage-2 design (do not change without asking):**
- Projector: `LayerNorm(H_in) → Linear(H_in, hidden) → GELU → Dropout(0.1) → Linear(hidden,
  k·d_model) → reshape → per-token unit-normalise × learnable scalar (init emb_norm)`.
  `hidden` is `--projector_hidden_dim` (default 256; **use 1024 when stacking positions** — at 256
  it is a 16–32× bottleneck and it measurably binds). Interface takes `[B, n_layers_in, H]` and
  flattens, which is what makes `--z_inputs` work with **no change to `model.py`**.
  *(The docstring's claim that the learnable scalar preserves z's magnitude is FALSE — see
  "Known wart" below. Measured cost ~0.01 Spearman, so it is left alone.)*
- **Separate model per arm** (`z` / `z_q` / `z_q_resp`), each trained on its own fixed,
  **null-free** sequence — no modality dropout, no z-dropout, no learned nulls. z-only =
  `[k soft][REG]`; z+q drops the response tokens; z+q+resp keeps both.
- **z input: pick the (position, layer) with `linear_ceiling_probe.py` (exact ridge sweep), NOT
  with the built-in 3B sweep.** ⚠️ The 3B sweep (600-example subsample, 3 epochs per candidate)
  is too noisy to rank layers — it selected TBG L12, costing ~0.12 Spearman. It is retained in
  the code but **should not be used**; pass `--reuse_selection` with an explicit
  `--z_inputs` / `--selected_*`. Current best input: **`TBG:22,SLT:15`** (the two positions are
  complementary; extra *layers* within a position are not). `k=4`.
- Target z-score standardised on train; metrics in original space: **Spearman (primary)**,
  RMSE, MAE, R², AUROC (via train `best_split`), per arm. *(R² is meaningless OOD — the label
  scale shifts, so it goes strongly negative. Rank metrics only, under shift.)*
- Frozen backbone, LoRA r16/α32/drop0.05 on q,k,v,o_proj, linear head, REG readout — **not to
  be changed**. bf16 backbone; projector/head fp32, cast at the backbone boundary.

## Current state (updated 2026-07-13)

**Stage 1 datasets (target LLM Llama-2-7b-chat):**
- `trivia_qa ..._n400_full/` — 400 records (mean_acc 0.5775, mean_CAE 0.6138). W&B artifact
  `stage1_records:v0` (run `4d2lvwzc`). Sanity probe: best test AUROC **0.805 (SLT L31)**.
- `trivia_qa ..._n2000_full/` — **2000 records** (mean_acc 0.5905, mean_CAE 0.5857). Built by
  reusing the 400 (verified: `random.sample` is nested, so the n2000 sample's first 400 == the
  n400 set) + generating 1600 new. Split 1440/360/200 (seed 42). W&B artifact
  `stage1_records_n2000`.
- `squad ..._n1000_full/` — **1000 records** (mean_acc 0.236, mean_CAE 1.498 — a real shift vs
  trivia's 0.59/0.59). Built for OOD evaluation only. Local; not pushed to W&B.

### ⚠️ RETRACTED (2026-07-13): the text-arm findings from the TBG-L12 runs

The 5-seed TBG-L12 run (2026-07-02) was the reference result and produced two headline
claims. **Both are now retracted.** They were artefacts of a badly chosen z, not real
effects. Kept here so they are not re-derived by accident:

- ~~"in-distribution **text HURTS**" (z+q+resp − z = −0.041 AUROC, 5/5 seeds)~~
- ~~"under domain shift the **canonical response HELPS**" (+0.027 AUROC / +0.045 Spearman, 5/5)~~

The 3B (pos, layer) sweep had selected **TBG L12**, which the ridge diagnostic shows is a
poor layer (ridge: 0.481 at L12 vs 0.600 at L22). With z starved of information, the text
arms partly *compensated* for it, producing consistent-looking text effects. Once z is fed
properly (TBG L22 + SLT L15, projector 1024), **every text effect collapses into noise**
(see the reference result below: |mean| ≤ 0.03, signs inconsistent at 2–3 / 5 seeds).
Lesson: sign-consistency across seeds proves the effect is not *seed* noise; it does **not**
prove the effect is real if the whole configuration is mis-specified.

The superseded numbers live in `stage2/runs/..._n2000_full/results_multiseed.json` and
`ood_results_squad_multiseed.json` (kept for provenance only — do not cite).

### Architecture change (2026-07-13) — what the input/projector dims actually became

The output side is unchanged (k=4 soft tokens × d_model 3072 = 12288). What changed: the input
**doubled** (two complementary positions instead of one) while the compression ratio **halved**
(16× → 8×), so the projector is strictly less lossy despite ingesting twice as much.

| | BEFORE (original) | AFTER (reference) |
|---|---|---|
| z input | 1 pos × 1 layer → **4096** (TBG L12) | 2 pos stacked `[2,4096]` → flat **8192** (TBG L22 + SLT L15) |
| projector | `LN(4096)→Lin(4096→256)→GELU→Drop→Lin(256→12288)` | `LN(8192)→Lin(8192→1024)→GELU→Drop→Lin(1024→12288)` |
| bottleneck | 256 (**16×** compression) | 1024 (**8×** compression) |
| projector params | 4,215,041 | 21,001,217 |
| total trainable | **13.4M** | **30.2M** |
| frozen backbone | 3.24B (untouched) | 3.24B (untouched) |

`h_in` is computed automatically by `Stage2Data.h_in_for(cfg)` = `len(z_inputs) · H`; the projector
flattens `[B, n, H]`, which is why **`model.py` needed no change**.

**Both changes are load-bearing (attribution ablation, E11, 5 seeds each — `runs/ablation{A,B}_*/`):**

| change isolated | ID Spearman | Δ vs E9 (0.517) |
|---|---|---|
| projector width only (TBG:22 @ 1024) | 0.539 ± 0.031 | **+0.022** |
| second position only (TBG:22,SLT:15 @ 256) | 0.559 ± 0.050 | **+0.042** |
| both (the reference config) | **0.602 ± 0.019** | **+0.085** |

The second position matters more than the width, and the two are **synergistic** (+0.085 > +0.065 if
additive): 8192 dims through a 256 bottleneck is a 32× compression, so the extra position only pays
off once the projector is wide enough to carry it. **Neither change alone suffices.** Not a
parameter-count effect — runs A (~21M) and B (~22M) have near-identical trainable params, yet B gains
twice as much. **⭐ The ridge diagnostic predicted the +0.042 exactly** (E8c: 0.600 → 0.642) — so use
`linear_ceiling_probe.py` as the **design oracle** for input choices; it is exact, costs seconds, and
is now prospectively validated.

### REFERENCE RESULT (2026-07-13): TBG L22 + SLT L15, projector 8192→1024, 5 seeds

Run: `--ood --ood_dataset squad --seeds 5 --reuse_selection --z_inputs TBG:22,SLT:15
--selected_k 4 --projector_hidden_dim 1024`.
Output: `stage2/runs/..._n2000_multipos_p1024/ood_results_squad_multiseed.json`;
log `stage2/logs/multipos_p1024.log`. **Spearman is primary; AUROC secondary.**

| arm | ID Spearman | OOD Spearman | ID AUROC | OOD AUROC |
|-----|-------------|--------------|----------|-----------|
| **z (hidden only)** | **0.602 ± 0.019** | 0.368 ± 0.033 | **0.807 ± 0.013** | 0.669 ± 0.014 |
| z + question | 0.590 ± 0.049 | 0.402 ± 0.033 | 0.808 ± 0.025 | 0.684 ± 0.018 |
| z + question + resp | 0.583 ± 0.015 | 0.398 ± 0.060 | 0.799 ± 0.012 | 0.682 ± 0.025 |

Paired (arm − z), Spearman: z_q **−0.013 ID (2/5)** / +0.034 OOD (3/5); z_q_resp **−0.020 ID
(2/5)** / +0.030 OOD (3/5). **No text effect is sign-consistent or larger than its own std —
all text arms are now indistinguishable from z-only.**

**Trajectory of the z arm (what the fixes bought):**

| config | ID | OOD | % of ID ceiling (0.914) |
|--------|-----|-----|------------------------|
| proxy TBG L12 (original sweep pick) | 0.467 | 0.289 | 51% |
| proxy TBG L22 (layer fixed) | 0.517 | 0.256 | 57% |
| **proxy TBG22+SLT15, projector 1024** | **0.602** | **0.368** | **66%** |
| *ridge, same input (TBG22+SLT15)* | *0.642* | *0.437* | *70%* |
| *ridge SLT L15 alone (best OOD)* | *0.584* | *0.495* | — |

**Honest interpretation (three parts):**
1. **The proxy's shortfall was input selection, not a bug.** Fixing the layer and feeding both
   positions through a wider projector moved ID 0.467 → 0.602 (+0.135) and recovered 51% → 66%
   of the label-noise ceiling. ID AUROC 0.807 now edges the direct SEP-style sanity probe (0.805).
2. **The text arms add nothing once z is well-fed.** They were compensating for a weak z.
3. **The proxy still LOSES to ridge on the same input** (0.602 vs 0.642 ID; 0.368 vs 0.437 OOD),
   even with the right layers and a 1024-wide projector. This is now a *fair fight* and a clean
   **negative result**: the frozen-3B + LoRA + soft-token design extracts nothing a linear
   readout cannot. Consistent with the MLP-vs-ridge test (below): the z→SE relation is linear,
   so there is no nonlinear signal for a backbone to add.

**Checkpointing (2026-07-02) — train once, evaluate anywhere.** Previously OOD *retrained* the
model (no weights were ever saved — only metrics JSON). Now `checkpoint.py` + `--save_checkpoints`
write one file per `(arm, seed)` under `run_dir/checkpoints/` holding **only the ~13–17M trainable
params** (projector/REG/head/LoRA) + metadata (selected `(pos,layer,k)`, target-model `h_in`,
training label transform, provenance) — **never the frozen 3B backbone** (~50 MB/ckpt). `--eval`
(`run_eval`) reloads them (reusing one backbone load) and scores the ID dataset's held-out **test**
split plus any `--eval_datasets name:N` on **all rows**, aggregating per arm across seeds — **no
retraining**. **Reload verified 2026-07-08** (`--eval --eval_datasets squad:1000`, log
`stage2/logs/eval_reload.log`, output `checkpoints/eval_summary.json`): reloaded ID-test AUROCs
reproduce the training log to 4 dp (z 0.7626±0.0101, z_q 0.7440±0.0323, z_q_resp 0.7218±0.0170 —
exact match), and OOD-squad matches the earlier retrain-based OOD run (z 0.622, z_q 0.586,
z_q_resp 0.650) — confirming the saved checkpoints ARE the trained models and the round-trip is
correct. (`run_eval` scores each reloaded checkpoint independently; it does not itself assert an
in-memory-vs-reload diff — the 4-dp reproduction is the evidence.) This is the
mechanism for "one proxy → many datasets" and, via a second training run, across target models
(the projector input dim is rebuilt from each checkpoint's `h_in`). `Trainer` now accepts a
prebuilt `model=` (eval reuses the backbone). Storage: local `/vol/bitbucket` (gitignored) +
optional `--push_wandb` versioned artifact (`type=model`, project `amortized_ue_stage2`).

## Diagnostics (2026-07-13) — the proxy is UNDERPERFORMING a ridge regression

Two read-only scripts, both run from the repo root in the **`se_probes`** env (they need no
GPU and touch nothing under `semantic_uncertainty/`). They were added to answer "is Spearman
0.467 any good?" and "why isn't it higher?". The answer to the second is uncomfortable and
should drive the next work.

**`label_noise_ceiling.py` — the SE label is RELIABLE, so noise does not excuse the gap.**
The label is estimated from N=10 samples, so it carries measurement noise, and no model can
rank against it better than it ranks against itself. Split-half reliability over the stored
`semantic_id`s (200 draws, Spearman–Brown corrected; the subset's ids are relabelled to
contiguous before calling SEP's `cluster_assignment_entropy`, because `np.bincount` on a subset
otherwise yields `0*log 0 = nan`):

| dataset | rows | split-half r | reliability | **ceiling = sqrt(rel)** |
|---------|------|--------------|-------------|-------------------------|
| trivia (ID test) | 200 | 0.717 ± 0.028 | 0.835 | **0.914** |
| trivia (all) | 2000 | 0.773 ± 0.007 | 0.872 | 0.934 |
| squad (all, OOD basis) | 1000 | 0.682 ± 0.013 | 0.811 | **0.901** |

So the achievable ceiling is ~0.91, not ~0.6 — **label noise accounts for only ~9 points**.
Recovered signal (z arm): **51%** ID at the original TBG L12 → **66%** ID at the fixed input
(0.602/0.914). **Useful control:** squad's labels are as reliable as trivia's (0.901 vs 0.914),
so the OOD drop is a *genuine transfer failure*, not noisier OOD labels.
Caveats: holds DeBERTa clustering fixed (measures sampling noise only → true ceiling slightly
lower → true recovered% slightly higher); zero-entropy prompts agree trivially across halves and
prop up `r_half`.

**IMPORTANT — two DIFFERENT ceilings; do not conflate them.**
- **Label-noise ceiling ≈ 0.91** — an upper bound imposed by the noisy target. Unreachable by
  anyone. Says nothing about whether the input *contains* the needed information.
- **Information ceiling ≈ 0.64 ID / ≈ 0.50 OOD** — how much SE is actually recoverable from
  hidden states at all (measured below). This is the ceiling that actually binds.
The 0.64 → 0.91 gap is **information genuinely absent from the hidden states**, not model
failure: SE is a property of the *distribution over 10 stochastic samples*, and one
deterministic forward pass cannot fully encode it. Chasing 0.91 with a bigger model is chasing
something that is not there.

**`linear_ceiling_probe.py` — plain ridge from ONE hidden state beats the 3B proxy.**
Same split, same continuous target, same Spearman; ID test + OOD (all squad rows):

| model | input | ID test Spearman | OOD Spearman |
|-------|-------|------------------|--------------|
| Stage-2 proxy (3B + LoRA + soft tokens) | TBG L12 | 0.467 (51% of ceiling) | 0.289 / 0.334 w/ text |
| **ridge** | **TBG L12** (identical input) | **0.481** | **0.301** |
| **ridge** | **TBG L22** (best ID) | **0.600** (66%) | 0.301 |
| **ridge** | **SLT L15** (best OOD) | **0.584** | **0.495** (55%) |

Conclusions:
- **The (pos, layer) selection was wrong.** The Stage-2 sweep chose TBG L12; ridge shows TBG
  L22–L32 are far stronger ID (0.59–0.60). The sweep trains the full 3B on a 600-example
  subsample for 3 epochs per (pos, layer) — too noisy/undertrained to rank layers. **Ridge selects
  exactly, in seconds — always pick the layer this way, never with the 3B sweep.**
- **ID-optimal ≠ OOD-optimal.** Late TBG layers are ID-strong but OOD-brittle (0.24–0.30); SLT
  L15 nearly wins both (ID 0.584 / OOD 0.495) and is still the best single OOD input known.
- **At equal input the backbone adds nothing** — see the fair-fight result above: even with the
  right layers and a 1024 projector the proxy (0.602 / 0.368) loses to ridge (0.642 / 0.437).

**`info_ceiling` experiment — the z→SE relation is LINEAR; there is no nonlinear headroom.**
MLP (unbottlenecked, on the full 4096/8192-dim z) vs ridge, same split:

| input | ridge ID | **MLP ID** | ridge OOD | **MLP OOD** |
|-------|----------|-----------|-----------|-------------|
| TBG L22 | 0.600 | **0.564** | 0.301 | **0.293** |
| SLT L15 | 0.584 | **0.567** | 0.495 | **0.463** |
| TBG L22 + SLT L15 | **0.642** | **0.584** | 0.437 | 0.420 |

**MLP LOSES to ridge at every input.** This is the root explanation for the whole Stage-2 story:
the backbone adds nothing because *there is nothing nonlinear to add*. Two more structural facts
from the same sweep:
- **Extra layers within a position are near-redundant** (TBG L22 alone 0.600 → TBG every-4th
  0.605, i.e. +0.005). A multi-layer *band* is NOT worth pursuing.
- **The two positions ARE complementary** (TBG L22 + SLT L15 → 0.642, +0.042). This is what
  motivated `--z_inputs` and it is where the real gain lives.

*Caveat:* only 1440 train examples for a 4096–8192-dim input, so the MLP is data-starved — the
honest claim is "**no nonlinear signal recoverable at N=2000**", not "none exists". Scaling
Stage-1 to N≈10k is the one experiment that could overturn this.

**Known wart (measured, minor):** `Projector` destroys z's magnitude twice — `LayerNorm` on the
input strips ‖z‖ per example, and the output is per-token unit-normalised then scaled by a
**single global learnable scalar** (`model.py:66,74`). The docstring claim that this preserves
z's magnitude is **false** (a learned constant scales all examples identically). Measured cost is
small (~0.01 Spearman: ridge on LayerNorm'd z scores 0.599 vs 0.600 at TBG L22), because ‖z‖
carries only weak SE signal (ρ(‖z‖, SE) ≈ −0.21 at TBG L22). Not worth fixing; do not cite the
docstring.

## ⭐ THE HEADLINE RESULT (E12/E13, 2026-07-14) — text-only arms

Every `z` arm needs a forward pass of the **target LLM**. But if you are running that pass anyway,
ridge/SEP on the hidden states already solves the problem *and beats this proxy*. So the SLM's
justification must come from what ridge **structurally cannot do** — running with **no hidden
states at all**. Two new arms (`q_only`, `q_resp_only`) drop `z` entirely; sequence = `[text][REG]`,
the projector is never called. Opt in via `--arms z,z_q,z_q_resp,q_only,q_resp_only`.

| arm | needs target LLM? | ID Spearman | OOD Spearman | % ID ceiling |
|---|---|---|---|---|
| z | yes (hidden states) | 0.602 ± 0.019 | 0.368 ± 0.033 | 66% |
| **q_only** | **NO — nothing at all** | **0.494 ± 0.049** | 0.259 ± 0.047 | **54%** |
| **q_resp_only** | answer text only, no hidden states | **0.521 ± 0.049** | **0.399 ± 0.073** | 57% |

**`q_only` predicts SE from the question alone, BEFORE running the target model** — 82% of the
hidden state's ID performance at zero cost. `q_resp_only` OOD (0.399) **beats z-only** (0.368).

**Control (E13, `text_baseline_probe.py`) — it is NOT a bag-of-words shortcut:** TF-IDF→ridge on the
same text gets ID 0.351 and **collapses to 0.037 (chance) OOD**, vs the 3B's 0.259 — a **7× gap**.
Question-length alone: 0.101. So the 3B is reading something *semantic* about question difficulty
that transfers across a domain shift; n-grams only memorise dataset vocabulary.

**The two-regime framing for the thesis:**
- **Hidden states available** → a linear probe (≈ SEP) is all you need; **the proxy is redundant**.
  Report this as a clean negative result.
- **No target-LLM forward pass** → **only the SLM can run at all**, and no trivial text baseline
  comes close. This is the regime that matters for routing / abstention / cascades.

## To-do list (pick up here)

**Done (2026-07-13):** ridge-based layer selection (#2), projector widened to 1024 (#3), arm
comparison re-run at the corrected input (#4 — all text effects turned out to be noise). The
multi-layer *band* ablation is **cancelled**: measured at +0.005, layers within a position are
redundant; positions are what matter, and `--z_inputs` already exploits that.

**The one thing that now matters: the proxy loses to ridge in a fair fight (0.602 vs 0.642 ID,
0.368 vs 0.437 OOD).** Everything below is framed around that.

1. **Decide the thesis framing — talk to the supervisor first.** A linear probe on hidden states
   predicting SE *is essentially SEP* (arXiv:2406.15927). Since ridge wins, the z-only branch of
   this project re-derives existing work. The SLM cannot be justified by "it models z better" —
   MLP < ridge proves there is no nonlinear signal to win with. Its value must come from something
   ridge **structurally cannot do**. This is a real pivot; surface it as a result, not a failure.
2. **(Highest value) Text-only arms — the one thing ridge cannot do.** Every current arm contains
   z, which requires a **forward pass of the target LLM** — but if you are running that pass
   anyway, SEP/ridge already solves the problem. Add **`q_only`** and **`q_resp_only`** arms (no z)
   to test whether the SLM can predict SE from **text alone, with no target-model forward pass**.
   Even ~0.3–0.4 Spearman would be a genuinely new capability (uncertainty *before* generation:
   routing, abstention, cascades). A hidden-state probe cannot do this by construction.
3. **Close the remaining ridge gap, or concede it.** Proxy 0.602 vs ridge 0.642 ID. Try
   `--projector_hidden_dim 2048` / `projector_type=linear` (pass-through). If it still loses,
   **report the negative result** — it is well-controlled and publishable.
4. **Beat ridge OOD, or concede.** Best OOD known is ridge on **SLT L15 alone (0.495)**, well
   above the proxy's 0.368. Try `--z_inputs SLT:15` (single position) — late TBG layers are
   OOD-brittle and may be *hurting* the OOD arm.
5. **Report every result against the ridge baseline.** Ridge on `[TBG L22 + SLT L15]` = **0.642 ID
   / 0.437 OOD** (and SLT L15 = 0.495 OOD) is the number to beat. Cheap, exact, seconds on CPU.
6. **Full 2×2 OOD matrix** — also train on squad and eval on trivia_qa. *Low priority.*
7. **Hyperparameter pass** — lr, LoRA rank, epochs. *Low priority* (the arms are now equivalent,
   so there is no regime-dependent arm choice to tune for).
8. **(Only if reviving the nonlinear story) Scale Stage-1 to N≈10k.** The MLP had 1440 examples
   for a 4096–8192-dim input, so "the relation is linear" is really "linear at N=2000". Expensive
   and speculative; do not lead with it.
9. **(Housekeeping)** rotate the HF token that was pasted in chat (security).
