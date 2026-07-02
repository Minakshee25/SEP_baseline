# CLAUDE.md — `amortized_ue/` (amortized UE: Stage 1 dataset + Stage 2 proxy)

> **Scope: amortized-UE Stage 1 (offline dataset) and Stage 2 (SLM proxy).** This file
> governs the `amortized_ue/` module only. The repo-root `../CLAUDE.md` is also in effect
> (it owns the SEP baseline, the Imperial-DoC machine quirks, wandb auth, model
> compatibility, and the `se_probes` env). Read both. Stage 2 runs in its **own separate
> conda env** (`amortized_stage2`, see the Stage 2 section) — `se_probes` stays pinned.

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
python -m amortized_ue.stage2.run             # full run -> stage2/runs/<name>/results.json (gitignored)
# OOD: train each arm on the ID dataset, evaluate on a 2nd Stage-1 dataset (eval-only)
python -m amortized_ue.stage2.run --ood --ood_dataset squad --ood_num_samples 1000
#   -> stage2/runs/<name>/ood_results_<dataset>.json  (reuses selected pos/layer/k from results.json)
```

**Where results are saved** (all gitignored — tensors/JSON are large / run-specific):
- Stage-1 records: `amortized_ue/data/stage1/<run_name>/records/<id>.pt` + `manifest.json`.
- Stage-2 ID run: `amortized_ue/stage2/runs/stage2_<model>_<dataset>_n<N>_full/results.json`
  (sweep, k-ablation, and per-arm train/val/test metrics).
- Stage-2 OOD run: `.../ood_results_<ood_dataset>.json` in the same run dir.
- W&B: Stage-1 datasets pushed as artifacts (`stage1_records:v0` for n400,
  `stage1_records_n2000` for n2000) in project `amortized_ue_stage1`.
- The numeric headline results are also recorded below and in the memory file
  `amortized-ue-stage2.md`.

**Locked Stage-2 design (do not change without asking):**
- Projector: `LayerNorm(H_in) → Linear(H_in,256) → GELU → Dropout(0.1) → Linear(256,k·d_model)
  → reshape → per-token unit-normalise × **learnable scalar** (init emb_norm)`. The learnable
  scale keeps soft tokens in embedding-norm range WITHOUT discarding z magnitude (an earlier
  hard norm-match did, and underperformed). Interface takes `[B, n_layers_in, H]` so a future
  multi-layer ablation needs no rewrite (this build uses 1 layer).
- **Separate model per arm** (`z` / `z_q` / `z_q_resp`), each trained on its own fixed,
  **null-free** sequence — no modality dropout, no z-dropout, no learned nulls. z-only =
  `[k soft][REG]`; z+q drops the response tokens; z+q+resp keeps both.
- z = one stored **(position, layer)** selected by **validation Spearman** via a z-only sweep
  over both positions × all 33 layers, trained on a fixed **600-example TRAIN-only** subsample
  (seed 42). `k∈{1,4,8}` ablated on the z-only arm; best k used for all arms.
- Target z-score standardised on train; metrics in original space: **Spearman (primary)**,
  RMSE, MAE, R², AUROC (via train `best_split`), per arm.
- Frozen backbone, LoRA r16/α32/drop0.05 on q,k,v,o_proj, linear head, REG readout — **not to
  be changed**. bf16 backbone; projector/head fp32, cast at the backbone boundary.

## Current state (updated 2026-07-02)

**Stage 1 datasets (target LLM Llama-2-7b-chat):**
- `trivia_qa ..._n400_full/` — 400 records (mean_acc 0.5775, mean_CAE 0.6138). W&B artifact
  `stage1_records:v0` (run `4d2lvwzc`). Sanity probe: best test AUROC **0.805 (SLT L31)**.
- `trivia_qa ..._n2000_full/` — **2000 records** (mean_acc 0.5905, mean_CAE 0.5857). Built by
  reusing the 400 (verified: `random.sample` is nested, so the n2000 sample's first 400 == the
  n400 set) + generating 1600 new. Split 1440/360/200 (seed 42). W&B artifact
  `stage1_records_n2000`.
- `squad ..._n1000_full/` — **1000 records** (mean_acc 0.236, mean_CAE 1.498 — a real shift vs
  trivia's 0.59/0.59). Built for OOD evaluation only. Local; not pushed to W&B.

**Stage 2 ID run COMPLETE (trivia_qa, N=2000) — SUCCESS.** Selected **TBG layer 12, k=4**.
z-only test AUROC **0.758** / Spearman 0.459 (was 0.596 at an earlier N=400 attempt — the
soft token is now used); **z+q+resp best: test AUROC 0.795 / Spearman 0.575 / R² 0.384**
(text helps); z+q 0.733 (question alone doesn't help — the canonical *response* carries the
signal). R² positive across arms. Code committed (`772340d`); results in
`stage2/runs/stage2_..._n2000_full/results.json` (gitignored). Smoke + `--report` pass.

**Stage 2 OOD run COMPLETE (trivia_qa → squad, N=1000 eval-only, `42a3d6c`).** Each arm trained
on trivia_qa n2000 (TBG/L12/k4), evaluated on all 1000 squad rows (`ood_results_squad.json`).
Spearman / AUROC:
- z (hidden only): trivia-test 0.466/0.757 → **squad 0.287/0.622** — the hidden-state signal
  transfers across a real distribution shift.
- z+q+resp: trivia-test 0.414/0.737 → **squad 0.291/0.618** — the in-distribution text advantage
  does **NOT** transfer (z ≈ z+q+resp OOD) → **z is the domain-robust feature** (headline finding).
- z+q: 0.372/0.709 → squad 0.081/0.513 (chance). OOD RMSE/R² are meaningless (label-scale shift).

**Caveat (open):** text-arm metrics show run-to-run variance — this OOD run's ID z+q+resp was
0.737 vs the committed ID run's 0.795 (z-only stable at ~0.758). Cause: `build_ood` skips the
sweep/k-ablation, so the shared RNG state entering arm-training differs. Treat single-run
text-arm magnitudes as noisy until reseeded + multi-seed'd (item 1 below).

## To-do list (pick up here)

1. **Per-arm reseeding + multi-seed run** — reseed each arm's training independently and run a
   few seeds to report mean±std, before any strong claim about the text-arm (z+q / z+q+resp)
   advantage. Motivated by the variance caveat above.
2. **Multi-layer projector ablation** — feed a band of layers (interface already supports
   `n_layers_in > 1`) instead of a single selected layer.
3. **Full 2×2 OOD matrix** — also train on squad and eval on trivia_qa (currently only
   trivia→squad is done).
4. **Hyperparameter pass on the winning z+q+resp arm** — lr, LoRA rank, epochs.
5. **(Housekeeping)** rotate the HF token that was pasted in chat (security).
