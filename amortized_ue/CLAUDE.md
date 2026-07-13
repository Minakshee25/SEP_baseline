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

**Stage 2 run COMPLETE — see the MULTI-SEED numbers below, which supersede the earlier
single-run figures.** Selected **TBG layer 12, k=4** (unchanged). The original single-run
`results.json` / `ood_results_squad.json` reported z-only ID 0.758, z+q+resp ID 0.795, and
OOD z 0.622 / z+q+resp 0.618 — but those individual seeds turned out to be **noise-dominated**
for the text arms (see the multi-seed section). The multi-seed run is the reference result.

**Stage 2 MULTI-SEED run COMPLETE (5 seeds, 2026-07-02) — to-do #1. This is the reference
result and it FLIPS both earlier single-run claims.** Each arm now trains on its own
deterministic `(seed, trial_seed, arm)` RNG stream (model re-init + batch shuffle + dropout),
decoupled from the sweep/k-ablation consumption; run via `--seeds N` (+ `--reuse_selection` to
skip the sweep and reuse TBG/L12/k4). Results in `results_multiseed.json` /
`ood_results_squad_multiseed.json`; logs in `stage2/logs/multiseed_{id,ood}.log`. Test AUROC
mean±std:

| arm | ID (trivia) | OOD (squad) |
|-----|-------------|-------------|
| z (hidden only)      | **0.763 ± 0.010** (best) | 0.622 ± 0.016 |
| z + question         | 0.744 ± 0.032 | 0.586 ± 0.045 (worst) |
| z + question + resp  | 0.722 ± 0.017 | **0.650 ± 0.005** (best) |

Paired per-seed (arm − z), sign consistent across all 5 seeds:
- **ID:** z+q+resp − z = **−0.041 AUROC, negative in 5/5 seeds** → in-distribution **text HURTS**;
  z-only is the strongest and most stable arm. (The committed 0.795 > 0.758 was a lucky seed.)
- **OOD:** z+q+resp − z = **+0.027 AUROC / +0.045 Spearman, positive in 5/5 seeds** → under a real
  shift the **canonical response HELPS**. (The committed single-run "text advantage doesn't
  transfer, z is domain-robust" was also noise: that seed had z+q+resp 0.618 < z 0.622.)
- **z+q (question alone) hurts in BOTH regimes** — the *response*, not the question, carries signal.

**Robust interpretation:** in-distribution the hidden state alone is best and added text is a
distractor; under domain shift z degrades (0.763→0.622) and the canonical response supplies
complementary, transferable signal. The variance caveat that motivated this task is resolved:
`build` and `build_ood` now give identical ID-test numbers for a given seed.

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
Recovered signal: proxy z = 0.467/0.914 = **51%** ID; 0.289/0.901 = **32%** OOD (z+q+resp 37%).
**Useful control:** squad's labels are as reliable as trivia's (0.901 vs 0.914), so the OOD drop
is a *genuine transfer failure*, not noisier OOD labels — this hardens the OOD finding.
Caveats: holds DeBERTa clustering fixed (measures sampling noise only → true ceiling slightly
lower → true recovered% slightly higher); zero-entropy prompts agree trivially across halves and
prop up `r_half`.

**`linear_ceiling_probe.py` — plain ridge from ONE hidden state beats the 3B proxy.**
Same split, same continuous target, same Spearman; ID test + OOD (all squad rows):

| model | input | ID test Spearman | OOD Spearman |
|-------|-------|------------------|--------------|
| Stage-2 proxy (3B + LoRA + soft tokens) | TBG L12 | 0.467 (51% of ceiling) | 0.289 / 0.334 w/ text |
| **ridge** | **TBG L12** (identical input) | **0.481** | **0.301** |
| **ridge** | **TBG L22** (best ID) | **0.600** (66%) | 0.301 |
| **ridge** | **SLT L15** (best OOD) | **0.584** | **0.495** (55%) |

Conclusions:
- **At equal input the backbone adds nothing.** Ridge 0.481 vs proxy 0.467 at TBG L12 — the
  frozen 3B, LoRA, and soft tokens buy no nonlinear signal over a linear readout.
- **The (pos, layer) selection is wrong.** The Stage-2 sweep chose TBG L12; ridge shows TBG
  L22–L32 are far stronger ID (0.59–0.60). The sweep trains the full 3B on a 600-example
  subsample for 3 epochs per (pos, layer) — too noisy/undertrained to select reliably. Ridge
  selects exactly, in seconds.
- **ID-optimal ≠ OOD-optimal, and SLT L15 nearly wins both** (ID 0.584 / OOD 0.495), beating the
  proxy on both axes. Late TBG layers are ID-strong but OOD-brittle (0.24–0.30).
- **This partly deflates the text finding:** "response helps OOD" is +0.045 Spearman, but simply
  picking a better layer is worth **+0.16 to +0.19**. The text arm may be compensating for a
  badly chosen z. Re-run the arm comparison at the ridge-selected layer before trusting it.

**Suspected causes, in order:** (a) bad layer selection; (b) the projector's **256-dim
bottleneck** (`projector_hidden_dim=256` compresses the 4096-dim z ~16× before expanding to soft
tokens — ridge keeps all 4096 dims linearly); (c) genuinely little nonlinear headroom to add.

## To-do list (pick up here)

1. **(DONE 2026-07-02)** Per-arm reseeding + multi-seed run — implemented and run (5 seeds, ID
   + OOD); results above. Superseded the noisy single-run text-arm claims.
2. **(NEW — do first) Fix (pos, layer) selection.** Replace the expensive, noisy 3B sweep with the
   exact ridge sweep in `linear_ceiling_probe.py`, then retrain the proxy at the ridge-selected
   layer (TBG L22 for ID; SLT L15 as a both-regimes operating point). Expect ID 0.467 → ~0.58–0.60.
3. **(NEW) Widen the projector.** `projector_hidden_dim=256` is a 16× bottleneck on a 4096-dim z.
   Ablate 1024/2048 (or a linear pass-through) and re-check whether the backbone then beats ridge.
   *This touches the locked projector spec — ask before changing.*
4. **(NEW) Re-run the arm comparison at the corrected layer.** The z-vs-text ID/OOD reversal was
   measured at the bad layer TBG L12; confirm it survives at TBG L22 / SLT L15.
5. **(NEW) Justify the 3B backbone.** If a widened projector at the right layer still doesn't beat
   ridge, the honest thesis result is that the SLM adds nothing for z-only, and its value is
   confined to the text arms / OOD regime. That is a publishable negative result — report it.
6. **Multi-layer projector ablation** — feed a band of layers (`n_layers_in > 1` already supported);
   the ID/OOD layer split (TBG-late vs SLT-mid) is direct motivation for a band.
7. **Full 2×2 OOD matrix** — also train on squad and eval on trivia_qa.
8. **Hyperparameter pass** — lr, LoRA rank, epochs. Winning arm is regime-dependent (z-only ID,
   z+q+resp OOD), so tune with the intended deployment in mind.
9. **(Housekeeping)** rotate the HF token that was pasted in chat (security).
