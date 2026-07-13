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
# preferred flow: train once WITH checkpoints, then evaluate any dataset without retraining
python -m amortized_ue.stage2.run --reuse_selection --seeds 5 --save_checkpoints [--push_wandb]
#   -> results_multiseed.json + checkpoints/<arm>_seed<s>.pt  (+ W&B artifact if --push_wandb)
python -m amortized_ue.stage2.run --eval --eval_datasets squad:1000 [--push_wandb]
#   -> loads the checkpoints, scores ID test + squad(all rows) -> checkpoints/eval_summary.json
# legacy: OOD by retraining (no checkpoints) — kept for reference, superseded by --eval
python -m amortized_ue.stage2.run --ood --ood_dataset squad --reuse_selection --seeds 5
```

**Checkpointing (train once, evaluate anywhere).** `--save_checkpoints` writes one file per
`(arm, seed)` under `run_dir/checkpoints/` holding **only the ~13–17M trainable params**
(projector, REG, head, LoRA) + metadata (selected `(pos,layer,k)`, target-model `h_in`, the
training label transform, provenance) — **never the frozen 3B backbone** (~50 MB/ckpt, not GB).
`--eval` reloads them and scores the checkpoints' own dataset on its held-out **test** split (ID)
plus any `--eval_datasets name:N` on **all rows** (OOD), aggregating per arm across seeds
(mean±std) — **no retraining**. Verified 2026-07-08: reloaded ID-test AUROCs reproduce the
training log to 4 dp (`--eval --eval_datasets squad:1000`, see `checkpoints/eval_summary.json`). This is
how one proxy is evaluated on many datasets, and (by training a second run) across target models
(the projector input dim `h_in` is read back from each checkpoint). Checkpoints live on
`/vol/bitbucket` (gitignored — never committed to git); `--push_wandb` mirrors the dir as a
versioned W&B artifact (`type=model`, project `amortized_ue_stage2`) for durable, shareable storage.

`--seeds N` runs N trial seeds; each arm trains on its own deterministic `(seed, trial, arm)`
RNG stream (init + shuffle + dropout), so the arms are decoupled from the sweep/k-ablation and
`build`/`build_ood` agree per seed. `--reuse_selection` skips the sweep and reuses the saved
(position, layer, k). Single-seed runs (`results.json` / `ood_results_<ds>.json`) still exist
but the text-arm figures there are noise-dominated — **use the multi-seed numbers below.**

### ⚠️ RETRACTED — the TBG-L12 text-arm findings (2026-07-02, withdrawn 2026-07-13)

The old reference run (TBG layer 12) produced two headline claims. **Both are retracted.**

- ~~"in-distribution **text HURTS**" (z+q+resp − z = −0.041 AUROC, 5/5 seeds)~~
- ~~"under domain shift the **response HELPS**" (+0.027 AUROC / +0.045 Spearman, 5/5 seeds)~~

The 3B sweep had picked **TBG L12**, a poor layer (ridge scores 0.481 there vs 0.600 at L22). With
`z` starved of information, the text arms partly *compensated* for it, manufacturing consistent-
looking text effects. Feed `z` properly (TBG L22 + SLT L15, projector 1024) and **every text effect
collapses into noise** (|mean| ≤ 0.03, signs inconsistent at 2–3 / 5 seeds).

Lesson worth keeping: sign-consistency across seeds shows an effect isn't *seed* noise — it does
**not** show the effect is real when the whole configuration is mis-specified.

### Results — REFERENCE (5 seeds; TBG L22 + SLT L15 stacked, projector 8192→1024, k=4)

ID = trivia_qa N=2000 (held-out test); OOD = train trivia → eval squad N=1000 (all rows).
**Spearman is the primary metric** (threshold-free); AUROC secondary.

| arm | ID Spearman | OOD Spearman | ID AUROC | OOD AUROC |
|-----|:---:|:---:|:---:|:---:|
| **z (hidden only)**     | **0.602 ± 0.019** | 0.368 ± 0.033 | **0.807 ± 0.013** | 0.669 ± 0.014 |
| z + question            | 0.590 ± 0.049 | 0.402 ± 0.033 | 0.808 ± 0.025 | 0.684 ± 0.018 |
| z + question + response | 0.583 ± 0.015 | 0.398 ± 0.060 | 0.799 ± 0.012 | 0.682 ± 0.025 |

Paired (arm − z), Spearman: z+q **−0.013 ID (2/5)** / +0.034 OOD (3/5); z+q+resp **−0.020 ID (2/5)**
/ +0.030 OOD (3/5). **No text effect is sign-consistent or larger than its own std.**

**What the fixes bought (z arm):**

| config | ID | OOD | % of ID ceiling (0.914) |
|--------|:---:|:---:|:---:|
| proxy TBG L12 (original sweep pick) | 0.467 | 0.289 | 51% |
| proxy TBG L22 (layer fixed) | 0.517 | 0.256 | 57% |
| **proxy TBG22+SLT15, projector 1024** | **0.602** | **0.368** | **66%** |
| *ridge, same input* | *0.642* | *0.437* | *70%* |
| *ridge SLT L15 alone (best OOD)* | *0.584* | *0.495* | — |

**Headline (three parts):**
1. **The shortfall was input selection, not a bug.** ID 0.467 → 0.602 (+0.135); recovered signal
   51% → 66% of the achievable ceiling. ID AUROC 0.807 now edges the direct probe reference (0.805).
2. **Text adds nothing once `z` is well-fed** — the text arms were compensating for a weak `z`.
3. **The proxy still LOSES to ridge on the same input** (0.602 vs 0.642 ID; 0.368 vs 0.437 OOD).
   With the right layers and a 1024 projector this is a *fair fight*, so it stands as a clean
   **negative result**: the frozen 3B + LoRA + soft tokens extract nothing a linear readout cannot.

squad is a genuine shift (mean acc 0.24 / mean CAE 1.50 vs trivia's 0.59 / 0.59). OOD RMSE/R² are
meaningless by design (label-scale shift → R² ≈ −2); use rank metrics under shift.

### Diagnostics — label-noise ceiling + linear probe (2026-07-13)

Two read-only diagnostics that answer "is Spearman 0.47 good?" and "why isn't it higher?".
Neither modifies Stage-1/Stage-2 code or data.

**1. Label-noise ceiling** (`python -m amortized_ue.label_noise_ceiling`, `se_probes` env).
The SE label is estimated from only N=10 samples, so it carries measurement noise, and no model
can rank against it better than it ranks against itself. Split-half reliability over the stored
`semantic_id`s (200 random draws, Spearman–Brown corrected) gives the achievable ceiling:

| dataset | rows | split-half r | reliability | **ceiling** |
|---------|------|--------------|-------------|-------------|
| trivia (ID test) | 200 | 0.717 ± 0.028 | 0.835 | **0.914** |
| trivia (all) | 2000 | 0.773 ± 0.007 | 0.872 | 0.934 |
| squad (all, OOD basis) | 1000 | 0.682 ± 0.013 | 0.811 | **0.901** |

**The label is highly reliable** — noise explains only ~9 points of the gap to 1.0, so it does
*not* excuse the proxy's shortfall. Two things this buys: (a) recovered-signal framing — the z arm
recovers **51%** of achievable ID at the old TBG L12, **66%** at the fixed input; (b) a **control
for the OOD story** — squad's labels are as reliable as trivia's (0.901 vs 0.914), so the OOD drop
is a genuine transfer failure, *not* noisier OOD labels.
Caveats: reusing stored `semantic_id`s holds the DeBERTa clustering fixed, so this measures
sampling noise only (true ceiling slightly lower → true recovered% slightly higher); and
zero-entropy prompts (all 10 samples in one cluster) agree trivially across halves, propping up
`r_half` somewhat.

> **Two DIFFERENT ceilings — do not conflate.** The **label-noise ceiling (0.91)** is an upper
> bound from the noisy target; it says nothing about whether the input *contains* the information.
> The **information ceiling (~0.64 ID / ~0.50 OOD)** is what is actually recoverable from hidden
> states (measured below) — and that is the one that binds. The 0.64 → 0.91 gap is information
> genuinely **absent** from a single forward pass (SE is a property of the distribution over 10
> stochastic samples), not model failure. Do not chase 0.91 with a bigger model.

**2. Linear probe — RIDGE IS THE BASELINE TO BEAT, and it still wins** (`python -m
amortized_ue.linear_ceiling_probe`). Ridge from hidden state(s) → continuous SE, same split, same
Spearman, ID test + OOD (all rows). **Use this to pick the (position, layer)** — never the 3B sweep.

| model | input | ID test Spearman | OOD Spearman |
|-------|-------|------------------|--------------|
| Stage-2 proxy (3B + LoRA + soft tokens) | TBG L12 (old sweep pick) | 0.467 | 0.289 |
| **Stage-2 proxy (fixed)** | **TBG22 + SLT15, proj 1024** | **0.602** | **0.368** |
| ridge | TBG L12 (same input as old proxy) | 0.481 | 0.301 |
| ridge | TBG L22 (best single, ID) | 0.600 | 0.301 |
| ridge | SLT L15 (best single, OOD) | 0.584 | **0.495** |
| **ridge** | **TBG22 + SLT15 (same input as fixed proxy)** | **0.642** | **0.437** |

Conclusions:
- **The layer selection was wrong.** The 3B sweep picked TBG L12; ridge shows TBG L22–L32 are far
  better ID (0.59–0.60). The sweep trains the full 3B on a 600-example subsample for 3 epochs per
  candidate — too noisy to rank layers. Ridge is exact and costs seconds.
- **ID-optimal ≠ OOD-optimal.** Late TBG layers are ID-strong but OOD-brittle (0.24–0.30); SLT L15
  nearly wins both (0.584 / 0.495) and is still the best OOD input known — better than the proxy.
- **Ridge beats the proxy even in a fair fight** (0.642 vs 0.602 ID; 0.437 vs 0.368 OOD), with the
  same input and a widened projector. The backbone is not earning its complexity.

**3. Is there nonlinear signal? No.** MLP (unbottlenecked, full 4096/8192-dim z) vs ridge:

| input | ridge ID | **MLP ID** | ridge OOD | **MLP OOD** |
|-------|:---:|:---:|:---:|:---:|
| TBG L22 | 0.600 | **0.564** | 0.301 | **0.293** |
| SLT L15 | 0.584 | **0.567** | 0.495 | **0.463** |
| TBG L22 + SLT L15 | **0.642** | **0.584** | 0.437 | 0.420 |

**MLP LOSES to ridge at every input** — the z→SE relation is linear, which is the root reason the
3B backbone adds nothing. Two structural facts from the same sweep:
- **Extra layers within one position are near-redundant** (+0.005) → a multi-layer *band* is not
  worth pursuing (this ablation is **cancelled**).
- **The two positions ARE complementary** (+0.042) → this is what `--z_inputs` exploits.

*Caveat:* 1440 train examples for a 4096–8192-dim input, so the MLP is data-starved. The honest
claim is "**no nonlinear signal recoverable at N=2000**", not "none exists".

### Where results are saved

- Stage-1 records: `amortized_ue/data/stage1/<run_name>/records/<id>.pt` + `manifest.json` (gitignored).
- **Reference result:** `.../stage2_..._n2000_multipos_p1024/ood_results_squad_multiseed.json`;
  log `stage2/logs/multipos_p1024.log`.
- **Superseded (provenance only, do not cite):** `.../stage2_..._n2000_full/*_multiseed.json`
  (TBG L12 — retracted text-arm claims); `.../stage2_..._n2000_TBG_L22/` (layer-only fix; its JSON
  was lost to the `build_ood` mkdir bug, since fixed — numbers survive in
  `stage2/logs/tbg_L22_multiseed.log`).
- W&B artifacts (project `amortized_ue_stage1`): `stage1_records:v0` (n400), `stage1_records_n2000`.
- Diagnostics: `label_noise_ceiling.py` / `linear_ceiling_probe.py` print to stdout; pass `--out`
  to write JSON.

### To-do

**Done 2026-07-13:** ridge-based layer selection, projector widened to 1024, arm comparison re-run
at the corrected input (all text effects turned out to be noise). Multi-layer *band* ablation
**cancelled** — layers within a position are redundant (+0.005); positions are what matter.

**The one thing that now matters: the proxy loses to ridge in a fair fight** (0.602 vs 0.642 ID,
0.368 vs 0.437 OOD). Everything below is framed around that.

1. **Decide the thesis framing — talk to the supervisor.** A linear probe on hidden states
   predicting SE *is essentially SEP* (arXiv:2406.15927), so the z-only branch re-derives existing
   work, and ridge does it better. The SLM cannot be justified by "it models z better" — MLP < ridge
   shows there is no nonlinear signal to win with. Its value must come from something ridge
   **structurally cannot do**. A real pivot; surface it as a result, not a failure.
2. **(Highest value) Text-only arms.** Every arm currently contains `z`, which needs a **forward
   pass of the target LLM** — but if you run that pass anyway, SEP/ridge already solves the problem.
   Add **`q_only`** / **`q_resp_only`** (no z): can the SLM predict SE from **text alone, with no
   target-model forward pass**? Even 0.3–0.4 Spearman is a genuinely new capability (uncertainty
   *before* generation → routing, abstention, cascades). A hidden-state probe cannot do this at all.
3. **Close the ridge gap, or concede it.** Try `--projector_hidden_dim 2048`, or
   `projector_type=linear` (pass-through). If it still loses, **report the negative result**.
4. **Beat ridge OOD, or concede.** Best OOD known is ridge on **SLT L15 alone (0.495)** vs the
   proxy's 0.368. Try `--z_inputs SLT:15` — late TBG layers are OOD-brittle and may be *hurting* it.
5. **Report every result against the ridge baseline** (`[TBG22+SLT15]` = 0.642 ID / 0.437 OOD).
6. **Full 2×2 OOD matrix** — also train on squad, eval on trivia_qa. *Low priority.*
7. **Hyperparameter pass** — lr, LoRA rank, epochs. *Low priority* (arms are now equivalent).
8. **(Only to revive the nonlinear story) Scale Stage-1 to N≈10k** — "the relation is linear" is
   really "linear at N=2000". Expensive and speculative.

### Stage 2 files

- `config.py` — `Stage2Config` (every knob, incl. `z_inputs`, `projector_hidden_dim`)
- `data.py`   — id-keyed load, split, target standardise, AUROC binarisation, `z_multi`/`h_in_for`
- `model.py`  — `Projector` + `ProxyModel` (frozen backbone + LoRA + soft tokens + REG head)
- `train.py`  — `Trainer`: per-arm train/eval, (pos,layer) sweep ⚠️ *(unreliable — use ridge)*, k-ablation
- `run.py`    — `--report` / `--smoke` / full-run CLI; `--z_inputs`, `--selected_*`, `--projector_hidden_dim`
- `checkpoint.py` — save/load trainable params only (never the frozen backbone)
