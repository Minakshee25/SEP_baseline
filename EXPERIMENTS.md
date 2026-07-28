# Experiment log — SEP baseline → amortized UE

Chronological record of **every experiment run on this project**, what was changed each time,
what came out, and which conclusions were later **retracted**. Newest results are at the bottom.

This is the narrative document — read it end-to-end to understand how we got here.
For *how to run* things see `amortized_ue/CLAUDE.md`; for raw numbers see the JSON/logs cited
in each entry.

---

## How to read this

**Metric conventions changed partway through the project — this matters when comparing entries.**

- **E1–E7 lead with AUROC.** **E8 onward leads with Spearman**, which is now the **primary
  metric**: the task is a *regression* on continuous semantic entropy, so a threshold-free rank
  metric is the honest one. AUROC is secondary and requires binarising the label at a
  variance-minimising threshold (`best_split`, fit on **train** for ID; refit on the eval rows for
  OOD, because the label scale shifts). Where both exist, both are given.
- **R² is meaningless OOD** (the label scale shifts: trivia mean CAE 0.59 vs squad 1.50), so it
  goes strongly negative there. Use rank metrics under shift.
- **Two different ceilings — do not conflate them** (established in E8a/E8c):
  - **Label-noise ceiling ≈ 0.91** — the SE label is estimated from only N=10 samples, so no model
    can rank against it better than it ranks against itself. An upper bound nobody can beat.
  - **Information ceiling ≈ 0.64 ID / ≈ 0.50 OOD** — how much SE is *actually recoverable* from
    hidden states at all. **This is the ceiling that binds.** The 0.64 → 0.91 gap is information
    genuinely absent from a single forward pass (SE is a property of the distribution over 10
    stochastic samples), not model failure.
- "% recovered" always means `observed Spearman / label-noise ceiling`.

**Datasets** (target LLM = Llama-2-7b-chat throughout):

| dataset | N | mean accuracy | mean CAE (the SE label) | role |
|---|---|---|---|---|
| trivia_qa | 2000 | 0.5905 | 0.5857 | in-distribution (ID). Split 1440/360/200, seed 42 |
| trivia_qa | 400 | 0.5775 | 0.6138 | first Stage-1 build (its 400 are the first 400 of the 2000) |
| squad | 1000 | 0.236 | 1.498 | OOD only — a **genuine** shift (accuracy halves, entropy ~2.5×) |

---

## Timeline at a glance

| # | date | what changed | headline | status |
|---|------|--------------|----------|--------|
| E0 | 06-29 | SEP baseline reproduced (Llama-2-7b-chat / trivia_qa) | SEP probe AUROC best 0.726 | ✅ |
| E1 | 06-30 | Stage-1 offline SE dataset (N=400) | 400 id-keyed records | ✅ |
| E2 | 06-30 | Sanity probe on Stage-1 hidden states | **AUROC 0.805** — became "the bar" | ✅ |
| E3 | 07-01 | Stage-2 proxy v1, N=400 | z-only AUROC **0.596** — below the bar | ❌ failed |
| E4 | 07-01 | Stage-2 v2: new projector, per-arm models, N=2000 | z 0.758 / z+q+resp **0.795** "text helps" | ⚠️ superseded |
| E5 | 07-02 | OOD: trivia → squad | "z is domain-robust, text doesn't transfer" | ⚠️ superseded |
| E6 | 07-02 | 5 seeds (fixes run-to-run variance) | "text HURTS ID, response HELPS OOD" (5/5 seeds) | ❌ **RETRACTED** |
| E7 | 07-08 | Checkpointing + `--eval` reload | reload reproduces training to 4 dp | ✅ |
| E8 | 07-13 | **Diagnostics: ceilings, ridge, MLP** | **ridge BEATS the 3B proxy; z→SE is linear** | ✅ pivotal |
| E9 | 07-13 | Fix the layer (TBG L12 → L22) | ID Spearman 0.467 → **0.517** | ✅ |
| E10 | 07-13 | Stack 2 positions + widen projector | ID Spearman → **0.602**; all text effects vanish | ✅ **reference** |
| E11 | 07-14 | Attribution ablation (isolate each change) | 2nd position +0.042, width +0.022, **synergistic** | ✅ |
| E12 | 07-14 | **Text-only arms** (`q_only`, `q_resp_only`, no z) | **SE from the question alone, no target-LLM pass: 0.494** | ✅ **breakthrough** |
| E13 | 07-14 | Bag-of-words control (TF-IDF → ridge) | TF-IDF 0.351 ID, **0.037 OOD (chance)** — 3B is not a shortcut | ✅ control passes |
| E14 | 07-14 | Proxy on SLT:15 only (OOD-optimal input) | OOD +0.032, ID −0.075; real ID/OOD trade-off | ⚠️ partial |
| E15 | 07-14 | Overfitting + learning-curve diagnostics | gap normal; **more data won't help** (ridge flat past 400) | ✅ pivotal |
| E16 | 07-16 | Regularisation sweep (`weight_decay`, projector form) | flat/noise; single-seed overfit claim RETRACTED | ✅ |
| E17 | 07-16 | Capacity curve + 20-fold CV | width flat past 1024; **proxy NOT over/under-fitting** | ✅ **confirmed** |
| E18 | 07-27 | Reference model SAVED (25 checkpoints) + 2 bugs | best model persisted; reproduces E12 to 4 dp | ✅ |
| E19 | 07-28 | Llama-3-8B dedicated env + Stage-1 smoke | Stage-1 runs, answers sane → **cross-LLM unblocked** | ✅ |
| E20 | — | Cross-LLM transfer (frozen proxy → Llama-3-8B) | all 5 arms; the PRH test | ⏳ **next** |

---

## E0 — SEP baseline reproduction (2026-06-29, commit `fbbc3c6`)

**Goal:** reproduce the SEP paper (arXiv:2406.15927) faithfully before building anything new.

Llama-2-7b-chat / trivia_qa, N=400, short-form. W&B run `095l3ou2`. Llama-2 was gated for this HF
account, so it loads via the byte-identical `NousResearch` mirror (one-line path change; no SE or
probe logic touched). A falcon-7b run (`9ddn5y2k`) validated the pipeline end-to-end first.

Per-layer test AUROC (33 layers, SE binarisation split 0.814):

| probe | mean | best |
|---|---|---|
| SEP (semantic entropy) TBG | 0.623 | **0.695** (L18) |
| SEP SLT | 0.608 | **0.726** (L22) |
| Accuracy probe TBG | 0.665 | 0.795 (L11) |
| Accuracy probe SLT | 0.642 | 0.731 (L20) |

**Open discrepancy (still unresolved):** the paper expects SEP to be *more* probeable than the
direct accuracy probe. Here the accuracy probe wins (0.795 > 0.726). Not yet reconciled — this is
outstanding task #4 in the root `CLAUDE.md`.

---

## E1 — Stage-1: offline SE dataset (2026-06-30, commit `0408cee`)

**Change:** new `amortized_ue/` module. For each prompt, build **one self-contained, id-keyed
record**: the low-temp canonical answer + its TBG/SLT hidden states (all 33 layers), the N=10
high-temp samples with their `semantic_id`s, and a **continuous** `cluster_assignment_entropy`
label. Reuses SEP's logic read-only; edits nothing under `semantic_uncertainty/`.

Deliberate fix vs SEP: everything is joined **by id inside one record**, never by list position
(SEP's positional join is fragile — see `SEP_TECHNICAL_REPORT.md` §7).

**Result:** 400 records, mean accuracy 0.5775, mean CAE 0.6138, ~26 min on one GPU.
(Later extended to N=2000 for E4, and squad N=1000 for E5.)

---

## E2 — Sanity probe: is the signal even there? (2026-06-30, commits `4f1fc51`, `def7aa3`)

**Change:** SEP-style logistic probe on the Stage-1 hidden states (binarised SE), per layer.

**Result: best test AUROC 0.805 (SLT L31).** The hidden states clearly carry SE signal.
**This 0.805 became "the bar"** the proxy was expected to reach — and it framed the next three
experiments. *(In hindsight this bar was the right instinct but the wrong statistic: it's an
AUROC on a binarised label, and it was never converted to Spearman. E8 finally did the
apples-to-apples version.)*

---

## E3 — Stage-2 proxy v1, N=400 (2026-07-01, commit `1e3de73`) — ❌ FAILED

**The design:** frozen Llama-3.2-3B backbone. The stored hidden vector `z` is mapped by a learned
projector into `k` soft tokens, and the sequence `[k soft] (+ [text]) + [REG]` is read in **one
forward pass**; a linear head on `[REG]` regresses standardised SE. Only projector + LoRA + REG +
head train. Runs in its own env `amortized_stage2` (transformers 4.52.4 — `se_probes` can't load
Llama-3.2's rope config).

v1 specifics: projector hidden 512, **hard norm-match** on the soft tokens, **ONE multi-arm model**
with modality dropout, `(pos, layer)` selected by **val MSE**.

**Result: z-only test AUROC 0.596 (Spearman 0.177) — well below the 0.805 bar.** Text arms were
*worse than chance* (z+q+resp AUROC 0.40). Red flag: "the soft token is being ignored."

**Diagnosed causes:** (a) multi-arm training + modality dropout diluted the z-only arm;
(b) selecting by val MSE was misaligned with the ranking metric we actually cared about;
(c) N=400 is far too small — test = 40 rows, so everything was noise.

---

## E4 — Stage-2 v2: rebuilt + N=2000 (2026-07-01, commit `772340d`) — ⚠️ superseded

**Changes (all three E3 causes addressed):**
1. **Projector redefined:** `LayerNorm(4096) → Linear(4096→256) → GELU → Dropout(0.1) →
   Linear(256→k·3072) → per-token unit-norm × learnable scalar`. (Replaced the hard norm-match.)
2. **Separate model per arm** (`z` / `z+q` / `z+q+resp`) — no modality dropout, null-free sequences.
3. **`(pos, layer)` selected by val Spearman**, not MSE — via a z-only sweep over 2 positions × 33
   layers on a 600-example train-only subsample.
4. **N=400 → N=2000** (split 1440/360/200).

Selected **TBG layer 12, k=4**.

**Result:** z-only AUROC **0.758** / Spearman 0.459 (soft token now used — red flag resolved);
z+q+resp **0.795** / Spearman 0.575; z+q 0.733.

**Claim made:** *"text HELPS in-distribution"* (0.795 > 0.758). ⚠️ **Later shown to be a lucky
seed** (E6), and the whole configuration was later shown to be mis-specified (E8).

---

## E5 — OOD: trivia → squad (2026-07-02, commit `42a3d6c`) — ⚠️ superseded

**Change:** `--ood` mode. Train each arm on trivia, evaluate (eval-only, all rows) on squad N=1000.
squad is a real shift: accuracy 0.236 vs 0.59, mean CAE 1.498 vs 0.586.

**Result (Spearman / AUROC):** z 0.287/0.622 · z+q+resp 0.291/0.618 · z+q 0.081/0.513 (chance).

**Claim made:** *"the in-distribution text advantage does NOT transfer → z is the domain-robust
feature."* ⚠️ **Also later shown to be noise** (E6).

**Caveat raised at the time (and it was the right instinct):** the text arms showed run-to-run
variance — ID z+q+resp was 0.737 here vs 0.795 in E4, while z-only was stable. Cause: `build_ood`
skipped the sweep, so the shared RNG state entering arm-training differed. **This motivated E6.**

---

## E6 — Multi-seed, 5 seeds (2026-07-02, commit `09c814f`) — ❌ **RETRACTED**

**Change:** each arm now trains under its own deterministic `(seed, trial_seed, arm)` RNG stream
(model re-init + shuffle + dropout), decoupled from the sweep. So `build` and `build_ood` agree for
a given seed, and 5 trial seeds give mean ± std. Still at **TBG L12, k=4**.

**Result** (mean ± std over 5 seeds):

| arm | ID Spearman | ID AUROC | OOD Spearman | OOD AUROC |
|---|---|---|---|---|
| z | 0.467 ± 0.011 | 0.763 ± 0.010 | 0.289 ± 0.027 | 0.622 ± 0.016 |
| z+q | 0.443 ± 0.070 | 0.744 ± 0.032 | 0.211 ± 0.070 | 0.586 ± 0.045 |
| z+q+resp | 0.423 ± 0.027 | 0.722 ± 0.017 | 0.334 ± 0.013 | 0.650 ± 0.005 |

**Claims made — both sign-consistent across all 5 seeds:**
- *"In-distribution **text HURTS**"* (z+q+resp − z = −0.041 AUROC, negative **5/5**)
- *"Under shift the **response HELPS**"* (+0.027 AUROC / +0.045 Spearman, positive **5/5**)

This correctly overturned E4's and E5's single-run claims (they *were* lucky seeds).

### ❌ Both of these are now RETRACTED (see E8/E10)

The 3B sweep had selected **TBG L12 — a poor layer** (E8b: ridge scores 0.481 there vs 0.600 at
L22). With `z` starved of information, **the text arms were partly compensating for it**, which
manufactured consistent-looking text effects. Feed `z` properly (E10) and every text effect
collapses into noise.

> **Methodological lesson — the most important thing this project has taught us.**
> Sign-consistency across seeds proves an effect is not **seed** noise. It does **NOT** prove the
> effect is real when the whole **configuration** is mis-specified. Five 3B training runs agreed
> with each other and were all wrong together. A 30-second ridge baseline caught it.
> **Always establish a cheap exact baseline before believing a deep model's ablation.**

---

## E7 — Checkpointing + `--eval` reload (2026-07-08, commit `b9655c6`)

**Change:** `checkpoint.py` saves only the ~13–17M **trainable** params (projector/REG/head/LoRA)
plus metadata — **never** the frozen 3B backbone (~50 MB/ckpt). `--eval` reloads them (one backbone
load, reused) and scores any dataset with **no retraining**. Enables "train once, evaluate anywhere".

**Verified:** reloaded ID-test AUROCs reproduce the training log to **4 decimal places**
(z 0.7626 ± 0.0101, exact match), and OOD-squad matches the earlier retrain-based run. The saved
checkpoints *are* the trained models.

---

## E8 — Diagnostics: the pivotal experiment (2026-07-13, commit `54b1cb5`)

Triggered by a simple question: **"is Spearman 0.467 actually good?"** Four sub-experiments. All
read-only, CPU-only, minutes to run. They reframed the entire project.

### E8a — Label-noise ceiling (`amortized_ue/label_noise_ceiling.py`)

**Method:** the SE label is estimated from only 10 samples, so it carries measurement noise, and no
model can rank against it better than it ranks against itself. Split each prompt's stored
`semantic_id`s into two halves, recompute CAE on each, correlate across prompts (200 random draws),
and Spearman–Brown correct. Needs no GPU and no LLM re-run.

| dataset | rows | split-half r | reliability | **ceiling** |
|---|---|---|---|---|
| trivia (ID test) | 200 | 0.717 ± 0.028 | 0.835 | **0.914** |
| trivia (all) | 2000 | 0.773 ± 0.007 | 0.872 | 0.934 |
| squad (all, OOD basis) | 1000 | 0.682 ± 0.013 | 0.811 | **0.901** |

**Finding: the label is highly reliable.** Noise explains only ~9 points of the gap to 1.0, so it
does **not** excuse the proxy's shortfall. Proxy z was recovering only **51%** of achievable ID.

**Bonus control (hardens the OOD story):** squad's labels are *as reliable as* trivia's
(0.901 vs 0.914) — so the OOD drop is a **genuine transfer failure**, not noisier OOD labels.

### E8b — Ridge baseline (`amortized_ue/linear_ceiling_probe.py`) — **the bombshell**

**Method:** plain ridge regression from ONE hidden state → continuous SE, same split, same metric.
Separates "the hidden state lacks the signal" from "our model isn't extracting it."

| model | input | ID Spearman | OOD Spearman |
|---|---|---|---|
| Stage-2 proxy (3B + LoRA + soft tokens) | TBG L12 | 0.467 | 0.289 |
| **ridge** | **TBG L12** — *the same input* | **0.481** | **0.301** |
| **ridge** | TBG L22 (best ID) | **0.600** | 0.301 |
| **ridge** | SLT L15 (best OOD) | 0.584 | **0.495** |

**Three findings:**
1. **At the same input, ridge ≥ the proxy.** The frozen 3B, LoRA, and soft tokens add **nothing**.
2. **The layer selection was wrong** — worth ~0.12 Spearman. The 3B sweep trains the full model on
   a 600-example subsample for 3 epochs *per candidate*: far too noisy to rank layers. Ridge ranks
   them exactly, in seconds. **⚠️ Never use the 3B sweep to pick the layer.**
3. **ID-optimal ≠ OOD-optimal.** Late TBG layers are ID-strong but OOD-brittle; SLT L15 nearly wins
   both and is *still* the best OOD input known — better than any proxy config.

### E8c — Is there any nonlinear signal? (MLP vs ridge) — **No.**

| input | ridge ID | **MLP ID** | ridge OOD | **MLP OOD** |
|---|---|---|---|---|
| TBG L22 | 0.600 | **0.564** | 0.301 | **0.293** |
| SLT L15 | 0.584 | **0.567** | 0.495 | **0.463** |
| TBG L22 + SLT L15 | **0.642** | **0.584** | 0.437 | 0.420 |

**MLP loses to ridge at every input.** This is the **root explanation** for the whole Stage-2 story:
the backbone adds nothing *because there is nothing nonlinear to add*. Two structural facts from the
same sweep:
- **Extra layers within one position are near-redundant** (+0.005) → a multi-layer *band* is
  pointless. **This planned ablation was cancelled.**
- **The two positions ARE complementary** (+0.042, → 0.642) → this is where the real gain is, and it
  directly motivated E10.

*Caveat:* 1440 train rows for a 4096–8192-dim input, so the MLP is data-starved. The honest claim is
"**no nonlinear signal recoverable at N=2000**", not "none exists". Scaling to N≈10k is the one
experiment that could overturn this.

### E8d — Is the projector's normalisation destroying signal? — **No** (hypothesis rejected)

`Projector` strips `z`'s magnitude twice (LayerNorm on input; per-token unit-norm × a single
**global** learnable scalar — so the docstring's claim that magnitude is preserved is **false**).
**But the measured cost is only ~0.01 Spearman** (ridge on LayerNorm'd z: 0.599 vs 0.600), because
`‖z‖` carries little SE signal (ρ(‖z‖, SE) ≈ −0.21). **Not the bug.** Left alone.

---

## E9 — Fix the layer: TBG L12 → L22 (2026-07-13)

**Change:** one config change, no architecture change. New CLI (`--selected_position/layer/k`,
`--run_name`) forces the ridge-selected layer, overriding the unreliable 3B sweep.
5 seeds, ID + OOD. Log: `stage2/logs/tbg_L22_multiseed.log`.

| arm | ID Spearman | OOD Spearman |
|---|---|---|
| z | **0.517 ± 0.048** (was 0.467) | 0.256 ± 0.034 |
| z+q | 0.505 ± 0.025 | 0.215 ± 0.063 |
| z+q+resp | 0.533 ± 0.051 | **0.367 ± 0.041** |

**Findings:**
- The layer fix is worth **+0.050 ID**, as predicted.
- **But the gap to ridge WIDENED**: 0.014 at L12 → **0.083** at L22 (ridge gets 0.600 here). The
  richer the layer, the more the 256-dim bottleneck costs. → **direct evidence the bottleneck binds**,
  which justified widening it in E10.
- The E6 claim *"text hurts ID"* already **fails to replicate** here (+0.016, only 3/5 seeds).

**Bug found and fixed** (`804b188`): `build_ood()` never created its output directory, so this run
trained all 5 seeds and then died writing the JSON. Pre-existing; only surfaced because this was the
first run with a fresh `--run_name`. Numbers were recovered from the log.

---

## E10 — Stack two positions + widen the projector (2026-07-13) — ✅ **REFERENCE RESULT**

**Change (architecture):**

| | before (E6/E9) | **after (E10)** |
|---|---|---|
| z input | 1 pos × 1 layer → **4096** | 2 pos stacked `[2,4096]` → flat **8192** (TBG L22 + SLT L15) |
| projector | `LN(4096)→Lin(4096→256)→GELU→Drop→Lin(256→12288)` | `LN(8192)→Lin(8192→1024)→GELU→Drop→Lin(1024→12288)` |
| bottleneck | 256 (**16×** compression) | 1024 (**8×** compression) |
| projector params | 4,215,041 | 21,001,217 |
| total trainable | 13.4M | 30.2M |
| frozen backbone | 3.24B | 3.24B (untouched) |

Output side unchanged (k=4 soft tokens × 3072 = 12288). `model.py` needed **no change** — its
`[B, n_layers_in, H]` interface already flattened, so `h_in` just widens to `n·H`.

Command (reproduces this result):
```bash
python -m amortized_ue.stage2.run --ood --ood_dataset squad --ood_num_samples 1000 \
  --seeds 5 --reuse_selection --z_inputs TBG:22,SLT:15 --selected_k 4 \
  --projector_hidden_dim 1024 --run_name stage2_..._n2000_multipos_p1024
```

**Result** (5 seeds; `runs/..._multipos_p1024/ood_results_squad_multiseed.json`):

| arm | ID Spearman | OOD Spearman | ID AUROC | OOD AUROC |
|---|---|---|---|---|
| **z** | **0.602 ± 0.019** | 0.368 ± 0.033 | **0.807 ± 0.013** | 0.669 ± 0.014 |
| z+q | 0.590 ± 0.049 | 0.402 ± 0.033 | 0.808 ± 0.025 | 0.684 ± 0.018 |
| z+q+resp | 0.583 ± 0.015 | 0.398 ± 0.060 | 0.799 ± 0.012 | 0.682 ± 0.025 |

Paired (arm − z), Spearman: z+q **−0.013 ID (2/5)** / +0.034 OOD (3/5); z+q+resp **−0.020 ID (2/5)**
/ +0.030 OOD (3/5). **No text effect is sign-consistent or larger than its own std.**

**Three findings:**
1. **The proxy works far better.** ID **0.467 → 0.602** (+0.135); recovered signal **51% → 66%** of
   the label-noise ceiling. ID AUROC 0.807 finally edges the E2 bar (0.805).
2. **All text effects are now noise** → the E6 claims are **retracted**. The text arms had been
   compensating for a starved `z`.
3. **The proxy STILL loses to ridge on the same input** (0.602 vs 0.642 ID; 0.368 vs 0.437 OOD).
   Right layers, both positions, widened bottleneck — a **fair fight**, and it loses. Exactly as
   E8c predicted: the relation is linear, so there is nothing for a backbone to add.

---

## E11 — Attribution ablation: which change caused E10's gain? (2026-07-14) — ✅

**Why:** E10 changed **two** things at once (2 positions **and** a wider projector, 13.4M → 30.2M
params), so its +0.085 was unattributed. Two intermediate runs isolate each variable. Runs A and B
have deliberately **similar parameter counts** (~21M vs ~22M), so comparing them also controls for
the "it was just more parameters" explanation.

5 seeds each, ID + OOD. Logs `stage2/logs/ablation{A,B}_*.log`; JSON in `runs/ablation{A,B}_*/`.

| run | input | proj | params | ID Spearman | OOD Spearman |
|---|---|---|---|---|---|
| E9 baseline | TBG:22 | 256 | 13.4M | 0.517 | 0.256 |
| **A** — width only | TBG:22 | **1024** | ~21M | 0.539 ± 0.031 | 0.293 ± 0.051 |
| **B** — 2nd position only | **TBG:22 + SLT:15** | 256 | ~22M | **0.559 ± 0.050** | **0.342 ± 0.076** |
| E10 — both | TBG:22 + SLT:15 | 1024 | 30.2M | **0.602 ± 0.019** | 0.368 ± 0.033 |

**Attribution of the +0.085 ID gain:**

| effect | ID | OOD |
|---|---|---|
| projector width alone (A − E9) | **+0.022** | +0.037 |
| second position alone (B − E9) | **+0.042** | +0.086 |
| both together (E10 − E9) | **+0.085** | +0.112 |
| *sum of individual effects* | *+0.065* | *+0.123* |

**Four findings:**
1. **The second position is the bigger driver (+0.042 ID)**; the projector width is real but
   secondary (+0.022 ID). So the missing *information* mattered more than the bottleneck did.
2. **The ridge diagnostic predicted this exactly.** E8c measured the two positions as complementary
   at **+0.042** in ridge (0.600 → 0.642); the proxy delivered **+0.042**. A CPU-only diagnostic
   predicted the 3B model's gain to three decimals — **strong evidence that ridge should be used as
   the design oracle** for future input choices, rather than expensive 3B sweeps.
3. **The two changes are SYNERGISTIC in-distribution** (+0.085 actual vs +0.065 if additive).
   Mechanically sensible: 8192 dims through a 256 bottleneck is a **32×** compression, so the second
   position can only pay off if the projector is wide enough to carry it. Neither change alone gets
   near 0.602 — **you need both.** (OOD is roughly additive / slightly redundant, but the OOD stds
   are large (±0.03–0.08), so read the OOD attribution as indicative only.)
4. **It is NOT just parameter count.** A (~21M) and B (~22M) have near-identical trainable params,
   yet B gains twice as much. Capacity is not the explanation; *what information reaches the model* is.

**Consequence:** E10's architecture is now fully justified — both of its changes are load-bearing,
and each is doing what the diagnostics predicted.

---

## E12 — Text-only arms: SE with NO target-LLM forward pass (2026-07-14) — ✅ **BREAKTHROUGH**

**Why:** every `z` arm needs a forward pass of the **target LLM** to produce `z`. But if you are
running that pass anyway, a linear probe on the hidden states (SEP / our ridge baseline) already
solves the problem — **and beats this proxy** (E8/E10). So the SLM can only be justified by
something ridge **structurally cannot do**. Hence: drop `z` entirely.

**Change (purely additive; the 3 existing arms are untouched and byte-identical).** Two new arms
skip the projector; the sequence is just `[text][REG]`:
- **`q_only`** — the question alone. **No target-LLM forward pass at all.** Uncertainty known
  *before generation* → routing, abstention, cascades.
- **`q_resp_only`** — question + the canonical answer text, but **no hidden states**.

*Regression test:* re-ran seed 0 of the E10 config on the patched code — all three z-arms reproduce
the pre-change log **to 4 dp**, ID and OOD. The working path is provably unperturbed.

**Result** (5 arms × 5 seeds; `runs/stage2_textonly_5arm_p1024/`):

| arm | needs target LLM? | ID Spearman | OOD Spearman | % of ID ceiling |
|---|---|---|---|---|
| z | yes (hidden states) | **0.602 ± 0.019** | 0.368 ± 0.033 | 66% |
| z_q | yes (hidden states) | 0.590 ± 0.049 | 0.402 ± 0.033 | 65% |
| z_q_resp | yes (hidden states) | 0.583 ± 0.015 | 0.398 ± 0.060 | 64% |
| **q_only** | **NO — nothing at all** | **0.494 ± 0.049** | 0.259 ± 0.047 | **54%** |
| **q_resp_only** | answer text only, no hidden states | **0.521 ± 0.049** | **0.399 ± 0.073** | 57% |

**Two findings:**
1. **`q_only` recovers 82% of the hidden state's ID performance (0.494 vs 0.602) at ZERO cost from
   the target model** — 54% of the achievable ceiling, predicting how uncertain Llama-2 will be from
   the question text alone, *before running Llama-2*. A hidden-state probe **cannot do this by
   construction**.
2. **Under shift, text-only matches the hidden state.** `q_resp_only` OOD **0.399** ≈ `z_q_resp`
   0.398, and **beats `z` (0.368)** (AUROC 0.684 vs 0.669). Coherent: hidden states are model- and
   domain-specific; text is not.

---

## E13 — Bag-of-words control: is `q_only` just a surface shortcut? (2026-07-14) — ✅ **control passes**

**Why:** `q_only` could be learning nothing more than *"long / rare / odd-looking questions are
hard"* — a shortcut a bag-of-words model would capture just as well, in which case the 3B adds
nothing and the E12 novelty collapses. **This is exactly the discipline that caught the retracted
findings (E6 → E8): establish the cheap exact baseline before believing the deep model.**

**Change:** `amortized_ue/text_baseline_probe.py` — TF-IDF (word 1-2 grams + char 3-5 grams) → ridge,
on the identical split and metric. Plus question-length-alone as a sanity floor. CPU, seconds.

| model | needs target LLM? | ID Spearman | **OOD Spearman** |
|---|---|---|---|
| **q_only (3B)** | no | **0.494** | **0.259** |
| TF-IDF(question) → ridge | no | 0.351 | **0.037** ← *chance* |
| **q_resp_only (3B)** | answer text only | **0.521** | **0.399** |
| TF-IDF(question+answer) → ridge | answer text only | 0.384 | **0.053** ← *chance* |
| question length alone | no | 0.101 | −0.031 |

**The control passes, and the OOD column is the decisive one.** In-distribution the 3B beats
bag-of-words by ~+0.14 — real but arguable. **Out-of-distribution TF-IDF collapses to chance
(0.037) while the 3B holds 0.259 — a 7× gap.** N-grams memorise dataset-specific vocabulary and
transfer nothing; the 3B is reading something **semantic** about what makes a question hard, and it
**survives a domain change**. Not a length shortcut either (length alone: 0.101).

**⭐ This is the first unambiguously positive result for the SLM in the project.** It has a
justified, unique niche: ridge/SEP **cannot run at all** without hidden states, and the trivial text
baseline **cannot transfer**. The 3B occupies exactly that gap.

---

## E14 — Proxy on SLT:15 only, the OOD-optimal input (2026-07-14) — ⚠️ partial

**Why:** ridge showed SLT L15 *alone* is better OOD (0.495) than TBG22+SLT15 (0.437) — late TBG
layers are OOD-brittle, so the proxy might be *hurt* OOD by its TBG input.
**Result** (5 seeds, `runs/stage2_SLT15only_p1024/`): z arm **ID 0.527 / OOD 0.400** vs the reference
0.602 / 0.368. Directionally right — OOD **+0.032**, but at **−0.075 ID** (a real ID/OOD trade-off in
which layer you feed), and the gap to ridge *widened* (−0.095 OOD). The proxy now trails ridge at
**every** input tested → a *modelling* cause, pinned down in E15–E17.

## E15 — Is the proxy over/under-fitting, and would more data help? (2026-07-14) — ✅ pivotal

Two CPU diagnostics. **(a) Overfitting via the train–test gap.** Ridge on TBG22+SLT15 swept over
alpha (20-fold CV): test **rises then falls**, peaking at alpha=1e4 (test 0.637, gap 0.213).
Regularising more makes test *worse* → **the best model still memorises heavily; a ~0.2 train–test
gap is what OPTIMAL looks like** at N=1440, D=8192, label reliability 0.835. **(b) Learning curve.**
Ridge test plateaus at ~400 rows (400→1440 = 3.6× data buys only +0.026), and the MLP trails ridge
by a constant margin at every size → **more data will NOT help, and "z→SE is linear" is solid, not a
small-N artifact.** ⛔ Do not build a bigger Stage-1 dataset. **SEP itself used only 2000 samples
across tasks** (arXiv:2406.15927 §B.3), so we are already at/above SEP's data scale.

## E16 — Regularisation sweep (2026-07-16) — ✅ (a self-correction)

**Prediction:** the proxy is under-regularised → raising `weight_decay` should close the gap to
ridge. **Falsified.** `weight_decay` ∈ {0.01,0.1,1.0} → flat (early stopping pre-empts it: a **dead
knob**). `projector_type` linear vs mlp, resolved at 5 seeds paired: **+0.007 ± 0.047 → noise** (the
projector's functional form does not matter).
> ⚠️ **RETRACTED (my own error, same session):** a single-seed reading (proxy train 0.891 vs ridge
> 0.856) had suggested "under-regularised". The **5-seed** mean train is **0.829**, gap **0.227 ≈
> optimal ridge's 0.213**. Drawing an overfitting conclusion from n=1 repeated the E6 mistake — caught
> by re-running at 5 seeds. `Trainer` now logs TRAIN metrics so the gap is always visible.

## E17 — Capacity curve + CV: IS it over/under-fitting? (2026-07-16) — ✅ **CONFIRMED: neither**

To hold the proxy to ridge's standard it needs its own rise-and-fall on a *working* dial.
`projector_hidden_dim` (5 seeds each): 256→0.559 · **1024→0.602** · 2048→0.585 · 4096→0.606 → **rises
then FLAT past 1024** ⇒ not underfitting from capacity. Ridge's own peak proven by 20-fold repeated
CV (peak at alpha=1e4, significant both sides, p<0.001).
**VERDICT — the proxy is neither over- nor under-fitting.** Every failure mode ruled out: overfitting
(gap 0.227 ≈ optimal 0.213), capacity (flat past 1024), data (plateau at 400), architecture form
(linear=mlp; MLP<ridge). The residual **−0.04 to ridge is structural** — routing `z` through 4 soft
tokens into a *frozen* backbone vs ridge reading all 8192 dims directly. **The architecture is sound
and correctly sized** — required before cross-LLM transfer, so a future transfer failure can't be
blamed on a mistuned/Llama-2-memorised projector.

## E18 — Reference model SAVED with checkpoints (2026-07-27) — ✅

The E10/E12 reference models had been trained **without `--save_checkpoints`** and discarded. Two
bugs fixed: (1) `Stage2Config.save_checkpoints` flipped **default → True**; (2) **`build_ood` never
wired the checkpoint dir at all** — so `--ood` runs discarded every model even with the flag on.
Retrained the reference (TBG22+SLT15, proj1024, k4, 5 arms, 5 seeds) → **25 checkpoints** at
`runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints/` (~30M trainable params each, no frozen
backbone). Numbers reproduce E12 **to 4 dp**. `q_only`/`q_resp_only` checkpoints (no hidden states)
are directly reusable on any target LLM; the 3 z-arms need a same-hidden-dim target.

## E19 — Llama-3-8B dedicated env + Stage-1 smoke PASSED (2026-07-28) — ✅ unblocks cross-LLM

**Goal:** a *different-family, 4096-dim* second target so **all 5 arms** can transfer (unlike 13b,
5120-dim, where only text arms transfer). Llama-3 can't load in `se_probes` (transformers 4.35.2
predates it; the tokenizer fails), and running Stage-1 in `amortized_stage2` (4.52.4) hit protobuf +
`torch.load(.bin)`-CVE walls with the DeBERTa entailment model. **Fix:** a dedicated env
`se_probes_llama3` = clone of `se_probes` + **transformers 4.44.2** (loads Llama-3 ≥4.40, loads
DeBERTa's `.bin` — predates the CVE block ~4.49), torch stays 2.1.1. Leaves `se_probes`,
`amortized_stage2`, and the shared cache untouched. Code (blocks-execution, Llama-3 only): added
`'8b'` to the model-load branch + redirected Llama-3 to the ungated **NousResearch** mirror (gated on
meta-llama), same pattern as Llama-2. **Smoke passed** (`bash amortized_ue/smoke_llama3.sh`, GPU with
~32GB free): 3 records, answers correct+clean-stopping (trump / hong kong / romania), SE signal
meaningful (record 2: 5 clusters, CAE 1.557 on a wrong answer vs 1 cluster on the two right ones).

## E20 — Cross-LLM transfer, the thesis experiment (2026-07-28) — ✅ DONE. Text transfers, hidden states do NOT.

Evaluated the **frozen Llama-2-trained proxy (25 reference checkpoints) on Llama-3-8B's data** — no
retraining. Built Llama-3-8B Stage-1 on the **200 held-out (Llama-2 test-split) questions** (same
questions the proxy never trained on → leakage-free, directly comparable), scored each arm on those
200 with Llama-3's own SE labels. Llama-3-8B is 4096-dim so all 5 arms transfer; the z-arm is the
**Platonic Representation Hypothesis** test.

**Result (Spearman, 5 seeds; control = the SAME harness on Llama-2's own 200 held-out records):**

| arm | control (→Llama-2, = ID) | **transfer (→Llama-3)** | retained |
|---|---|---|---|
| **z** (hidden only) | 0.602 ± 0.019 | **0.056 ± 0.082** | ~0% (**chance**) |
| z_q | 0.590 ± 0.049 | 0.116 ± 0.042 | ~20% |
| z_q_resp | 0.583 ± 0.015 | 0.102 ± 0.217 | ~17% |
| **q_only** (no target LLM) | 0.494 ± 0.049 | **0.436 ± 0.048** | **88%** |
| **q_resp_only** (answer text) | 0.521 ± 0.049 | **0.562 ± 0.040** | **>100%** |

**Two findings — the thesis:**
1. **Hidden states do NOT transfer.** z collapses 0.602 → **0.056 (chance)**: a projector fit on
   Llama-2's hidden geometry carries *no* SE signal to Llama-3. The naive PRH reading ("reuse the
   frozen z-pathway across models") **fails** for SE across these two families. Bolting text onto the
   broken z-pathway (z_q, z_q_resp ≈ 0.1) does not rescue it — the misaligned z actively drags those
   arms *below* the pure-text ones.
2. **Text DOES transfer.** `q_only` keeps **88%** of its ID signal (0.436) with **no target-LLM
   forward pass**; `q_resp_only` transfers **fully** (0.562, above its own ID 0.521). The
   question-difficulty signal is intrinsic and **model-agnostic** → validates the *text-reading*
   proxy design. A per-model probe (SEP/ridge) cannot run cross-model at all; a learned hidden-state
   projector doesn't survive the jump either — only the text pathway does.

**Control validates the harness:** re-running the identical eval on Llama-2's own 200 held-out
records reproduces the reference ID numbers to 4 sig figs (z 0.6024, q_only 0.4939, q_resp_only
0.5208), so the z-collapse is a true model-swap effect, not a harness artifact.

**Blocks-execution fix (approved) needed to build Llama-3 Stage-1:** Llama-3's tokenizer normalises
" ?"→"?" on decode, so `full_answer.startswith(input_data)` failed on 9/200 prompts (space-before-
punctuation) → the `huggingface_models.py` else-branch raised. Fix recovers the offset from the
**token boundary** (`decode(input_tokens)`); Llama-2 keeps the startswith path byte-identical. No
SE/clustering/TBG-SLT/probe logic touched. Verified: the 9 affected prompts extracted correct answers
(e.g. "…Republican President of the United States ?" → "abraham lincoln").

**Artifacts:** Llama-3 data `amortized_ue/data/stage1/Meta-Llama-3-8B-Instruct_trivia_qa_n200_full/`
(mean_acc 0.685, mean_CAE 0.448); transfer JSON + control JSON under the reference checkpoints dir
(`cross_llm_Meta-Llama-3-8B-Instruct_trivia_qa.json`). Tooling: `amortized_ue/stage1.py --only_ids`
(build another run's held-out ids on a new target), `amortized_ue/stage2/eval_cross_llm.py`.

---

## Where we stand (2026-07-28)

**Cross-LLM transfer is DONE (E20): the thesis holds — text transfers, hidden states do not.**

**In-distribution result (reference model, saved, all 5 arms — Spearman / AUROC):**

| arm | needs target LLM? | ID | OOD | **Llama-3 transfer (Spearman)** |
|---|---|---|---|---|
| z (hidden only) | yes | 0.602 / 0.807 | 0.368 / 0.669 | **0.056 (chance)** |
| z_q · z_q_resp | yes | ~0.59 / ~0.80 | ~0.40 / ~0.68 | ~0.10 |
| **q_only** | **NO — nothing** | 0.494 / 0.758 | 0.259 / 0.614 | **0.436 (88%)** |
| q_resp_only | answer text only | 0.521 / 0.768 | 0.399 / 0.684 | **0.562 (full)** |

**Settled conclusions:**
1. **The proxy is neither over- nor under-fitting** (E15–E17); its −0.04 gap to ridge is structural.
2. **Negative result (single LLM):** with hidden states available, ridge (≈ SEP) beats the proxy
   (0.642 vs 0.602), MLP loses to ridge, more data won't help. The z-branch re-derives SEP, worse.
3. **Positive result / the thesis (E12/E13):** `q_only` predicts SE from the **question alone, no
   target-LLM forward pass** (0.494, 54% of ceiling), which a hidden-state probe cannot do; a
   bag-of-words baseline collapses to chance OOD (0.037) while the 3B holds (0.259).
4. **⭐ Cross-LLM transfer (E20):** the frozen Llama-2 proxy on Llama-3-8B — **hidden-state transfer
   FAILS** (z 0.602→0.056, chance; PRH does not hold for SE across Llama-2→Llama-3), **text transfer
   SUCCEEDS** (q_only 88%, q_resp_only full). This is the argument for the text-based proxy: only the
   model-agnostic text pathway survives a target-model swap.

**Next:** a 2nd cross-LLM target to test generality of E20 (e.g. a non-Llama family); compile
`amortized_ue/RESULTS.md`. **The consolidated to-do list lives in `amortized_ue/CLAUDE.md`.**
