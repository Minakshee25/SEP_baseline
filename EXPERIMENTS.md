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

## E21 — Cross-LLM #2: Mistral-7B (different family) — ✅ REPLICATES E20. Generality confirmed.

Repeated E20 with a **non-Llama** 2nd target to test generality. **Mistral-7B-Instruct-v0.2** chosen:
4096-dim (= Llama-2 → all 5 arms transfer) AND a different family. It loads in `se_probes_llama3`
(transformers 4.44 — the old 4.35.2 tokenizer failure is gone) with **no code change** (the Mistral
load branch already existed) and no fix-A trigger. Built Stage-1 on the **same 200 held-out ids**
(`stage1.py --only_ids`; mean_acc 0.67 / mean_CAE 0.476), scored the frozen Llama-2 proxy on it.

**Result (Spearman) — replicates E20 on a different family:**

| arm | **Mistral transfer** | Llama-3 transfer (E20) | in-dist (ID) |
|---|---|---|---|
| **z** (hidden only) | **0.044 (chance)** | 0.056 | 0.602 |
| z_q / z_q_resp | 0.104 / 0.065 | 0.116 / 0.102 | ~0.59 |
| **q_only** (no target LLM) | **0.410 (76% of ceiling)** | 0.436 (86%) | 0.494 |
| **q_resp_only** (answer text) | **0.511 (~full)** | 0.562 | 0.521 |
| SE-label ceiling (corr w/ Llama-2) | **0.540** | 0.505 | — |

**Generality confirmed.** The E20 pattern is not a Llama-3 (same-family) quirk: on a *different family*
(Mistral) it holds identically — **hidden states do NOT transfer** (z ≈ chance, 0.044, despite matching
4096 dims → dimension-match ≠ representational alignment) and **text DOES** (q_only 76%, q_resp_only
~full). The shared cross-model difficulty ceiling is again ~0.5 (0.540 vs Llama-3's 0.505). q_only
recovers slightly less of the ceiling than for Llama-3 (76% vs 86%), consistent with Mistral being a
more distant model. Two-family evidence for "text transfers, hidden-state geometry doesn't."

**Infra:** `push_to_wandb` default flipped **→ True** (smoke builds excluded) and W&B artifact names
made auto-distinct per (model, dataset, N) so each dataset is independently reloadable; both cross-LLM
datasets pushed (`stage1_records_{Meta-Llama-3-8B-Instruct,Mistral-7B-Instruct-v0.2}_trivia_qa_n200`).
New helper `amortized_ue/push_dataset_wandb.py` back-fills existing datasets. All Stage-1 datasets live
on `/vol/bitbucket` (source of truth) + now W&B (extra copy). Tooling: `build_mistral_*.sh`.

## E22 — Role swap: train the proxy on Mistral → test on Llama-2 — ✅ transfer is DIRECTIONALLY SYMMETRIC

E20/E21 trained on Llama-2 and transferred *out*. E22 reverses it: **train the proxy on Mistral,
test on Llama-2**, to check the finding isn't an artifact of "Llama-2 is a good source model." Built
**Mistral n2000** (same seed as Llama-2's n2000 → identical questions + 1440/360/200 split; the 200
test records reused from E21 so the test set is unchanged), re-picked Mistral's best z-layers with
`linear_ceiling_probe` (**TBG L31 + SLT L20**, in-dist ridge ceiling ~0.62 — Mistral's z→SE is as
probeable as Llama-2's), and trained a fresh 5-arm × 5-seed proxy (proj 1024, k=4) on Mistral's 1440.

**Result (Spearman):**

| arm | Mistral in-dist (test-200) | **Mistral→Llama-2 transfer (E22)** | Llama-2→Mistral (E21) |
|---|---|---|---|
| **z** (hidden) | 0.638 ± 0.024 | **−0.002 (chance)** | 0.044 |
| z_q / z_q_resp | 0.597 / 0.630 | 0.004 / 0.037 | 0.116 / 0.102 |
| **q_only** | 0.414 ± 0.060 | **0.476 (88% of the 0.540 ceiling)** | 0.410 |
| **q_resp_only** | 0.528 ± 0.026 | **0.509** | 0.511 |

**Directionally symmetric.** Swapping which model is source vs target changes nothing: **hidden
states don't transfer either way** (z ≈ chance both directions), **text transfers both ways** (q_only
0.41↔0.48, q_resp_only ~0.51 both). The phenomenon is a property of the *model pair*, not the
direction — so "text transfers, hidden geometry doesn't" is not because Llama-2 happens to be a good
source. Mistral's own in-distribution proxy (z 0.638) is a touch stronger than Llama-2's (0.602),
confirming Mistral is a full-strength counterpart, not a weak one.

**Infra (all in amortized_ue/):** `stage2/run.py` gained `--stage1_model_name`/`--stage1_dataset` (train
the proxy on any target's records). Fixed a wandb-push CLI bug — `stage1.py --push_to_wandb` was a
store_true defaulting False that silently overrode the new True config default; now push-by-default
with `--no_push_to_wandb` to opt out. Fixed the recurring **home-quota** break: wandb's artifact cache
(`~/.cache/wandb`, 4.4 GB) was filling the 12 GB home quota → redirected `WANDB_CACHE_DIR`/
`WANDB_DATA_DIR` to `/vol/bitbucket` (added to `~/.bashrc` above the `$PS1` guard, matching HF_HOME).
Datasets + checkpoints saved: Mistral n2000 (`stage1_records_Mistral-7B-Instruct-v0.2_trivia_qa_n2000`)
and the E22 proxy (`runs/E22_Mistral_proxy_p1024_5arm_ckpt/`, 25 ckpts) both on /vol/bitbucket + W&B.

## E23 — Replication on a FRESH 1000-question held-out batch (zero overlap) — ✅ confirms E20–E22 at 5× power

E20–E22 all used the same 200 held-out test questions. E23 stress-tests the finding on a **fresh,
larger, fully-disjoint** batch. Built **1000 trivia_qa questions drawn from the 1074 validation
questions in NO existing build** — proven zero overlap with n2000 and its train(1440)/val(360)/test(200)
splits and every prior build (all existing trivia builds are subsets of n2000, so unique "seen" = 2000;
the batch is the complement). Same Stage-1 procedure (seed-10 few-shot prompt, `--selection_num_samples
3074 --only_ids`). Built for **both** targets in their faithful envs — Llama-2 in `se_probes` (4.35.2,
mean_acc 0.609 / CAE 0.608), Mistral in `se_probes_llama3` (4.44, mean_acc 0.649 / CAE 0.451). **No
retraining** — scored the existing frozen checkpoints (Llama-2 REFERENCE proxy, E22 Mistral proxy) on
the fresh batch, all 5 arms, via `eval_cross_llm` (split=all).

**Result (Spearman) — 4 combos, N=1000; SE-label ceiling Llama-2↔Mistral = 0.524:**

| combo | z | z_q | z_q_resp | q_only | q_resp_only |
|---|---|---|---|---|---|
| **REF proxy → Llama-2** (in-dist) | 0.562 | 0.561 | 0.545 | 0.489 | 0.558 |
| **E22 proxy → Mistral** (in-dist) | 0.628 | 0.604 | 0.620 | 0.487 | 0.572 |
| **REF proxy → Mistral** (transfer) | **0.014** | 0.073 | 0.124 | **0.474** | **0.531** |
| **E22 proxy → Llama-2** (transfer) | **0.031** | 0.015 | 0.042 | **0.477** | **0.523** |

(std tight: ~0.006–0.05, vs ~0.02–0.09 on the n200 evals — 5× the questions.)

**Confirms E20–E22 with higher confidence:** hidden-state transfer ≈ **chance** both directions (z
0.014 / 0.031), text transfer **holds** both directions (q_only ~0.475 = ~90% of the 0.524 ceiling,
q_resp_only ~0.52). New angle this batch adds: **in-distribution z stays high on FRESH questions**
(0.562 / 0.628) → the proxy generalises to unseen *questions* fine; it is specifically the *model swap*
that destroys the z-arm, not question novelty. Datasets on /vol/bitbucket + W&B
(`stage1_records_{Llama-2-7b-chat,Mistral-7B-Instruct-v0.2}_trivia_qa_n1000`). Tooling:
`build_e23_fresh.sh` (parametrised build+waiter). Fresh ids: `scratch_xllm/e23_fresh_ids.txt`.

## E24 — Procrustes alignment RECOVERS hidden-state transfer — ✅ PRH holds for SE (up to a rotation)

E20–E23 showed raw hidden-state (z) transfer is ~chance while each model's OWN ridge reads its SE fine
(~0.62). Is that a *basis* mismatch (fixable by an orthogonal map → Platonic) or genuine
incompatibility? Surgical ridge-level test (`amortized_ue/procrustes_alignment.py`, CPU, additive,
reuses `linear_ceiling_probe` helpers read-only): **TBG only, never SLT; NO SE labels in the fitting.**
Fit an orthogonal Procrustes map W from **Mistral's TBG → Llama-2's TBG** on the shared **1440 train**
questions (both mean-centered), translate Mistral's TBG for the **200 held-out test** ids
(`(x−m̄)·W + l̄`), feed through **Llama-2's frozen ridge**, Spearman-score vs **Mistral's** SE.

**Result + controls (Spearman, N_test=200):**

| variant | Spearman | note |
|---|---|---|
| raw z transfer (floor) | **−0.051** | naive transfer fails (as E20–E23) |
| ctrl: mean-shift only (no rotation) | −0.051 | identical to floor → NOT an offset artifact |
| ctrl: random orthogonal rotation | +0.071 ± 0.077 | chance → NOT "any rotation works" |
| **aligned transfer (learned Procrustes)** | **+0.545** | only the geometry-aligned rotation recovers it |
| native Mistral ridge (skyline) | +0.620 | source ridge on own TBG L31 |
| Llama-2 in-dist ridge (context) | +0.585 | target ridge on own TBG L30 |
| **fraction of floor→skyline gap recovered** | **88.8%** | |

**Controls make it defensible.** Mean-shift alone does nothing (−0.051 = floor); a random orthogonal
map stays at chance (0.07); only the LEARNED, label-free alignment recovers 0.545. No leakage path — W
never sees an SE label, the ridge is frozen, train/test disjoint. Mechanically the aligned transfer is
`x_mistral·(W·β_llama2)` = a linear SE probe on Mistral built WITHOUT any Mistral SE label, so it
correctly sits below Mistral's own supervised skyline (0.620), not above.

**Reconstruction diagnostic on held-out PAIRS:** per-row cosine **0.001 → 0.399** (the rotation
genuinely aligns paired vectors), rel Frobenius recon error 1.035 → 0.927, **linear CKA 0.865**
(orthogonal-INVARIANT → before==after by construction; measures that the two TBG spaces are highly
alignable-by-rotation in the first place). Not a null.

**Both directions — the alignment is SYMMETRIC (added).** Repeated with source/target swapped
(`--source Llama-2-7b-chat --target Mistral-7B-Instruct-v0.2`): align **Llama-2 TBG → Mistral TBG**,
feed Mistral's frozen ridge, score vs Llama-2 SE.

| variant | reverse (L2→Mistral) | forward (Mistral→L2) |
|---|---|---|
| raw z transfer (floor) | −0.139 | −0.051 |
| ctrl: mean-shift only | −0.139 (=floor) | −0.051 (=floor) |
| ctrl: random orthogonal | −0.048 ± 0.08 (chance) | +0.071 ± 0.08 (chance) |
| **aligned transfer** | **+0.555** | **+0.545** |
| native ridge (skyline) | +0.585 | +0.620 |
| **gap recovered** | **95.9%** | 88.8% |

(reverse: CKA 0.849, per-row cosine 0.000 → 0.406.) So the recovery holds ~90–96% *whichever* model is
source vs target → not a one-way fluke; the two models' uncertainty geometry is genuinely the same up
to a rotation, both ways. Reverse JSON: `amortized_ue/procrustes_alignment_llama2_to_mistral.json`.

**Interpretation — reframes E20–E23.** The naive "hidden states don't transfer" was a **basis
mismatch, not incompatibility.** A label-free orthogonal rotation makes Llama-2's frozen ridge read
Mistral's SE at 0.545 — near Mistral's own ridge (0.620). **The Platonic Representation Hypothesis
holds for SE: two different-family LLMs encode semantic uncertainty in the same geometry up to a
rotation.** Cross-LLM story is now: **text transfers directly (E20–E23), and hidden states transfer
after a cheap UNSUPERVISED orthogonal alignment (E24).** Practical payoff: build an SE probe for a NEW
model with **no N-sample SE labels** for it — only paired forward passes on shared questions to fit W,
then reuse a reference model's probe. Caveats: needs paired hidden states (same questions through both
models — cheap, label-free); shown at one TBG layer / one model pair / N_test=200; the E20–E23 negative
stands WITHOUT the map. Result JSON: `amortized_ue/procrustes_alignment_result.json`.

> ⚠️ **QUALIFIED by E25 (Mechanism-A control).** Most of this "88.8% recovery" is the shared
> question-difficulty confound (target's own states predict source SE at ~0.45–0.56 with no cross-model
> geometry). The rotation's genuine model-specific increment is small but significant (+0.032 at
> N=1000). Read E25 before citing the E24 headline — the effect is real but modest, not the dramatic
> PRH triumph the raw floor→skyline framing implies.

## E25 — Mechanism-A control QUALIFIES E24: mostly shared difficulty + a small real model-specific increment

E24 reported hidden-state SE transfer "recovers 88.8%" after orthogonal alignment. Missing control:
how much of that is just **shared question-difficulty** (both models find the same questions hard),
which needs no cross-model geometry at all? **Mechanism-A control:** score Llama-2's frozen ridge on
**Llama-2's OWN TBG** states (same eval ids), Spearman vs **Mistral's** SE — pure shared-difficulty
readout, uses zero Mistral states. If aligned clearly beats it → the rotation carries Mistral-specific
uncertainty (PRH-positive); if they match → shared-difficulty only. Re-ran the whole test (W fit on the
same 1440 train pairs, no labels) evaluating on **both** the N=200 test split and the **E23 fresh
n1000** batch (both models' TBG already on disk) to shrink CIs. Additive
(`amortized_ue/procrustes_alignment.py` extended; `--fresh_num_samples`).

**Result (Mistral→Llama-2, Spearman):**

| | N=200 (test) | N=1000 (fresh) |
|---|---|---|
| raw z transfer (floor) | −0.051 | +0.037 |
| **control: shared-difficulty (Mech-A)** | **+0.451** | **+0.557** |
| aligned transfer (learned W) | +0.545 | +0.590 |
| native Mistral ridge (skyline) | +0.620 | +0.632 |
| **aligned − control** (paired bootstrap 95% CI) | **+0.095 [+0.020, +0.172]** | **+0.032 [+0.001, +0.063]** |
| separated from 0? | yes (P=1.00) | yes, barely (P=0.98) |

**Interpretation — qualifies E24.** The control is HIGH (0.45–0.56): Llama-2's own uncertainty predicts
Mistral's SE at ~0.5 with **no Mistral states at all**, so **most of E24's "88.8% recovery" is the
shared question-difficulty confound**, not model-specific geometry transfer. **But** the aligned
transfer still beats the control by a small, **statistically significant** margin (+0.032 at N=1000, CI
excludes 0) → the rotation carries a genuine but **modest** Mistral-specific uncertainty signal on top
of the large shared-difficulty base. Corrected claim: **hidden-state alignment is weakly PRH-positive**
— real model-specific transfer (~+0.03), not the dramatic geometry-alignment triumph E24's headline
implied. The n1000 batch pinned the increment to ±0.03 (vs the noisy ±0.10 at n200). *(Aside: the
denoised ridge control 0.557 slightly exceeds the raw label-correlation ceiling 0.524 (E23) because the
ridge prediction is a smoothed estimate of the shared difficulty.)* JSON:
`amortized_ue/procrustes_e25_mistral_to_llama2.json`.

## E26 — Decompose the aligned transfer: real but small & mostly redundant new info (confirms E25)

E25 left the increment (aligned − control ≈ +0.03) established but not characterised. E26 asks *is the
rotation's signal genuinely new, or redundant with shared difficulty?* Two tests on the E23 fresh n1000
batch, reusing the same E25 fit (W + Llama-2 ridge on the n2000 1440-train pairs, no SE labels in W).
Additive, CPU (`amortized_ue/procrustes_e26_decomposition.py`); touches nothing existing.

**(1) Semi-partial correlation** — regress the control prediction out of Mistral SE (OLS residuals),
then Spearman(aligned, residuals):

| | value | 95% CI (bootstrap) | verdict |
|---|---|---|---|
| semi-partial (as specified) | **+0.042** | [−0.000, +0.086], P(>0)=0.97 | **borderline** (touches 0) |
| symmetric rank-based partial Spearman | **+0.267** | — | clearly positive |

**(2) Ensemble** (control + aligned; score vs Mistral SE, fresh n1000):

| predictor | Spearman | (− control), 95% CI |
|---|---|---|
| control alone | 0.557 | — |
| aligned alone | 0.590 | +0.033 |
| ensemble (avg) | 0.582 | +0.025 [+0.016, +0.033], P=1.00 |
| **ensemble (2-input ridge)** | **0.598** | +0.041 [+0.019, +0.063], P=1.00 |

ridge meta-weights: control +0.32, aligned +2.09.

**Interpretation — confirms E25 ("weakly PRH-positive").** Both tests agree the rotation carries *some*
genuinely new signal, and that it is **small and largely redundant** with shared difficulty: (i) the
ensemble beats control significantly (+0.041) but beats *aligned-alone* by only **+0.008** (0.598 vs
0.590) → almost all of the ensemble's edge is just "aligned > control" (the E25 point); the genuine
complementarity bonus is tiny. (ii) The semi-partial/partial gap is diagnostic, not contradictory: the
FULL aligned prediction barely correlates with residualised SE (0.042, borderline) because aligned is
*dominated* by the shared component it shares with control; but the UNIQUE parts of aligned and SE
clearly correlate (partial 0.267). So a real unique signal exists, it is just a small fraction of the
total. Net: the rotation's model-specific contribution is **real but modest and mostly redundant** — no
overclaim survives, the effect does not vanish. JSON: `amortized_ue/procrustes_e26_decomposition.json`.

## E27 — Does hidden-state alignment HELP uncertainty estimation (beyond text)? Full decomposition

E24–E26 asked whether z transfers; E27 asks the *useful* question: does the aligned hidden state give a
better cross-model **uncertainty estimator** than the model-agnostic text, and how do the pieces
combine? All Mistral→Llama-2 (unless noted), fresh n1000, vs Mistral SE. Additive scripts
`procrustes_e27*.py`; the only code change is two z-free/text-map lines in `stage2/train.py` for the new
`z_resp`/`resp_only` arms.

**E27a — the aligned hidden state carries SE info BEYOND the question text (robust).** Control the
aligned-z ridge against the `q_only` TEXT prediction (3B proxy). Semi-partial(aligned, SE | text
removed) = **+0.091 [95% CI +0.046, +0.135], P=1.00** — clearly > 0; ensemble(text+aligned) beats text
by +0.057. **Robustness battery** (`procrustes_e27a_robustness.py`) — 2 directions × 2 eval sets, the
semi-partial CI **excludes 0 in all 4 cells** and **all 5 text-seeds are positive in every cell**;
20 anchor-resamples (refit W+ridge) stay positive (±0.01–0.02). So "hidden adds over text" is not a
one-config fluke. *(Caveat: part of the gain over text is that hidden states read difficulty more richly
than raw text; vs a hidden-state difficulty control the model-specific increment was the smaller E26
+0.042.)*

**E27b — proxy (all 5 arms) vs ridge on the SAME aligned `[TBG:22,SLT:15]`** (layers validated
ridge-optimal for Llama-2: 0.600/0.584). Raw → aligned Spearman: **z 0.014→0.545**, z_q 0.073→0.478,
z_q_resp 0.124→0.510, q_only 0.474, q_resp_only 0.531; **ridge 0.091→0.580**. Findings: (1) alignment
rescues the *proxy's* z-arm too (not just the ridge); (2) the **ridge beats every proxy arm** (0.580 >
best arm z 0.545) — z→SE is linear, the 3B adds nothing; (3) **adding text to the z-arm HURTS** (z_q,
z_q_resp < pure z) — z and question are redundant, so fusing dilutes.

**E27 gate + E27c/d — the response, and early vs late fusion.** Gate (`_zresp_gate`): the *response*
text is *mildly* complementary to aligned-z — combining aligned-z (0.580) + q_resp_only (0.531) beats
both significantly. **⚠️ combiner note:** the gate/AUROC scripts used a 2-input *ridge* combiner fit on
Mistral SE labels → **0.608**, so that specific number is NOT label-free. A genuinely **LABEL-FREE**
combiner (standardized average of the two predictions, no target labels) **matches it: 0.609 Spearman /
0.867 AUROC** (`procrustes_e27_labelfree_ensemble.py`) — so the labels buy ~nothing and the combined
result stands label-free. **E27c** trained the actual **`z_resp` arm** (hidden+response, no question):
aligned **0.523 ± 0.054 — BELOW pure z (0.545)**. **E27d** trained **`resp_only`** (response text, no
question): **0.455 — BELOW q_resp_only (0.531) and even q_only (0.474)**. Two clean lessons: (i) **late
fusion wins, early fusion loses** — stacking two well-built predictors (→0.609 label-free) beats forcing
z+response into one arm (z_resp 0.523) or one text arm; (ii) **the question helps in the text-only
regime** (drop it and resp_only falls to 0.455) — "q hurts" was only ever a z-arm redundancy effect.
Unified rule: *use difficulty once — from z if you have it, from the question if you don't.*

**AUROC vs the supervised baseline** (`procrustes_e27_auroc.py` + `_labelfree_ensemble.py`,
thr=best_split 0.814, 23% positive). **Labels column is exact — the ridge combiner uses target labels;
the averages do not:**

| predictor | AUROC | Spearman | Mistral SE labels? |
|---|---|---|---|
| q_only (text) | 0.828 | 0.537 | no |
| q_resp_only (text) | 0.852 | 0.587 | no |
| aligned-z ridge | 0.850 | 0.580 | no |
| avg (raw) z + q_resp | 0.862 | 0.602 | **no (LABEL-FREE)** |
| **avg (standardized) z + q_resp** | **0.867** | **0.609** | **no (LABEL-FREE)** |
| ridge combiner z + q_resp | 0.866 | 0.608 | **YES (fits weights on labels)** |
| **Mistral supervised ridge (BASELINE, same features)** | 0.863 | 0.587 | **yes** |

**On AUROC the LABEL-FREE ensemble — a standardized average, no target labels — reaches 0.867, matching
the supervised baseline (0.863)** and matching the label-fitted ridge combiner (0.866), so the labels
buy ~nothing for combining. On Spearman the label-free average (0.609) beats the same-feature baseline
(0.587) and sits just under the best-layer skyline (0.632). Like-for-like hidden-only, the supervised
ridge (0.863) edges label-free aligned-z (0.850), as expected — the ensemble's edge comes from adding
the text modality. *(Correction: an earlier version reported the ridge-combiner 0.608/0.866 as
"label-free"; that combiner uses Mistral labels. The genuinely label-free average (0.609/0.867) is used
here and lands in the same place.)* *(Aggregation note: proxy arms here are 5-seed
**prediction-averaged** (denoised), so `q_resp_only` reads 0.587 here vs the **per-seed-mean** 0.531 in
E27b/gate — same arm, two aggregations. The ensemble averages predictions, so the prediction-averaged
number is the correct input; cf. `q_only` 0.537 pred-avg vs 0.474 per-seed-mean.)*

**vs the OFFICIAL SEP baseline (matched).** The E27 "supervised ridge" above is a stacked-ridge proxy;
the actual **SEP** is single-layer LogisticRegression on binarized SE (`best_split`, same convention as
`semantic_entropy_probes/run_llama2_probe.py`). Ran the real SEP method on the E27 data (n2000 train →
fresh n1000 eval), same binarization:

| SEP (in-model, supervised, single-layer logistic) | best AUROC |
|---|---|
| saved official SEP — Llama-2, **N=400** (run 095l3ou2) | 0.726 |
| SEP method — Llama-2, **N=1000** (matched) | 0.795 (TBG L31) |
| **SEP method — Mistral, N=1000** (matched, the in-model baseline for our target) | **0.857 (SLT L15)** |

**Our label-free ensemble (0.867) is on par with — slightly above — the actual Mistral SEP (0.857)**, on
a fully matched comparison (same data/binarization/eval), and the SEP uses Mistral's labels while ours
uses none. Hidden-only, aligned-z (0.850) ≈ SEP (0.857), SEP marginally ahead (in-model supervised). Two
lessons: (i) the earlier saved-SEP number (0.726) was **N=400-underpowered** — same Llama-2 SEP reaches
0.795 at N=1000; (ii) **Mistral SE is more predictable than Llama-2's** (SEP 0.857 vs 0.795), a genuine
model difference. So the E27 AUROCs (~0.85–0.87) are NOT inflated vs SEP — they match the real SEP method
on matched data. JSON: `procrustes_e27_sep_comparison.json`.

**E27 bottom line.** The best **label-free** cross-model uncertainty estimator is a **standardized
average of the aligned-z ridge + `q_resp_only` → Spearman 0.609 / AUROC 0.867** — matching the
supervised baseline on AUROC (0.863) and recovering ~96% of the Spearman skyline (0.632), with **no
target SE labels** (needs paired anchor forward passes to fit W; the text arm needs nothing). A
label-*fitted* ridge combiner matches it (0.608/0.866) but requires target labels, so it is not
preferred. Alignment genuinely *helps* (E27a), a ridge exploits aligned z best (E27b), the response adds
a little only via late fusion (gate/c), and the question earns its keep only when there's no z (d).
Result JSONs: `procrustes_e27{a_vs_text,a_robustness,b_proxy_vs_ridge,_zresp_gate,c_zresp_arm,
d_resp_only,_auroc,_labelfree_ensemble}.json`.

## E27 (addendum) — rank-fusion variant + squad OOD shape-robustness

Adds a **rank-fusion** label-free combiner to the E27 ensemble (`procrustes_e27_rank_fusion.py`, additive):
map each predictor (aligned-z ridge, `q_resp_only`) to normalized ranks via its **empirical CDF fit on
TRAIN predictions only** (no labels), average the two. Reported next to the raw and standardized
averages; also run on **squad OOD** (built Mistral squad n1000 for this — aligned-z is Mistral→Llama-2 so
it needs Mistral squad hidden states) to test shape-robustness under distribution shift.

| fusion (all label-free) | ID trivia_qa fresh n1000 | **OOD squad n1000** |
|---|---|---|
| floor (raw z, **NO W**) | — | −0.026 / 0.491 |
| aligned-z ridge | 0.580 / 0.850 | 0.481 / 0.743 |
| q_resp_only (text) | 0.587 / 0.852 | 0.505 / 0.753 |
| avg (raw) | 0.602 / 0.862 | 0.526 / 0.764 |
| avg (standardized) | 0.609 / 0.867 | 0.538 / 0.770 |
| **RANK FUSION (empirical-CDF avg)** | 0.608 / 0.866 | 0.541 / 0.771 |

(Spearman / AUROC; ID `best_split` 0.814, OOD 1.233 — squad SE is higher, mean_acc 0.228 vs trivia 0.649,
a real shift.)

**Findings.** (1) **Rank fusion TIES the standardized average on OOD** — paired bootstrap (1000
resamples, E25/E26 convention) of Δ(rank fusion − std-avg): **Spearman +0.002 [−0.000, +0.005], AUROC
+0.001 [−0.001, +0.003]**, both CIs include 0. So the earlier "best OOD combiner" read was noise; rank
fusion is a *valid, tied* label-free combiner, not a better one. (Its CDF normaliser is at least as
shape-robust as mean/std — no ID cost either, 0.608/0.866 tie.) (2) **Floor control confirms the
trivia-fit W transfers cross-domain**: raw (unaligned) Mistral squad states through the same Llama-2
ridge (NO W) are at **chance (−0.026 / 0.491)**, while applying the *trivia-fit* Procrustes W lifts it to
**0.481 / 0.743** — so the alignment learned on trivia still aligns Mistral→Llama-2 geometry on the
shifted squad domain; **cross-domain transfer of W is confirmed.** (3) The **ensemble gain is robust to
shift** — every label-free fusion still beats both components OOD (~0.54 vs 0.48/0.51). Data: built
`Mistral-7B-Instruct-v0.2_squad_n1000` (on /vol/bitbucket + W&B). JSON:
`procrustes_e27_rank_fusion.json`.

---

## E28 — Add a 4th target LLM: DeepSeek-LLM-7B-Chat (Stage-1 dataset built) — ✅ pipeline clean, layers re-picked

**Goal (infrastructure, no downstream analysis yet):** onboard a 4th cross-LLM target so the
alignment / transfer line (E24–E27) can later be extended beyond the Llama-2 ↔ Mistral pair.
DeepSeek-LLM-7B-Chat (`deepseek-ai/deepseek-llm-7b-chat`) is a plain `LlamaForCausalLM` but a
distinct pre-training lineage — **30 layers, 4096-dim** (matches the 4096 of Llama-2/Mistral, so
z-arms are dimension-compatible for future alignment).

**Code (additive, blocks-execution only; no SE/clustering/probe logic touched):** (1) a new load
branch in `huggingface_models.py` (`elif 'deepseek' in model_name.lower()` → `deepseek-ai/{name}`,
same minimal AutoTokenizer/AutoModelForCausalLM pattern as the Mistral branch); (2) one entry added
to the `init_model` dispatch whitelist in `utils/utils.py` (`or 'deepseek' in mn.lower()`) so the
new name reaches the load branch. Loads in **`se_probes_llama3`** (transformers 4.44.2). No decode
quirk (unlike Llama-3's " ?" issue): the `startswith` fast-path holds; `pad_token_id` auto-sets to
eos `100001`.

**Smoke (3 records, `smoke_deepseek.sh`):** answers extract cleanly and stop correctly — e.g.
"Donald Trump" (acc 1.0), "moldova"/"vauxhall" — no run-on, no leaked special tokens; hidden states
`(31, 1, 4096)` = embedding + 30 layers; SE labels non-degenerate (CAE 0.33 / 0.67, n_clusters 2).
mean_acc 0.333, mean_CAE 0.676.

**Stage-1 build (E23 fresh ids, `stage1.py --only_ids scratch_xllm/e23_fresh_ids.txt`, resumable):**
**`deepseek-llm-7b-chat_trivia_qa_n1000_full`** — 1000/1000 records, **mean_acc 0.527**,
**mean_CAE 0.8035**, elapsed ~70 min (GPU 1). On /vol/bitbucket + W&B (artifact
`stage1_records_deepseek-llm-7b-chat_trivia_qa_n1000`, run `c6ijifxe`, 974 MB). Same 1000 held-out
trivia_qa ids as the E23 Llama-2/Mistral fresh batches → directly comparable, zero overlap with any
earlier build. (Context: mean_acc 0.527 sits between Llama-2's ~0.65 and Mistral's fresh-batch level;
mean_CAE 0.80 is a moderately higher-entropy target than Llama-2's ~0.59.)

**z-layer re-pick (`linear_ceiling_probe.py`, ridge → continuous SE, ID test Spearman; no OOD — no
DeepSeek squad build yet; `deepseek_layer_pick.json`):** Llama-2's TBG:22/SLT:15 do **not** carry
over (30-layer model, signal sits deeper). Per-position ID-test-Spearman argmax (the Mistral/Llama-2
convention):

| position | chosen layer | ID-test Spearman | notes |
|---|---|---|---|
| **TBG** | **28** | 0.670 | broad plateau L24–28 |
| **SLT** | **16** | 0.680 | sharp peak L15–17; global best |

→ downstream z-input for DeepSeek is **`--z_inputs TBG:28,SLT:16`**. (Ridge ID ceiling ≈ 0.68, in
line with Llama-2/Mistral ~0.62–0.64.)

**No downstream analysis run yet** (no proxy training, no alignment/transfer eval) — this entry only
establishes the target + dataset + chosen layers. Next candidates: replicate the E24/E25 Procrustes
controls + the E27 SEP comparison with DeepSeek as a 3rd alignment target.

---

## E29 — Run the E24–E27 alignment chain on DeepSeek; four-model master table — ✅ recovery generalises to a 3rd family; label-free ensemble ≈/> supervised SEP; increment underpowered at n1000

**Goal:** extend the Llama-2↔Mistral alignment line (E24–E27) to the new DeepSeek target and assemble a
four-model picture, to ask whether the geometric recovery + the model-specific increment track FAMILY
relatedness. Prediction-reuse + CPU (plus frozen-proxy GPU inference for `q_resp_only`); no new Stage-1
builds. All of Llama-2/Mistral/DeepSeek share the **identical E23 fresh 1000 ids** (verified L∩M∩D=1000).

**Tooling (additive; no training logic touched):** parametrised `procrustes_alignment.py` with
`--position/--source_layer/--target_layer` (defaults reproduce the exact TBG-auto E24/E25). New
`procrustes_e29_ensemble_sep.py` (label-free ensemble vs supervised SEP, within-n1000, reusing
`arm_preds`/`ecdf`/`boot_delta`/`fit_probe`/`best_split`/`binarize_entropy`) and
`procrustes_e29_master_table.py` (read-only assembler). DeepSeek uses its **re-picked SLT:16**; ensemble
uses the Llama-2 **reference** layers TBG:22/SLT:15 (the reference ridge + REFERENCE proxy).

**Data-regime honesty:** DeepSeek has ONLY n1000 → the alignment is fit on its 720-row train and
evaluated on the **100-row test split** (vs the official Mistral run's n2000-fit / 1000-row-eval). So
DeepSeek's CIs are inherently ~3× wider. **Llama-3 cannot enter the alignment table at all** — only
n200 exists, on the *E20* ids (not the E23 split); 144 train pairs can't fit a stable 4096-dim W. Its
E20 result stands (text transfers, raw z is chance) but the Procrustes line was never run for it. This
is a scope boundary, reported not buried.

**(1) E24 recovery — the hidden-state geometry aligns for DeepSeek too (both directions):**
predict-DeepSeek-SE (Llama-2 ridge ← aligned DeepSeek z, SLT): floor −0.136 → aligned **+0.572** →
skyline 0.680 = **86.8% recovery**; predict-Llama-2-SE (DeepSeek ridge ← aligned Llama-2 z): floor
+0.076 → aligned **+0.532** → skyline 0.646 = **80.2%**. In line with Mistral (official 92.8%). So raw
z fails but an unsupervised orthogonal W recovers most of the gap — now on a **3rd model family**.

**(2) E25 increment — underpowered at N=100, NOT a family signal.** All within-n1000 (N=100)
`aligned − control` CIs include 0: DeepSeek→L2 −0.049 [−0.201,+0.087]; L2→DeepSeek +0.023
[−0.125,+0.170]; **Mistral calibration** +0.066 [−0.017,+0.155]. **The Mistral calibration is the
key control:** its KNOWN-significant increment (+0.032 [+0.001,+0.063] at the official N=1000) *also*
loses significance at N=100 → the DeepSeek non-significance is a **power limitation of n1000-only data,
not weaker transfer**. No family-relatedness claim on the increment is possible without DeepSeek n2000
(a GPU build, out of scope). Shared question-difficulty (the control) is large for all pairs (0.45–0.62).

**(3) E29 label-free ensemble vs supervised SEP (within-n1000, N=100) — label-free never loses:**
rank-fusion(aligned-z + q_resp_only), no target SE labels:

| target (ref=Llama-2) | ensemble AUROC / ρ | supervised SEP AUROC | Δ(ens−SEP) AUROC [95% CI] |
|---|---|---|---|
| **DeepSeek** | **0.916 / 0.758** | 0.809 (SLT L17) | **+0.108 [+0.038, +0.186] — excludes 0 (BEATS)** |
| **Mistral** | **0.925 / 0.654** | 0.878 (TBG L30) | +0.047 [−0.024, +0.122] — includes 0 (on par) |

Both different-family ensembles land **~0.92 AUROC**, strikingly consistent; the ensemble ≥ SEP every
time. DeepSeek's ensemble *beats* its SEP because DeepSeek's own SEP is weaker (0.809 < Mistral 0.878),
not because the ensemble is better there. Consistent with E27's "label-free ≈ supervised SEP".

**Four-model picture (`procrustes_e29_master_table.json`):** the alignment recovery + the label-free
ensemble generalise to a 3rd family (DeepSeek); the ensemble matches or beats the supervised SEP with no
target labels; the model-specific increment could not be resolved at n1000 (power), and the same-family
Llama-3 comparison is blocked by data. JSONs: `procrustes_e29_{deepseek_to_llama2,llama2_to_deepseek,
mistral_to_llama2_n1000,ensemble_sep_*,master_table}.json`.

---

## E30 — FULL-POWER four-model alignment table (built DeepSeek + Llama-3 n2000) — ✅ label-free ensemble > supervised SEP on all 3 targets; alignability tracks CKA, not family; DeepSeek is a low-CKA outlier

**Goal:** remove E29's two caveats (DeepSeek's N=100 power; Llama-3 not computable) by building the two
missing **n2000** datasets on the shared seed-10 selection, then redo the alignment + ensemble chain at
full power. Built `deepseek-llm-7b-chat_trivia_qa_n2000` (mean_acc 0.523 / CAE 0.794) and
`Meta-Llama-3-8B-Instruct_trivia_qa_n2000` (mean_acc 0.655 / CAE 0.486); both on /vol/bitbucket + W&B,
verified by fetch. (Infra: Llama-3's build OOM'd once when a co-tenant grabbed GPU-1 slack mid-run →
added **GPU memory fencing** `gpu_reserve.py` + `build_n2000_waiter.sh`, which resumed it cleanly; 2
records corrupted by the OOM's partial writes were detected by a torch-load scan and regenerated.)
New tooling: `procrustes_e30_ensemble_sep.py` (fit-on-one-set / eval-on-another) + `procrustes_e30_master_table.py`.

**Regime:** DeepSeek & Mistral fit on n2000 → eval the disjoint **fresh n1000 (N=1000)**; Llama-3 fits+evals
its n2000 (**N=200** test split — it has no fresh n1000). Reference = Llama-2 (ensemble at TBG:22/SLT:15;
DeepSeek alignment at its SLT:16, Mistral/Llama-3 at TBG).

**(1) Alignment recovery + CKA + E25 increment (master table):**

| pair | N | recovery | **CKA** | increment [95% CI] | sig |
|---|---|---|---|---|---|
| Mistral→Llama-2 | 1000 | 92.8% | 0.795 | **+0.032 [+0.001,+0.063]** | **YES** |
| DeepSeek→Llama-2 | 1000 | 94.7% | **0.248** | +0.009 [−0.028,+0.044] | no |
| Llama-2→DeepSeek | 1000 | 94.1% | **0.273** | +0.006 [−0.032,+0.044] | no |
| Llama-3→Llama-2 | 200 | 91.8% | **0.871** | +0.069 [−0.004,+0.143] | no |
| Llama-2→Llama-3 | 200 | 94.6% | **0.872** | −0.023 [−0.107,+0.056] | no |

**Recovery is uniformly high (~92–95%) for ALL pairs** — it's dominated by shared question-difficulty,
so it does NOT discriminate. **The discriminator is CKA (rotational alignability): Llama-3 (same family)
0.87 > Mistral (different) 0.80 ≫ DeepSeek (different) 0.25.** Family is at best a *weak* predictor
(Llama-3 highest, but Mistral close behind); **DeepSeek is a striking low-CKA outlier despite matching
4096 dims.** The genuine **model-specific increment tracks CKA, not family**: at N=1000 Mistral is +0.032
(significant) but **DeepSeek is ~0 with a tight CI** (+0.008, [−0.03,+0.04]) — so E29's DeepSeek "null"
was NOT merely power; at full power DeepSeek genuinely has almost **no model-specific geometric SE
component** (pure shared-difficulty). Llama-3 (N=200) is noisy (+0.069 / −0.023) — its increment can't be
resolved without a fresh n1000, the one residual data limit.

**(2) Label-free ensemble vs supervised SEP — label-free WINS on all three targets (full power):**

| target | ensemble AUROC / ρ | supervised SEP AUROC | Δ(ens−SEP) AUROC [95% CI] |
|---|---|---|---|
| Mistral (N=1000) | 0.866 / 0.608 | 0.832 | **+0.035 [+0.006,+0.063] — beats** |
| DeepSeek (N=1000) | 0.869 / 0.711 | 0.805 | **+0.065 [+0.041,+0.088] — beats** |
| Llama-3 (N=200) | 0.892 / 0.672 | 0.839 | +0.054 [−0.003,+0.115] (ρ beats, AUROC on par) |

**The label-free ensemble (aligned-z + `q_resp_only`, no target SE labels) matches or beats the target's
OWN supervised SEP for every model** — the thesis result, now at full power on 3 targets + the reference.
Mechanistically robust even for DeepSeek, whose hidden geometry barely transfers (CKA 0.25): the text arm
carries it and aligned-z still adds via shared difficulty. *(SEP-AUROC fix: the within-split SEP AUROC was
first scored over all rows incl. train — 0.977; corrected to eval-only, 0.839; deltas were always vs the
correct test-only SEP.)* JSONs: `procrustes_e30_*.json`; table `procrustes_e30_master_table.json`.

---

## E31 — Correctness-based evaluation: do the SE predictors actually detect *wrong answers*? — ✅ additive; SE-fidelity ≠ correctness, but rankings mostly agree

*(The task brief called this "E30"; **E30 was already taken** by the four-model alignment table above, so
this is logged as **E31**.)*

**Motivation.** Every headline number in this repo (E0–E30) is scored against the **semantic-entropy
label**. Nothing was scored against whether the target LLM's canonical answer was actually **correct**.
Stage-1 records already store `canonical.accuracy`, so this needs no GPU-generation and no retraining —
only the hidden states / SE labels / existing REFERENCE proxy checkpoints already on disk. New script
**`amortized_ue/correctness_eval.py`** (strictly additive: modifies no existing script, training logic,
or prediction artifact). Detection label **`incorrect = 1`**.

**Accuracy distribution (reported first, per the brief).** `canonical.accuracy` is **already binary
{0,1}** on every Stage-1 set (the stored `squad` metric is thresholded at generation, not a continuous
F1), so the 0.5 binarisation is a **no-op**. Incorrect-rate per eval set: Mistral 0.351, DeepSeek 0.473,
Llama-3 0.315, Llama-2 0.391.

**Regime mirrors E30 exactly** (reference = Llama-2 at TBG:22/SLT:15): Mistral/DeepSeek/Llama-2 fit n2000
→ eval **fresh n1000**; Llama-3 fit n2000 → eval **test split (N=200)**. Predictors scored against
`incorrect` **and** against binarised SE, side by side. Rejection metrics (PRR + accuracy@coverage) and a
paired bootstrap (B=10000, shared resample indices) vs the two SEP baselines are reported per predictor.

### Correctness-AUROC (detect `incorrect`) with SE-AUROC alongside

| predictor (higher ⇒ more uncertain) | Mistral inc / SE | DeepSeek inc / SE | Llama-3 inc / SE | Llama-2 inc / SE |
|---|---|---|---|---|
| **true semantic entropy** (10-sample, upper bound) | **0.747** / 1.000 | **0.795** / 1.000 | **0.775** / 1.000 | **0.760** / 1.000 |
| SEP, single best layer (val-sel, = E30) | 0.705 / 0.832 | 0.716 / 0.805 | 0.720 / 0.839 | 0.661 / 0.746 |
| SEP, 5-layer concat (L28-32; arXiv Tbl 4) | 0.714 / 0.840 | 0.721 / 0.812 | 0.729 / 0.848 | 0.681 / 0.783 |
| aligned-z ridge (label-free) | 0.720 / 0.850 | 0.758 / 0.849 | 0.705 / 0.877 | 0.706 / 0.816¹ |
| q_resp_only (label-free) | 0.725 / 0.852 | 0.764 / 0.857 | 0.739 / 0.874 | 0.715 / 0.835 |
| **rank-fusion ensemble** (label-free) | 0.731 / 0.866 | 0.772 / 0.869 | 0.730 / 0.892 | 0.719 / 0.840¹ |
| random control | 0.498 / 0.522 | 0.529 / 0.515 | 0.511 / 0.463 | 0.513 / 0.494 |

(inc = AUROC vs `incorrect`; SE = AUROC vs binarised SE. ¹Llama-2 is the reference, so its aligned-z /
rank-fusion use Llama-2's own SE to fit the ridge — **not** label-free there; flagged in the JSON.
AUPRC, PRR and accuracy@{1.0,0.9,0.75,0.5} for every cell are in `correctness_eval_<model>.json`.)

### Findings

1. **SE-fidelity is NOT correctness.** Every method is a **much weaker correctness detector than SE
   predictor** — AUROC drops ≈0.10–0.15 from the SE target to the correctness target (rank-fusion:
   Mistral 0.866→0.731, Llama-2 0.840→0.719). SE-based UE ranks *its own label* far better than it ranks
   answer correctness. This is a real caveat for the whole line: the ~0.85 SE-AUROCs overstate how well
   these estimators flag wrong answers (~0.70–0.77).

2. **Sampling beats amortization for correctness.** The **true 10-sample SE is the best correctness
   detector on all four targets** (0.747–0.795), above every one-forward-pass proxy — significantly over
   the single-layer SEP on Mistral/DeepSeek/Llama-2 (paired-bootstrap Δ excludes 0: +0.042 / +0.080 /
   +0.099), not on Llama-3 (N=200, CI includes 0). Amortizing to one pass has a measurable
   correctness-detection cost that the SE-target numbers hide.

3. **The label-free ensemble still ≥ supervised SEP on the correctness target.** rank-fusion − SEP(single)
   Δ AUROC_incorrect: Llama-2 **+0.058 [+0.032,+0.083]**, DeepSeek **+0.057 [+0.031,+0.082]** (both exclude
   0); Mistral +0.026 [−0.001,+0.054] and Llama-3 +0.010 [−0.058,+0.077] (include 0). So E30's thesis
   (label-free ≥ supervised SEP, no target labels) is **not an artifact of scoring against SE** — it holds
   when re-targeted to actual correctness.

4. **Method ORDERING: does the SE-AUROC ranking match the correctness-AUROC ranking? MOSTLY, NOT ALWAYS.**
   The two orderings **MATCH on 3/4 targets** (Mistral, DeepSeek, Llama-2 — identical permutation
   true_SE > rank_fusion > q_resp > aligned_z > sep_5layer > sep_single > random). They **DIFFER on
   Llama-3** (N=200): there **aligned-z ridge is 3rd by SE-fidelity (0.877) but next-to-last by correctness
   (0.705, below even single-layer SEP)** — the aligned hidden state tracks Llama-3's SE well yet its
   correctness poorly. **Stated plainly, not smoothed:** better SE fidelity does not guarantee better
   correctness detection, and on the smallest split the ranking visibly reorders.

**SEP reproduction (verification).** The single-layer SEP AUROC-vs-SE **reproduces the committed E30
numbers exactly** (Mistral 0.832, DeepSeek 0.805, Llama-3 0.839; leak-free val-selected = E30
`sep_fit_eval`). The older ad-hoc `procrustes_e27_sep_comparison.json` values (Llama-2 0.795 / Mistral
0.857, "best test layer") **do not reproduce under a leak-free split**: a best-layer-on-eval selection
gives Mistral **0.834** / Llama-2 **0.785** (~0.02 below the 0.857/0.795 in that hand-made JSON, whose
exact selection I could not recover). Reported honestly rather than reconciled away. **id-set check:** the
ids used for accuracy and for every predictor are identical per target (asserted in-script + re-verified).

**Artifacts:** `amortized_ue/correctness_eval.py`, `correctness_eval_{Mistral-7B-Instruct-v0.2,
deepseek-llm-7b-chat,Meta-Llama-3-8B-Instruct,Llama-2-7b-chat}.json`, `correctness_eval_master.json`.

---

## E32 — Correctness eval, qualitative follow-ups: label noise, confusion matrix / genuine FNs, model-specific signal — ✅ exploratory

Follow-on to E31 (which was aggregate AUROC). Three question-level analyses on the trivia_qa correctness
target. **All exploratory** (run from throwaway scratchpad scripts, since deleted; methods below are
sufficient to reproduce). Detector for the confusion/model-specific parts = **aligned-z ridge** (CPU, label-free)
used as the stand-in for the full rank-fusion proxy — the proxy (`q_resp_only` via `arm_preds`) run was
abandoned after crawling ~37 min on a degraded `/vol/bitbucket` NFS; aligned-z's AUROC is within ~0.01 of the
ensemble (0.706 vs 0.719 Llama-2; 0.720 vs 0.731 Mistral) so the qualitative buckets are unchanged. The LLM
judge used throughout = **`NousResearch/Meta-Llama-3-8B-Instruct`** (ungated mirror; `meta-llama/Llama-3.1-8B`
is gated for this acct), greedy decoding, YES/NO grading of (question, gold aliases, model answer).

### A. Label-noise quantification (Llama-2, trivia_qa n1000; 391 strict-wrong rows)
`canonical.accuracy` is **already binary {0,1}** (exact-match/SQuAD-F1≥0.5 thresholded at generation). Many
"wrong" answers are actually correct (plural/spelling/synonym). Two matchers on the strict-wrong rows:
- **rule-based** (SQuAD-normalise + containment + Snowball-stemmed token-F1≥0.5 + difflib ratio≥0.85): **38/391
  flip → 3.8%**, a **hard, 100%-verified floor**.
- **LLM judge** (Meta-Llama-3-8B): **173/391 flip → 17.3%**. All 38 rule-flips ⊆ the 173 judge-flips.
- **Sanity check of the judge** (random 80 rule-negative rows re-judged): 26 flipped; on eyeball ~10 clearly
  correct, ~5 borderline, **~11 genuinely WRONG** → the judge's **exclusive flips are only ~50% precise** (it
  rubber-stamps e.g. *oranges=shoes*, *Holiday Inn=Motel 6*). Its **"NO" (wrong) verdicts are reliable**; its
  "YES" flips are not. **⇒ 17.3% is inflated.**
- **Best estimate ≈ 10% label noise (bracket 3.8%–17.3%).** Llama-2 exact-match accuracy 0.609 → ~0.71 once real
  labelling artefacts are removed (naïve judge-corrected 0.78 is too high). **Implication: the E31
  correctness-AUROCs are a mild *under*estimate** of the detectors' true skill.

### B. Confusion matrix + GENUINE false negatives (aligned-z detector, Youden's-J threshold)
positive = "flagged likely WRONG". **genuine FN = an FN row still wrong after a lenient re-check (rules ∪ judge)**;
FN rows a lenient check flips to correct were label noise, not real misses. (Judge "NO" reliable ⇒ genuine-FN list
is high-precision, slight under-count; e.g. *pol pot* vs gold *saloth sar* is a mislabel the judge kept, so counts
are ~1–2 high.)

| target | detector | Youden J | TP | FP | FN | TN | FN that are LABEL NOISE | **GENUINE misses** |
|--------|----------|----------|----|----|----|----|-------------------------|--------------------|
| **Llama-2** | aligned-z (self ridge) | 0.487 | 293 | 260 | 98 | 349 | 63 (64%) | **35 (36%)** |
| **Mistral** | aligned-z (Procrustes→Llama-2, label-free) | 0.548 | 274 | 278 | 77 | 371 | 63 (82%) | **14 (18%)** |

**Finding: the "missed error" bucket is dominated by exact-match label noise on both models** (64% / 82%), so the
detector's true miss rate is ~⅓ (Llama-2) / ~⅕ (Mistral) of the raw FN count. Genuine misses are real
hallucinations of a *different entity* (Llama-2: brazil≠colombia, Zeus≠atlas, dove≠raven; Mistral: ferrari≠fiat,
van gogh≠rembrandt, manhattan≠queens). FP bucket = correct answers to *obscure* questions (detector reads
difficulty and hedges even when right). The label-noise pattern is model-independent → it's a property of the
**grader**, not the model.

### C. Is the hidden state MODEL-SPECIFIC? (divergent-correctness test)
Isolate model-specific signal by holding difficulty constant: on the **held-out test split (n=200)** take the
**42 "divergent" questions** where exactly one of {Llama-2, Mistral} is correct. A per-model uncertainty score =
each model's OWN ridge (fit on train) reading its OWN hidden state at its picked layers (Llama-2 TBG22/SLT15,
Mistral TBG31/SLT20) → predicted SE. Test: does the score of the *wrong* model exceed the *right* model's?

| ranker of "which model failed" | correct rate | note |
|--------------------------------|--------------|------|
| question-only / difficulty | **50.0%** | null — identical score per model by construction |
| **hidden-state reader (per model)** | **54.8%** | **> null ⇒ a real, weak model-specific signal** |
| true 10-sample SE (upper bound) | 61.9% | how model-specific the signal can be |

**Finding: the hidden state carries a genuine but SMALL model-specific increment** (54.8% vs 50% null, vs 61.9%
ceiling) — the individual-question view of the E25/E26 aggregate ("mostly shared difficulty + a small real
model-specific increment"). Clean examples where the same question yields different per-model z correctly pointing
at the failing model: *"Caledonian Brewery city?"* Llama-2 Inverness✗ z=1.02 / Mistral edinburgh✓ z=0.32;
*"Film of White Christmas?"* Llama-2 holiday-inn✓ z=0.58 / Mistral hollywood-cafeteria✗ z=0.97.
**Caveats: underpowered** (n=42, 23/42 → 95% CI ~[40%,70%], includes 50% — suggestive, not significant), and the
**divergent set is label-noise-contaminated** (some "disagreements" are grading artefacts, e.g. Denali, Carson City).
Firming-up plan (not yet run): expand to val+test (~560 → ~3× divergent), lenient-filter the divergent set, add a
bootstrap CI.

**Bottom line (E32):** the E31 correctness numbers are a mild under-estimate (~10% label noise); the detectors'
genuine misses are far fewer than the raw FN count (grading artefacts dominate); and the hidden state does encode a
small, real, model-specific uncertainty signal on top of shared question difficulty — but proving its significance
needs the (cheap, CPU) firming-up run above.

---

## Where we stand (2026-08-12)

**Cross-LLM transfer characterised end-to-end (E20–E27).** Text transfers directly; **raw** hidden
states do not, but they transfer after a **label-free Procrustes alignment** (mostly shared difficulty +
a small genuine model-specific increment). The best **label-free** cross-model uncertainty estimator is
**on par with the supervised SEP baseline** on AUROC, using **no target SE labels**, and the alignment
map even **transfers across domains** (trivia→squad).

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
4. **⭐ Cross-LLM transfer (E20–E23) — the thesis: two-family, directionally symmetric, and replicated
   at scale:** the frozen Llama-2 proxy on **Llama-3-8B** (same family) and **Mistral-7B** (different
   family), the reverse **Mistral proxy → Llama-2** (E22), and a **fresh 1000-question held-out batch**
   both directions (E23) — **hidden-state transfer FAILS every time** (z ≈ chance: 0.056 / 0.044 /
   −0.002 / 0.014 / 0.031; despite matching 4096 dims → naive/raw z transfer fails), **text transfer
   SUCCEEDS every time** (q_only 76–90% of ceiling, q_resp_only ~full). Shared cross-model difficulty
   ceiling ≈ 0.5 (0.505 / 0.540 / 0.524, symmetric). E23 adds: in-distribution z stays high on *fresh
   questions* (0.56–0.63) → it's the model swap, not question novelty, that kills z. The phenomenon is a
   property of the model *pair*, not direction. *(Superseded framing: the raw z-failure is a basis
   mismatch, fixable — see conclusion 5 / E24.)*

5. **Hidden-state alignment is WEAKLY PRH-positive (E24 + E25 control) — mostly shared difficulty, a
   small real increment.** A label-free orthogonal Procrustes map (source TBG → target TBG, no SE
   labels) makes the target's frozen ridge read the source's SE at ~0.55–0.59 (E24: "88.8% of the
   floor→skyline gap"), symmetric both directions (95.9% reverse). **But the Mechanism-A control (E25)
   qualifies this:** the target's OWN states already predict the source's SE at **0.45–0.56** (shared
   question-difficulty, no cross-model geometry), so **most of the apparent recovery is that confound,
   not model-specific transfer.** The rotation's genuine model-specific increment (aligned − control) is
   small but **significant**: +0.032 at N=1000 (95% CI [+0.001, +0.063]). So: two LLMs' uncertainty is
   largely a shared, model-agnostic "difficulty" signal (which the TEXT arms already capture); alignment
   adds a modest genuine model-specific component. The strong E24 "PRH holds / hidden states transfer"
   headline is **tempered** — real but small.

6. **⭐ Alignment DOES help uncertainty estimation — a label-free estimator on par with the supervised
   baseline (E27).** The aligned hidden state carries SE info beyond the question text (E27a semi-partial
   +0.091, robust across directions/eval-sets/seeds/anchor-resamples). Best **label-free** cross-model
   recipe: **standardized average of the aligned-z ridge + `q_resp_only` → Spearman 0.609 / AUROC 0.867**
   — **on par with the actual Mistral SEP baseline** (single-layer logistic, matched data: 0.857 AUROC;
   the supervised ridge proxy 0.863), recovering ~96% of the Spearman skyline (0.632), with **no target
   SE labels**. *(The saved official SEP 0.726 was N=400-underpowered — the same Llama-2 SEP hits 0.795 at
   N=1000; Mistral SE is more SEP-predictable than Llama-2's.)* *(A label-fitted ridge combiner matches it, 0.608/0.866,
   but uses target labels to set the weights — so the average, not the ridge combiner, is the label-free
   result; an earlier note mislabeled the ridge combiner as label-free.)* Mechanistic lessons: a linear
   ridge beats the 3B proxy on aligned z (z→SE is linear); **late fusion beats early fusion** (stacking >
   a fused `z_resp` arm 0.523 > forcing text into z); and the question helps only when there's no z
   (`resp_only` 0.455 < 0.531). Needs paired anchor forward passes to fit W (label-free, not sample-free);
   the text arm needs nothing.

7. **⭐ Label-free fusions + OOD robustness + cross-domain W (E27 rank-fusion addendum).** Three
   label-free combiners of aligned-z + `q_resp_only` — raw average, standardized average, and
   **rank fusion** (empirical-CDF rank average, CDF fit on train predictions only) — all land ~0.61 /
   0.867 on ID. **Rank fusion TIES the standardized average** (paired bootstrap Δ: Spearman +0.002
   [−0.000, +0.005], AUROC +0.001 [−0.001, +0.003], both CIs include 0 — an earlier "best OOD" read was
   noise; it's a valid *tied* combiner, no ID cost). **Under a real trivia→squad shift** (best_split
   0.814→1.233, mean_acc 0.649→0.228) everything drops ~0.10 Spearman but the **ensemble gain survives**
   — every fusion still beats both components OOD (~0.54 vs 0.48/0.51). **Floor control confirms the
   trivia-fit Procrustes W transfers CROSS-DOMAIN:** raw (unaligned) Mistral-squad states through the
   Llama-2 ridge are at chance (−0.026 / 0.491 AUROC); the *trivia-fit* W lifts that to 0.481 / 0.743 on
   squad — the alignment learned on one dataset still aligns the geometry on a shifted one. (Built
   `Mistral-…_squad_n1000` for this OOD test.)

**Next (Procrustes line E24–E27 is DONE):** a 3rd-family alignment (build Llama-3 n2000 → replicate
E24/E25 controls + E27 SEP comparison on Llama-3); an **anchor-count efficiency sweep** for W (how few
paired anchors suffice — quantifies the only label-free cost); multi-target / leave-one-out proxy
training; and compile `amortized_ue/RESULTS.md`. **Consolidated to-do list: `amortized_ue/CLAUDE.md`.**
