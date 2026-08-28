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

## E33 — Is `z_aligned` worth it GIVEN the text proxy `q_r_proxy`? — ✅ no: marginal on SE, negative on correctness, CKA-independent

**Motivation.** Every "alignment helps" headline (E27/E29/E30) compares the label-free ensemble
(aligned-z + `q_resp_only`) against the **supervised SEP**. But `q_resp_only` — the model-agnostic
**text** arm, trained once on the Llama-2 reference — needs **zero** target-side fitting and **zero**
target sampling, whereas the aligned-z arm needs a per-target anchor set + a Procrustes **W**. So the
sharp question is not "ensemble vs SEP" but **"ensemble vs `q_resp_only`-ALONE"**: does the
hidden-state arm earn its per-target cost? And the "model-specific increment" (E25/E26) is measured
against a shared-**difficulty** control *inside the z pathway* — but the text arm already reads
difficulty, so that increment may be **redundant** with text.

**New data built.** `Meta-Llama-3-8B-Instruct_trivia_qa_n1000_full` on the E23 fresh shared ids
(mean_acc 0.651 / CAE 0.466 / incorrect-rate 0.349; integrity-scanned, 0/1000 corrupt) — the fresh
disjoint eval Llama-3 lacked in E30, so all three targets now evaluate at **N=1000**. Built with the
new **`build_e23_fresh_fenced.sh`** (E23 fresh-ids recipe + `gpu_reserve` fencing from
`build_n2000_waiter.sh`, so a co-tenant can't OOM the run mid-flight).

**(1) SE-fidelity: `ensemble − q_resp_only-ALONE`** (`procrustes_e33_ens_vs_qresp.py`; reference
Llama-2 TBG:22/SLT:15; fit n2000 → eval fresh n1000; 1000-resample paired bootstrap; rank-fusion):

| target | CKA (E30) | model-specific increment (E30) | q_resp AUROC | **Δ AUROC (ens − q_resp)** | **Δ Spearman** |
|---|---|---|---|---|---|
| DeepSeek | 0.25 | +0.008 (n.s.) | 0.857 | **+0.012 [+0.003, +0.023]** | +0.029 [+0.014, +0.045] |
| Mistral | 0.80 | +0.032 (sig) | 0.852 | **+0.014 [+0.002, +0.026]** | +0.022 [+0.005, +0.041] |
| Llama-3 | 0.87 | +0.069 (N=200 n.s.) | 0.827 | **+0.018 [+0.006, +0.029]** | +0.035 [+0.016, +0.052] |

All Δ CIs exclude 0 — z_aligned adds *real* SE signal over text — but the magnitude is **small and
essentially flat across CKA** (0.012→0.018 over CKA 0.25→0.87, overlapping CIs, and non-monotonic on
Spearman where DeepSeek +0.029 > Mistral +0.022). **The "model-specific increment" does NOT translate
into a proportional gain over the text arm** because that arm already carries the shared-difficulty
signal. *(Bonus: the fresh-n1000 Llama-3 `q_resp` AUROC is 0.827, vs the E30 within-set N=200 split's
optimistic 0.874 — the disjoint fresh eval is the honest number.)*

**(2) Correctness: does z_aligned catch WRONG-but-fluent answers text misses?** All three targets at
**fresh N=1000** (Llama-3 now on the new n1000, no longer the E31 N=200 within-set split), 10k-resample
paired bootstrap of the *incorrect*-label AUROC deltas (`correctness_e33_ens_vs_qresp.py`; reproduces
E31's Mistral/DeepSeek point estimates exactly):

| target | z_aligned AUROC_inc | q_resp AUROC_inc | ensemble | **Δ(z_aligned − q_resp) [95% CI]** | Δ(ensemble − q_resp) [95% CI] |
|---|---|---|---|---|---|
| Mistral | 0.720 | 0.725 | 0.731 | −0.005 [−0.028, +0.017] (incl 0) | +0.006 [−0.006, +0.018] (incl 0) |
| DeepSeek | 0.758 | 0.764 | 0.772 | −0.005 [−0.025, +0.015] (incl 0) | +0.008 [−0.003, +0.020] (incl 0) |
| Llama-3 | 0.701 | 0.697 | 0.709 | **+0.003** [−0.020, +0.026] (incl 0) | +0.012 [−0.000, +0.025] (incl 0) |

**Every CI includes 0** → on wrong-answer detection, `z_aligned` is **statistically indistinguishable
from the text proxy** (point deltas −0.005 / −0.005 / +0.003), and the ensemble's positive trend over
text (+0.006 / +0.008 / +0.012) never reaches significance either. **Correction of the earlier N=200
Llama-3 read:** the within-set split showed `z_aligned − q_resp = −0.034` ("z much worse"), but that
was a **small-sample artifact** — at full N=1000 it is +0.003. So the honest statement is *not* "z is
worse on correctness" but "**z adds no significant correctness signal over text.**" *(Bug fixed en
route: the driver called `arm_preds` inside a per-id list comprehension — ~1000× recompute, the ~30
min/target slowness; a single call + index cut it to ~2 min/target. `arm_preds` reloads are near-free —
the OS-cached 3B backbone; the cost is the proxy forward passes.)*

**Conclusion.** Given `q_resp_only`, `z_aligned` is **not worth its per-target cost**: on SE it is a
small (+0.012–0.018) top-up that does **not** scale with representational compatibility (CKA
0.25→0.87), and on correctness it adds **nothing statistically significant** (all Δ CIs include 0).
`q_resp_only` needs zero target fitting/sampling; `z_aligned` needs a per-target anchor set + Procrustes
W. The transferable uncertainty signal lives in the **model-agnostic text**, not the aligned
hidden-state geometry — the Platonic-alignment result (E24–E30) is scientifically real but
**operationally marginal**. **`q_resp_only` is the right primitive for a deployable amortized-UE
proxy.** Artifacts: `procrustes_e30_ens_vs_qresp_<slug>.json` (SE deltas),
`correctness_ens_vs_qresp.json` (correctness deltas, all 3 at N=1000), `build_e23_fresh_fenced.sh`,
`procrustes_e33_ens_vs_qresp.py`, `correctness_e33_ens_vs_qresp.py`.

---

## E34 — After alignment, do the models share the same UNCERTAINTY DIRECTION? (readout agreement) — ✅ yes, validated against a *fair* ceiling (Mistral/Llama-3 tight; DeepSeek same in the dominant dims)

**Question.** E24–E30 showed the aligned hidden state *reads* SE cross-model. E34 asks a sharper,
geometric question: once all four models' SE readouts are carried into one shared basis (Llama-2
anchor, orthogonal Procrustes W as in `procrustes_alignment.py`), **do they rank questions the same
way, and do their uncertainty *directions* (the readout weight vectors) point the same way?**
Diagnostic only — new `amortized_ue/readout_agreement.py` (kept) reuses `linear_ceiling_probe` +
`procrustes_alignment` read-only; deep-dives ran as throwaway scratch scripts (logs kept under
`amortized_ue/*.log`). TBG, trivia_qa **n2000**, four id-aligned targets, **one shared layer L_a**
throughout (W only connects one layer). CPU; thread-capped (see lesson 2).

**(A) Prediction agreement reaches the within-model reliability ceiling.** `readout_agreement.py`
fits each model's ridge readout at L_a, maps each into the anchor basis, applies all to one fixed
held-out set (anchor test-200), and takes pairwise Spearman between prediction vectors + bootstrap
CIs; references = split-half ceiling (same model, two half-readouts) and random-orthogonal floor.
The layer was **auto-picked L_a=30** first (best val Spearman, *not* the documented 22 — see lesson
3); re-run pinned at **L_a=22** for robustness. Both layers give the *same* conclusion.

Pairwise Spearman between carried predictions (diagonal=1.000 by construction):

| L22 | Llama-2 | Mistral | Llama-3 | deepseek |   | L30 | Llama-2 | Mistral | Llama-3 | deepseek |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-2 | 1.000 | 0.807 | 0.829 | 0.800 | | Llama-2 | 1.000 | 0.889 | 0.855 | 0.882 |
| Mistral | 0.807 | 1.000 | 0.780 | 0.790 | | Mistral | 0.889 | 1.000 | 0.797 | 0.893 |
| Llama-3 | 0.829 | 0.780 | 1.000 | 0.796 | | Llama-3 | 0.855 | 0.797 | 1.000 | 0.788 |
| deepseek | 0.800 | 0.790 | 0.796 | 1.000 | | deepseek | 0.882 | 0.893 | 0.788 | 1.000 |

Cross-model-vs-anchor **meets or exceeds** the split-half ceiling at both layers (L22: Mistral 0.807
vs ceil 0.766; Llama-3 0.829 vs 0.773; DeepSeek 0.800 vs 0.849 — L30: 0.889/0.866, 0.855/0.837,
0.882/0.913). Floors ≈ 0 (random-orthogonal W). Native self-Spearman 0.59–0.65 (sanity PASS). The
built-in "ceiling > cross" sanity check reports **FAIL — that FAIL is the finding**: different models
rank questions no more differently than one model does against itself. Absolute numbers scale with the
layer (L22 noisier readout → ~0.80; L30 → ~0.88) but the ratio (and conclusion) is invariant.

**(B) A dissociation, then its cause.** Prediction agreement is high (~0.80) yet the **weight-direction
cosine** in the shared basis is only ~0.43–0.49 (L22). Resolved by a direct diagnostic (anchor only,
own standardized space, no W, K=15 random half-splits): two ridge fits on disjoint halves of the *same
model* only agree in direction at cosine ~0.4, and it is a pure **collinearity** effect —

| alpha | coef cosine | pred Spearman |   (D=4096, but effective rank of train cov ≈ **218** → 19× redundancy) |
|---|---|---|---|
| 1 | 0.074 ±0.012 | 0.386 ±0.057 | sanity: identical data twice → cosine **1.000** (pipeline correct) |
| 100 | 0.084 | 0.437 | |
| 1000 | 0.146 | 0.557 | |
| **10000** (operating) | **0.407 ±0.017** | **0.799 ±0.020** | reproduces the 0.415 ceiling across 15 splits |
| 100000 | 0.757 | 0.947 | heavy shrink → direction stabilises onto the dominant axis |

→ ridge **coefficients are unstable under multicollinearity while predictions are stable**; the low
full-vector cosine is the noisy low-variance tail, not a model difference. So full-vector cosine is the
wrong instrument — measure the direction **inside the well-determined subspace**.

**(C) Cutoff sweep — same direction in the reliably-estimated subspace.** Project each carried readout
onto the anchor's top-k PCs (label-free), sweep k, standardized L22 space:

| k | Mistral | Llama-3 | deepseek | anchor split-half ceiling | var captured |
|---|---|---|---|---|---|
| 10 | 0.972 | 0.969 | 0.964 | 0.969 | |
| 50 | 0.886 | 0.892 | 0.903 | 0.915 | 35% |
| 100 | 0.774 | 0.812 | 0.822 | 0.779 | 49% |
| 200 | 0.687 | 0.762 | 0.696 | 0.671 | 65% |
| 1440 (full) | 0.472 | 0.514 | 0.498 | 0.416 | reproduces the ~0.47 full-vector number |

Cross ≈ the same-model ceiling at *every* k; the full-vector k=1440 collapses to ~0.47 for cross AND
~0.42 for the same model against itself → the "low" number is noise, shared.

**(D) Principal-angle test — the whole subspace coincides, not just the dominant axis.** SE is a scalar
→ one readout direction; so the "whole subspace" question is on the *representation* subspace the
direction lives in. Principal angles between the anchor's and each aligned model's top-k **state**
subspaces (matched n=720, W fit on 720):

| k | Mistral | Llama-3 | deepseek | #dims coinciding (cos>0.7), of k |
|---|---|---|---|---|
| 10 | 0.863 | 0.895 | 0.924 | 9–10 / 10 |
| 50 | 0.838 | 0.850 | 0.825 | 40–43 / 50 |
| 100 | 0.813 | 0.823 | 0.804 | 78–81 / 100 |
| 200 | 0.805 | 0.813 | 0.808 | 153–155 / 200 |

→ ~80% of the top-100 directions genuinely coincide — broadly aligned geometry, not one lucky axis.
**Caveats stated, not hidden:** the split-half "ceiling" here is *not* a fair reference (it perturbs
the *questions* while cross perturbs the *model* — different perturbations), and W is fit in-sample; the
orthogonal-invariant CKA (0.91) corroborates the alignability is intrinsic.

**(E) ⭐ Decisive matched same-vs-different ceiling.** The valid test (the earlier ceilings were both
invalid — lesson 1). Split train into disjoint halves h1/h2; **both** comparisons span h1↔h2 so they
carry *identical* sampling noise, and the ONLY difference is same-model vs different-model. Direction
cosine in the top-k PC subspace, `cross-model / that model's own same-model ceiling`:

| k | Mistral (cross/self) | Llama-3 (cross/self) | deepseek (cross/self) | anchor ceiling |
|---|---|---|---|---|
| 10 | 0.951 / 0.953 | 0.941 / 0.943 | 0.939 / 0.945 | 0.969 |
| 50 | 0.822 / 0.802 | 0.814 / 0.821 | 0.839 / 0.831 | 0.915 |
| 100 | 0.669 / 0.645 | 0.703 / 0.706 | 0.717 / 0.760 | 0.779 |
| 200 | 0.546 / 0.505 | 0.615 / 0.629 | 0.594 / 0.677 | 0.671 |
| 1440 | 0.303 / 0.272 | 0.355 / 0.384 | 0.368 / 0.441 | 0.416 |

**cross ≈ self-ceiling at every k, for every model** → swapping the model costs no more than a single
model's own measurement wobble. The full-vector 0.30–0.37 that first looked alarming is exactly the
same-model noise floor (0.27–0.44). **DeepSeek nuance:** matches in the top ~50 dims but its cross runs
a hair below its self-ceiling in the deeper dims (k≥100: 0.594 vs 0.677 at 200) → a *small genuine*
model-specific residual, consistent with it being the CKA/scale outlier; Mistral/Llama-3 show no gap.

**Conclusion.** After label-free Procrustes alignment, the models **share the same uncertainty
direction, up to estimation noise** — tight for Mistral and Llama-3, and same-in-the-dominant-directions
for DeepSeek (with a small real residual in the fine detail). Validated against a matched, fair ceiling.
**Scope (do not overstate):** trivia_qa, TBG L22, the **variance-ranked (label-free)** subspace, and this
"direction" is largely the **shared question-difficulty** signal (E25/E26/E33), not a proven model-private
uncertainty axis; "same" means "no detectable difference beyond noise," not an identical crisp axis.

**Methodological lessons (carried into memory):**
1. **A "same vs different" claim is only valid if the reference (ceiling) is matched on every nuisance
   except the factor of interest.** Two ceilings were caught invalid before the right one: (i) the
   split-half ceiling estimated subspaces from *half* the samples while cross used the *full* train
   (sample-size artifact — made cross look better than the ceiling); (ii) the matched-n ceiling still
   perturbed a *different* nuisance (questions) than cross (model). Only the disjoint-halves
   same-vs-different design isolates the model factor.
2. **Cap BLAS threads on the shared node** (`OMP_/OPENBLAS_/MKL_NUM_THREADS`) — the first run burned
   3228 CPU-s in 675 wall-s (~4.8× oversubscription thrash); and **never let a `grep` pipe block-buffer
   away a background job's progress** (write unbuffered to a log).
3. `readout_agreement.py` **auto-picked L_a=30** (best val Spearman on n2000), not the documented
   Llama-2 TBG:22 — the late TBG layers are near-tied, val noise decides; the L22 robustness re-run gave
   the identical conclusion.

**Artifacts.** `amortized_ue/readout_agreement.py`; JSONs `readout_agreement_result.json` (L30),
`readout_agreement_L22_result.json` (L22, `split_half_ceiling` entries carry a `cosine` field); logs
`amortized_ue/{cutoff_sweep,principal_angles,matched_ceiling}.log`; scratch analyses (cosine_check,
cutoff_sweep, principal_angles, matched_ceiling) in the session scratchpad.

---

## E35 — Can we POOL multiple aligned models' data into one ridge that beats a single-model ridge? — ⚠️ small yes (ties oracle best-single, beats a fixed anchor by ~+0.015), but data-saturated and marginal vs the text baseline

**Question.** E34 showed the 4 models share the same uncertainty *direction* after alignment — which makes **pooling** statistically valid (you'd be averaging noisy estimates of the *same* direction, not blending different targets). So: train one ridge on several aligned models' data, does it transfer to an *unseen* model better than a single-model ridge? Leave-one-out: train on 3 models (aligned into the Llama-2 TBG-L22 frame, label-free Procrustes), test on the held-out 4th's held-out questions (`te`, disjoint from train questions). Diagnostic only; the target LLMs are pre-existing (we did **not** select them by direction — E34 *verified* the direction, which is the eligibility check, not a selection step).

**⚠️ A real bug was found mid-experiment (user-caught) and fixed.** v1 pooled the models' **raw** SE labels while the states were per-model **centered** — but the models have very different SE scales (mean CAE: Llama-3 0.48, Mistral 0.48, Llama-2 0.58, **DeepSeek 0.78**). So pooling injected per-model label **offsets** the centered states couldn't explain → the ridge over-regularized and pooling looked *worse* than single (Δ −0.02 to −0.03). **Fix:** z-score each source's SE labels (own train stats) + a per-model feature scaler; standardize the held-out target with its **own** scaler. That lifted pooling by up to **+0.03** (Mistral +0.021, Llama-3 +0.029). Lesson filed in memory (see `pooling-per-model-normalization`).

**(1) LOO pilot (fixed).** Pooled-3 vs the single-source ridges (Spearman on held-out `te`, n=200):

| held-out | pooled-3 | best single (oracle) | Llama-2 single (fixed, deployable) |
|---|---|---|---|
| Llama-2 | 0.613 | 0.631 | — |
| Mistral | 0.558 | 0.569 | 0.543 |
| Llama-3 | 0.591 | 0.590 | 0.564 |
| DeepSeek | 0.576 | 0.595 | 0.576 |

Pooled **ties the oracle best-single** (Δ −0.017 to 0.000, all CIs include 0) and **beats the fixed Llama-2 anchor** (+0.01–0.03). ("Best single" is an oracle — you only know which source is best *after* seeing the target's answers, so you can't use it for a genuinely unseen model.)

**(2) Data-size sweep (single Llama-2 `s` / pooled-3 `p`, per-model normalized).** Less data hurts; pooled ≥ single at every size:

| n_sub | Mistral s/p | Llama-3 s/p | DeepSeek s/p |
|---|---|---|---|
| 50 | 0.463/0.493 | 0.506/0.546 | 0.510/0.547 |
| 200 | 0.486/0.524 | 0.509/0.526 | 0.535/0.581 |
| 800 | 0.551/0.556 | 0.563/0.570 | 0.580/0.580 |
| 1440 | 0.543/0.558 | 0.564/0.591 | 0.576/0.576 |

Both curves rise to a **plateau at ~800 questions** (revises the old "~400" heuristic) — no regime where less data helps.

**(3) ⭐ Matched-total PARTITIONED control (the clean test).** The pilot/sweep gave pooled 3× the rows, so its low-data lead was confounded with row count. Clean design: **same question set AND same total rows** on both arms — single routes all `total` questions through Llama-2; pooled **partitions the SAME questions** into 3 groups, one per source. Only 1-model vs 3-model routing differs:

| total rows | Mistral s/p | Llama-3 s/p | DeepSeek s/p |
|---|---|---|---|
| 150 | 0.458/0.462 | 0.541/0.536 | 0.494/0.517 |
| 600 | 0.533/0.538 | 0.567/0.576 | 0.549/0.564 |
| 1200 | 0.539/0.544 | 0.571/0.577 | 0.569/0.571 |
| 1440 | 0.543/**0.560** | 0.564/**0.584** | 0.576/**0.588** |

Pooled ≥ single in 14/15 cells; at full data all 3 targets +0.012–0.020. **So the genuine diversity effect is small (~+0.015) but real and never negative** — and the *large* low-data lead in sweep (2) was **mostly the 3× rows**, not diversity (at matched rows the low-data gap collapses to ~+0.005).

**(4) 1440 (matched, 1 view/Q) vs 4320 (unmatched, 3 views/Q):** Mistral 0.560 vs 0.558, Llama-3 0.584 vs 0.591, DeepSeek 0.588 vs 0.576 → **tripling the rows changes nothing** (Δ avg ≈ −0.002). Both pools cover the **same 1,440 unique questions**; extra rows are redundant model-views. **What determines performance is the number of unique questions, not rows.**

**Conclusions.**
1. **Pooling ties the oracle best-single and beats a fixed anchor by a small, safe ~+0.015** (never hurts). Much of the earlier apparent advantage was row count; the genuine model-diversity effect is small but consistent.
2. **Both single and pooled ridges are data-saturated** (~800 questions). More rows / more model-views on the **same** questions don't help — **only more distinct questions would**, which needs new Stage-1 generation.
3. **Structural limit:** all 4 models were run on the **same** questions (required for alignment), so pooling can only add **model-diversity, not question coverage** — capping how much it can ever help on this data.
4. **Not better than the strong baselines.** It's a marginal top-up over the text proxy `q_resp_only` (E33), which needs **no** target forward pass / white-box access / alignment. So E35 refines the hidden-state-transfer line but doesn't change E33's verdict that text is the deployable primitive.

**Inference use (how the ridge would deploy) + model criteria.** Offline: train the ridge on aligned source states→SE. Per new target T: **one-time label-free calibration** — run T *and* the anchor on shared "anchor questions," fit `W_T` (orthogonal Procrustes, no SE labels); then **one forward pass per query** → grab T's TBG-L22 state → `W_T` → standardize → ridge → SE. **A target qualifies iff:** white-box hidden-state access, **matching hidden dim (4096** — square Procrustes; other widths need a rectangular map we haven't built), and **adequate alignability = high CKA after `W`** (E30/E34: CKA, not family; verify label-free before trusting — DeepSeek is the low-CKA cautionary case).

**Caveats (honesty).** 3–4 seeds, **no CIs** → ~+0.015 is "small and consistent," not "significant." Minor asymmetry: pooled selects α on a 3× larger val set. **One bug found+fixed**; a `/code-review` was started then **stopped before emitting findings** (token cost) — so these scripts are **not** independently review-verified. Artifacts: `amortized_ue/e35_pooling_{loo_pilot,datasize_sweep,matched_partition}.py`; logs `amortized_ue/{loo_pilot,datasize_sweep,loo_pilot_v2,loo_datasize_v2,matched_partition}.log`. E34 direction deep-dive scripts saved as `amortized_ue/e34_{cosine_instability,cutoff_sweep,principal_angles,matched_ceiling}.py`.

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

## E36 — E35 pooling re-run with LEAK-FREE BEST layers (source side) + a layer-selection leak fix — ✅ magnitudes corrected up for Mistral/Llama-3, pooling conclusion holds, anchor 22-vs-30 a wash

**Why.** Prepping the Exp-2 multi-target proxy, the user asked to reconfirm each target's best layer.
This surfaced a **selection leak** and made the E35 baseline provisional.

**(1) Layer-selection leak (fixed).** `linear_ceiling_probe.main` picked the best *layer* by
`id_test_spearman` — the **test set** (the per-layer ridge alpha was correctly val-selected, but the
cross-layer argmax was on test). The saved `scratch_xllm/*_layer_pick.json` inherited this. New
`reconfirm_layers.py` selects (position, layer) on **val / 5-fold CV**, never test, with a printed
4-point leakage self-audit; validated on synthetic data. **Leak-free best TBG layers:** Llama-2 **30**
(≈22 plateau, tied), Mistral **31**, Llama-3 **31** (CV; the old "SLT:31 0.708" was a test-selection
artifact — ranks #24/66 under CV — RETRACTED), DeepSeek **28** (flat plateau L18–29; unlike the others
it does *not* peak at its final layer). Nothing downstream actually used the leaked `best_id` (E30
aligned Llama-3 on TBG:31, the correct layer), and Llama-2's fixed-layer work sat on the 22↔30 plateau,
so **no prior conclusion is overturned** — but E35 ran on suboptimal *source* layers.

**(2) E35 best-layer re-run** (`e35_pooling_matched_partition_bestlayer.py`; audited clone, only layer
indexing changed — sources at their best TBG, aligned into Llama-2 anchor; anchor 30 and 22 both run).
- **Best-source lifts the hobbled models.** Pooled@1440: Mistral 0.560→**0.594** (+0.032), Llama-3
  0.584→**0.603** (+0.017), DeepSeek ~0 (already near-best at 22). Clean isolation (best-source vs
  shared-22, both anchor 22): +0.032 / +0.017 / +0.001. So E35's magnitudes **were understated** for
  Mistral/Llama-3.
- **Pooling conclusion HOLDS.** Diversity effect (pooled−single) still ~+0.02, always positive — proper
  layers raised absolute numbers but not the qualitative story.
- **Depth-matching hypothesis REJECTED.** Anchor 30 vs 22 is a **wash** (all deltas ≤0.010, 4-seed noise;
  22 even marginally ahead on single-source, plausibly because Llama-2 L22 generalizes slightly better on
  test). Chose **anchor 30** for internal best→best consistency; carry 22 as a cheap comparison.

**Corrected Exp-2 baseline** (best-layer pooled@1440, anchor 30): **Mistral 0.594 / Llama-3 0.603 /
DeepSeek 0.579**. ⚠️ 4 seeds, **no CIs** ("small & consistent", not "significant"). Artifacts:
`amortized_ue/reconfirm_layers.py`, `e35_pooling_matched_partition_bestlayer.py`,
`scratch_xllm/reconfirm_*`, `scratch_xllm/e35_bestlayer_matched_anchor{30,22}.json`. **Still open:** redo
loo_pilot + datasize_sweep with best layers; bootstrap CIs. Lesson filed: [[audit-leakage-before-presenting]].

## E37 — Multi-target cross-LLM PROXY, leave-one-LLM-out (the thesis experiment) — ✅ label-free text+aligned-z FUSION beats the supervised-on-sources ridge on all 4 targets

**The Exp-2 build.** Train ONE proxy (frozen Llama-3.2-3B + LoRA, projector→k=4 soft tokens, ~26M
trainable) on several target LLMs' **aligned** hidden states + text, then **leave-one-LLM-out**: hold out
each of 4 targets (Llama-2/Mistral/Llama-3/DeepSeek, trivia_qa n2000), train on the other 3, test on the
held-out target's 200-question `te`. `z` = per-source best-TBG → Llama-2 TBG:30 frame (label-free
Procrustes, E36 layers), never raw. **Per-model normalization** (feature scaler + SE-label z-score, train
only). **Pooling = SAME questions to all sources** (each Q seen through all 3 source models → the
model-invariance signal; NOT a disjoint partition — that denies invariance and confounds model with the
question set). Arms: z / z_q / z_q_resp / q_only / q_resp_only + **fuse = rank-fusion(z ⊕ q_resp_only)**.
Baseline: ridge on the same aligned pooled z (also label-free on the target). 3 seeds. Same-questions
pooling verified (disjoint 480×3 vs same-480-to-all-3); step-1 slice (`exp2_step1_zarm.py`) verified the
pipeline first (proxy z transfers to unseen Llama-3 at 0.586 ± 0.007, ridge 0.607, chance 0.056).

**Results — 4-fold LOLO, 3-seed means (Spearman):**

| arm | Llama-2 | Mistral | Llama-3 | DeepSeek | MEAN | needs target z? |
|-----|---------|---------|---------|----------|------|-----------------|
| ridge_z | 0.604 | 0.586 | 0.607 | 0.565 | 0.591 | yes |
| z | 0.564 | 0.600 | 0.586 | 0.531 | 0.570 | yes |
| z_q | 0.574 | 0.586 | 0.599 | 0.572 | 0.583 | yes |
| z_q_resp | 0.587 | 0.596 | 0.612 | 0.550 | 0.586 | yes |
| q_only | 0.578 | 0.502 | 0.546 | 0.573 | 0.550 | **no** |
| **q_resp_only** | 0.680 | 0.630 | 0.622 | 0.662 | **0.648** | **no** |
| **fuse(z⊕qresp)** | 0.679 | 0.667 | 0.659 | 0.650 | **0.664** | partial |

**Per-seed Spearman [seed0, seed1, seed2]** (ridge is deterministic — no seeds; from `exp2_lolo_full.json`):

| arm | Llama-2 (ridge 0.604) | Mistral (ridge 0.586) | Llama-3 (ridge 0.607) | DeepSeek (ridge 0.565) |
|-----|------------------------|------------------------|------------------------|-------------------------|
| z | [0.551, 0.593, 0.547] | [0.606, 0.591, 0.602] | [0.585, 0.578, 0.595] | [0.509, 0.560, 0.523] |
| z_q | [0.583, 0.548, 0.591] | [0.589, 0.595, 0.573] | [0.635, 0.598, 0.564] | [0.543, 0.601, 0.573] |
| z_q_resp | [0.617, 0.556, 0.587] | [0.635, 0.579, 0.574] | [0.601, 0.627, 0.608] | [0.569, 0.559, 0.523] |
| q_only | [0.601, 0.585, 0.548] | [0.506, 0.512, 0.489] | [0.547, 0.563, 0.528] | [0.572, 0.583, 0.565] |
| q_resp_only | [0.679, 0.673, 0.688] | [0.625, 0.623, 0.641] | [0.603, 0.641, 0.622] | [0.690, 0.641, 0.656] |
| fuse(z⊕qresp) | [0.671, 0.698, 0.669] | [0.669, 0.658, 0.675] | [0.648, 0.671, 0.657] | [0.659, 0.641, 0.651] |

Seed spreads are tight (std ~0.007–0.03; the early-fusion arms z_q/z_q_resp are the noisiest, up to ~0.03).
The recovery re-run reproduced the original run's per-seed values byte-for-byte (deterministic seeds).

**Findings.** (1) **Label-free fusion ≥ supervised-on-sources ridge on all 4 by mean** (0.664 vs 0.591)
— replicates E27's late-fusion headline across targets. (2) **`q_resp_only` (text, no target hidden
states) beats the ridge on all 4 by mean** (0.648 vs 0.591) — the model-agnostic pathway transfers across
every LLM swap (the thesis). (3) **z tracks CKA** (E30): proxy z beats ridge only on high-CKA Mistral
(0.600 vs 0.586); on low-CKA DeepSeek the text arm dominates (0.662) and z is weak (0.531). (4) **Late >
early fusion** (fuse 0.664 ≫ z_q/z_q_resp 0.583/0.586). (5) `q_only` (question alone, zero target forward
pass) 0.550 mean.

**Significance (bootstrap over 200 examples, seed-avg preds, vs ridge scalar — CONSERVATIVE unpaired):**
**fuse BEATS ridge on 3/4** (Mistral/DeepSeek/Llama-2; overlaps Llama-3), q_resp_only on 2/4, z never.
⚠️ Paired bootstrap (arm−ridge per resample, more powerful) PENDING — needs ridge per-example preds
recomputed (not saved).

**Caveats.** 3 seeds (spreads tight, std ~0.007–0.03); **per-seed data was LOST once (missing json.dump)
then fully recovered by a deterministic re-run** — `exp2_run.py` now saves per-fold incrementally +
per-example predictions; Llama-2 fold has a native-frame advantage (anchor → no cross-align; its ridge
0.604 is inflated), text still beats it; unique-question coverage fixed at 1440 (all models share
questions for alignment) — the proxy is more question-hungry than the ridge (structural cap, step-1).

**Deployable proxy (running):** `--deploy` trains the reusable proxy on ALL 4 models (5760 rows) with
ALL checkpoints saved (`results/deploy_checkpoints/`) + full training log (`results/deploy_curves.json`:
per-step loss/lr/grad-norm, per-epoch train/val-mse/val-spearman, best-epoch/wall-time/params, config).

**Artifacts.** `exp2_run.py`, `exp2_step1_zarm.py`, `results/exp2_lolo_full.json` (per-seed + preds),
`results/exp2_lolo_foldmeans.json`, `scratch_xllm/stage_to_data2.sh` (parallel /data2 staging). Lessons:
[[persist-results-before-done]], [[always-save-checkpoints]].

---

## E38 — Correctness eval of the E37 LOLO proxy: does it actually catch WRONG answers? — ✅ yes; `q_resp_only` is statistically ON PAR with 10-sample sampling and beats supervised SEP on 3/4

**Motivation.** E37 scored the leave-one-LLM-out proxy against the **semantic-entropy label only**.
E31 had shown SE-fidelity ≠ correctness (−0.10 to −0.15 AUROC) for the *E27/E30-era* predictors
(closed-form ridge / rank-fusion), but the **trained multi-target proxy was never re-scored that way**.
New additive script **`amortized_ue/correctness_eval_e37.py`** (trains nothing; reads E37's saved
per-example predictions). Detection label **`incorrect = 1`**; eval rows = the **same 200 held-out `te`
rows E37 reported**, per fold, so every number is directly comparable to the E37 Spearman table.

**Audits run before reading any number** (all passed):
- **ID-mapping audit:** re-derived each fold's `te` ids (`sorted(manifest ids)` → `splits(2000)`) and
  checked their SE labels against E37's saved `target_y` → **max deviation 0.000e+00 on all 4 folds.**
- **Ridge rebuild:** recomputed E37's ridge baseline end-to-end → Spearman reproduces the saved scalar
  to **4 dp on all 4 folds** (0.5861/0.6070/0.5648/0.6038). Its **per-example preds are now saved**
  (`ridge_te_preds`), closing E37's "paired bootstrap PENDING — ridge preds not saved".
- **Independent reproduction of E31:** Llama-3's fold scores the *same* 200 rows E31 used, and this
  script reproduces E31's baselines **exactly** (true SE 0.775, SEP 0.720, SEP-5 0.729) via a different
  code path.

**Results — AUROC for detecting a WRONG answer (`incorrect`), per held-out target (N=200 each):**

| predictor | Mistral | Llama-3 | DeepSeek | Llama-2 | MEAN | label-free on target? |
|---|---|---|---|---|---|---|
| true semantic entropy (10-sample) | 0.762 | 0.775 | 0.821 | 0.783 | 0.785 | NO (is the label) |
| SEP, single best layer (supervised) | 0.721 | 0.720 | 0.740 | 0.611 | 0.698 | NO |
| SEP, 5-layer concat (supervised) | 0.739 | 0.729 | 0.736 | 0.679 | 0.721 | NO |
| ridge_z (E37 baseline) | 0.739 | 0.720 | 0.724 | 0.733 | 0.729 | yes |
| z (proxy) | 0.748 | 0.708 | 0.723 | 0.733 | 0.728 | yes |
| z_q_resp (proxy, early fusion) | 0.755 | 0.748 | 0.764 | 0.752 | 0.755 | yes |
| q_only (proxy) | 0.725 | 0.709 | 0.766 | 0.751 | 0.738 | yes |
| **q_resp_only (proxy, text only)** | **0.796** | **0.767** | **0.844** | **0.797** | **0.801** | **yes** |
| fuse(z ⊕ q_resp) | 0.790 | 0.755 | 0.800 | 0.781 | 0.781 | yes |
| random control | 0.498 | 0.511 | 0.490 | 0.472 | 0.493 | yes |

**Per-seed spread (3 seeds, seed-mean ± std; the table above uses the seed-averaged prediction):**
`q_resp_only` 0.782±0.013 / 0.761±0.003 / 0.825±0.029 / 0.789±0.007; `fuse` 0.785±0.013 / 0.750±0.001 /
0.790±0.014 / 0.775±0.005. Spreads are tight; seed-averaging the *predictions* adds ~+0.01 over the mean
of per-seed AUROCs (ensembling), which is why the headline row sits slightly above the seed means.

**Paired bootstrap (B=10000, shared resample indices), Δ AUROC_incorrect:**

| arm | vs supervised SEP (single) | vs true 10-sample SE |
|---|---|---|
| **q_resp_only** | **+0.074\* / +0.047 / +0.103\* / +0.186\*** (sig. on 3/4) | +0.033 / −0.008 / +0.023 / +0.014 — **all 4 CIs include 0** |
| fuse(z ⊕ q_resp) | +0.068\* / +0.034 / +0.059\* / +0.170\* (sig. on 3/4) | +0.027 / −0.020 / −0.021 / −0.002 — all 4 include 0 |
| ridge_z | +0.018 / −0.000 / −0.016 / +0.122\* (sig. on 1/4) | −0.023 / −0.054 / −0.096\* / −0.050 |
| z | +0.027 / −0.013 / −0.017 / +0.122\* (sig. on 1/4) | −0.015 / −0.067 / −0.097\* / −0.050 |

(order Mistral / Llama-3 / DeepSeek / Llama-2; \* = 95% CI excludes 0.)

**Findings.**
1. **⭐ The text arm closes the gap to sampling.** `q_resp_only` is **statistically indistinguishable
   from the true 10-sample SE** on all four targets (every paired CI includes 0), while needing **no
   sampling, no target hidden states, and no target labels**. State this as *on par*, **not** "beats" —
   the mean is nominally ahead (0.801 vs 0.785) and it leads on 3/4, but no CI excludes 0.
   **This UPDATES E31 finding #2** ("sampling beats amortization for correctness"), which held for the
   E27/E30-era closed-form predictors but does **not** hold for the E37 multi-target trained proxy.
2. **Label-free proxy > supervised SEP on correctness**, significantly on 3/4 (`q_resp_only`; Llama-3's
   CI includes 0) — E37's SE-fidelity headline is **not an artifact of scoring against SE**. Mean
   0.801 vs SEP-single 0.698 / SEP-5layer 0.721.
3. **Multi-source training genuinely helped, measured apples-to-apples.** On Llama-3's *identical* 200
   rows, `q_resp_only` goes **0.739 (E31: single-source Llama-2 reference proxy) → 0.767 (E37: 3-source
   LOLO proxy)**, +0.028, with every baseline in the column reproducing exactly.
4. **The SE→correctness drop is smallest for the text arms.** Mean AUROC_SE − AUROC_inc: true SE +0.215,
   SEP +0.113, ridge_z/z ≈ +0.100, **`q_resp_only` +0.062**, `q_only` +0.073. The hidden-state pathway
   over-fits *SE fidelity* relative to what actually predicts wrongness; text degrades least.
5. **Orderings DIFFER on all 4 targets** (vs E31's "match on 3/4"). Consistently, ranking by SE-fidelity
   **over-ranks `fuse` and under-ranks `q_resp_only`**: `fuse` is 1st-or-2nd by SE on every target but
   `q_resp_only` is 1st-or-2nd by correctness on every target. **Practical consequence: if the goal is
   catching wrong answers, pick `q_resp_only`, not the SE-optimal `fuse`.**
6. **z stays the weak arm** and is significantly *below* true SE on DeepSeek (the low-CKA outlier) —
   consistent with E33 ("z_aligned adds nothing significant on correctness over text").

**Caveats (stated, not smoothed).** **N=200 per fold** → CIs are wide (±0.06–0.08); this is the main
limit and is why "on par with sampling" is the honest read rather than a win. **The Llama-2 fold is not
a clean cross-model test** — Llama-2 is the alignment anchor (native frame, no cross-align) *and* its
SEP baseline is anomalously weak (0.611; selected layer TBG:21 vs TBG:28–31 for the others), so every
Llama-2 Δ-vs-SEP is inflated; the 3/4 significance claim rests on Mistral + DeepSeek, which are clean.
3 seeds. Correctness labels are ~10% noisy (E32) ⇒ these AUROCs are mild **under**-estimates.

**Bug fixed en route (`correctness_eval.py`, E31).** `label_free["q_resp_only"]` was hardcoded `True`;
on the **reference target (Llama-2)** the `q_resp_only` predictor *is* the Llama-2-trained REFERENCE
proxy, so it is not label-free there — same caveat the script already applied to `aligned_z_ridge` /
`rank_fusion_ensemble` (`not is_reference`). **Metadata-only: no reported AUROC changes** (eval ids were
always disjoint from fit ids — re-verified: n2000 ∩ fresh-n1000 = 0 for all 4 targets). The committed
`correctness_eval_*.json` still carry the pre-fix flag (re-running E31 needs the GPU + `amortized_stage2`).

**Artifacts.** `amortized_ue/correctness_eval_e37.py`, `amortized_ue/results/correctness_eval_e37.json`
(per-target metrics + per-seed AUROCs + bootstrap CIs + **ridge per-example preds**),
`amortized_ue/correctness_e37.log`. Runs on CPU in `se_probes` (~25 min, `--data_dir /data2/mn1025/stage1`).

---

## E39 — OOD (cross-DATASET) correctness eval: trivia-fit → squad — ⚠️ E38's parity with sampling does NOT survive the shift; the proxy still beats SEP

**Motivation.** E38 found the E37 proxy's text arm statistically **on par with 10-sample sampling** at
detecting wrong answers — but only **in-distribution** (trivia_qa, the dataset everything was fit on).
This asks the harder question: **under a dataset shift, how well does the proxy detect wrong answers vs
SE and SEP?** New additive script **`amortized_ue/correctness_eval_ood.py`**. Everything is fit/trained on
trivia_qa and evaluated on **squad n1000 (all 1000 rows)**. squad is a real shift: mean accuracy **0.236 /
0.228** vs trivia ~0.65 (incorrect rate **0.77** vs 0.35), mean CAE 1.50 vs 0.59. **squad records exist for
Llama-2 and Mistral only** → a 2-target study.

**⚠️ Which proxy — E37's LOLO run saved NO checkpoints** (`--ckpt_dir` was never passed), so the
leave-one-LLM-out proxy cannot be run on new data without retraining ([[always-save-checkpoints]] again).
The two proxies that DO exist are both reported and answer different questions:
- **DEPLOY** (`results/deploy_checkpoints/`, 3 seeds, all-4-model trivia-trained) — pure **cross-DATASET**
  test; the target **was** in the training pool, so **not** cross-LLM and **not** label-free w.r.t. the
  target. It is the fair peer of SEP (both saw the target's own trivia labels; both tested on squad).
- **REFERENCE** (`runs/REFERENCE_multipos_p1024_5arm_ckpt/`, 5 seeds, Llama-2-only) — **text arms only**
  (its z arm is in Llama-2's native basis, at chance on a model swap without Procrustes W). On **Mistral**
  this is **cross-LLM AND cross-dataset, fully label-free** — the strict thesis test.

**Results — AUROC_incorrect on squad (N=1000/target):**

| predictor | Llama-2 | Mistral | MEAN |
|---|---|---|---|
| **true semantic entropy (10-sample)** | **0.784** | 0.774 | **0.779** |
| SEP, single best layer (supervised) | 0.603 | 0.667 | 0.635 |
| SEP, 5-layer concat (supervised) | 0.631 | 0.665 | 0.648 |
| ridge_z (pooled 3-source aligned) | 0.641 | 0.703 | 0.672 |
| deploy z | 0.636 | 0.714 | 0.675 |
| deploy z_q_resp | 0.657 | 0.735 | 0.696 |
| deploy q_only | 0.647 | 0.662 | 0.655 |
| **deploy q_resp_only** | 0.716 | **0.763** | 0.739 |
| deploy fuse(z ⊕ q_resp) | 0.701 | 0.761 | 0.731 |
| **reference q_resp_only** (label-free on Mistral) | 0.692 | **0.713** | 0.703 |
| random control | 0.537 | 0.522 | 0.529 |

**⭐ Finding 1 — sampling is robust to the shift, amortization is not.** Restricting E38's ID numbers to
these same two targets (a fair matched comparison; true SE / SEP / ridge_z are constructed identically in
both, so those rows are strictly apples-to-apples):

| predictor | ID (trivia) | OOD (squad) | Δ |
|---|---|---|---|
| **true 10-sample SE** | 0.773 | 0.779 | **+0.007 — flat** |
| q_resp_only | 0.797 | 0.739 | −0.058 |
| fuse | 0.786 | 0.731 | −0.055 |
| ridge_z | 0.736 | 0.672 | −0.064 |
| z | 0.741 | 0.675 | −0.066 |
| SEP single | 0.666 | 0.635 | −0.031 |

**True SE barely moves under the dataset swap while every amortized predictor loses 0.03–0.07.** Paired
bootstrap vs true SE: `q_resp_only` **−0.068\*** (Llama-2, CI excludes 0) / −0.011 (Mistral, includes 0).
**So E38's "on par with sampling" is an IN-DISTRIBUTION phenomenon and does NOT survive the shift** —
this **restores E31's "sampling beats amortization"** in the OOD regime. *(⚠️ the proxy rows here are the
DEPLOY proxy, not E38's LOLO proxy, so the proxy Δ is indicative, not strictly matched; the true-SE /
SEP / ridge_z rows are.)*

**Finding 2 — the proxy still beats supervised SEP out of distribution, significantly on BOTH targets.**
Δ AUROC_incorrect vs SEP-single: `q_resp_only` **+0.113\*** (Llama-2) / **+0.096\*** (Mistral); `fuse`
+0.098\* / +0.095\*; `z_q_resp` +0.055\* / +0.068\*; `deploy_z` +0.034 / +0.048\*; `ridge_z` +0.038 /
+0.036 (both include 0, marginally). Amortization degrades under shift, but **less than the in-model
probe it replaces**.

**⭐ Finding 3 — the strict thesis test PASSES.** `reference_q_resp_only` on **Mistral** is the Llama-2-trained
proxy that **never saw Mistral and never saw squad**, fully label-free: **0.713 vs Mistral's own supervised
SEP 0.667, Δ +0.046 [+0.004, +0.089] (excludes 0)**. A proxy trained on a *different model* and a
*different dataset* beats the target's own supervised probe.

**Finding 4 — `q_only` collapses OOD** (deploy 0.655, reference 0.628 — barely above `ridge_z`, and on
Mistral *below* SEP). The **response** text carries the transferable signal, not the question; consistent
with the ID `q_resp_only` ≫ `q_only` gap. **Finding 5 — target-in-pool is worth ~0.05:** on Mistral,
deploy (saw Mistral trivia) 0.763 vs reference (never saw Mistral) 0.713.

**PRR tells the same story more starkly** (true SE 0.522/0.495 · q_resp_only 0.451/0.501 · SEP 0.206/0.304);
on Mistral `fuse` PRR 0.519 nominally exceeds true SE's 0.495 while its AUROC is slightly lower.

**Caveats.** **2 targets only** (squad exists for Llama-2/Mistral alone). The DEPLOY rows are **not
label-free** w.r.t. the target — only the `reference_*` rows are, which is why Finding 3 rests on the
Mistral reference row. **Llama-2's SEP is anomalously weak again** (0.603, selected layer TBG:21 vs
Mistral's TBG:30) exactly as in E38, so its Δ-vs-SEP is inflated — **Mistral is the clean column**.
squad's base rate is 0.77 incorrect ⇒ PRR / acc@coverage are more informative than AUPRC.
AUROC_SE uses `best_split` refit on the squad rows (the documented OOD convention — the label scale shifts).

**Second latent bug found + fixed.** `exp2_run.py` wrote the `checkpoint/v1` format tag but omitted the
`k` and `transform` meta keys, so `stage2.checkpoint.load_checkpoint` **KeyErrors on its own checkpoints**.
Fixed in `exp2_run.py` for future runs (targets are z-scored per model upstream, so the identity transform
is the honest record); `correctness_eval_ood.py` carries a compat loader (`_load_exp2_ckpt`) for the files
already on disk.

**Artifacts.** `amortized_ue/correctness_eval_ood.py`, `amortized_ue/results/correctness_eval_ood.json`,
`amortized_ue/correctness_ood.log`. Env `amortized_stage2` + GPU (~15 min);
`--trivia_dir /data2/mn1025/stage1` (squad always reads the default path — /data2 holds only trivia).

---

## E40 — Is the pooled multi-model RIDGE model-SPECIFIC, or only a question-difficulty detector? — ⚠️ genuinely model-specific but thin (12.6% of the attainable), gated by alignment quality; **and the leave-one-out null is NEGATIVE, not zero**

**Question (user's framing).** Find questions where the target LLMs genuinely disagree in semantic
entropy (SE_Llama-2(x)=1.8 vs SE_Mistral(x)=1.2) and ask whether the shared probe reproduces that
disagreement. If it does, the probe has not merely learned "this is a hard question" — it has preserved
"THIS model is uncertain about this question". This is the sharp version of E25's Mechanism-A control,
applied to the **E35/E36/E37 pooled ridge** (aligned hidden states → SE), not to the SLM proxy.

**Design.** 4 targets × the SAME 2000 trivia questions, id-joined; `splits(2000)` → tr/va/te, and `te`
(200 questions) is IDENTICAL across models, which is what makes a matrix analysis valid. Leave-one-LLM-out:
for target T one ridge trains on the other 3 aligned models (label-free Procrustes into the Llama-2
best-TBG frame, per-model feature scaler + per-model SE z-score, α on the pooled source val) → **P[4,200]**
vs **Y[4,200]**, every column produced by a probe that never saw that model. Per-model normalization is
**mandatory, not a choice**: the ridge trains on per-model z-scored labels, so the per-model SE offset
(mean CAE: Llama-3 .486, Mistral .492, Llama-2 .586, DeepSeek .794) is unrepresentable by construction —
comparing raw SE across models would score the probe on a constant it was built not to emit. Primary
normalization = within-model rank→normal quantile over `te`; robustness = z-score on the target's own VAL.

**Audits (all passed, before reading any headline).** Rebuilt ridge te-Spearman 0.6036/0.5865/0.6070/0.5649
vs E37's 0.6038/0.5861/0.6070/0.5648, and its **per-example preds correlate 1.000 with E38's saved
`ridge_te_preds`** → the refit IS the E37 ridge. E37 `target_y` vs freshly loaded labels: max dev 5.9e-08
(float32 storage in E37's JSON, not a mismatch) — this also validates the `/data2` copy. Checkpoint
reload dev 2.4e-07. Analysis primitives were validated on synthetic data first (a difficulty-only
predictor must score 0.000 residual / 0.500 pair-accuracy; a model-specific one must score high).

**⚠️ The measurement was broken, and the `q_only` CONTROL caught it.** `q_only` (E37 proxy arm, question
text only — its input is IDENTICAL for all four targets, so it cannot know which model it predicts for)
must score ~0 on any model-specificity metric. It scored **−0.097 (p=0.013)**. Not a coding bug — a
structural property of leave-one-out: for target T the probe is trained on the other 3, so it estimates
*their* SE, and since the model-specific residuals sum to zero, `mean_{k≠T} s_k = −s_T/3`. **A predictor
carrying no information about T is ANTI-correlated with it.** Proved exactly (E40b): the perfect
pure-difficulty LOO predictor `D_T = mean_{k≠T} Y_k` satisfies `R_D = −(1/3)R_Y` identically →
**residual corr = −1.0000** (numerically confirmed). ⇒ **"chance = 0" is WRONG for any leave-one-out
model-specificity metric**, and E40's [C]/[D]/[F] are all biased DOWNWARD.

**The clean estimand (E40b) — leave-TWO-out.** One ridge (trained on the other two models) scores BOTH
members of the held-out pair, so `dP = P_A − P_B` uses the SAME weights and the fold-composition artifact
cancels; a question-only predictor gives `dP = 0` exactly ⇒ **the null IS 0**. Bootstrap over the 200
questions + an exact sign-flip permutation test:

| held-out pair | r(dP,dY) | boot95 | sign-flip p | pair-acc |
|---|---|---|---|---|
| Mistral vs Llama-3 | **+0.262** | [+0.145, +0.380] | 0.0000 | 0.558 |
| Llama-2 vs Mistral | +0.153 | [+0.017, +0.289] | 0.0426 | 0.502 |
| Mistral vs DeepSeek | +0.112 | [−0.042, +0.252] | 0.1564 | 0.543 |
| Llama-2 vs Llama-3 | +0.085 | [−0.038, +0.202] | 0.1844 | 0.492 |
| Llama-3 vs DeepSeek | +0.057 | [−0.070, +0.180] | 0.3858 | 0.505 |
| Llama-2 vs DeepSeek | **+0.001** | [−0.125, +0.127] | 0.9840 | 0.487 |
| **POOLED (n=1200)** | **+0.110** | **[+0.027, +0.192]** | **0.0002** | 0.515 [0.477, 0.550] |

**Findings.**
1. **There IS disagreement to predict.** Question effect = **65.7%** of normalized SE variance,
   **model-specific residual 34.3%**; cross-model SE Spearman only **0.486–0.583**; raw |SE_a − SE_b|
   mean 0.464 / median 0.325 / p90 1.234, with **40.1% of pairs > 0.5 nats**. The user's 1.8-vs-1.2 case
   is common, not exotic.
2. **⭐ The ridge is genuinely model-specific — but thin.** Pooled clean-frame r = **+0.110**
   (p=0.0002) against a **matched split-half ceiling of 0.870** (E40c, computed for the pair-difference
   estimand, not borrowed from the LOO residual) ⇒ it recovers **12.6% of the attainable disagreement**.
   So the probe is *mostly* a question-difficulty detector with a small, real per-model component.
3. **The signal lives only in the LARGE gaps.** Magnitude-weighted correlation is significant while
   unweighted pair-ordering accuracy is not (0.515 [0.477, 0.550]) — it gets the direction right when the
   models differ a lot and is at chance when they differ a little. Same shape in the biased LOO frame,
   where accuracy climbs monotonically with gap size: 0.509 (all) → 0.531 (top 50%) → 0.547 (top 25%) →
   **0.600 (top 9%)**.
4. **Model-specificity is gated by alignment quality, not uniform.** One pair carries most of it
   (Mistral↔Llama-3 +0.262); the **low-CKA DeepSeek pair is exactly zero** (Llama-2↔DeepSeek +0.001) —
   consistent with E30's "alignability tracks CKA, not family". Same ordering per-model in the LOO frame
   (Mistral 0.195, Llama-3 0.107, Llama-2 0.022, DeepSeek −0.024).
5. **Response TEXT is a far more model-specific channel than the aligned hidden state.** In the (biased,
   ordinal-only) LOO frame: `q_resp_only` **+0.237** ≫ `z_q_resp` +0.097 ≈ `z` +0.090 ≈ `ridge_z` +0.075
   ≫ `q_only` −0.097 (the no-information baseline). Counterintuitive but coherent with E33/E38: the
   sampled answer IS the model's own output, whereas alignment rotates hidden states into a shared frame
   that washes out much of what makes each model distinctive.
6. **[E] semi-partial vs a difficulty oracle is BIASED UP and must not be read alone** — the oracle
   (mean true SE of the other 3) is itself noisy, so even a *noiseless* pure-difficulty predictor keeps a
   positive partial correlation (+0.27 in synthetic control). The matched empirical null `r(Pbar,Y|D)` is
   printed alongside; it exceeds the ridge's value on Llama-2 and DeepSeek.

**Robustness.** val-z-score normalization instead of rank-qnorm: +0.061 (vs +0.075), pair-acc(top25)
0.520 (vs 0.547) — same conclusion.

**Caveats (stated, not smoothed).** **N=200 questions** ⇒ wide CIs throughout; only 1 of 6 pairs is
individually significant and the pooled result rests largely on Mistral↔Llama-3. **The leave-two-out
probes train on only 2 source models**, so they are weaker than the 3-source LOO ridge — the clean design
costs power, and +0.110 is likely a mild UNDER-estimate of what a 3-source probe preserves. The [F] arm
comparison sits in the biased frame and is **ordinal only**. Llama-2 is the alignment anchor (identity W,
native frame), so its pairs are not clean cross-model tests. Ridge is deterministic (no seed variance);
the E37 proxy arms are 3-seed-averaged predictions.

**⚠️ Checkpoint gap found + fixed (user-caught).** The **entire E35/E36/E37 hidden-state line had never
persisted a single fitted ridge** — `e35_pooling_*.py` write JSON only, and `exp2_run.py`'s `--ckpt_dir`
covers `train_arm` (the proxy) but not `ridge_on_z`, which fits, returns a Spearman scalar and drops the
model; the only surviving trace was E38's `ridge_te_preds`. "Checkpoint" had been read too narrowly as
"neural checkpoint". Fixed: `save_ridge_bundle()`/`load_ridge_bundle()` persist the **full inference
chain** — per-model Procrustes W + centering mean (the expensive part), feature scalers, per-model SE
label z-stats, all 10 fitted ridges (4 LOO + 6 LTO) with alphas, and `meta.json` (splits/ids/layers/
inference recipe) — to **`stage2/runs/E40_pooled_multimodel_ridge/checkpoints/`** (179MB), i.e. the same
`stage2/runs/<RUN>/checkpoints/` convention as every other trained artifact here, already covered by the
blanket gitignore on that path → W&B, not git. E40b **reuses the saved bundle with no refit**, which
doubles as the proof it loads.

**Artifacts.** `amortized_ue/e40_model_specificity.py` (build + [A]–[G] + checkpointing),
`amortized_ue/e40b_lto_significance.py` (the negative-null proof + CIs on the clean test),
`amortized_ue/e40c_lto_ceiling.py` (matched ceiling); `results/e40_model_specificity.json`,
`results/e40b_lto_significance.json`, `results/e40c_lto_ceiling.json`;
`stage2/runs/E40_pooled_multimodel_ridge/checkpoints/`. Env `se_probes`, CPU, minutes. **Run with
`--data_dir /data2/mn1025/stage1`** (node-local ext4: 3.5s to read 2000 records vs minutes on the NFS).

---

## E41 — Fix Llama-2's SEP layer-selection variance (user-caught) + rerun E38/E39 — ✅ correction applied, no headline overturned

**Why.** Comparing the E37/E38 proxy against each target's own supervised SEP, Llama-2's SEP was an
outlier: 0.611 AUROC_incorrect vs 0.72–0.74 for the other three targets, inflating the proxy's
apparent edge on Llama-2 to +0.186. User asked to check the layer selection rather than take the
number at face value ("we already have the best layer for Llama-2, ~30 — check E36/EXPERIMENTS.md").

**Diagnosis (`sep_layer_diag.py`, scratch, not committed — dumps val/test AUROC for all 66
(pos,layer) combos on all 4 targets).** `sep_single_val_selected` picks the layer by AUROC on a
360-row val split — leak-free, but on Llama-2 the late-TBG band is a genuine near-tie: **TBG:21 and
TBG:23 tie at val 0.7763 to 4dp**, and the whole TBG L18-32 band spans only 0.036 in val AUROC.
E36's TBG:30 (chosen separately, on 5-fold CV, for the ridge/z arms) sits at val-rank 8/66 — 0.017
off the "winner" — but scores **0.669 vs 0.611 on test AUROC_incorrect**, while the leaky
test-oracle over all 66 combos is only **0.687**. So Llama-2's SEP is genuinely the weakest of the
four (no layer reaches Mistral/DeepSeek's ~0.74), but 0.611 specifically was selection noise, not a
property of the model. Mistral's val-selected layer also moved (TBG:30→31, val-rank 2/66); Llama-3
and DeepSeek were already at their CV-optimal layer (val-rank 1/66 each) — matches E36's original
finding "≈22 plateau, tied" for Llama-2 and confirms E34's lesson #3 (late TBG layers near-tied, val
noise decides).

**Fix (additive).** New `sep_single_fixed_layer()` in `correctness_eval.py` — identical
fit/binarisation to `sep_single_val_selected`, layer passed in rather than re-selected.
`correctness_eval_e37.py` and `correctness_eval_ood.py` both now score `sep_single_e36_layer`
(E2.BEST_TBG per target — Llama-2 30, Mistral 31, Llama-3 31, DeepSeek 28) alongside the original
`sep_single_best_layer`, and use the fixed-layer version as the primary bootstrap baseline.
`sep_single_val_selected` itself is untouched (E31 stays reproducible). Outputs go to new files
(`correctness_eval_e41_fixedlayer.json`, `correctness_eval_e41_ood_fixedlayer.json`) — E38/E39's
committed jsons are untouched.

**E38 rerun (ID, trivia, `se_probes`/CPU, `--data_dir /data2/mn1025/stage1`):**

| target | SEP old | SEP fixed | Δ | Δ(q_resp_only − SEP), old | Δ(q_resp_only − SEP), fixed |
|---|---|---|---|---|---|
| Mistral | 0.721 | 0.738 | +0.017 | +0.074\* | +0.058 (now includes 0) |
| Llama-3 | 0.720 | 0.720 | 0 | +0.047 | +0.047 |
| DeepSeek | 0.740 | 0.740 | 0 | +0.103\* | +0.103\* |
| Llama-2 | 0.611 | 0.669 | +0.058 | +0.186\* | +0.128\* |
| **MEAN** | **0.698** | **0.717** | **+0.019** | +0.103 | **+0.084** |

Only real change to a significance verdict: Mistral's proxy-vs-SEP gap is no longer significant on
its own (was inflated by Mistral's SEP also landing 1 layer off its CV-optimum). DeepSeek and
Llama-2 stay significant → **honest claim is "significant on 2/4 individually, positive on all 4,"
not "3/4."** Everything else holds: `q_resp_only` still leads every target by mean AUROC_incorrect,
still statistically on par with true 10-sample SE (all CIs include 0), "SE-fidelity over-ranks
`fuse`, under-ranks `q_resp_only`" unchanged.

**E39 rerun (OOD, trivia→squad, `amortized_stage2`/GPU):**

| predictor | Llama-2 | Mistral | MEAN |
|---|---|---|---|
| true 10-sample SE | 0.784 | 0.774 | 0.779 |
| SEP old | 0.603 | 0.667 | 0.635 |
| **SEP fixed** | 0.621 | 0.669 | **0.645** (+0.010) |
| deploy `q_resp_only` | 0.716 | 0.763 | 0.739 |
| reference `q_resp_only` | 0.692 | 0.713 | 0.703 |

Correction is much smaller OOD (+0.010 mean vs +0.019 ID) — plausible, since every hidden-state
method is already degraded by the dataset shift, so a better layer buys less. **All three E39
findings survive unchanged:** proxy still loses to true SE (deploy_q_resp_only Δ vs SE: Mistral
−0.011 n.s. / Llama-2 −0.068\*); proxy still beats SEP-fixed significantly on both targets (+0.094\*
Llama-2, +0.096\* Mistral); the strict thesis test still passes (reference_q_resp_only on Mistral
0.713 vs SEP-fixed 0.669, Δ +0.046\* — essentially identical to the pre-fix number, since Mistral's
own layer barely moved).

**Takeaway.** The fix was worth doing — it replaces a baseline with a known selection-variance flaw
with one chosen leak-free on CV — but it narrows rather than overturns the headline: the proxy's
edge over SEP shrinks from "3/4 significant, mean +0.103" to "2/4 significant, mean +0.084" ID, and
is essentially unchanged OOD. Lesson: [[audit-leakage-before-presenting]] extends to
**selection-variance audits**, not just leakage — a leak-free selector chosen on a small val split
can still be high-variance, and a near-tied plateau (E34/E36's recurring Llama-2 finding) is a sign
to check before trusting a single-run selection.

**Artifacts.** `correctness_eval.py::sep_single_fixed_layer` (new fn, additive), edits to
`correctness_eval_e37.py`/`correctness_eval_ood.py` (additive predictor + baseline, new output
paths). `results/correctness_eval_e41_fixedlayer.json`, `results/correctness_eval_e41_ood_fixedlayer.json`,
`correctness_e41.log`, `correctness_e41_ood.log`.

---

## E42 — a proxy trained ONLY on Mistral's TriviaQA, tested on Mistral's squad (fills a gap E39 left open) — ✅ dataset-shift-only result, the cleanest single-source case yet

**Why.** E39's OOD eval had a Llama-2-trained-only proxy on Llama-2's own squad (Reference,
0.692 — pure dataset shift) but the matching Mistral case didn't exist: the only Mistral-side
numbers were Deploy (all 4 models pooled, so not single-source) and Reference *evaluated on*
Mistral (which stacks a model shift on top of the dataset shift). The direct mirror of "Llama-2
proxy on Llama-2 squad" for Mistral was never run — checked first (`grep`'d `results/` and
EXPERIMENTS.md for any prior Mistral+squad+proxy combination): confirmed absent, not just
undocumented. It exists to build: the E22 role-swap proxy
(`stage2/runs/E22_Mistral_proxy_p1024_5arm_ckpt/checkpoints`, 5 seeds, trained on Mistral trivia_qa
n2000, all 5 arms saved, `checkpoint/v1` format identical to REFERENCE) was built for E22/E23's
cross-LLM trivia test and simply never pointed at squad.

**Method (additive, `mistral_trained_proxy_ood.py`).** `procrustes_e27_rank_fusion.arm_preds` got
one new optional `ckpt_dir` param (default = REFERENCE's dir, so every existing caller — E27/E38/E39
— is untouched) so it can load a DIFFERENT proxy's checkpoints. Text arms only (`q_only`,
`q_resp_only`) — matches what's usable from Reference cross-model (z arms aren't run: E33 already
established z isn't worth its cost relative to text, and this keeps the comparison to Reference
apples-to-apples). Independently re-derives Mistral's squad ids/accuracy/SE via `load_records` (not
`load_matrix`, a different code path) and asserts them against `correctness_eval_e41_ood_fixedlayer.json`'s
frozen values before reusing SEP/SE from there — id-mapping audit **max dev 5.9e-08, MATCH** — so
the true-SE/SEP numbers aren't refit (saves ~an hour of SEP layer-fitting) but the comparison is
still verified apples-to-apples. Env `amortized_stage2` + free GPU, ~2 min.

**Result:**

| predictor | AUROC_incorrect | what it is |
|---|---|---|
| true 10-sample SE | 0.774 | frozen, reused from E41 |
| **Mistral-trained proxy, `q_resp_only`** | **0.748** | **this run** |
| reference (Llama-2-trained) proxy on Mistral | 0.713 | frozen, reused from E41 |
| Mistral SEP, E41-fixed layer | 0.669 | frozen, reused from E41 |
| Mistral-trained proxy, `q_only` | 0.647 | this run (control) |
| Mistral SEP, old val-selected layer | 0.667 | frozen, reused from E41 |

Internal check: Δ(q_resp_only − q_only) = **+0.101 [+0.068, +0.133]**, excludes 0 — the gain comes
from reading the model's answer, not just the question, consistent with every other OOD result.

**Findings.**
1. **This is the smallest dataset-shift penalty seen anywhere in the OOD line.** vs true SE: −0.026,
   vs Llama-2's single-source case (Reference on Llama-2, −0.092) and the cross-model case (Reference
   on Mistral, −0.061). Single-source, single-model dataset shift costs less on Mistral than on
   Llama-2 — consistent with Mistral being the higher-CKA, more "well-behaved" model throughout E29–E40.
2. **Model-matched training beats cross-model training, dataset shift held constant.** Mistral-trained
   → Mistral squad (0.748) beats Llama-2-trained → Mistral squad (0.713) by **+0.035**, isolating a
   "knowing the target model helps" effect from the dataset shift itself (both rows are evaluated on
   the identical 1000 Mistral/squad rows, only the training source differs).
3. **Still beats SEP by a clear margin** (+0.079 vs the fixed-layer SEP, +0.081 vs the old one) even
   in this single-source, no-target-labels setting.
4. **The four-point dataset-shift-only ladder is now complete and orderly:** Llama-2-on-itself
   (0.692) < Llama-2-on-Mistral / cross-model (0.713) < **Mistral-on-itself (0.748, this run)** <
   Deploy/all-4-on-Mistral (0.763) — more or better-matched training data monotonically shrinks the
   dataset-shift penalty, closing in on true SE (0.774) but never quite reaching it.

**Caveats.** Single dataset-shift comparison (Mistral only — Llama-2's own-model number, 0.692, is
the only other point in the same "single-source, own-model" cell); bootstrap CI only computed
internally (q_resp_only vs q_only) — the deltas vs the frozen SEP/SE baselines are point estimates,
not independently reconfirmed with paired CIs against this run's specific predictions (the frozen
CIs from E41 apply to the *frozen* rows, not new pairings against this run's `q_resp_only`). No
z/ridge arm run (deliberate, matches Reference's usable arm set, but means this isn't a full 5-arm
picture the way E38/E39 are).

**Artifacts.** `amortized_ue/mistral_trained_proxy_ood.py`, `amortized_ue/results/e42_mistral_trained_proxy_ood.json`,
`amortized_ue/e42_mistral_proxy_ood.log`. One-line additive change to `procrustes_e27_rank_fusion.py`
(`arm_preds` gains `ckpt_dir=None`, defaults preserve every existing caller).

## E44

**Goal:** extend the target-LLM roster from 4 to (eventually) 14 by adding two new families —
Qwen (5 checkpoints spanning generations) and Gemma (5 checkpoints spanning generations) — as a
breadth test of the cross-LLM transfer thesis across a much wider range of architectures/vendors
than the existing Llama-2/Mistral/Llama-3/DeepSeek set.

**Models chosen** (after several rounds of narrowing with the user): small tier (~7-9B, one from
each of the two most recent generations per family) — `Qwen3-8B`, `Qwen3.5-9B`, `gemma-7b-it`,
`gemma-2-9b-it`; big tier (~27-31B, one per generation) — `Qwen3.5-27B`, `Qwen3.6-27B`,
`Qwen3.8-27B`, `gemma-2-27b-it`, `gemma-3-27b-it`, `gemma-4-31B-it`. Several of these
(Qwen3.5/3.6/3.8, Gemma 4) were released within days-to-months of this session and postdate every
existing conda env's `transformers` — required a new environment.

**Infra work (all additive, no SE/probe logic touched):**
- `qwen` load branch + `gemma` gated→`unsloth`-mirror redirect added to `huggingface_models.py`;
  `init_model()` dispatch whitelist in `utils.py` widened to accept `qwen`/`gemma` (same pattern
  already used for `deepseek`).
- New env `se_probes_v5`: a plain **venv** (not conda), bootstrapped from `/data/sv/miniconda3`
  directly to sidestep a bad NFS window that was hanging both `conda create --clone` and even
  `pip freeze` on the existing NFS-hosted envs. `transformers==5.15.1`, `torch==2.13.0+cu130`,
  covers every new architecture (`qwen3`, `qwen3_5`, `gemma`, `gemma2`, `gemma3`, `gemma4`).
  `HF_HOME` and all Stage-1 output moved to `/data2` (non-NFS) for the same reason.
- Considered relocating the whole repo to `/data2` for speed; **rejected** — Claude Code's
  project/session/memory identity is keyed to the literal working-directory path (confirmed:
  `~/.claude/projects/<slug-of-cwd>/`), and a symlink workaround was assessed as unsafe (evidence
  Claude Code resolves realpath before slugging, which would silently fork history). Repo stays on
  NFS; only the new env + HF cache + Stage-1 output moved.

**Two model-scoped bugs found and fixed** (`Qwen3.8-27B` first, `Qwen3.5-9B` found later at n1000
scale after its 3-question smoke test passed clean — **a 3-question smoke test is not enough to
catch a ~5% failure rate**):
1. Both models sometimes open completions with a blank line then a multi-line
   `<think>...</think>` reasoning block; the pipeline's default "stop at first `\n`" rule fires on
   the leading blank line, or (once a first fix skipped leading whitespace) on the first newline
   *inside* the still-open think block — captured answer ends up empty or literally `"<think>"`.
   **Two wrong turns before the real fix:** raising `model_max_new_tokens` alone does nothing (the
   stop fires within the first few tokens regardless of budget — verified twice); and killing a
   build partway through and reading `manifest.json` to check whether a fix worked is misleading,
   because `write_manifest()` only runs once at the very end of the batch while individual `.pt`
   files save incrementally — a stale manifest looked identical to "the fix failed" when the fix
   had actually worked. **Real fix:** `StoppingCriteriaSub.tolerate_thinking` — suppress the stop
   match entirely while more `<think>` than `</think>` have been seen so far, so generation runs
   through the whole reasoning block uninterrupted and only stops at a newline *after* the tag
   closes. Model-scoped via `_LEADING_WHITESPACE_MODELS`/`_EXTRA_TOKEN_BUDGET_MODELS` — off
   (byte-identical) for every other model.
2. `gemma-4-31B-it` — a much deeper problem, not fixed: degenerate output (echoes the last
   few-shot example's answer verbatim, or produces repetitive gibberish) even on a clean
   3-question smoke test. **Not attempted at n1000; needs real diagnosis in a future session.**

**⚠️ Dual-GPU sharding is reproducibly broken for the Qwen3.5+ hybrid (Gated-DeltaNet + Gated-
Attention) architecture.** `device_map='auto'` split across both GPUs crashes every time —
`torch.multinomial`: "probability tensor contains inf/nan", CUDA device-side assert — fast,
deterministic, reproduced 2/2 on real n1000 builds (`Qwen3.5-27B`, `Qwen3.6-27B`) and once more
cheaply on `Qwen3.5-9B` forced into an artificial 2-GPU split via a tight `max_memory` cap.
**Root cause diagnosed (not fixed):** the device map is clean at layer granularity (no layer's
parameters split mid-way) — the crash is specifically at the boundary between a Gated-Attention
layer on one GPU and a Gated-DeltaNet layer on the other. DeltaNet layers carry recurrent state
across generation steps; `accelerate`'s generic multi-GPU dispatch hooks evidently don't re-sync
that state correctly when a layer's input activation arrives from a different GPU than the
layer's home device. Almost certainly an upstream `transformers`/`accelerate` bug (architecture
is ~1 week old) — patching it for real means editing vendored modeling code, out of proportion to
the payoff; **not attempted.** **Workaround:** single-GPU + CPU offload (proven reliable in every
build, ~45-75s/record for a 27B model — slow but correct). One user-caught near-incident:
investigating the crash via a small forced-split reproduction on `Qwen3.5-9B` briefly grabbed
~1.6GB on the same GPU hosting the live `Qwen3.5-27B` build, which only had ~1.5GB truly free,
and OOM-killed it. No data lost (Stage-1 builds are resumable, records save incrementally) but a
reminder to check headroom more conservatively before touching a GPU with a live job on it.

**Results at session end:**

| Model | Status |
|---|---|
| Qwen3-8B, Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it | ✅ done, 1000/1000 verified |
| Qwen3.5-27B, Qwen3.6-27B | 🔄 running (single-GPU+offload, parallel across the 2 GPUs) |
| Qwen3.8-27B, gemma-2-27b-it, gemma-3-27b-it | ⏳ queued |
| gemma-4-31B-it | ❌ broken, excluded until diagnosed |

**Caveats.** Big-tier builds were still in progress when this entry was written — verify final
record counts before citing. The Qwen3.5-9B dataset has ~46/1000 records from its hardest
question subset where the model doesn't converge on an answer within the 250-token budget even
after the stop-logic fix — kept as-is per user decision (treated as legitimate "model didn't
converge" signal, not corrupted data, since the fix already guarantees no more truncated-garbage
records). Dual-GPU root cause is diagnosed but unverified against the actual `transformers`
source (inferred from the device map + crash location, not from reading the modeling code
directly) — treat as a strong hypothesis, not a confirmed upstream bug report.

**Artifacts.** `amortized_ue/build_small_tier_n1000.sh` (done), `amortized_ue/build_big_tier_n1000.sh`
(running at session end), shared id file `/data2/mn1025/stage1_meta/shared_n1000_ids.txt`, per-model
logs under `amortized_ue/logs/`. Code changes in `semantic_uncertainty/uncertainty/models/huggingface_models.py`
and `.../utils/utils.py` (both additive, see diff).

## E45

**Goal:** the sharpest cross-LLM-transfer test run so far. Score the existing DEPLOY proxy (frozen
Llama-3.2-3B + LoRA, trained by pooling Llama-2/Mistral/Llama-3/DeepSeek's n2000 —
`results/deploy_checkpoints/`, 3 seeds) **zero-shot** — no retraining, no calibration, no target
labels, no target hidden states — on the 4 E44 small-tier targets (`Qwen3-8B`, `Qwen3.5-9B`,
`gemma-7b-it`, `gemma-2-9b-it`). Unlike E20-23/E37-39's cross-LLM tests, which all swap among the
Llama-2/Mistral/Llama-3/DeepSeek set (related lineage, overlapping training-data eras), Qwen and
Gemma are genuinely different vendors/architectures the deploy proxy has never been near in any
form — the closest thing to an out-of-family generalization test the project has run.

**Method (additive, `e45_qwen_gemma_zeroshot.py`).** Only the text arms (`q_only`, `q_resp_only`)
— no Procrustes alignment exists yet for Qwen/Gemma, so the z-pathway isn't reachable zero-shot;
scoped out rather than faked. Scores each target's existing E44 `*_trivia_qa_n1000_full` (shared
ids, verified **zero overlap** with the n2000 the deploy proxy trained on) against
`canonical.accuracy` (correctness-eval convention, matches E31/E38/E39/E42): true 10-sample SE,
`q_only`, `q_resp_only`, random control, plus paired bootstrap deltas (`q_resp_only` vs true SE,
`q_resp_only` vs `q_only`).

**Three infra/bugs hit and fixed en route (none touch SE/probe logic):**
1. **NFS degraded window blocked the run entirely at first** — `amortized_stage2` (unlike
   `se_probes_v5`) still lives on NFS, so every `import torch`/`transformers`/etc. during a
   degraded window turned into a multi-minute stall (`state D`, `wchan: rpc_wait_bit_killable`,
   confirmed via `/proc/<pid>/status`; a bare `ls` on `/vol/bitbucket` took 2.6-3.9s instead of
   instant). **Fixed properly, not just waited out:** built `amortized_stage2_v5`, a `/data2`
   venv mirroring `se_probes_v5`'s NFS-bypass pattern, with `transformers==4.52.4`/
   `peft==0.19.1`/`accelerate==1.14.0` pinned exactly to the live NFS env's versions (read via a
   background version-check once NFS un-stalled). **`torch` could not be held at the original
   2.1.1** — `bitsandbytes` pulled a newer torch transitively even with `torch==2.1.1` installed
   first; ended up on `torch==2.13.0+cu130` (same as `se_probes_v5`). Verified this doesn't break
   checkpoint loading (`torch.load` compat check passed) before trusting any result. Also copied
   `deploy_checkpoints/` (~1.5GB) to `/data2/mn1025/stage2_checkpoints/` so `torch.load` on the
   ~100MB shards doesn't hit NFS either.
2. **`arm_preds`'s checkpoint glob didn't match `deploy_checkpoints`' filenames** — it looks for
   `{arm}_seed*.pt`, but deploy's files are `deploy_{arm}_seed*.pt` (a naming convention the E42
   `ckpt_dir` param never needed to handle, since REF/E22 checkpoints have no prefix) → silent
   `paths[0]` `IndexError` on an empty glob result. **Fix:** glob pattern widened to
   `*{arm}_seed*.pt` (verified the leading wildcard can't false-match a *longer* arm name's file,
   e.g. `z_q_seed` never matches the `z_seed` pattern).
3. **`deploy_checkpoints` (also `exp2_run.py`-saved, same era as E39's `k`/`transform`-omission
   bug) additionally omit `position`/`layer` in their meta** — `load_checkpoint` (the *generic*
   loader `arm_preds` uses, distinct from `correctness_eval_ood.py`'s narrower `_load_exp2_ckpt`
   compat shim) had no fallback for any of the four. **Fixed in the shared loader itself** (not
   just one caller): `meta.get("k", cfg.k_soft_tokens)`, `meta.setdefault("position",
   cfg.selected_position)`, `meta.setdefault("layer", cfg.selected_layer)`, and
   `transform = TargetTransform(0.0, 1.0)` (identity) when `meta["transform"]` is absent —
   AUROC/rank metrics are invariant to that identity fallback, only `.decode()`'s absolute scale
   would differ, and this run never uses decoded values. Now every caller of `load_checkpoint`
   (not just `arm_preds`) tolerates exp2-line checkpoints.
4. **`Qwen3.5-9B`'s manifest was stale at 46/1000 records** (the exact write-once-at-batch-end
   trap flagged in E44's own CLAUDE.md note, hit for real this time): a later resume run that only
   rebuilt the ~46 hardest-tail stragglers called `write_manifest()` with just *its own* batch's
   entries, overwriting the earlier complete-1000 manifest rather than merging into it — even
   though all 1000 `.pt` files were genuinely present and undamaged on disk. **Fixed
   mechanically, not guessed:** rebuilt `manifest.json` by loading all 1000 records directly
   (`load_record`) and re-deriving entries via the pipeline's own `manifest_entry`/
   `write_manifest`, after confirming zero corrupt files. No `.pt` data was touched.

**Result (AUROC_incorrect, N=1000/target, deploy proxy zero-shot):**

| target | true SE | q_only | q_resp_only | Δ(q_resp_only − true SE) |
|---|---|---|---|---|
| Qwen3-8B | 0.787 | 0.777 | **0.840** | **+0.053 [+0.030, +0.077]** — excludes 0, beats |
| Qwen3.5-9B | 0.810 | 0.765 | 0.818 | +0.009 [−0.012, +0.031] — includes 0, on par |
| gemma-7b-it | 0.771 | 0.753 | **0.848** | **+0.078 [+0.052, +0.104]** — excludes 0, beats |
| gemma-2-9b-it | 0.769 | 0.704 | 0.722 | **−0.047 [−0.075, −0.020]** — excludes 0, loses |

`q_resp_only` also beats `q_only` on every target (deltas +0.018 to +0.095, excludes 0 on 3/4) —
the response text, not just the question, is where the signal is on every target, consistent with
every OOD/cross-LLM result since E20.

**Findings.**
1. **The thesis extends to genuinely new vendors/architectures, but not uniformly.** 2/4 targets
   (Qwen3-8B, gemma-7b-it) show the zero-shot proxy *significantly beating* true 10-sample
   sampling-based SE — stronger than anything seen even for the proxy's own training families
   (E38's best was "on par", never "beats"). 1/4 (Qwen3.5-9B) replicates the "on par" pattern.
   1/4 (gemma-2-9b-it) is a **significant loss**, the first target anywhere in the project where
   `q_resp_only` is clearly worse than true SE.
2. **`q_resp_only` never drops below `q_only` on any target** — the model-agnostic text pathway
   is never actively harmful, matching E20-23/E37-39's universal finding that response text
   carries transferable signal.
3. **gemma-2-9b-it is an outlier on its own stats, not just on the delta:** mean accuracy 0.684
   and incorrect-rate 0.316, both far outside the other 3 targets' range (mean_acc 0.42-0.56,
   incorrect-rate 0.44-0.58) — a model that's simply *better* at trivia_qa has fewer wrong answers
   to detect, which could explain a weaker/noisier signal independent of any cross-family transfer
   failure. **Flagged as a hypothesis, not established** — would need a matched-difficulty or
   matched-base-rate re-analysis to confirm vs. a genuine family-transfer effect.

**Caveats.** Single run per target (the 3-seed ensemble mean is baked into `arm_preds`, but there's
no additional resampling across question subsets); z-arm not run (would need a label-free
Procrustes fit per new target first — natural next step per the original E45 scoping conversation);
`gemma-2-9b-it`'s negative result is reported as-is, not yet root-caused past the accuracy-outlier
hypothesis above; the 3 infra bugs fixed here (glob pattern, checkpoint meta fallback, manifest
rebuild) were all latent and could plausibly affect other not-yet-run E42-style
`ckpt_dir`-override or exp2-line-checkpoint use cases — worth a quick audit if one comes up.

**Artifacts.** `amortized_ue/e45_qwen_gemma_zeroshot.py`,
`amortized_ue/results/e45_qwen_gemma_zeroshot.json`, `amortized_ue/logs/e45_qwen_gemma_zeroshot.log`.
Additive fixes: `amortized_ue/procrustes_e27_rank_fusion.py` (`arm_preds` glob pattern + new
`data_dir` param), `amortized_ue/stage2/checkpoint.py` (`load_checkpoint` meta fallbacks),
`amortized_ue/stage2/config.py`/`stage2/data.py` (`stage1_output_dir` override plumbing). New env
`amortized_stage2_v5` (`/data2/mn1025/conda_envs/amortized_stage2_v5`, NFS-free, versions pinned
to match the live `amortized_stage2` conda env) + checkpoint copy at
`/data2/mn1025/stage2_checkpoints/deploy_checkpoints/`.

## E46

**Goal:** E45 scored each of the 4 new Qwen/Gemma targets against its OWN correctness labels, in
isolation — strong AUROCs there are consistent with the proxy just reading question difficulty +
each model's own answer-text tells, without ever checking whether it can tell "model A is
uncertain here but model B isn't" on the SAME question. This asks that directly: does
`q_resp_only` distinguish genuine CROSS-MODEL disagreement, not just within-model correctness?

**Method (additive, `e46_qwen_gemma_pairwise_disagreement.py`), following E40's design but
simpler.** For every one of the 6 pairs among the 4 targets, on the shared 1000 question ids:
`dY = SE_A − SE_B` (or `incorrect_A − incorrect_B` for the correctness framing), `dP = q_resp_only(A)
− q_resp_only(B)`, then `rho(dP, dY)` (paired bootstrap CI) and pairwise accuracy
(`sign(dP)==sign(dY)`) on rows where the two targets' correctness genuinely diverges. **Crucially,
E40's negative-null correction does NOT apply here** — that correction existed because E40's
pooled ridge was trained via leave-one-out (asymmetric: 3 models in training, 1 held out), so a
predictor with zero real signal was provably anti-correlated with the true gap by construction.
Here, the deploy proxy was trained ONLY on Llama-2/Mistral/Llama-3/DeepSeek — **none** of the 4
new targets were in its training set, so every pair is symmetric (both members equally unseen);
the null really is 0, no correction needed. `q_only`'s `dP` is included only as a determinism
sanity check (identical input across models ⇒ should be exactly 0), not a statistical null.

**Result (all 6 pairs, `q_resp_only`):**

| pair | SE-gap corr (95% CI) | pairwise accuracy on divergent rows | n |
|---|---|---|---|
| Qwen3-8B vs Qwen3.5-9B | +0.259 [+0.198,+0.321] | 69.1% | 191 |
| Qwen3-8B vs gemma-7b-it | **+0.417** [+0.357,+0.474] | **80.5%** | 262 |
| Qwen3-8B vs gemma-2-9b-it | +0.268 [+0.198,+0.337] | 74.3% | 249 |
| Qwen3.5-9B vs gemma-7b-it | +0.325 [+0.265,+0.384] | 78.5% | 275 |
| Qwen3.5-9B vs gemma-2-9b-it | +0.225 [+0.160,+0.290] | 75.5% | 200 |
| gemma-7b-it vs gemma-2-9b-it | +0.323 [+0.260,+0.385] | 79.7% | 305 |

`q_only`'s `dP` was identically 0 for every question on every pair, exactly as predicted (sanity
check passed).

**Findings.**
1. **Every pair is significant and clearly above chance** — no cherry-picking, this is all 6 of
   6 possible pairs among the 4 targets, not a selected subset.
2. **Substantially stronger than the closest prior test.** E40 ran the analogous check on the
   *original* 4 models using the aligned hidden-state pathway and found only 51.5% pairwise
   accuracy — not statistically significant, barely above chance. Here: 69–81%, clean and
   consistent. Two likely reasons: (a) this uses the *response text* pathway, which E40 itself
   flagged as "far more model-specific than the aligned hidden state" (+0.237 vs +0.090); (b) no
   LOO asymmetry to correct for here (see Method).
3. **No family clustering** — Qwen-vs-Qwen (0.259) is *weaker* than several cross-vendor pairs
   (e.g. Qwen3-8B vs gemma-7b-it at 0.417), consistent with E30/E40's "alignability tracks CKA,
   not family" pattern, though CKA itself hasn't been computed for Qwen/Gemma yet.
4. **Concrete examples pulled** (`e46_examples.py`, Qwen3-8B vs gemma-7b-it, the strongest pair):
   211/262 divergent rows called correctly (matches the aggregate 80.5% exactly, a useful internal
   consistency check). Illustrative misses are almost all off-topic or made-up-sounding answers
   ("the jem and the holograms" for a ThunderCats question; "luna" for a locomotive) correctly
   flagged as the more-uncertain side.

**Caveats.** 6 pairs from only 4 targets are not independent draws (each target appears in 3
pairs), so treat the 6-pair table as one coherent picture, not 6 independent replications; no
z-arm (no alignment fit for Qwen/Gemma yet); bootstrap CIs are per-pair, no multiple-comparison
correction applied (unnecessary here — every CI is far from 0, correction would not change any
conclusion).

**Artifacts.** `amortized_ue/e46_qwen_gemma_pairwise_disagreement.py`,
`amortized_ue/e46_examples.py`, `amortized_ue/results/e46_qwen_gemma_pairwise.json`,
`amortized_ue/results/e46_examples.json`.

## E47

**Goal:** E45/E46 both measure *ranking* quality (AUROC, pairwise accuracy) — neither directly
answers "how well does the proxy's raw score track the actual continuous SE value," the project's
primary SE-fidelity metric used throughout E12–E37. Per E31, SE-fidelity and correctness are
established to be different things, not two views of the same number — so this needed its own
check, not an inference from E45/E46.

**Method (additive, `e47_qwen_gemma_se_fidelity.py`).** Spearman `rho(q_resp_only, true_SE)` per
target (bootstrap CI), plus `q_only` for comparison, on the same 4 targets/1000-id records as
E45/E46.

**Result:**

| target | `q_only` rho | `q_resp_only` rho (95% CI) |
|---|---|---|
| Qwen3-8B | 0.601 | **0.719** [0.684,0.750] |
| Qwen3.5-9B | 0.662 | **0.749** [0.721,0.774] |
| gemma-7b-it | 0.537 | **0.670** [0.632,0.705] |
| gemma-2-9b-it | 0.628 | **0.674** [0.638,0.707] |

**Findings.**
1. **SE-fidelity is strong on all 4 new targets (0.67–0.75) — at or above the proxy's own
   training-family benchmark.** E37 reported `q_resp_only` Spearman ≈0.648 (mean, leave-one-out)
   on the 4 *training* models. Zero-shot on brand-new families, the proxy matches or beats its own
   in-training-distribution number.
2. **gemma-2-9b-it's SE-fidelity (0.674) is essentially on par with gemma-7b-it's (0.670)** —
   despite gemma-2-9b-it being the one target where E45 found a significant correctness-detection
   *loss* (0.722 vs true SE's 0.769). This is direct, target-specific evidence for E31's finding
   that SE-fidelity ≠ correctness: the proxy reads gemma-2-9b-it's uncertainty pattern just as well
   as the others; the weaker link is specifically between gemma-2-9b-it's own SE and its own
   correctness (plausibly related to it being the accuracy outlier — mean_acc 0.684 vs 0.42–0.56
   for the other 3), not a proxy failure.
3. **Concrete examples** (`e47_examples.py`, Qwen3-8B vs gemma-7b-it, top-10 by |true SE gap|):
   9/10 correct direction (matches E46's 80.5% pairwise accuracy within noise). The one miss is
   diagnostic: Qwen3-8B answered "Last Tango in Paris" director as "francis ford coppola" —
   **wrong but confidently/consistently so (true SE=0.00)** — while gemma-7b-it answered correctly
   but with high sample-to-sample variance (true SE=2.16). The proxy predicted the reverse
   (Qwen higher, gemma lower) — it read Qwen's answer as *wrong-sounding* and inferred high
   uncertainty, missing that the model was actually confident. This is the proxy's structural
   blind spot: it sees one response, never the 10 samples that define true SE, so it cannot
   observe resampling consistency directly — only infer a correlate from single-answer plausibility.
4. **Scale caveat, not a finding:** decoded "predicted SE" values shown in the examples are
   rescaled onto each target's own true-SE mean/std for readability only (the deploy checkpoint's
   original decode stats are genuinely unrecoverable — see E45 finding #2's root cause, clarified
   further below) — rank and relative gaps are the trustworthy part; absolute numbers are
   illustrative.

**Root cause of the missing decode scale (clarified, not just worked around).** Checked
`exp2_run.py`'s current (already-partially-fixed-post-E39) checkpoint-save code directly: it
explicitly writes `"transform": {"mean": 0.0, "std": 1.0}` — an *intentional* identity placeholder,
not an oversight. The comment explains why: targets are z-scored **per source model** before
pooling into one training set (so DeepSeek's naturally-higher SE scale doesn't dominate the
shared proxy, the [[pooling-per-model-normalization]] lesson from E35). There is no single
absolute SE scale for a pooled multi-source proxy to save — the identity transform is described in
the code itself as "the honest record" of that fact. **Also newly noticed: the same fixed code
still omits `position`/`layer` in meta** — any *future* exp2-line checkpoint would hit the exact
crash `load_checkpoint`'s E45 fallback now silently absorbs; not yet fixed at the source.

**Caveats.** No ceiling computed for these 4 targets (the ~0.90 label-noise ceiling from prior
work is target-specific and hasn't been re-derived here, so "% of ceiling recovered" can't be
stated, only the raw rho); single run, no seed variance beyond the 3-seed ensemble baked into
`arm_preds`; the rescaling used for readable examples uses eval-time target statistics the proxy
itself never has access to — do not read it as evidence of proxy calibration.

**Artifacts.** `amortized_ue/e47_qwen_gemma_se_fidelity.py`, `amortized_ue/e47_examples.py`,
`amortized_ue/results/e47_qwen_gemma_se_fidelity.json`, `amortized_ue/results/e47_examples.json`.

## E48

**Goal:** a sharp methodological worry raised mid-session: is the LoRA fine-tuning on our
SE-labeled data actually contributing anything on Qwen/Gemma, or is the strong zero-shot
performance mostly just Llama-3.2-3B's own pretrained factual knowledge leaking through — a
fact-checker the backbone already had before any SE-specific training, with our training adding
little? The existing TF-IDF-vs-3B control (E12/E13) tests a *different* axis (is the 3B smarter
than keyword-matching); nothing in the project isolates whether the LoRA training itself, on OUR
SE labels, is doing real work.

**Method (additive, `e48_frozen_backbone_baseline.py`).** Skip the trained proxy. Load the SAME
frozen backbone (`meta-llama/Llama-3.2-3B`) with **no LoRA, no projector, no head, no training at
all** — just the raw pretrained model. Score it with a standard few-shot "Is this answer True or
False" prompt (the classic `p_true` self-verification format already used in the SEP baseline —
`semantic_uncertainty/uncertainty/uncertainty_measures/p_true.py`, reused as a template only,
adapted here as a cross-model judge rather than self-verification). Few-shot examples drawn from
Llama-2's own records (disjoint from every evaluated target). Read `P(token=" B" i.e. "False")`
from the next-token logits after "Answer:" as the predicted-incorrectness score — no generation,
one forward pass per question, no training whatsoever. Scored with the exact same
AUROC_incorrect/SE-fidelity-rho metrics as E45/E47 for direct comparison.

**Result:**

| target | untrained AUROC | **trained AUROC** (Δ) | untrained SE-rho | **trained SE-rho** (Δ) |
|---|---|---|---|---|
| Qwen3-8B | 0.760 | **0.840** (+0.080) | 0.477 | **0.719** (+0.242) |
| Qwen3.5-9B | 0.752 | **0.818** (+0.066) | 0.509 | **0.749** (+0.240) |
| gemma-7b-it | 0.797 | **0.848** (+0.051) | 0.410 | **0.670** (+0.260) |
| gemma-2-9b-it | 0.667 | **0.722** (+0.055) | 0.390 | **0.674** (+0.284) |

**Findings.**
1. **Training clearly adds real, substantial signal — the worry does not hold as stated.** The
   trained proxy beats the untrained backbone on every target, on both metrics, with **zero
   exceptions**.
2. **But the worry wasn't baseless either — the untrained backbone alone is well above chance**
   (0.67–0.80 AUROC vs 0.50 chance), confirming pretrained knowledge *does* carry real,
   independent signal. Training is additive on top of a real baseline, not manufacturing signal
   from nothing.
3. **The SE-fidelity gap (+0.24 to +0.28) is far larger and much more uniform across targets than
   the AUROC gap (+0.05 to +0.08).** Interpretation: the untrained backbone gives a coarse,
   roughly binary "does this look right" signal; the SE-labeled training data teaches something
   *graded* on top — translating "looks wrong" into a properly scaled *degree* of uncertainty that
   tracks the continuous entropy structure. That graded skill is what shows up as the large,
   consistent rho gap, and it transfers to models never seen in training.
4. **gemma-2-9b-it is the weakest target for BOTH the untrained baseline (0.667/0.390, both the
   lowest of the 4) and, per E45, the trained proxy's correctness edge over true SE** — consistent
   with gemma-2-9b-it being a genuinely harder-to-judge target in general (its own higher accuracy
   / lower incorrect-rate, per E45's finding #3), not a training-specific weakness.

**Caveats.** `p_true`-style prompting of a *base* (non-instruct) 3B model is a coarse instrument —
few-shot format-following from a non-chat model is noisier than an instruction-tuned judge would
be, so the untrained baseline's numbers are plausibly a slight underestimate of "best achievable
from pretrained knowledge alone" (a stronger untrained baseline would only shrink the measured
gap, not reverse the conclusion given its current size). Single run, 4-example few-shot prompt not
tuned/ablated. This is a different information setting than the classic `p_true` (which also reads
the brainstormed high-temperature samples, not just one answer) — deliberately restricted to one
answer here to match `q_resp_only`'s own input exactly, for a fair comparison.

**Artifacts.** `amortized_ue/e48_frozen_backbone_baseline.py`,
`amortized_ue/results/e48_frozen_backbone_baseline.json`.

## E49

**Goal:** while sanity-checking the squad build's slow pace for Qwen3.5-9B (9.2× slower than its
own trivia_qa build, see infra note below), found that the trivia_qa eval set used throughout
E45-E48 has 58/1000 (5.8%) records where `canonical_response` is literally `"<think>"` or
`"<think>\n\n</think>"` — the model exhausted its 250-token thinking budget without producing a
real answer (the documented "hardest tail," now precisely counted; the earlier "~46" figure was an
estimate). User asked directly: do E45-E48's Qwen3.5-9B numbers need to be redone excluding these?

**Method (additive, `e49_qwen35_9b_think_leak_check.py`).** Re-scored all 4 predictors used across
E45/E47/E48 (true SE, `q_only`, `q_resp_only`, frozen-backbone `p_false`) on the full 1000 rows AND
on the clean 942 (excluding the 58), side by side — AUROC_incorrect and SE-fidelity rho both ways.

**Result:**

| predictor | AUROC (all 1000) | AUROC (clean 942) | Δ | rho (all) | rho (clean) | Δ |
|---|---|---|---|---|---|---|
| true SE | 0.810 | 0.787 | −0.022 | 1.000 | 1.000 | — |
| `q_only` | 0.765 | 0.746 | −0.020 | 0.662 | 0.633 | −0.030 |
| `q_resp_only` | 0.818 | 0.797 | −0.021 | 0.749 | 0.724 | −0.025 |
| frozen backbone | 0.752 | 0.729 | −0.023 | 0.509 | 0.468 | −0.041 |

**Findings.**
1. **Removing the 58 rows makes every number WORSE, not better — the opposite of the initial
   hypothesis.** Before running this, the working theory (by analogy to E47's "Last Tango in
   Paris" confidently-wrong example) was that these degenerate rows would be "confidently wrong"
   (low SE, inflating everyone's apparent skill). Checked directly: **mean true SE for the 58 bad
   rows is 2.00, more than double the clean rows' 0.97** — the opposite of confidently wrong.
   Likely cause: the entailment model can't cleanly judge two `"<think>"` fragments as
   semantically equivalent (there's no real claim to entail), so instead of clustering into one
   low-entropy group, they scatter into many small clusters — high, not low, computed entropy.
2. **These rows are the easiest wrong-answer cases in the dataset, not the hardest.** High SE +
   accuracy=0 (all 58) is a trivial case for every predictor. Removing them leaves a harder
   remaining pool, so every metric drops by a small, remarkably uniform ~0.02-0.04 — consistent
   with removing "free points" rather than removing noise.
3. **E45-E48's original Qwen3.5-9B numbers (on all 1000) stand as reported — no correction
   needed.** The relative ordering between predictors is essentially unchanged either way (the
   drop is uniform across all four), so no headline claim from E45-E48 is affected.

**Caveats.** Single target only (Qwen3.5-9B — the one target with this failure mode at meaningful
scale); doesn't rule out a *different* mechanism at play for a hypothetical model with truly
"confidently wrong" degenerate outputs (that would need its own check, not assumed from this one).

**Infra fixed alongside this (not part of the experiment, but discovered while investigating it):**
1. **Stale manifest metadata for `Qwen3.5-9B_trivia_qa_n1000_full`, found and fixed.** The
   manifest's nested `meta.mean_accuracy`/`meta.mean_cluster_assignment_entropy`/`meta.n_records`
   still reported `0.0217`/`2.053`/`46` — leftover from an earlier manifest rebuild (fixing the
   documented [[stage1-manifest-write-once-trap]]) where the 1000 individual record *entries* were
   correctly rebuilt from disk, but the summary-stats fields in `meta` were carelessly copied
   verbatim from the old 46-record partial manifest instead of being recomputed. **The actual
   per-record data was never wrong** — all of E45-E48 read per-record fields directly, never this
   summary, so no result was affected — but the summary now correctly reads `mean_accuracy=0.56`
   (matching E45's own live log at the time), `n_records=1000`. Fixed by recomputing from the 1000
   records already on disk, no `.pt` files touched.
2. **`squad_v2` dataset loading broke under `se_probes_v5`'s newer `huggingface_hub`.**
   `semantic_uncertainty/uncertainty/data/data_utils.py:13` called
   `datasets.load_dataset("squad_v2")` (legacy unnamespaced short form) — the newer
   `huggingface_hub` in `se_probes_v5` rejects it (`HfUriError`), which is why the squad-for-new-
   models attempt failed immediately, 4/4, on first try. **Fixed** (stopped and asked first, per
   this repo's rule for anything under `semantic_uncertainty/uncertainty/`): changed to the
   fully-qualified `datasets.load_dataset("rajpurkar/squad_v2")`. Verified byte-identical
   (130,319 train examples) under BOTH the old pinned `se_probes` env and the new `se_probes_v5`
   before applying, so this doesn't risk the existing Llama-2/Mistral squad reproducibility.
3. **Built squad n1000 OOD test data for all 4 small-tier Qwen/Gemma targets** (same recipe as the
   existing Llama-2/Mistral squad sets — no `--only_ids`, default `random_seed=10` reproduces the
   identical squad question selection) — `Qwen3-8B`, `Qwen3.5-9B`, `gemma-7b-it`, `gemma-2-9b-it`,
   all 1000/1000, on GPU0 in parallel with the big-tier n1000 queue on GPU1 (zero contention, GPU0
   was otherwise idle). Qwen3.5-9B's squad build took 9.2× longer than its own trivia_qa build
   (188.7 vs 20.6 min) — squad questions are harder on average (lower base accuracy across every
   model tested so far), which triggers Qwen3.5-9B's long-`<think>` behavior far more often; the
   other 3 targets were only 1.2-1.4× slower, in line with squad's generally longer contexts.
4. **Big-tier queue parallelized across both GPUs.** `gemma-2-27b-it`/`gemma-3-27b-it` pulled out
   of the sequential GPU1 queue (`build_big_tier_n1000.sh`, still running Qwen3.5-27B →
   Qwen3.6-27B → Qwen3.8-27B unmodified) into a new independent lane
   (`build_gemma_bigtier_gpu0.sh`) that starts the moment GPU0 frees up, running single-GPU (not
   the originally-planned dual-GPU attempt — GPU1 is fully occupied by the Qwen queue so dual-GPU
   isn't available as an option right now regardless, and this sidesteps that untested risk for a
   new architecture). Roughly halves the big-tier queue's total wall time.

**Artifacts.** `amortized_ue/e49_qwen35_9b_think_leak_check.py`,
`amortized_ue/results/e49_qwen35_9b_think_leak_check.json`,
`amortized_ue/build_small_tier_squad_n1000.sh`, `amortized_ue/build_gemma_bigtier_gpu0.sh`. One-line
dependency-fix change to `semantic_uncertainty/uncertainty/data/data_utils.py` (dataset repo id
only, no SE/probe logic touched).

## E51 — the direct proxy-vs-SEP SE-fidelity head-to-head, across every regime built so far — ✅ proxy beats SEP on Spearman in 13/14 settings, ties on 1/14, never loses

**Goal:** every prior script scored the `q_resp_only` proxy and SEP against SE **separately** (E37's
`te_spearman`, E47's rho) or compared them on a **different** target (correctness,
`auroc_incorrect` — E38/E39/E45), but nothing put the two side by side on the SAME held-out rows
against the SAME continuous SE label with a paired-bootstrap CI on the delta. User asked for
exactly that: one script, one final table, per-seed AND ensemble reported separately, across every
regime the project has already built data for (LOLO, squad OOD, fresh trivia, Qwen/Gemma
zero-shot) — using the E41-corrected SEP layers, retraining nothing.

**Method (additive, `amortized_ue/se_fidelity_proxy_vs_sep.py`).** Four settings, each scored with
the same recipe: **Spearman(pred, continuous SE)** and **AUROC(pred, high-vs-low SE)** with the
`best_split` threshold fit on the FIT-side TRAIN split only (never on eval — the standing
convention throughout the repo), plus a paired bootstrap (10,000 resamples, one shared index set
reused across every predictor) giving a 95% CI on (proxy − SEP) for both metrics. SEP predictions
are id-joined onto the proxy's rows (never positional).
1. **LOLO trivia_qa** (Llama-2/Mistral/Llama-3/DeepSeek): proxy = the E37/E43 leave-one-LLM-out
   checkpoints' saved per-seed `te_pred_by_seed` (no proxy forward pass needed — CPU/`se_probes`
   only); SEP = E41 fixed-layer (`exp2_run.BEST_TBG`), fit on the target's own n2000 train split,
   evaluated on the identical 200 `te` rows (id-mapping re-audited: max deviation `0.000e+00` on
   all 4 folds).
2. **Squad OOD** (Llama-2 + Mistral, the only 2 targets with squad records): DEPLOY proxy
   (all-4-pooled, trivia-trained) run fresh on squad via a new `arm_preds_per_seed` (a copy of
   `procrustes_e27_rank_fusion.arm_preds` that keeps every seed's prediction instead of only the
   mean — same checkpoints, same forward pass, no retraining); SEP fit on trivia n2000, evaluated
   OOD on squad (mirrors E39's setup exactly).
3. **Fresh trivia n1000** for all 4 training models: DEPLOY proxy vs SEP, both fit/eval on
   genuinely disjoint id sets. Verified on-disk that **Llama-3 now has a fresh n1000 with 0 id
   overlap against its n2000 training set** — the earlier `correctness_eval.py` TARGETS-dict note
   ("Llama-3: eval=test split, no fresh n1000 exists") is **stale**; the fresh set has since been
   built and all 4 models get the fresh-set treatment.
4. **Qwen/Gemma zero-shot** (Qwen3-8B, Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it — never in the deploy
   proxy's training pool): SEP here is a **genuinely fair, target-specific** probe — fit on that
   model's own n2000 **training** tier, evaluated on its **disjoint** n1000 eval tier (0 id overlap
   confirmed programmatically for all 4, per E44's split). No E41/E36 CV-picked layer exists yet
   for these families, so the layer is chosen by leak-free **validation**-selection
   (`sep_single_val_selected`, selects on the fit-side val split, never on eval) rather than a
   fixed CV layer — flagged explicitly per target, since this is a noisier selection than the
   fixed-layer SEP used in settings 1-3 (the exact failure mode E41 fixed for the original 4
   models).

**Result (ensemble vs SEP; ρ = Spearman, AU = AUROC-vs-SE; bold = CI excludes 0):**

| setting | target | SEP ρ | proxy ρ | Δρ [95% CI] | SEP AU | proxy AU | ΔAU [95% CI] |
|---|---|---|---|---|---|---|---|
| LOLO | Mistral | 0.599 | 0.658 | +0.056 [−0.03,+0.15] | 0.865 | 0.854 | −0.011 [−0.07,+0.04] |
| LOLO | Llama-3 | 0.518 | 0.643 | **+0.114** [+0.01,+0.22] | 0.839 | 0.874 | +0.034 [−0.04,+0.11] |
| LOLO | DeepSeek | 0.597 | 0.703 | **+0.104** [+0.01,+0.20] | 0.812 | 0.862 | +0.050 [−0.01,+0.11] |
| LOLO | Llama-2 | 0.424 | 0.701 | **+0.266** [+0.15,+0.39] | 0.778 | 0.862 | **+0.084** [+0.01,+0.16] |
| squad OOD | Llama-2 | 0.236 | 0.590 | **+0.352** [+0.29,+0.41] | 0.622 | 0.797 | **+0.175** [+0.13,+0.22] |
| squad OOD | Mistral | 0.425 | 0.594 | **+0.168** [+0.12,+0.22] | 0.686 | 0.812 | **+0.126** [+0.08,+0.17] |
| fresh trivia | Llama-2 | 0.523 | 0.660 | **+0.131** [+0.09,+0.17] | 0.779 | 0.863 | **+0.083** [+0.06,+0.11] |
| fresh trivia | Mistral | 0.548 | 0.653 | **+0.096** [+0.05,+0.14] | 0.834 | 0.891 | **+0.057** [+0.03,+0.09] |
| fresh trivia | Llama-3 | 0.596 | 0.652 | **+0.052** [+0.01,+0.09] | 0.843 | 0.872 | **+0.028** [+0.00,+0.05] |
| fresh trivia | DeepSeek | 0.583 | 0.764 | **+0.178** [+0.14,+0.22] | 0.805 | 0.901 | **+0.097** [+0.07,+0.12] |
| Qwen/Gemma | Qwen3-8B | 0.623 | 0.719 | **+0.089** [+0.05,+0.13] | 0.867 | 0.910 | **+0.042** [+0.02,+0.07] |
| Qwen/Gemma | Qwen3.5-9B | 0.700 | 0.749 | **+0.049** [+0.02,+0.08] | 0.874 | 0.893 | +0.019 [−0.00,+0.04] |
| Qwen/Gemma | gemma-7b-it | 0.509 | 0.670 | **+0.157** [+0.11,+0.21] | 0.760 | 0.838 | **+0.078** [+0.05,+0.11] |
| Qwen/Gemma | gemma-2-9b-it | 0.623 | 0.674 | **+0.047** [+0.01,+0.08] | 0.853 | 0.885 | **+0.031** [+0.01,+0.06] |

Per-seed numbers (all settings) sit 0.02-0.07 below the ensemble on both metrics, consistently —
e.g. squad/Llama-2 individual seeds ρ 0.505-0.605 vs ensemble 0.590; full per-seed table in
`results/se_fidelity_proxy_vs_sep.json`.

**Findings.**
1. **The proxy beats SEP on Spearman in 13/14 settings (every CI excludes 0, every delta
   positive) and ties on the 14th (Mistral-LOLO, CI includes 0 but still positive) — it never
   loses on SE-fidelity, in any regime tested.** On AUROC-vs-SE the picture is slightly softer:
   10/14 CIs exclude 0 (all positive), 4 include 0 (Mistral-LOLO, Llama-3-LOLO, DeepSeek-LOLO,
   Qwen3.5-9B) — **AUROC is the noisier of the two metrics here**, consistent with it collapsing a
   continuous relationship into a binary threshold at N as low as 200 (the LOLO rows).
2. **The proxy's edge is LARGEST exactly where SEP is weakest: cross-dataset and cross-LLM
   shift.** Squad OOD gives the two biggest deltas in the whole table (Δρ +0.352 Llama-2, +0.168
   Mistral) — SEP degrades badly under dataset shift (Llama-2 SEP ρ collapses to 0.236, matching
   E39's finding that SEP degrades more than the proxy under shift) while the proxy holds up
   (0.590). LOLO-Llama-2 is the next largest (+0.266) — SEP's Llama-2 layer is the one E41 flagged
   as an outlier/high-variance pick, so this table's Llama-2 rows are consistent with, not
   independent evidence beyond, that known SEP weakness.
3. **The result generalizes to a zero-shot regime with NO training-pool overlap at all.** The
   Qwen/Gemma deltas (+0.047 to +0.157 ρ) are smaller than squad's but every one is positive, and 3
   of 4 have CIs excluding 0 on both metrics — even against a SEP that is fit and evaluated
   entirely within that same unseen model family (the fairest possible SEP baseline for those
   targets).
4. **Smallest gaps cluster on the targets where SEP itself is already strong** (Mistral-LOLO SEP ρ
   0.599 — the highest SEP score of any LOLO row; Qwen3.5-9B SEP ρ 0.700, AU 0.874 — the highest
   Qwen/Gemma SEP score) — the proxy's advantage shrinks, but never reverses, as the supervised
   in-model baseline gets better. Consistent with the project's running theme (E27/E33/E38): the
   label-free/model-agnostic pathway is most valuable exactly where a per-model supervised probe
   is weakest, and merely competitive (not dominant) where the probe is already strong.
5. **This closes a specific methodological gap E38 left open** — `correctness_eval_e37.py` already
   computed `auroc_binarised_se`/`spearman_se` per predictor per fold, but never bootstrapped a
   proxy-vs-SEP delta on the SE label itself (only on `incorrect`). This experiment is the first
   place that comparison exists, for SE-fidelity specifically, across all four LLM-family regimes
   the project has built.

**Caveats.** Qwen/Gemma SEP uses val-selection, not a fixed CV layer (no E36-style multi-fold CV
has been run for these families yet — a fixed layer would need that first, mirroring how E41 fixed
the original 4). LOLO is N=200/fold (widest CIs in the table); squad/fresh/Qwen-Gemma are
N=1000. AUROC-vs-SE is inherently noisier than Spearman at these sample sizes — treat the 4
CI-includes-0 AUROC rows as "not distinguishable from SEP on this metric," not as evidence of a
real proxy weakness (their Spearman deltas are still positive, 2 of the 4 still excluding 0).

**Infra note (not part of the experiment).** Both GPUs were saturated with live Stage-1 builds
(`Qwen3.5-27B` on GPU1 at 731/2000 records, `gemma-3-27b-it` on GPU0 at only 132/2000) when the
GPU-dependent settings (2-4) needed to run. With explicit user go-ahead, paused the
least-progressed job (`gemma-3-27b-it`, SIGTERM — the build is resumable, records save
incrementally, `--overwrite` not passed means nothing already on disk is redone), ran settings 2-4
on the freed GPU0, then relaunched the identical `build_gemma3_n2000_gpu0.sh` command
(`nohup ... &`, disowned) once done. Confirmed it resumed from 132/2000 (not from scratch) before
moving on.

**Artifacts.** `amortized_ue/se_fidelity_proxy_vs_sep.py`,
`amortized_ue/results/se_fidelity_proxy_vs_sep.json` (includes the full per-seed breakdown and the
`_final_table` summary rows). Summary metrics logged to W&B (`amortized_ue/log_e51_wandb.py`,
project `amortized_ue_stage2`, run `E51_proxy_vs_sep_se_fidelity` — no new dataset/checkpoint, this
run only carries the comparison table + per-setting deltas for tracking).

## E52 — the LOLO proxy (never saw this target's data at all) tested on squad OOD — fills the one combination E51 left untested; ✅ proxy still beats SEP under the hardest combined shift yet

**Why.** E51's table has a `lolo` setting (LOLO proxy, but only ever eval'd on trivia_qa, same
dataset it was trained on across the other 3 models) and a `squad` setting (squad OOD, but with the
DEPLOY proxy, which pools ALL 4 models including the target — so squad is the only shift). Nobody
had run the LOLO proxy — trained on the *other* 3 targets, zero exposure to this target's data in
any form — **and** evaluated it on squad, a dataset it also never saw. User asked directly whether
this combination had been tested; it had not. This is the single hardest transfer regime available
in the project: simultaneously cross-LLM (never this target) and cross-dataset (never this
distribution), with no fitting/calibration on either axis.

**Method (additive, extends `se_fidelity_proxy_vs_sep.py` with a new `lolo_squad` setting).** Reuses
every piece of existing infrastructure — nothing retrained. For each of the 2 targets with squad
records (Llama-2, Mistral — the only ones with squad n1000 built): load that target's E37/E43 LOLO
fold checkpoints (`q_resp_only` arm, 3 seeds, trained on the other 3 models' trivia_qa only, saved at
`amortized_ue/stage2/runs/E37_LOLO_ckpt/checkpoints/`) and run a fresh forward pass on the target's
squad n1000; score against the same E41 fixed-layer SEP used throughout E51 (fit on the target's own
trivia n2000, evaluated OOD on squad — identical SEP recipe to E51's `squad` setting, so the SEP
column is directly comparable across settings). New helper `arm_preds_per_seed_prefixed` — the LOLO
checkpoint directory holds all 4 folds' files together (`<HeldOutTarget>_<arm>_seed<N>.pt`), so
`arm_preds_per_seed`'s glob (`*{arm}_seed*.pt`) would silently pull in all 4 targets' checkpoints;
the new helper filters the glob to `{target_prefix}_{arm}_seed*.pt` first. Same scoring
(`score_block`, paired bootstrap, 10,000 resamples) as every other E51 setting.

**Infra.** Both GPUs were saturated with live big-tier Stage-1 builds when this needed a GPU
(`gemma-3-27b-it` on GPU0 at 1089/2000, only ~3.3GB free; `Qwen3.6-27B` on GPU1 at 498/2000, ~8.4GB
free — the thinner-margin GPU1 job was also the less-progressed one). Asked the user first; with
explicit go-ahead, paused `Qwen3.6-27B` (SIGTERM on both the python process and its bash driver, to
stop the queue from auto-advancing to the next queued model) to free GPU1's full 46GB, ran the eval
(~2 min), then relaunched a scoped resume script
(`build_bigtier_n2000_gpu1_resume_qwen36.sh`, same queue minus the already-completed Qwen3.5-27B)
and confirmed via a monitored record count that it resumed from 498 (499 observed shortly after
relaunch), not from scratch.

**Result (ensemble vs SEP; N=1000 squad questions per target):**

| target | SEP ρ | proxy ρ | Δρ [95% CI] | SEP AU | proxy AU | ΔAU [95% CI] |
|---|---|---|---|---|---|---|
| Llama-2 | 0.236 | 0.616 | **+0.378** [+0.32,+0.44] | 0.622 | 0.808 | **+0.186** [+0.14,+0.23] |
| Mistral | 0.425 | 0.548 | **+0.123** [+0.07,+0.18] | 0.686 | 0.779 | **+0.093** [+0.05,+0.14] |

Per-seed (ensemble in bold for reference): Llama-2 ρ 0.551/0.603/0.583 (**ens 0.616**), AU
0.777/0.806/0.789 (**ens 0.808**); Mistral ρ 0.462/0.566/0.456 (**ens 0.548**), AU
0.734/0.801/0.727 (**ens 0.779**).

**Findings.**
1. **The proxy beats SEP on both metrics for both targets, every CI excludes 0** — the LOLO proxy
   holds up even under the hardest combined shift tested anywhere in the project (never saw this
   target's hidden states/text/labels in training, never saw this data distribution either).
2. **Llama-2's Δρ (+0.378) is the largest single delta recorded across every E51/E52 setting**,
   nominally edging out even the DEPLOY-proxy squad row (E51: +0.352) — consistent with SEP's
   already-known weak/high-variance Llama-2 layer (E41) collapsing further under dataset shift
   (0.236, same SEP number as E51's `squad` row, since it's the identical SEP recipe) while the
   LOLO proxy (0.616) actually reads slightly *higher* than the DEPLOY proxy did on the same squad
   data (E51: 0.590) — despite having strictly less information (no Llama-2 training data at all
   vs DEPLOY's full inclusion). Read this as noise on a single N=1000 draw, not as "excluding the
   target's own data helps" — nothing in the design supports that claim.
3. **Mistral's margin (+0.123) is smaller than the DEPLOY-proxy squad row (E51: +0.168)** — in the
   direction E42 would predict (having *any* trivia data from the target model, or even related
   models, in the pool narrows the dataset-shift penalty), though this is one comparison, not a
   controlled ablation of "in-pool vs LOLO" holding everything else fixed.
4. **Per-seed spread is the widest seen in the project** (individual seeds trail the ensemble by up
   to 0.09 on Mistral ρ) — expected, since this is the least-informed regime (3-source LOLO
   checkpoint, zero target signal, cross-dataset on top).
5. **Closes the one setting/target combination E51's table left untested** — LOLO trivia (E51),
   squad OOD via DEPLOY (E51), and now LOLO×squad (E52) together cover the full cross-product of
   {model-seen, model-unseen} × {trivia, squad} for the 2 targets with squad data; the proxy has
   not lost to SEP in any cell.

**Caveats.** Only 2 targets have squad records (Llama-2, Mistral) — DeepSeek/Llama-3 LOLO×squad
remains untestable without a new squad build for those. N=1000 (tighter CIs than the N=200 `lolo`
rows, comparable to the other N=1000 settings). The LOLO checkpoints' saved transform is an identity
fallback (per-model z-scoring happens before pooling at train time, so there's no single absolute SE
scale to decode to) — irrelevant here since both metrics (Spearman, AUROC) are rank-based and
invariant to a fixed linear rescaling.

**Artifacts.** `amortized_ue/se_fidelity_proxy_vs_sep.py` (new `lolo_squad` setting +
`arm_preds_per_seed_prefixed` helper), `amortized_ue/results/se_fidelity_proxy_vs_sep.json` (new
`lolo_squad` key + updated `_final_table`), `amortized_ue/logs/lolo_squad_eval.log`,
`amortized_ue/build_bigtier_n2000_gpu1_resume_qwen36.sh`.

---

## E53 — reverse-E45: a proxy trained ONLY on the 4 Qwen/Gemma small-tier models, zero-shot on Llama-2/Mistral — ✅ beats SEP on both metrics, ties true SE on correctness, never saw either target

**Goal.** E45 trained on Llama-2/Mistral/Llama-3/DeepSeek and tested zero-shot on the 4 new
Qwen/Gemma small-tier targets (mixed result: 2/4 beat true SE, 1/4 on par, 1/4 lost). This is the
reverse direction: pool `q_resp_only` (question+response text, no hidden states, no alignment)
from **Qwen3-8B / Qwen3.5-9B / gemma-7b-it / gemma-2-9b-it**'s n2000 trivia_qa train/val splits
(deploy-style, no held-out — matches `exp2_run.build_deploy`'s convention minus the z/alignment
machinery, which `q_resp_only` never needs), train ONE proxy, and score it **zero-shot on
Llama-2 and Mistral** — the proxy never sees either target's hidden states, labels, or text in
any form during training.

**⚠️ OOM at the established batch_size=32 recipe — root-caused, not just worked around.**
Every prior `q_resp_only` deploy run (E37/E45) used batch_size=32 without incident because the
original 4 models' trivia_qa answers are short. Qwen3.5-9B leaves `<think>...</think>` reasoning
traces in `canonical.response` (E44/E49) that can run to hundreds of characters — a single batch
containing one such row hits the full `max_seq_len=256` token cap, and a batch=32 forward pass at
T=256 needs far more activation memory than any batch this arm had ever actually been asked to
process before. Confirmed via bisection (a hand-copied inline replica of `train_arm`'s exact logic
succeeded at low memory; the real function call failed identically every time) that the crash was
genuinely about batch content, not a bug — `torch.cuda.memory_summary()` at the point of failure
showed ~390GB of *cumulative* allocation activity for what should be one forward pass, consistent
with the true worst-case (32, 256) tensor shape, confirmed by an isolated single-call test scaling
linearly from a (32, 53) baseline.

**Fix: gradient accumulation, not "hope a smaller batch is just as good."** Lowering `batch_size`
alone to fit memory would silently change the training recipe with unknown effect on quality (a
correction to an earlier claim in this session's own code comment: batch_size was NEVER one of the
knobs E16/E17 swept — those tested `weight_decay` and projector width/type — so "confirmed inert"
was an overclaim). Instead, added grad-accum support to `exp2_run.train_arm` (reads
`cfg.grad_accum`, additive — byte-identical at the default grad_accum=1, verified by inspection of
the zero_grad/backward/clip/step ordering). This project's `ProxyModel` has no batchnorm anywhere
(only LayerNorm), so K micro-batches of size B, each loss divided by K before `backward()` and
accumulated across K steps before one `opt.step()`, is **mathematically exact**, not approximate,
for reproducing one true batch=B·K step. **Verified numerically** on a toy linear model + MSELoss:
gradients and post-step weights matched a true batch=32 step to float32 precision (~1e-7/1e-8, pure
summation-order noise). Ran with `batch_size=8, grad_accum=4` (effective batch=32, the established
recipe, exactly). `Stage2Config.grad_accum` already existed as an unused field; now it does
something and is captured in checkpoint provenance for free via the existing `cfg.as_dict()`.

**⚠️ Separate incident: a co-tenant raced into a resumable GPU0 build mid-load (user-caught,
root-caused, fixed).** While training was queued, `build_bigtier_n2000_gpu0_resume.sh` (a bare
"poll for ≥40GB free, then launch" loop, no fencing) saw GPU0 clear and started loading
`Qwen3.5-27B`; another user (`sh2419`, `router-tuning.py`, unrelated to this project) started a job
on the same GPU during the ~44s load window and grabbed memory concurrently, OOM'ing our load 44s
in before a single new record was written — a classic check-then-act race, **not** anyone stopping
our process. **This project already built the fix for exactly this failure mode** after an earlier
Llama-3 incident (`amortized_ue/gpu_reserve.py`, used by `build_n2000_waiter.sh`) but the GPU0
resume script never adopted it. Patched `build_bigtier_n2000_gpu0_resume.sh` to fence with
`gpu_reserve.py` before each model launch (same holder pattern: grab `free − budget − safety` MiB
immediately after the free-memory check clears, before the real job starts loading), and restored
`Qwen3.5-27B` to the front of its queue (the old unfenced loop had silently dropped it — moved on
to the next model rather than retrying after the OOM). Separately, `gpu_reserve.py` was also used
live to fence GPU1 for the E53 training run itself (holding the slack free memory, parented to the
training process so it self-releases the instant training exits — confirmed working: the fence
vanished on its own within seconds of training completing, no manual cleanup needed).

**Training result.** 3 seeds, effective batch=32, ~24 min total on one GPU (fenced). In-distribution
sanity Spearman (val pool, same 4 training models — NOT the real zero-shot result, just a pipeline
check): **[0.703, 0.702, 0.693], mean 0.699**, tight across seeds. Checkpoints (3×87MB) + full
per-step/per-epoch training curves saved — the earlier draft of this script had forgotten to
persist `train_arm`'s returned curves at all (would have repeated the exact "missing `json.dump`"
mistake that lost E37's per-seed data, [[persist-results-before-done]]); caught and fixed before
the real run, not after.

**Zero-shot eval on Llama-2 and Mistral (fresh n1000 each, disjoint from any training data) — the
actual result:**

| target | true SE AUROC_inc | proxy AUROC_inc | Δ(proxy−true SE) | proxy Spearman-vs-SE | SEP Spearman | SEP AUROC_inc |
|---|---|---|---|---|---|---|
| Llama-2-7b-chat | 0.760 | 0.748 | −0.013 [−0.041,+0.016] (incl. 0 — **on par**) | **0.632** | 0.523 | 0.681 |
| Mistral-7B-Instruct-v0.2 | 0.747 | 0.746 | −0.000 [−0.029,+0.028] (incl. 0 — **on par**) | **0.634** | 0.548 | 0.714 |

Both targets: proxy vs random excludes 0 by a wide margin (Δ +0.234/+0.248, clearly non-chance).
**The proxy beats SEP on both metrics, both targets** (SE-fidelity Spearman +0.109/+0.086;
correctness AUROC_inc +0.067/+0.032) — same direction/magnitude as every one of E51's 14 settings
(proxy beat SEP on Spearman 13/14) — and is **statistically on par with true 10-sample SE on
correctness** on both targets (a stronger result than E45's original direction, where 1/4 targets
lost to true SE). This proxy has **zero exposure** to either target in any form.

**Ridge context (CONTEXT ONLY, not a fair opponent — needs full target access, so cannot run
zero-shot by construction; see the conversation, ridge structurally cannot be evaluated without
being fit on the target's own hidden states):**

| target | ridge, same layer as SEP | ridge, TBG+SLT ceiling | proxy (zero access) |
|---|---|---|---|
| Llama-2-7b-chat | 0.596 | 0.585 | **0.632 — beats both** |
| Mistral-7B-Instruct-v0.2 | 0.632 | 0.647 | 0.634 — ties same-layer, ~on par with ceiling |

**Striking, precisely-scoped finding:** on Llama-2, the zero-access proxy's Spearman (0.632)
exceeds even ridge's in-distribution ceiling (0.585) — the best possible *linear* read of Llama-2's
own hidden states, with full access. This does **not** mean the proxy beats ridge in general (E8-E10
already established ridge beats even this project's own 3B proxy when both have full target
access) — it means the access-vs-no-access gap is smaller than expected, small enough that a
text-only proxy trained on four unrelated models closes it entirely on one target and nearly closes
it on the other. Llama-2's ridge ceiling (0.585, the established `TBG:22+SLT:15` reference combo)
came in *below* its own single-layer number (0.596, `TBG:30`) — the older reference layer combo
isn't necessarily optimal against E36's later leak-free layer pick; reported as-is, not re-tuned for
a maximally favorable ridge number.

**⭐ Also built (user-requested, after the SEP-Spearman numbers were initially — and correctly —
challenged as looking too low): a canonical, non-scattered reference for these baselines.**
`build_sep_reference.py` extracts (programmatically, not hand-typed) EVERY SEP Spearman/AUROC_se
value ever computed in this project (8 targets × 3 settings) from `se_fidelity_proxy_vs_sep.json`
into `results/sep_reference_values.json` — one file any future script reads instead of
hand-copying numbers into a local dict (which an earlier draft of `e53_eval_on_llama2_mistral.py`
had done). **The "too low" concern resolved to a metric mismatch, not a bug**: SEP is a
single-layer LOGISTIC classifier (AUROC-native; fit to separate binarized SE at one threshold), so
its Spearman is a repurposed use of a classifier probability against a full continuous-scale
ranking question it was never optimized for — mechanically why it reads lower than `ridge`
(a proper regressor for the same task, hence the higher, ~0.6, numbers). SEP and ridge are
deliberately separate, differently-named baselines in this project since E8 specifically to
prevent this exact conflation (which caused the E6 retraction). Independently re-verified the
Llama-2/Mistral fresh-trivia SEP numbers from scratch (`compute_sep`, CPU-only) — matched the
`se_fidelity_proxy_vs_sep.json` values to 4 dp, so not stale/buggy.

`e53_full_comparison.py` **consolidates everything into ONE output file** (true SE / SEP / ridge
context / proxy, both metrics, both targets) — folds in what were briefly three separate scripts
(proxy eval, an ad-hoc SEP-correctness recompute, and a standalone ridge-context script) after the
user asked for the numbers to be saved properly in one place rather than scattered; the standalone
ridge script was deleted once its logic was merged in, not left as a redundant duplicate.

**Caveats.** No paired-bootstrap CI computed for proxy-vs-SEP or proxy-vs-ridge specifically (only
proxy-vs-true-SE and proxy-vs-random are rigorously bootstrapped, via the original eval run) —
would need a GPU pass to save per-row proxy predictions and could be added later; the point-estimate
gaps are clean (not borderline) and consistent with E51's already-significant pattern across 14
settings. Only 2 targets (Llama-2, Mistral — the only ones with existing fresh n1000 builds
available without new Stage-1 generation). Ridge's "reference ceiling" layers were not re-tuned for
this specific comparison (reported as the established config, not a maximized one).

**Artifacts.** `amortized_ue/e53_train_qwengemma_deploy.py`, `amortized_ue/e53_eval_on_llama2_mistral.py`,
`amortized_ue/e53_full_comparison.py`, `amortized_ue/build_sep_reference.py`,
`amortized_ue/results/{e53_qwengemma_deploy_train_curves,e53_qwengemma_deploy_qresp_on_llama2_mistral,
e53_full_comparison,sep_reference_values}.json`,
`amortized_ue/stage2/runs/E53_qwengemma_deploy_qresp/checkpoints/` (3 seeds). Additive edit to
`amortized_ue/exp2_run.py` (`train_arm` gains grad-accum support via `cfg.grad_accum`, byte-identical
at the default). Patched `amortized_ue/build_bigtier_n2000_gpu0_resume.sh` (adds `gpu_reserve.py`
fencing + restores the dropped `Qwen3.5-27B`).

---

## E54 — the TRUE LOLO proxy's correctness (not just SE-fidelity) on squad OOD, the last gap in the {model-seen/unseen} × {trivia/squad} × {SE-fidelity/correctness} cube — ✅ still beats SEP and ridge, real (not "on par") gap to true SE

**Why.** Two prior experiments each covered half of this cell. **E39** ran the squad *correctness*
eval (`incorrect`, not SE) but had to substitute the **DEPLOY** proxy (all-4-pooled, target's own
trivia data WAS in its training pool) because at the time E37's leave-one-LLM-out run had saved no
checkpoints. **E52** later scored the **TRUE LOLO** proxy (checkpoints now exist:
`stage2/runs/E37_LOLO_ckpt/checkpoints/`, trained on the *other* 3 targets, zero exposure to this
target's data OR to squad) on squad — but only against the continuous SE label, never against
actual wrong answers. User asked directly for the missing combination: true LOLO proxy, scored on
squad, against correctness. Only Llama-2 and Mistral have squad records, so this is a 2-target study
(same limitation as E39/E52).

**Method (additive, new script `amortized_ue/correctness_eval_lolo_squad.py`).** Trains nothing;
reuses `se_fidelity_proxy_vs_sep.{compute_sep, arm_preds_per_seed_prefixed}` for the SEP fit (E41
fixed layer) and the LOLO `q_resp_only` forward pass (3 seeds, `E37_LOLO_ckpt` checkpoints), and
`correctness_eval.{load_accuracy, accuracy_coverage, prediction_rejection_ratio,
paired_bootstrap_auc, ci}` for the correctness scoring — the same recipe `correctness_eval_ood.py`
already used for the DEPLOY/REFERENCE proxies, applied here to the true LOLO one. SEP + true SE +
LOLO predictions are id-joined (never positional) onto the same squad n1000 rows; paired bootstrap
(10,000 resamples, shared indices) gives a 95% CI on (LOLO − SEP) and (LOLO − true SE).

**⚠️ Infra: got stuck on NFS twice while launching this, both traced and fixed.**
1. First launch forgot `--trivia_dir /data2/mn1025/stage1` (a rule already in memory from
   [[use-data2-not-nfs]]) — the process sat in kernel disk-wait (`STAT=D`,
   `wchan=folio_wait_bit_common`) reading Mistral's n2000 trivia hidden states off NFS for 5+
   minutes before being caught (Llama-2's block had already finished and saved correctly by then).
   Killed and relaunched with the flag.
2. Second launch (with `--trivia_dir` correctly passed) stalled the **same way** on **squad**
   records instead — squad had never been staged on `/data2` at all (trivia-only, a known
   limitation — see [[use-data2-not-nfs]]). Verified NFS was genuinely in one of its documented
   degraded windows at the time (a plain `cat` of the squad `.pt` files itself timed out at 30s).
   **Fix, at the user's explicit request to "make the copy fast, keep it parallel":** parallel
   per-file copy (`find ... -print0 | xargs -0 -P 32 -I{} cp -n {} ...`, both models' 1000-record
   squad dirs copied concurrently) — completed in **under 15 seconds**, because NFS's bulk-read
   stall is a latency problem for large sequential reads, not a bandwidth ceiling; many small
   parallel requests route around it. `squad_n1000_full` for Llama-2 and Mistral now permanently
   live on `/data2/mn1025/stage1/` alongside trivia — **this closes the trivia-only limitation
   [[use-data2-not-nfs]] flagged**, at least for these two models. The script itself was updated to
   route `eval_data_dir`/`load_accuracy`'s `output_dir`/`arm_preds_per_seed_prefixed`'s `data_dir`
   through the same `--trivia_dir` override for squad too, not just trivia. Re-ran cleanly end to
   end afterward (whole 2-target run <2 min). **Memory updated** ([[use-data2-not-nfs]]) with a
   mandatory pre-launch habit (grep the script for a data-dir flag and pass it every time; check
   `ps -o stat,wchan` within 30-60s of a silent launch) so this specific mistake doesn't recur.

**Audit before trusting the numbers:** the rerun's SE-fidelity side (`auroc_binarised_se`/
`spearman_se`, computed as a byproduct of scoring the same LOLO predictions) reproduces E52's saved
values **exactly** — Llama-2 ρ 0.616 / AUROC_SE 0.808, Mistral ρ 0.548 / AUROC_SE 0.779 — confirming
the forward pass, id-join, and SEP fit are unchanged from the already-verified E52 pipeline; only the
scoring target (`incorrect` vs SE) is new.

**Result (AUROC_incorrect, N=1000 squad questions per target):**

| target | true SE | SEP (E41 fixed layer) | ridge_z† | **LOLO `q_resp_only`** | DEPLOY `q_resp_only`† |
|---|---|---|---|---|---|
| Llama-2 | 0.784 | 0.621 | 0.641 | **0.729** | 0.716 |
| Mistral | 0.774 | 0.669 | 0.703 | **0.735** | 0.763 |

† ridge_z and DEPLOY are reported for context from the existing E39/E41-corrected results (not
recomputed here) — DEPLOY's target model WAS in its training pool, so it is not a true LOLO
comparison; ridge is a full-access baseline, also not label-free/LOLO on the target.

**Paired bootstrap, Δ AUROC_incorrect (B=10000):**

| target | LOLO vs SEP | LOLO vs true SE |
|---|---|---|
| Llama-2 | **+0.108** [+0.061, +0.154] (excludes 0) | **−0.055** [−0.087, −0.022] (excludes 0) |
| Mistral | **+0.066** [+0.026, +0.108] (excludes 0) | −0.039 [−0.077, −0.002] (excludes 0, barely) |

Per-seed LOLO AUROC_incorrect: Llama-2 [0.713, 0.719, 0.713] (tight); Mistral [0.703, 0.749, 0.688]
(wider spread — only 1 of 3 individual seeds beats SEP significantly, though the 3-seed ensemble
does).

**Findings.**
1. **The true LOLO proxy beats both SEP and ridge at catching wrong answers on both targets, under
   the hardest combined shift tested anywhere in the project** (never saw this target's hidden
   states/text/labels in training, never saw squad either) — extending E52's SE-fidelity result to
   the correctness target, closing the last open cell of the {model-seen, model-unseen} ×
   {trivia, squad} × {SE-fidelity, correctness} cube for the 2 targets with squad data.
2. **Unlike E38's in-distribution result, the gap to true 10-sample SE here is REAL, not "on par."**
   Both CIs exclude 0 (Llama-2 comfortably, Mistral by a hair) — this matches E39's general OOD
   finding that amortization degrades under dataset shift while sampling stays robust, now confirmed
   for the true LOLO proxy specifically rather than only the DEPLOY/REFERENCE stand-ins E39 had to
   use.
3. **Llama-2's LOLO number (0.729) nominally edges out DEPLOY's (0.716)** despite having strictly
   less information (zero Llama-2 data anywhere in training) — the same noise pattern E52 flagged
   for the SE-fidelity metric on this exact target/setting. **Not evidence that excluding the
   target's data helps**; read as noise on one N=1000 draw, consistent with E52's caveat.
4. **Mistral's LOLO number (0.735) sits below DEPLOY's (0.763)**, in the direction E42/E52 would
   predict (in-pool data narrows the dataset-shift penalty) — again one comparison, not a controlled
   ablation.

**Caveats.** Only 2 targets have squad records (Llama-2, Mistral) — DeepSeek/Llama-3 remain
untestable here without a new squad build. N=1000 (same power as E39/E51/E52's squad rows). No
`sep_single_best_layer`/`sep_5layer_concat`/`ridge_z` bootstrap computed fresh in this script (only
`sep_single_e36_layer` and `true_semantic_entropy` — the two established bases); the ridge_z/DEPLOY
columns in the table above are read from the existing E41-corrected results, not recomputed, so no
CI is reported for LOLO-vs-ridge or LOLO-vs-DEPLOY specifically.

**Artifacts.** `amortized_ue/correctness_eval_lolo_squad.py`,
`amortized_ue/results/correctness_eval_lolo_squad.json`,
`amortized_ue/logs/correctness_eval_lolo_squad.log`. `/data2/mn1025/stage1/{Llama-2-7b-chat,
Mistral-7B-Instruct-v0.2}_squad_n1000_full/` (new local copies, 1000/1000 records + manifest each,
verified against the NFS originals by record count).

## E55 — data-generation status: DeepSeek/Llama-3 squad builds (done) + Qwen "nothink" regeneration across all 5 Qwen targets (🔄 in progress) — no experiment run yet, this is a data-readiness snapshot

**Goal:** two data gaps opened by earlier sessions. (1) E39/E52/E54's squad correctness studies were
stuck at 2 targets (Llama-2, Mistral) because DeepSeek and Llama-3 had never had a squad build at
all. (2) Qwen3.8-27B was known to stall on `<think>` generation (E44: 65/1000 records over >40h) and
Qwen3.5-9B's manifest history separately showed a ~5-6% "never finishes thinking" tail even after
E44's `tolerate_thinking` stop-logic fix — both point at needing to disable thinking mode outright
rather than just tolerating it.

**Fix 1 — squad coverage.** `build_deepseek_llama3_squad_n1000.sh` ran `deepseek-llm-7b-chat` and
`Meta-Llama-3-8B-Instruct` on squad, N=1000, no `--only_ids` (default `random_seed=10,
num_few_shot=5` reproduces the exact question selection already used for the Llama-2/Mistral squad
sets, so all 4 original targets now share the same squad questions). **Both done, 1000/1000
records**, written straight to `/data2` (not staged via NFS copy, unlike E54's Llama-2/Mistral
fix). Squad coverage is now: all 4 original targets (Llama-2, Mistral, DeepSeek, Llama-3) + the 4
Qwen/Gemma small-tier models (Qwen3-8B, Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it, built earlier under
E44/E49) = **8 of 14 targets have squad_n1000**. Big-tier Qwen (27B) and gemma-2/3-27b remain
trivia-only by design (matches E44's original scope, not extended here).

**Fix 2 — disable Qwen thinking mode entirely.** New `_DISABLE_THINKING_MODELS` tuple in
`huggingface_models.py` (`Qwen3-8B`, `Qwen3.5-9B`, `Qwen3.5-27B`, `Qwen3.6-27B`, `Qwen3.8-27B` — all
5 Qwen targets; confirmed offline that no Gemma tokenizer has an equivalent switch, so Gemma is
untouched). For these models only, `predict()` now wraps the raw few-shot prompt as a single chat
turn via `apply_chat_template(..., enable_thinking=False)` before tokenizing — this pipeline never
called `apply_chat_template` before, so Qwen's official thinking-disable switch had never actually
been reached by E44's fix. A second bug surfaced immediately: the chat-template path returns a
string containing literal special-token text, and decoding the *output* with the default
`skip_special_tokens=True` (while `input_data` still has them) desyncs the token-count bookkeeping
used to slice the generated continuation, driving `n_generated` negative → `IndexError` on
`hidden[n_generated-1]` (confirmed live on Qwen3.5-9B and Qwen3.6-27B). Fixed by decoding with
`skip_special_tokens=False` for exactly this model set. Both changes are additive and model-scoped
— every other model's code path (including the `tolerate_thinking`/`_LEADING_WHITESPACE_MODELS`
machinery from E44) is untouched.

**Writes to NEW `_nothink`-suffixed run names** — none of E44-E49's existing `_full` dirs were
touched or overwritten, so every already-published result (E44-E54) stays reproducible against
what's still on disk under the old names.

**Infra: dual-lane GPU0/GPU1 build with a shared work-stealing queue + a live fencing bug caught and
fixed.** 12 builds total (5 Qwen models × up to 3 dataset variants each, small tier gets all 3 —
trivia n1000/n2000 + squad n1000 — big tier gets the 2 trivia sizes only, matching E44's scope).
Small-tier (Qwen3-8B, Qwen3.5-9B) run one per lane in parallel; the 6 big-tier 27B jobs
(Qwen3.5/3.6/3.8-27B × {n1000,n2000}) live in one shared file that both lanes race to claim via
atomic `mkdir` (exactly one caller wins per line), so whichever lane finishes small-tier first
naturally absorbs more of the shared backlog instead of sitting idle. **Live bug (user-caught):**
the first version of each lane script computed its GPU memory fence (`gpu_reserve.py` hold) *once*,
before the small-tier phase even started; when free memory at that instant happened to be below the
big-tier budget (as it was on GPU0 right after the squad job exited), the hold went negative and the
`if HOLD > 512` guard silently skipped fencing for the **entire lane** — 20GB+ of genuinely free
memory sat completely unprotected through all of small-tier. Fixed with per-phase dynamic
re-fencing (`refence()`, resized whenever the job size changes small→big) in both lane scripts.
`watchdog_lanes.sh` supervises both lanes across crashes (e.g. a neighbouring OOM), releasing a
crashed job's `mkdir` claim (never auto-released on failure) from the lane's own log before letting
either lane retry it — `stage1.py` itself skips already-completed records on disk, so a resumed job
continues rather than restarting.

**Status at the time this entry was written (2026-08-26, mid-build):**

| Build | Status |
|---|---|
| DeepSeek + Llama-3, squad n1000 | ✅ done, 1000/1000 each |
| Qwen3-8B, Qwen3.5-9B — trivia n1000/n2000 + squad n1000 (6 builds) | ✅ done, verified counts (1000/1000, 2000/2000, 1000/1000) |
| Qwen3.5-27B — trivia n1000 | ✅ done, 1000/1000 |
| Qwen3.5-27B — trivia n2000 | 🔄 running (lane A, GPU0), 360/2000 |
| Qwen3.6-27B — trivia n1000 | 🔄 running (lane B, GPU1), 14/1000 |
| Qwen3.6-27B trivia n2000, Qwen3.8-27B trivia n1000/n2000 | ⏳ queued (shared work-stealing queue) |

**Caveats.** This entry documents infrastructure and in-progress data, not a result — no proxy has
been trained or evaluated on any `_nothink` data yet. Record counts for the two running builds will
have moved by the time this is read; check `ls .../records | wc -l` before citing. The lane/watchdog
scripts (`lane_a_gpu0.sh`, `lane_b_gpu1.sh`, `watchdog_lanes.sh`, `resume_lane_a_tomorrow_noon.sh`,
`build_qwen_nothink_regen.sh`, `build_deepseek_llama3_squad_n1000.sh`) are launched via `nohup ...
& disown` and are designed to survive independently of any Claude Code session or SSH connection.

**W&B: all datasets built so far auto-pushed, verified via the API** (Stage-1's `push_to_wandb`
defaults `True`; none of these scripts pass `--no_push_to_wandb`). Every completed build above has a
corresponding new version in `amortized_ue_stage1` under its existing `stage1_records_<model>_<dataset>_n<N>`
collection name (the artifact name does not encode the `_nothink`/`_full` run-name suffix, so each
regenerated dataset lands as a new version of the same collection, e.g.
`stage1_records_Qwen3.5-9B_squad_n1000` v0→v2) — confirmed by listing versions and their
`created_at`/`n_records` metadata directly from `wandb.Api()`, not from build logs. No manual push
was needed.

**Artifacts.** `amortized_ue/build_deepseek_llama3_squad_n1000.sh`, `build_qwen_nothink_regen.sh`,
`lane_a_gpu0.sh`, `lane_b_gpu1.sh`, `watchdog_lanes.sh`, `resume_lane_a_tomorrow_noon.sh`. Code
changes in `semantic_uncertainty/uncertainty/models/huggingface_models.py` (`_DISABLE_THINKING_MODELS`,
both additive, see diff). New data under `/data2/mn1025/stage1/{deepseek-llm-7b-chat,
Meta-Llama-3-8B-Instruct}_squad_n1000_full/` and `/data2/mn1025/stage1/*_nothink/` (per table above).
Queue state: `/data2/mn1025/stage1_meta/nothink_bigtier_jobs.txt` +
`nothink_bigtier_claims/`. Logs: `amortized_ue/logs/{lane_a,lane_b}_driver*.log`,
`amortized_ue/logs/watchdog.log`, per-model `*_nothink.log`.

## E56 — how much of SE's wrong-answer signal survives cheaper supervision? (SE vs n_clusters / MC seq-entropy / perplexity / binarised SE) — ✅ diagnostic; clustering carries the signal, the entropy weighting barely does, binarising SE costs a significant ~0.07 AUROC

**Goal.** SE (the Stage-2 training target) needs 10 high-temp samples *and* an entailment model to
cluster them. This asks, purely as a **wrong-answer detector on the label set itself**, how much of
that discrimination is retained by signals that need strictly less machinery, all read from the same
Stage-1 records at zero extra cost:

| signal | needs sampling? | needs entailment model? | needs the SE formula? |
|---|---|---|---|
| SE (`cluster_assignment_entropy`) | yes (10) | yes | yes |
| SE_binary (SE thresholded via `stage2.data.best_split`) | yes (10) | yes | yes + binarise |
| `n_clusters` (raw count of distinct semantic ids) | yes (10) | yes | **no** |
| MC sequence entropy (−mean over the 10 samples of each sample's length-normalised mean token log-lik) | yes (10) | **no** | **no** |
| perplexity (`exp(-mean(canonical.token_log_likelihoods))`) | **no** (1 canonical pass) | no | no |

**Method.** New standalone `amortized_ue/supervision_signal_compare.py` — read-only over Stage-1, no
GPU, no target-LLM calls, no new generation. Loads trivia_qa **n2000** (`load_records`, id-sorted,
same convention as `linear_ceiling_probe.py`) for **all 4 original targets** (Llama-2, Mistral,
Llama-3, DeepSeek). Target = `incorrect = 1 - canonical.accuracy` (accuracy is exact-match
squad-metric, asserted already binary {0,1}; 0.5 threshold per `correctness_eval.py`'s fixed
convention, a no-op here). AUROC on the **full n2000, no train/test split** (a descriptive
discrimination measure on the label set, not held-out generalisation). SE_binary threshold from
`best_split` fit on the same n2000 SE array (reused verbatim from `stage2/data.py`, not reinvented).
Paired bootstrap for the deltas: **10000 resamples, one shared set of row indices reused for all 5
signals**, via `paired_bootstrap_auc` + `ci` imported unchanged from `correctness_eval.py` (the
E25/E26/E31/E38 convention). Four deltas per target, all `SE_continuous − X`.

**AUROC vs `incorrect`, full n2000:**

| signal | Llama-2 | Mistral | Llama-3 | DeepSeek |
|---|---|---|---|---|
| **SE (continuous)** | **0.7874** | **0.7521** | 0.7729 | **0.8151** |
| SE_binary | 0.7227 | 0.6851 | 0.7090 | 0.7271 |
| `n_clusters` | 0.7819 | 0.7504 | 0.7685 | 0.8101 |
| MC sequence entropy | 0.7491 | 0.7465 | **0.7841** | 0.7821 |
| perplexity | 0.6285 | 0.6842 | 0.5654 | 0.6102 |
| best_split threshold | 0.698 | 0.814 | 0.698 | 0.954 |
| incorrect rate | 0.4095 | 0.3675 | 0.3450 | 0.4770 |

**Paired-bootstrap 95% CI on `SE_continuous − X` (mean [lo, hi]; "excl 0" = CI excludes zero):**

| delta | Llama-2 | Mistral | Llama-3 | DeepSeek |
|---|---|---|---|---|
| − SE_binary | +0.0646 [+0.053, +0.076] **excl 0** | +0.0669 [+0.053, +0.081] **excl 0** | +0.0639 [+0.051, +0.077] **excl 0** | +0.0880 [+0.077, +0.100] **excl 0** |
| − `n_clusters` | +0.0055 [+0.003, +0.008] **excl 0** | +0.0017 [−0.001, +0.004] n.s. | +0.0043 [+0.002, +0.007] **excl 0** | +0.0050 [+0.002, +0.008] **excl 0** |
| − MC seq-entropy | +0.0382 [+0.023, +0.053] **excl 0** | +0.0056 [−0.008, +0.020] n.s. | −0.0112 [−0.025, +0.002] n.s. | +0.0330 [+0.021, +0.046] **excl 0** |
| − perplexity | +0.1587 [+0.135, +0.182] **excl 0** | +0.0679 [+0.048, +0.088] **excl 0** | +0.2075 [+0.185, +0.231] **excl 0** | +0.2049 [+0.183, +0.227] **excl 0** |

**Findings.**
1. **The clustering step carries almost all of SE's signal; the entropy weighting on top of it barely
   does.** `n_clusters` — just the integer count of distinct entailment clusters among the 10 samples,
   no `logsumexp`/entropy at all — is within **+0.002 to +0.006 AUROC** of full continuous SE on every
   target (significant on 3/4 by the tightest of margins, statistically tied on Mistral). Whatever the
   SE formula adds over "how many different answers did it give" is real but negligible for
   wrong-answer detection.
2. **Binarising SE costs a significant ~0.06–0.09 AUROC on all 4 targets** — a much larger loss than
   clustering-vs-`n_clusters` (~0.005) or dropping the entailment model (~0.03). Binarised SE is the
   **weakest** of the entropy-family signals here, scoring *below* even raw `n_clusters` on all four.
   This is a cost the SEP objective (predict binarised SE) pays that E31 never isolated: the continuous
   ranking within each side of the `best_split` threshold is genuine wrong-answer information, and the
   threshold throws it away.
3. **Dropping the entailment model (MC sequence entropy) costs ~0.03 on the two Llama-2-family models
   and DeepSeek, but is a wash on Mistral and nominally *ahead* on Llama-3** (−0.011, CI includes 0).
   A sampling-based lexical uncertainty signal that needs the 10 generations but not DeBERTa is roughly
   competitive with SE and, on one of four targets, better.
4. **One forward pass is not enough.** Canonical-answer perplexity — the only signal that needs no
   sampling — is **0.07 to 0.21 AUROC below SE, significant on every target**, and collapses to near
   chance on Llama-3 (0.565). Consistent with the whole project's "sampling beats a single pass" line
   (E31/E39).
5. **DeepSeek has the most separable SE→correctness signal** (SE 0.815, highest of the four; also the
   highest incorrect rate at 0.477) and Mistral the least (SE 0.752) — Mistral is also where every
   delta is smallest/least significant.

**Consistency check.** Llama-2 SE AUROC 0.787 here ≈ E31/E38's "true 10-sample SE" ≈ 0.783 (different
code path, different row set — E38 was 200 held-out, this is full n2000) — the two agree, so the
numbers are on the same footing as the correctness-eval line.

**Caveats.** Full-set AUROC, **no train/test split** — these are label-set discrimination numbers, not
held-out generalisation; treat the *ranking* of signals and the *deltas* as the result, not the
absolute AUROCs. Accuracy is exact-match squad-metric with ~10% label noise (E32) ⇒ absolute AUROCs
are mild under-estimates uniformly. trivia_qa n2000 only — no squad / no OOD. SE_binary uses a single
`best_split` on the full array (matches how this script has no split); a train-fit threshold could
differ slightly.

**Artifacts.** `amortized_ue/supervision_signal_compare.py` (new, standalone),
`amortized_ue/results/supervision_signal_compare_{llama2,mistral,llama3,deepseek}_trivia.json` (per
target: 5 point estimates + all 4 bootstrap deltas + `best_split` threshold). Run:
`python -m amortized_ue.supervision_signal_compare --model_name <M> --data_dir /data2/mn1025/stage1
--out amortized_ue/results/supervision_signal_compare_<tag>_trivia.json` in `se_probes`.

## E57 — combined two-position ridge ceiling for Mistral-v0.2 / Llama-3-8B / DeepSeek-7B (extending E8c's Llama-2 TBG:22+SLT:15) — ✅ diagnostic; a second position lifts every model's ID ridge ceiling to ~0.66–0.68 Spearman

**Why.** E8c established Llama-2's linear ceiling by concatenating two hidden-state positions
(TBG:22 + SLT:15 → ID Spearman 0.642) and showed the two positions are complementary (+0.042 over the
best single). E10 built the reference proxy on exactly that combo. The other three targets only ever
had a **single-position** leak-free ceiling (`reconfirm_layers.py`: Mistral TBG:31, Llama-3 TBG:31,
DeepSeek SLT:16). This fills that gap — same ridge-sweep method as `linear_ceiling_probe.py`, extended
to two positions.

**Method** (`amortized_ue/two_pos_ceiling.py`, new, read-only, CPU, `se_probes`, `--data_dir
/data2/mn1025/stage1`). Reuses `linear_ceiling_probe`'s `load_matrix / splits / fit_probe / rho`
verbatim (ridge → continuous `cluster_assignment_entropy`, α on val Spearman, trivia_qa n2000, split
1440/360/200 seed 42). Per model: (1) fix the first position at the model's leak-free best single
`(pos, layer)` from `reconfirm_layers.py --cv 5`; (2) sweep every layer of the complementary position,
**concatenate the two hidden-state vectors feature-wise** (exactly E8c's construction — verified: this
code reproduces Llama-2 TBG:22+SLT:15 at ID 0.6425 / OOD 0.437, matching E8c's 0.642), fit one ridge,
**select the complement layer on validation Spearman** (leak-free, same philosophy as
`reconfirm_layers.py`); (3) report held-out TEST Spearman (ID) + all-rows squad n1000 Spearman (OOD).
All four targets now have squad n1000 (E55 built DeepSeek + Llama-3's).

**Results** (ID = held-out test Spearman, OOD = squad n1000 all-rows; val-selected complement):

| target | 1st pos (leak-free) | single ID / OOD | + val-sel complement | two-pos ID | two-pos OOD | ID gain |
|---|---|---|---|---|---|---|
| Mistral-7B-Instruct-v0.2 | TBG:31 | 0.620 / 0.534 | SLT:6  | **0.656** | 0.571 | +0.036 |
| Meta-Llama-3-8B-Instruct | TBG:31 | 0.623 / 0.483 | SLT:11 | **0.672** | 0.540 | +0.049 |
| deepseek-llm-7b-chat | SLT:16 | 0.629 / 0.529 | TBG:28 | **0.683** | 0.534 | +0.054 |
| *Llama-2-7b-chat (ref, leak-free 1st pos)* | TBG:30 | 0.585 / 0.269 | SLT:13 | 0.603 | 0.423 | +0.018 |
| *Llama-2-7b-chat (E8c/E10 fixed combo)* | TBG:22 | 0.600 / 0.301 | SLT:15 (fixed) | 0.642 | 0.437 | +0.042 |

**Findings.**
1. **The second position helps every target** — ID ridge ceiling +0.036 to +0.054, matching the
   magnitude E8c/E11 measured for Llama-2 (+0.042). The three non-Llama-2 targets land at a **higher
   absolute ceiling (~0.66–0.68) than Llama-2 (~0.60–0.64)** — consistent with their higher
   single-position ceilings.
2. **OOD also improves for all** (+0.04 to +0.15 over the single position), largest for Llama-2 and
   Llama-3 whose single-TBG OOD was weakest. The OOD-optimal complement is a slightly different (later
   SLT / earlier TBG) layer than the val-selected one — reported as `ood_selected_context` in the JSON
   (e.g. Mistral SLT:14 → OOD 0.623; DeepSeek TBG:24 → 0.563; Llama-3 SLT:17 → 0.562).
3. **The complement plateau is flat** (val within ~0.003 across a wide band): Mistral's SLT band
   SLT:4–15 is all ~0.67 val / 0.64–0.67 ID; DeepSeek's TBG:19–29 all ~0.70 val. So the exact
   complement layer is val-noise; the ceiling is the ~0.66–0.68 plateau, not the argmax.
4. **Llama-2's 22-vs-30 wobble reappears:** at the leak-free first position TBG:30 the two-pos ceiling
   is only 0.603, below the E8c/E10 fixed TBG:22+SLT:15 combo (0.642) — the val scores are near-tied
   (0.6325 vs 0.6328) but TBG:22 generalises better on this test split (already noted E34/E36). This is
   why E10's reference proxy uses TBG:22, not the CV-picked TBG:30.

**Caveats.** Held-out test n=200 (ID) so ±0.05-ish; complement selected on a single 360-row val split
(no CV) — flat plateau means the argmax is noisy, cite the band. No retraining — these are linear
ridge ceilings, not proxy or SEP numbers ([[sep-vs-ridge-different-baselines]]). trivia→squad only.

**Artifacts.** `amortized_ue/two_pos_ceiling.py`, `scratch_xllm/two_pos_ceiling.json` (per target:
single-position baseline, full complement sweep, val-selected + OOD-selected combos). Run:
`python -m amortized_ue.two_pos_ceiling --data_dir /data2/mn1025/stage1` in `se_probes`.

---

## E58 — label-noise ceiling for all 4 targets (extends E8a's Llama-2-only ceiling to Mistral-v0.2 / Llama-3-8B / DeepSeek-7B, on both the n2000 ID set and the fresh n1000 set) — ✅ diagnostic; the SE label is ~equally reliable (~0.93–0.95) across all 4 targets

**Why.** E8a established the split-half label-noise ceiling on the SE target for **Llama-2 only**
(trivia ID-test 0.914, all-2000 0.934, squad 0.901). Every cross-LLM comparison since (E29–E31,
E37–E38, E45–E54) reports "% of achievable signal recovered" against *that* number, implicitly
assuming the other targets' labels are equally noisy. This closes the gap: same method, run for all 4
training targets on the two trivia_qa sets they share.

**Method** (`amortized_ue/label_noise_ceiling.py`, unchanged computation — only additive `--data_dir`
flag added to read the fast off-NFS `/data2` copy). Verbatim E8a: split each prompt's stored
`semantic_id`s into two disjoint halves, recompute `cluster_assignment_entropy` on each, Spearman
across prompts, **200 random split-half draws**, seed 42, Spearman–Brown up-correct to the n-sample
reliability, `ceiling = sqrt(reliability)`. The 200-/100-row sub-rows are the held-out test slice
(`train_test_split` `test_size=0.1`, `split_seed=42` — matches `Stage2Config`). CPU, no GPU, no
LLM/entailment re-run (holds the DeBERTa clustering fixed, so this is a mild *over*-estimate of the
true ceiling per the script docstring). Llama-3 + DeepSeek n1000 records parallel-copied to `/data2`
first ([[use-data2-not-nfs]]); byte-identical to NFS.

**Results — n2000 ID set** (200-row = held-out ID test slice, matches E8a's basis):

| target | rows | split-half r | reliability_n | **ceiling** |
|---|---|---|---|---|
| *Llama-2-7b-chat (E8a, for reference)* | 200 / 2000 | 0.717 / 0.773 | 0.835 / 0.872 | **0.914** / 0.934 |
| Mistral-7B-Instruct-v0.2 | 200 | 0.770 ± 0.023 | 0.870 | **0.933** |
|  | 2000 | 0.765 ± 0.007 | 0.867 | 0.931 |
| Meta-Llama-3-8B-Instruct | 200 | 0.803 ± 0.023 | 0.891 | **0.944** |
|  | 2000 | 0.817 ± 0.007 | 0.899 | 0.948 |
| deepseek-llm-7b-chat | 200 | 0.826 ± 0.018 | 0.904 | **0.951** |
|  | 2000 | 0.819 ± 0.007 | 0.900 | 0.949 |

**Results — fresh n1000 set** (the E23 disjoint held-out batch; 100-row = its 0.1 test slice, wider CI):

| target | rows | split-half r | reliability_n | **ceiling** |
|---|---|---|---|---|
| Llama-2-7b-chat | 1000 | 0.782 ± 0.010 | 0.877 | **0.937** |
|  | 100 | 0.758 ± 0.033 | 0.862 | 0.929 |
| Mistral-7B-Instruct-v0.2 | 1000 | 0.754 ± 0.012 | 0.860 | **0.927** |
|  | 100 | 0.734 ± 0.042 | 0.847 | 0.920 |
| Meta-Llama-3-8B-Instruct | 1000 | 0.818 ± 0.010 | 0.900 | **0.949** |
|  | 100 | 0.830 ± 0.030 | 0.907 | 0.952 |
| deepseek-llm-7b-chat | 1000 | 0.828 ± 0.008 | 0.906 | **0.952** |
|  | 100 | 0.848 ± 0.024 | 0.918 | 0.958 |

**Findings.**
1. **All 4 targets' SE labels are about equally reliable — ceiling ~0.93–0.95 everywhere.** Noise
   explains only ~5–7 points of the gap to 1.0 for any target, same as E8a found for Llama-2. The
   "% recovered" framing used across E29–E54 is sound for all 4, not just Llama-2.
2. **Ordering is stable across both sets:** DeepSeek ≈ Llama-3 (0.95) > Llama-2 (0.937) > Mistral
   (0.927–0.933). Llama-3 and DeepSeek labels are marginally *less* noisy than Llama-2's; Mistral's
   are marginally *more*. Small effect — within ~0.02 across the board.
3. **The two sets agree to ~0.005 per target** (Llama-2 0.937 vs 0.934, Mistral 0.927 vs 0.931,
   Llama-3 0.949 vs 0.948, DeepSeek 0.952 vs 0.949) — the fresh n1000 batch carries the same label
   quality as the n2000 ID set.
4. Unlike E8a's Llama-2 (ID-test 0.914 sitting *below* all-rows 0.934), the other 3 targets show the
   sub-row slice within ~0.002–0.006 of the full-set number — no meaningful test-split effect. The
   Llama-2 ID-test dip is a 200-row sampling artifact, not a property of its labels.

**Caveats.** Sub-row slices are n=200 (ID set) / n=100 (fresh set) so their `r_half` CI is wide
(±0.02–0.04) — cite the full-set ceiling. Entailment clustering held fixed → true ceiling is slightly
*lower* than reported (unmeasured DeBERTa noise). No squad ceilings re-run here (E8a's Llama-2 squad
0.901 stands; E57 covers the squad *ridge* ceiling for the other 3). These are label-reliability
ceilings, not proxy/SEP/ridge numbers ([[sep-vs-ridge-different-baselines]]).

**Artifacts.** `amortized_ue/results/label_noise_ceiling_{Mistral-7B-Instruct-v0.2,Meta-Llama-3-8B-Instruct,deepseek-llm-7b-chat}_trivia.json`
(n2000) + `label_noise_ceiling_{Llama-2-7b-chat,Mistral-7B-Instruct-v0.2,Meta-Llama-3-8B-Instruct,deepseek-llm-7b-chat}_trivia_freshn1000.json`.
Reproduce (per target, `se_probes`, CPU):
`python -m amortized_ue.label_noise_ceiling --model_name <M> --dataset trivia_qa --num_samples <2000|1000> --data_dir /data2/mn1025/stage1 --out <path>`.
