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

## Where we stand

**What improved:** ID Spearman **0.467 → 0.602** (+29% relative), recovering **66%** of achievable
signal (from 51%). OOD **0.289 → 0.368**. Both came from **fixing the input**, not from the model.

**What was retracted:** every text-arm claim (E4, E5, E6). Text adds nothing once `z` is well-fed.

**The unresolved problem:** a **ridge regression still beats the 3B proxy** (0.642 vs 0.602 ID;
0.437 vs 0.368 OOD), and an MLP loses to ridge, so there is **no nonlinear headroom** to win with.

This matters beyond a leaderboard: *a linear probe on hidden states predicting SE is essentially
**SEP** (arXiv:2406.15927)* — so the z-only branch of this project **re-derives existing work, and
ridge does it better.** The SLM cannot be justified by "it models `z` better."

**Therefore the open question is: what can the SLM do that a linear probe structurally cannot?**
The leading candidate — and the highest-value next experiment — is **text-only arms** (`q_only`,
`q_resp_only`, with **no `z`**): can the proxy predict SE from the **question alone, with no target-LLM
forward pass at all**? A hidden-state probe cannot do this by construction. Even 0.3–0.4 Spearman
would be a genuinely new capability (uncertainty *before* generation → routing, abstention, cascades).

**Attribution (E11, now closed):** E10's gain is fully accounted for — the second position is worth
+0.042 ID, the wider projector +0.022, and they are **synergistic** (+0.085 together). Both changes
are load-bearing and neither alone suffices. It is not a parameter-count effect.

**A methodological asset worth using:** the ridge diagnostic **predicted the proxy's gain to three
decimals** (E8c said positions were worth +0.042; E11 measured +0.042). Use ridge as the **design
oracle** for future input choices — it is exact, costs seconds on CPU, and has now been validated
prospectively. Do **not** spend 3B sweeps on questions ridge can answer.

The full to-do list lives in `amortized_ue/CLAUDE.md`.
