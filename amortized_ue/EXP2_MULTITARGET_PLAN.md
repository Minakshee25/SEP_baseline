# Exp 2 — Multi-target proxy training (the SLM-proxy analog of E35 pooling)

**Status:** PLAN (not yet implemented). Written 2026-08-17.
**One-line question:** *Does training the SLM proxy jointly on several target LLMs' (aligned)
hidden states induce a **model-agnostic** uncertainty code that transfers to an **unseen** target
better than a single-source proxy — and does it earn its place over the simpler E35 pooled ridge?*

---

## 1. Why this experiment, and what already constrains it

The thesis is cross-LLM transfer. We have established:

- **Raw hidden states do NOT transfer** across a model swap (E20–E23: `z` → chance, e.g. Llama-2
  proxy → Llama-3 `z` = **0.056**). **Text transfers** (`q_only` 88%, `q_resp_only` full).
- **Aligned hidden states DO transfer** (E24; E27b: frozen proxy on aligned Mistral→Llama-2 z rises
  `0.014 → 0.545`). Alignment is a **label-free orthogonal Procrustes** fit on shared anchor questions.
- **On aligned z, a plain ridge BEATS the proxy** (E27b: ridge **0.580** > proxy `z` **0.545**), and
  **early fusion loses** (adding text inside a z-arm hurts: z_q 0.478, z_q_resp 0.510 < pure z 0.545).
  The proven combined recipe is **late fusion** (rank-fuse pure-`z` ⊕ pure-`q_resp_only`).
- **E35 pooled RIDGE** (train on 3 aligned models → test held-out 4th, label-free) is a **small, safe
  win**: pooled ties the oracle best-single source, beats a fixed Llama-2 anchor by ~+0.015, but is
  **data-saturated (~800 Q)** and only marginal over the text proxy `q_resp_only`.
- **E34:** after alignment the 4 models share the **same uncertainty direction** up to noise —
  including **DeepSeek**, which is a **low-CKA (0.25) outlier** yet aligns in the top ~50 dims.

**What has NOT been tested — the gap Exp 2 fills:** every result above is either single-source→
single-target (E27b) or a *ridge* (E35). **No experiment has trained the SLM proxy jointly on multiple
aligned sources and tested leave-one-out on an unseen target.** That is Exp 2.

**Burden of proof (stated up front):** given E27b + E35, the proxy is *behind* on aligned z. Exp 2's
job is to find whether **joint multi-source training** gives the SLM something a per-pair ridge can't:
(a) better transfer to an *unseen* target, (b) a *single* model serving all targets, and (c) whether
the text pathway lets it degrade gracefully on a hard (low-CKA) target. If the proxy cannot beat the
E35 pooled ridge, that is itself a clean, reportable result.

---

## 2. Models & data (no new Stage-1 generation)

Four targets, all **trivia_qa n2000**, all **4096-dim** (required to share the projector width and a
square Procrustes W). Datasets already on `/vol/bitbucket` + W&B:

| model | role | leak-free best (pos:layer) | selection | val / test Spearman |
|---|---|---|---|---|
| Llama-2-7b-chat | anchor frame / source / target | **TBG:30** (≈22, tied); SLT:15 for SLT arm | val | 0.610 / 0.585 |
| Mistral-7B-Instruct-v0.2 | source / target | **TBG:31** | val | 0.644 / 0.620 |
| Meta-Llama-3-8B-Instruct | source / target | **TBG:31** | **CV (5-fold)** | 0.672cv / 0.623 |
| deepseek-llm-7b-chat | source / target (**hard**) | **SLT:16** | val | 0.680 / 0.629 |

**Confirmed leak-free (2026-08-17) via `reconfirm_layers.py`** (selection on val / 5-fold CV, never
test; audit-clean). Supersedes the old `scratch_xllm/*_layer_pick.json`, whose `best_id` was selected
on the **test set** (a leak). Two corrections this produced: **(a) Llama-3 is TBG:31, NOT SLT:31** — the
old SLT:31 (test 0.708) was a test-selection artifact (ranks #24/66 under CV, val 0.613); CV cleanly
picks TBG:31 (test 0.623), consistent with E30's TBG:31→Llama-2 alignment. **(b) each model's best is
NOT layer 22** — three of four peak at late TBG (30/31/31), DeepSeek at SLT:16 — so best-layer-per-source
is the right source choice; shared-index-22 (E27b/E35) is the weaker recipe. See §3.

**DeepSeek stays in** — it is the only model that separates the two competing explanations of transfer
("needs high raw CKA" vs "needs shared uncertainty direction after alignment", E34). It is both the
hardest held-out target and the most informative diversity source. Replacing it with a high-CKA
Llama-family model would only prove the weak version of PRH.

**Question split (E35 discipline):** one global `tr / va / te` split over the shared id order
(`linear_ceiling_probe.splits`). **All sources train only on `tr`; every held-out target is evaluated
only on `te`.** Because `te` questions are absent from every source's training, this is clean for the
`z` arms *and* the text arms (no question-difficulty leakage through the model-agnostic text pathway).

---

## 3. Alignment — which layer from each model (aligned-z primary; raw-z a cheap contrast)

**The proxy reads ONE fixed input frame**, so unlike E30/E34 (which scored each pairing at its own
best-agreeing Llama-2 layer) we **fix the common frame** and align every source into it. Layers below
are **leak-free, reconfirmed 2026-08-17** (`reconfirm_layers.py`, val / 5-fold CV; see §2).

- **MAIN RUN = single TBG position.** All four models had TBG best-or-near-best (DeepSeek's TBG plateau
  is only 0.008 below its SLT, so a shared TBG frame handicaps no one). Single position ⇒ one Procrustes
  map per source, one frame — simplest correct design, and it shares machinery with the E35 re-run.
- **Common frame: primary = Llama-2 TBG:30** (best→best — sources at their best late layers → anchor at
  Llama-2's best), **but keep testing both 30 and 22.** The E35 re-run found 30 vs 22 a **performance
  wash** (all deltas ≤0.010, within 4-seed noise; depth-matching hypothesis unsupported — Procrustes maps
  late sources into either anchor equally well). 30 is chosen for internal consistency, not because it
  wins; carry 22 as a cheap comparison to confirm the wash holds on the proxy too.
- **Source layers (own leak-free best TBG) → the frame:** Mistral **TBG:31**, Llama-3 **TBG:31**,
  DeepSeek **TBG:28** (flat plateau L18–29; late end chosen to depth-match — DeepSeek does *not* peak at
  its final layer). Llama-2's own W = identity. Fit label-free `orthogonal_procrustes` on shared `tr`
  anchors (mean-centre, as E27b `fit_align_at`). Sources use their labels only to pick their own layer
  (they're training data — fine).
- **⚠️ Held-out TARGET layer — the label-free catch.** Picking the target's own best layer uses its SE
  labels, so report **two numbers**: **(oracle)** target's own best TBG layer = the ceiling, flagged
  non-deployable; **(deployable)** a **label-free** target-layer pick (max post-alignment CKA /
  anchor-agreement to the frame) + fixed-index as the simplest baseline. The **oracle − deployable gap
  = the cost of not knowing the target's best layer.**

**Regimes and ablations:**
- **Aligned regime (PRIMARY):** every z-arm consumes the Procrustes-aligned, per-model-scaled state.
- **Naive regime (cheap contrast):** same run with the W step removed — "did joint multi-source training
  reduce the *need* for explicit alignment?" Expected ≈ chance on z (E20–E23); run once, not the headline.
- **TBG+SLT stacking (ablation):** add each source's best SLT (Llama-2 SLT:15, Mistral SLT:18, Llama-3
  SLT:15, DeepSeek SLT:16) aligned into a Llama-2 SLT:15 frame, stacked with TBG (the reference proxy's
  [TBG,SLT] input). The single-model reference gained +0.042 from SLT; whether that complementarity
  **survives cross-model alignment** is unknown — measure it, don't assume it.
- **Shared-index-22-for-all (comparison):** the E27b/E35 recipe — quantifies what best-layer alignment buys.

---

## 4. Normalization (mandatory — the E35 bug)

Per **source model**, fit on `tr` only:
- **SE-label z-score** — each source's target standardised on its own train mean/std before pooling
  (Mistral/Llama-2/Llama-3/DeepSeek have different SE scales; skipping this is the exact bug that made
  E35 pooling look falsely *worse*). Rank metrics are transform-invariant, so this does not distort the
  held-out ranking.
- **Feature scaler** — `StandardScaler` on the aligned state (E35). The projector's `LayerNorm(H_in)`
  already absorbs per-example scale; treat "LayerNorm-only vs explicit per-model scaler" as a small
  ablation, default = with scaler (matches E35).
- Held-out target uses **its own** `tr`-fit label-stats + feature scaler (label-free: SE stats only for
  decoding RMSE; Spearman/AUROC need none).

---

## 5. Arms (all 5 + a late-fusion estimator)

In the aligned regime `z` means **aligned z** — the **single-TBG** aligned state for the main run
(§3), with **TBG+SLT** as the stacking ablation:

| arm | input | expectation (from E27b, single-pair) |
|---|---|---|
| `z` | aligned z | primary; alignment rescues it |
| `z_q` | aligned z + question | early fusion — expected ≤ pure z |
| `z_q_resp` | aligned z + question + response | early fusion — expected ≤ pure z |
| `q_only` | question text (no z) | transfers well; no target forward pass |
| `q_resp_only` | question + response text (no z) | transfers ~fully |
| **late fusion** | rank-fuse pure `z` ⊕ `q_resp_only` | **E27's proven winner (headline combined)** |

We train **all 5** (GPU is not a constraint) so the multi-target LOO regime can *re-test* "late > early"
on hard unseen targets rather than assume it. **Headline combined = late fusion**; `z_q`/`z_q_resp` are
carried as the early-fusion comparison. Reason to keep the early-fusion arms despite E27b: E27b was on
Llama-2 (z aligns well); on a **low-CKA target (DeepSeek)** the text pathway — the only thing that
transferred across models — may need to carry the load, which could flip the sign.

---

## 6. The matched source-count sweep (isolate diversity from volume)

E35's core lesson: a naive pool confounds **diversity** with **3× rows**. So hold **total train rows
fixed** across source counts (the E35 matched-partition design):

| condition | rows / source | total train rows |
|---|---|---|
| 1 source | 1440 | 1440 |
| 2 sources | 720 + 720 | 1440 |
| 3 sources | 480 + 480 + 480 | 1440 |

Same questions, same row count, only **model-routing** differs → a rising held-out `z` across 1→2→3 is
**diversity**, not volume. Data saturates ~800 Q (E35), so 1440 is safely above saturation. Also run one
**unmatched full-data** pass (each source contributes its full 1440) as the practical best-case; the
*claim* rests on the matched sweep.

---

## 7. Leave-one-out design

4 rotating folds — hold out each model in turn, train on the other 3 (and the 1-/2-source subsets for
the sweep), evaluate on the held-out target's `te`.

- **Start with hold-out-Llama-3** (directly comparable to E20's chance 0.056).
- Then Mistral, Llama-2, and **DeepSeek** (the CKA stress test).

---

## 8. Baselines to beat (report the proxy against all)

1. **Chance / single-source raw z** — E20 = 0.056 (the floor the proxy must clear).
2. **Single-source aligned proxy** — the 1-source point of the matched sweep (does pooling *add*?).
3. **E35 pooled RIDGE (best-layer re-run — DONE 2026-08-17).** The simpler competitor the proxy must beat.
   Corrected best-layer pooled Spearman @1440 (anchor 30, matched partition): **Mistral 0.594, Llama-3
   0.603, DeepSeek 0.579** (single-source: 0.571 / 0.579 / 0.568). Best-source layers lifted Mistral
   **+0.032** / Llama-3 **+0.017** over the old shared-22 (DeepSeek ~0, already near-best at 22); the
   pooling diversity effect (~+0.02) and its conclusion are unchanged. JSONs:
   `scratch_xllm/e35_bestlayer_matched_anchor{30,22}.json`. ⚠️ 4 seeds, no CIs — deltas "small &
   consistent", not "significant".
4. **Matched-target SEP** (single-layer logistic on the target's own data) — the supervised upper
   reference the label-free proxy is trying to reach.

---

## 9. Metrics

- **Spearman (primary)** and **AUROC** (train-`best_split` threshold) on the held-out target's `te`.
- **5 seeds** per (fold × source-count × arm); report mean ± std.
- **Bootstrap CIs on the key deltas** (pooled − single; proxy − E35 ridge) — E35 was explicitly
  criticised for having no CIs; fix that here so we can say "significant", not just "consistent".

---

## 10. Outcome interpretation (pre-registered)

| result | meaning |
|---|---|
| held-out `z` climbs 1→2→3 sources, above single-source | **multi-target training induces a model-agnostic code** — the PRH-positive headline |
| `z` flat across source count | joint training adds no shared code beyond what alignment already gives; the ridge story stands |
| proxy `z` ≥ E35 pooled ridge on unseen target | the SLM earns its place on hidden states |
| proxy `z` < E35 ridge (expected from E27b) | clean negative — the SLM's value is only the **text pathway + one-model-for-all** |
| DeepSeek (low CKA) transfers ≈ high-CKA models after alignment | **shared uncertainty direction, not raw CKA, governs transfer** (confirms E34) — a strong thesis result |
| late fusion > early fusion holds on unseen targets | E27's "late > early" generalises cross-model |
| early fusion (`z_q_resp`) wins on DeepSeek only | text carries transfer when z is weak — a new, target-difficulty-dependent finding |

---

## 11. Implementation sketch (reuse-heavy; no edits under `semantic_uncertainty/`)

New files under `amortized_ue/` (additive):

- **`exp2_multitarget_data.py`** — multi-source adapter. Loads each model via `linear_ceiling_probe.
  load_matrix` (reuses the exact split), fits per-source label z-score + feature scaler + Procrustes W
  from each source's own best-TBG (Mistral 31, Llama-3 31, DeepSeek 28) into the Llama-2 **TBG:30**
  frame on `tr` anchors (main run: single TBG; stacking ablation adds a second W into an SLT:15 frame),
  and exposes a `Stage2Data`-compatible view
  (aligned+scaled `hidden[pos][layer][rows]`, `questions`, `responses`, pooled standardized labels)
  over the routed/partitioned pooled rows so the existing `Trainer._forward_batch` works unchanged.
- **`exp2_multitarget_run.py`** — driver: builds the pooled adapter per (fold, source-count, seed),
  trains all 5 arms with the existing `ProxyModel` + `Trainer` loop, evaluates the frozen proxy on the
  held-out target's `te` (aligned with the target's own W), computes late fusion (reuse
  `procrustes_e27_rank_fusion.py` logic), and writes `exp2_multitarget_{fold}.json` + a master table.
- Reused **read-only**: `ProxyModel`, `Trainer`, `train.py` arm/tokenize logic, `linear_ceiling_probe`
  (`load_matrix`, `splits`, `fit_probe`, `rho`), `orthogonal_procrustes`, `checkpoint.py`.

**Env:** `amortized_stage2` (proxy). **Checkpoints save by default.** Push datasets/checkpoints + JSONs
to W&B and verify by fetch; log to `EXPERIMENTS.md` (new E-number) + `amortized_ue/CLAUDE.md` + memory.

---

## 12. Open design choices deliberately deferred (ablations, not the main run)

- **Anchor 30 vs 22** for the TBG frame (E35 re-run: a wash, ≤0.010; main run uses 30 for consistency,
  carries 22 to confirm the wash holds on the proxy — depth-matching hypothesis was unsupported).
- **TBG-only vs TBG+SLT stacking** (main run: TBG-only; stacking = ablation, two W/source — §3).
- **Shared-index-22-for-all** vs best-layer-per-source (main run: best-layer; shared-22 = comparison row).
- **LayerNorm-only** vs explicit per-model feature scaler (main run: with scaler).
- Adding a **5th target** (needs new Stage-1 generation) — only if the 4-model trend is promising.
- The **naive raw-z** regime beyond the single cheap contrast pass.
