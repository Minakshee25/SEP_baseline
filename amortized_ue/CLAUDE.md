# CLAUDE.md — `amortized_ue/` (amortized UE: Stage 1 dataset + Stage 2 proxy)

> **Scope:** this file governs the `amortized_ue/` module (Stage 1 offline dataset + Stage 2 SLM
> proxy). The repo-root `../CLAUDE.md` is also in effect (SEP baseline, Imperial-DoC machine quirks,
> wandb auth, model compatibility, the `se_probes` env). Read both.

> 📓 **The chronological experiment log is `../EXPERIMENTS.md`** (E0→E20) — what changed each step,
> what came out, what was **retracted**, and why. Read it for *how we got here*. **This file is
> current-state + how-to-run + the single to-do list.** When you run a new experiment, add an
> `EXPERIMENTS.md` entry.

## What this module is

MSc project: **amortized uncertainty estimation** — train a small model to predict a large LLM's
semantic entropy in **one forward pass**, avoiding the multi-sample cost at inference. Two stages:

- **Stage 1 (dataset):** for one target LLM + QA dataset, produce one **self-contained, id-keyed
  record per prompt** (canonical answer + TBG/SLT hidden states all layers, N high-temp samples,
  continuous `cluster_assignment_entropy` label) so Stage 2 never re-runs the target LLM.
- **Stage 2 (proxy):** train a frozen decoder-only SLM (Llama-3.2-3B) to regress that continuous SE
  label from the stored hidden state (as soft tokens) plus optional text. Consumes Stage-1 read-only.

## 🎯 Long-term goal — CROSS-LLM TRANSFER (the thesis)

**A proxy whose uncertainty estimates carry across target LLMs** — train on one LLM's data, evaluate
on another. This is *why* the proxy is an SLM taking **text** alongside the hidden state: text is
model-agnostic, so a text-reading proxy is not tied to one target. Scientific backing to test: the
**Platonic Representation Hypothesis** (different LLMs' internal representations align as training
scale grows), so the hidden-state input may transfer too. A per-model probe (SEP/ridge) *cannot* do
this — a probe fit on model A's hidden states can't even be applied to model B without retraining.

**Status (2026-07-28): cross-LLM experiment #1 DONE (E20) — the thesis holds.** The frozen Llama-2
proxy was evaluated on Llama-3-8B's 200 held-out questions: **hidden-state transfer FAILS** (z
0.602→0.056, chance) but **text transfer SUCCEEDS** (q_only 88% retained, q_resp_only full). Only the
model-agnostic text pathway survives a target-model swap — the core argument for a text-reading proxy.
See Current state + E20. Next: a 2nd cross-LLM target (ideally a non-Llama family) to test generality.

## Relationship to the SEP repo (read-only reuse)

`amortized_ue/` imports SEP's logic read-only via `sys.path` (`sep_bridge.py` adds
`../semantic_uncertainty`). **Nothing under `semantic_uncertainty/` or `semantic_entropy_probes/` is
edited** (except the sanctioned blocks-execution model-loading redirects in `huggingface_models.py`
— NousResearch mirrors for gated Llama-2/Llama-3, and the `'8b'` load branch; no SE/probe logic
touched). Do not modify `get_semantic_ids`, `cluster_assignment_entropy`, `logsumexp_by_id`, the
entailment model, TBG/SLT extraction, or the sampling.

Reused unchanged: `HuggingfaceModel.predict(return_latent=True)`, `load_ds`, prompt construction,
`get_metric`, `get_reference`, `split_dataset`, `EntailmentDeberta`, `get_semantic_ids`,
`cluster_assignment_entropy`.

## Files

**Stage 1:** `config.py` (`Stage1Config`), `sep_bridge.py` (path + reused-fn re-exports),
`record.py` (schema `stage1-v1`, save/load), `stage1.py` (builder + `--smoke` CLI),
`loaders.py` (`load_records`, local|wandb switch), `wandb_io.py`, `data/stage1/` (gitignored).

**Diagnostics (se_probes env, no GPU) — run BEFORE any Stage-2 training:**
- **`linear_ceiling_probe.py`** — ridge from hidden state → continuous SE. **Use it to pick the
  (position, layer)** and as the baseline every Stage-2 result is reported against.
- **`label_noise_ceiling.py`** — split-half reliability of the SE label → achievable ceiling
  (turns raw Spearman into "% of achievable signal recovered").
- `text_baseline_probe.py` — TF-IDF→ridge control for the text-only arms.
- `sanity_probe.py` — throwaway SEP-style classification probe (binarised SE, per-layer AUROC).

**Stage 2:** `stage2/{config,data,model,train,run,checkpoint}.py`,
`stage2/proxy_learning_curve.py` (drives Trainer read-only to measure data-appetite),
`smoke_llama3.sh` (Llama-3 Stage-1 smoke in the `se_probes_llama3` env).

## Record schema (`stage1-v1`, one `.pt` per prompt, keyed by `id`)

```
id, question, context, reference
canonical:                       # low-temperature (0.1) "most likely" answer
  response, accuracy, token_log_likelihoods
  hidden_states: { TBG: [L+1,1,H], SLT: [L+1,1,H] }   # all layers, native dtype
samples: [ {response, token_log_likelihoods, semantic_id}, ... ]   # N high-temp
labels: cluster_assignment_entropy (PRIMARY, CONTINUOUS float), semantic_ids, n_clusters, n_samples
meta: { model, dataset, temperatures, entailment settings, git_commit, positions... }
```

Joined **by id**, never by list position (fixes SEP's positional-join fragility, `SEP_TECHNICAL_REPORT.md` §7).

### Hidden-state positions — true-position labelling

| record key | position | HF index |
|------------|----------|----------|
| `TBG` | token before generation (last input token) | `hidden[0]` |
| `SLT` | second-last generated token | `hidden[n_gen-2]` |

`predict()` returns `(scalar, sec_last=SLT, last_tok_before_gen=TBG)`; `stage1.py` unpacks
`(embedding, slt_emb, tbg_emb)` — so our keys are correct. **⚠️ SEP's own stored keys are inverted**:
amortized `TBG` == SEP `emb_tok_before_eos` == SEP probe `slt_dataset`; amortized `SLT` == SEP
`emb_last_tok_before_gen` == SEP probe `tbg_dataset`. Mind this when comparing to SEP.

## Environments (three; keep separate)

| env | transformers | used for |
|-----|-------------|----------|
| `se_probes` | 4.35.2 (pinned baseline) | Stage-1 generation for **Llama-2**; the diagnostics |
| `amortized_stage2` | 4.52.4 (+peft) | **Stage-2 proxy** training (Llama-3.2-3B; se_probes rejects its rope config) |
| `se_probes_llama3` | 4.44.2 | Stage-1 generation for **Llama-3** (E19; se_probes too old, amortized_stage2 hit DeBERTa `.bin`/protobuf walls) |

All are clones of `se_probes` (hardlinks) with transformers upgraded; torch stays 2.1.1. Llama-2 and
Llama-3 load via ungated **NousResearch** mirrors (meta-llama is gated for acct Minakshee25).

## Commands

**Stage 1** (repo root, `se_probes` env for Llama-2 / `se_probes_llama3` for Llama-3):
```bash
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3      # smoke, prints a record
python -m amortized_ue.stage1 --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 2000
bash amortized_ue/smoke_llama3.sh                                # Llama-3 smoke (se_probes_llama3 env)
```
Build is **resumable** (`overwrite=False` skips existing records). Shared GPUs are often full —
pin `CUDA_VISIBLE_DEVICES` to a GPU with ≥~16 GB free (Llama-3-8B loads fp32, needs ~32 GB).

**Stage 2** (repo root, `amortized_stage2` env, free GPU). **Checkpoints save by DEFAULT now**
(`--no_save_checkpoints` to opt out). **Do NOT use the built-in 3B (pos,layer) sweep** — it is
unreliable (picked TBG L12, costing ~0.12 Spearman); pick the layer with `linear_ceiling_probe.py`.
```bash
# THE REFERENCE COMMAND — reproduces the current best result + saves 25 checkpoints
python -m amortized_ue.stage2.run \
  --ood --ood_dataset squad --ood_num_samples 1000 \
  --seeds 5 --reuse_selection \
  --arms z,z_q,z_q_resp,q_only,q_resp_only \
  --z_inputs TBG:22,SLT:15 --selected_k 4 --projector_hidden_dim 1024 \
  --run_name REFERENCE_multipos_p1024_5arm_ckpt
#  -> z ID Spearman 0.602±0.019 / OOD 0.368±0.033  (full 5-arm table in Current state)

python -m amortized_ue.stage2.run --eval --eval_datasets squad:1000   # reload checkpoints, no retrain
```
**Key flags:** `--z_inputs POS:LAYER,...` stacks positions (h_in widens to n·H automatically);
`--selected_position/layer/k` force the input (override wins over a saved `results.json`);
`--projector_hidden_dim` (default 256; use 1024 when stacking); `--arms` (adds `q_only`/`q_resp_only`,
the text-only arms — no hidden states); `--weight_decay`/`--projector_type`/`--lora_r` (regularisation
knobs, all swept — none is a live dial, see E16/E17); `--run_name`.

## Locked design (do not change without asking)

**Stage 1:** SE stored **continuous** (never binarised); keep `semantic_ids` + per-sample log-probs
so the label is recomputable. Hidden states at **TBG & SLT, all layers**, canonical answer only.
Joined by id. Local disk is source of truth (offline-first); W&B is an extra copy.

**Stage 2:**
- Projector: `LayerNorm(H_in) → Linear(H_in, hidden) → GELU → Dropout(0.1) → Linear(hidden, k·d_model)
  → per-token unit-norm × learnable scalar`. `hidden` = `--projector_hidden_dim` (**1024 when
  stacking positions**; 256 is a 16–32× bottleneck that measurably binds). Interface takes
  `[B, n_layers_in, H]` and flattens → `--z_inputs` works with **no `model.py` change**. *(The
  docstring's "preserves z magnitude" claim is false; measured cost ~0.01 Spearman, left alone.)*
- **Separate model per arm** (`z`/`z_q`/`z_q_resp`/`q_only`/`q_resp_only`), each on its own fixed,
  null-free sequence — no modality dropout. Text-only arms skip the projector entirely (`[text][REG]`).
- **z input picked with `linear_ceiling_probe.py` (exact ridge sweep), NOT the 3B sweep.** Current
  best: **`TBG:22,SLT:15`, k=4** (the two positions are complementary; extra *layers* within a
  position are not, +0.005).
- Target z-scored on train; metrics in original space: **Spearman (primary)**, AUROC (via train
  `best_split`), RMSE/MAE/R². *(R² meaningless OOD — label scale shifts; use rank metrics under shift.)*
- Frozen backbone, LoRA r16/α32/drop0.05 on q,k,v,o_proj, linear head, REG readout. bf16 backbone;
  projector/head fp32.

## Current state (updated 2026-08-18)

**Target LLMs (4):** Llama-2-7b-chat (reference), Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct,
and **DeepSeek-LLM-7B-Chat (E28, NEW)**. Per-target z-layers **re-confirmed LEAK-FREE
(2026-08-17, `reconfirm_layers.py`, selection on val / 5-fold CV, never test)**: Llama-2 **TBG:30**
(≈22, tied; SLT:15 for SLT arm), Mistral **TBG:31**, **Llama-3 TBG:31** (5-fold CV; test 0.623), DeepSeek
**SLT:16** (0.629). **⚠️ Correction:** the old `scratch_xllm/*_layer_pick.json` picked the best layer on
the **TEST set** (`best_id` = `id_test_spearman`, a leak) — so the earlier "**Llama-3 best SLT:31 (0.708)**"
was a **test-selection artifact** (SLT:31 ranks #24/66 under CV; CV picks TBG:31, matching E30's
TBG:31→Llama-2 alignment). Also each model's best is NOT layer 22 (three peak at late TBG). JSONs:
`scratch_xllm/reconfirm_{<model>}_val_single_split.json` + `reconfirm_Meta-Llama-3-8B-Instruct_cv5.json`.

**Stage-1 datasets (target LLM Llama-2-7b-chat):** trivia_qa n400 (`stage1_records:v0`), **n2000**
(`stage1_records_n2000`; split 1440/360/200 seed 42; the ID dataset), squad n1000 (OOD; mean_acc
0.236 / mean_CAE 1.498 — a real shift vs trivia 0.59/0.59). **Llama-3-8B Stage-1 (E20):** trivia_qa
**n200** on the exact Llama-2 held-out test ids (`Meta-Llama-3-8B-Instruct_trivia_qa_n200_full`;
mean_acc 0.685 / mean_CAE 0.448) — built with `stage1.py --only_ids` (reproduces the seed-10 n2000
selection, keeps the 200 test ids). A Llama-2 `n200_full` copy of the same ids exists as the eval
control. **DeepSeek-LLM-7B-Chat Stage-1 (E28):** trivia_qa **n1000** on the E23 fresh held-out ids
(`deepseek-llm-7b-chat_trivia_qa_n1000_full`; mean_acc 0.527 / mean_CAE 0.804; on /vol/bitbucket +
W&B `stage1_records_deepseek-llm-7b-chat_trivia_qa_n1000`, run `c6ijifxe`) — same 1000 ids as the E23
Llama-2/Mistral fresh batches, zero overlap.

**E29→E30 — full-power four-model alignment table (DeepSeek + Llama-3 n2000 built).** Extended the
E24–E27 line to DeepSeek and Llama-3 at full power (built both **n2000** on the shared seed-10 selection;
on /vol/bitbucket + W&B, verified). Two settled results:
1. **Label-free ensemble ≥ supervised SEP on ALL 3 targets** (no target labels): rank-fusion(aligned-z +
   `q_resp_only`) AUROC — Mistral 0.866 (SEP 0.832), DeepSeek 0.869 (0.805), Llama-3 0.892 (0.839); Δ
   excludes 0 for Mistral & DeepSeek, ρ-only for Llama-3. **The thesis holds at full power on 3 targets.**
2. **Recovery is high for all (~92–95%) but is shared question-difficulty; the discriminator is CKA
   (alignability): Llama-3 0.87 > Mistral 0.80 ≫ DeepSeek 0.25.** Family is only a *weak* predictor —
   **DeepSeek is a low-CKA outlier despite matching 4096 dims**, and its model-specific increment is
   genuinely ~0 at N=1000 (+0.008), vs Mistral's significant +0.032. The increment tracks CKA, not family.
   (Llama-3 increment unresolved — N=200 only, no fresh n1000.)

Tooling (all additive): parametrised `procrustes_alignment.py` (`--position/--source_layer/--target_layer`);
`procrustes_e29/e30_ensemble_sep.py` (E30 = fit-one-set/eval-another for N=1000 power);
`procrustes_e29/e30_master_table.py`; **GPU fencing** `gpu_reserve.py` + `build_n2000_waiter.sh` (waits for
a free GPU, fences the slack so co-tenants can't OOM a build mid-run — added after Llama-3 OOM'd once).
JSONs: `procrustes_e29_*.json` (n1000-preliminary), `procrustes_e30_*.json` (full power). Full arc:
EXPERIMENTS.md E29–E30.

**E31 — correctness-based eval (do the SE predictors detect WRONG ANSWERS?).** Everything E0–E30 is scored
vs the SE label; E31 re-scores the same predictors (E30 regime, all 4 targets) vs `incorrect = 1` using the
stored `canonical.accuracy` (**already binary {0,1}** — the 0.5 threshold is a no-op). Additive script
`correctness_eval.py` (CPU for the hidden-state arms; needs the proxy env+GPU for `q_resp`/rank-fusion);
JSONs `correctness_eval_{<model>,master}.json`. **Three results:** (1) **SE-fidelity ≠ correctness** — every
method drops ≈0.10–0.15 AUROC from the SE target to the correctness target (rank-fusion Mistral 0.866→0.731),
so the ~0.85 SE-AUROCs overstate wrong-answer detection (~0.70–0.77). (2) **True 10-sample SE is the best
correctness detector on all 4 targets** (0.747–0.795) — amortizing to one pass has a real correctness cost
(sig. over single-layer SEP on Mistral/DeepSeek/Llama-2). (3) **Label-free rank-fusion ≥ supervised SEP on
correctness too** (sig. on Llama-2 +0.058 / DeepSeek +0.057; ties on Mistral/Llama-3) → E30's thesis isn't an
SE-scoring artifact. **Ordering (SE-AUROC vs correctness-AUROC): MATCHES on 3/4 (Mistral/DeepSeek/Llama-2),
DIFFERS on Llama-3** — there aligned-z is 3rd by SE but next-to-last by correctness (good SE ≠ good
correctness). **SEP repro:** single-layer SEP-vs-SE reproduces E30 exactly (0.832/0.805/0.839); the ad-hoc
`procrustes_e27_sep_comparison.json` 0.795/0.857 does NOT reproduce leak-free (best-on-eval 0.785/0.834,
~0.02 below — flagged, not smoothed). Full arc: EXPERIMENTS.md E31.

**E32 — correctness-eval qualitative follow-ups (exploratory).** (A) **Label noise ≈10%** (bracket 3.8%
rule-verified floor → 17.3% raw LLM-judge, but the judge — `NousResearch/Meta-Llama-3-8B-Instruct` — is
**over-lenient, ~50% precise on its exclusive flips**; its "NO"/wrong verdicts are reliable). So exact-match
accuracy under-credits the model (~0.61→~0.71 for Llama-2) ⇒ **the E31 correctness-AUROCs are a mild
under-estimate**. (B) **Confusion matrix + genuine-FN** (aligned-z detector, Youden's-J; genuine FN = FN still
wrong after a lenient re-check): the "missed error" bucket is mostly label noise — **only 36% (Llama-2) / 18%
(Mistral) of FNs are genuine**. (C) **Model-specific signal:** on divergent-correctness questions (same Q, one
model right one wrong; difficulty held constant so text/`q_only` is pinned at 50%), each model's own
hidden-state reader picks the failing model **54.8%** (vs 50% null, 61.9% SE ceiling) — a **real but small,
underpowered (n=42)** model-specific increment, the question-level view of E25/E26. Full arc: EXPERIMENTS.md E32.

**E33 — is `z_aligned` worth it GIVEN the text proxy `q_resp_only`? ➜ NO.** The E27–E30 headlines
compare the ensemble vs the *supervised SEP*; but `q_resp_only` (model-agnostic text, trained once on
the Llama-2 reference) needs **zero target fitting and zero target sampling**, while `z_aligned` needs a
per-target anchor set + Procrustes W. The sharp test is **ensemble vs `q_resp_only`-ALONE**. Built the
missing **`Meta-Llama-3-8B-Instruct_trivia_qa_n1000_full`** (fresh E23 shared ids; mean_acc 0.651 / CAE
0.466; 0/1000 corrupt; on /vol/bitbucket + W&B `stage1_records_Meta-Llama-3-8B-Instruct_trivia_qa_n1000:v0`,
verified by fetch) so all 3 targets eval at **N=1000**. (1) **SE-fidelity** (`procrustes_e33_ens_vs_qresp.py`,
paired bootstrap): Δ AUROC(ens − q_resp) = DeepSeek **+0.012** / Mistral **+0.014** / Llama-3 **+0.018**
(CIs exclude 0 but tiny; **FLAT across CKA 0.25→0.87**, non-monotonic on Spearman → the E25/E26
model-specific increment is largely **redundant with the difficulty signal text already carries**). Fresh
Llama-3 `q_resp` AUROC 0.827 vs the optimistic within-set N=200 0.874. (2) **Correctness** (all 3 at fresh
N=1000, 10k paired bootstrap, `correctness_e33_ens_vs_qresp.py` → `correctness_ens_vs_qresp.json`):
Δ(z_aligned − q_resp) = −0.005 / −0.005 / **+0.003**, **every CI includes 0** → z_aligned is
statistically **indistinguishable** from text on wrong-answer detection. **Correction:** the earlier E31
N=200 within-set Llama-3 showed −0.034 ("z worse"); at N=1000 it is +0.003 — a **small-sample artifact**,
so say "no significant correctness benefit", NOT "worse". **Verdict: `q_resp_only` is the right primitive
for a deployable proxy; `z_aligned` is a small SE-only top-up not worth its per-target cost.** Tooling:
`build_e23_fresh_fenced.sh` (E23 fresh-ids build + `gpu_reserve` fencing — NOTE fencing reserves *memory*,
not compute; a co-tenant's kernels still time-slice yours). **Driver gotcha (fixed):** the correctness
driver called `arm_preds(...)` *inside a per-id list comprehension* → ~1000× recompute (~30 min/target);
one call + index → ~2 min/target. `arm_preds` reloads are near-free (OS-cached 3B backbone); the cost is
the proxy forward passes. Full arc: EXPERIMENTS.md E33; results table `RESULTS.md`.

**E34 — do the aligned models share the same UNCERTAINTY DIRECTION? ➜ YES, up to noise** (diagnostic;
`readout_agreement.py`, kept; TBG, trivia_qa n2000, 4 id-aligned targets, one shared anchor layer L_a).
(A) Cross-model **prediction agreement** between readouts carried into the Llama-2 basis **meets/exceeds
the within-model split-half ceiling** (L22 0.80–0.83, L30 0.85–0.89; floors ≈0) — the built-in
"ceiling>cross" sanity FAIL *is* the finding. *(Script auto-picked L_a=**30**, not the documented TBG:22
— late TBG layers near-tied on val; L22 re-run identical conclusion.)* (B) Raw weight-direction cosine
looked "low" (~0.47) only because of **ridge coefficient instability under collinearity** (D=4096 but
effective rank ≈**218**; two ridges on disjoint halves of the SAME model agree at cosine ~0.41 at
α=10000; identical-data sanity = 1.000). (C/D) In the **top-k PC subspace** cross ≈ same-model ceiling at
every k (top-10 cosine 0.96–0.97), and **~80% of the top-100 state-subspace directions coincide** across
models (principal angles) — broadly aligned, not one axis. (E) **⭐ Decisive matched same-vs-different
ceiling** (disjoint halves h1/h2; both cross and ceiling span h1↔h2 so noise is identical, only the model
differs): **cross ≈ self-ceiling at every k for every model** → the model swap costs no more than one
model's own wobble; the alarming full-vector 0.30 is just the same-model noise floor (0.27–0.44).
**DeepSeek nuance:** same in the top ~50 dims, small *genuine* residual in deeper dims (k≥100) — the
CKA/scale outlier again; Mistral/Llama-3 no gap. **Scope (do not overstate):** trivia_qa/TBG L22,
variance-ranked (label-free) subspace; the direction is largely shared question-**difficulty** (E25/E26/
E33), not a proven model-private axis; "same" = no detectable difference beyond noise, not a crisp axis.
**Lesson (memory):** a same-vs-different claim needs a ceiling matched on every nuisance but the tested
factor — **two ceilings were caught invalid** (half-vs-full sample size; then different-question vs
different-model perturbation) before the disjoint-halves design. Full arc + all tables: EXPERIMENTS.md
E34. Artifacts: `readout_agreement.py`, `readout_agreement_L22_result.json`,
`amortized_ue/e34_{cosine_instability,cutoff_sweep,principal_angles,matched_ceiling}.py` + `.log`.

**E35 — POOL multiple aligned models into one ridge? ➜ small yes, but data-saturated + marginal vs text.**
E34's shared direction makes pooling *valid* (averaging estimates of the same direction). Leave-one-out
(train on 3 aligned models → test held-out 4th's `te`; Llama-2 frame; label-free). **⚠️ user-caught bug:**
v1 pooled **raw** SE labels across models with different SE scales (DeepSeek mean 0.78 vs Llama-3 0.48)
while states were per-model centered → per-model label offsets over-regularized the ridge, making pooling
look *worse*. **Fix:** per-model SE-label z-scoring + per-model feature scaler (target uses its OWN scaler);
lifted pooling +0.02–0.03. **Results:** pooled **ties oracle best-single**, **beats a fixed Llama-2 anchor
by ~+0.015** (never hurts). The clean **matched-partition control** (SAME questions + SAME rows, only
1-model vs 3-model routing) → pooled ≥ single by +0.012–0.020 at full data: a **small but real diversity
effect**; the big low-data lead in the naive sweep was **mostly 3× rows**. **1440 rows == 4320 rows** (Δ≈0)
→ ridge **data-saturated (~800 Q)**; only *more unique questions* would help, but all models share the same
questions (needed for alignment) so pooling adds **model-diversity, not question coverage**. **Not better
than the strong baseline:** a marginal top-up over the text proxy `q_resp_only` (E33, needs no target
forward pass). **Inference:** per new target — one-time **label-free** calibration (run target+anchor on
shared anchor Qs, fit `W_T`), then **one forward pass/query** → `W_T` → standardize → ridge → SE. **Target
criteria:** white-box states, **hidden dim 4096** (square Procrustes), **high CKA after `W`** (E30/E34: CKA
not family; verify label-free first). **Caveats:** 3–4 seeds, **no CIs** (so "small & consistent", not
"significant"); pooled α-val 3× larger; one bug fixed; a `/code-review` was **stopped before findings** →
scripts not review-verified. Full arc + tables: EXPERIMENTS.md E35. Artifacts:
`amortized_ue/e35_pooling_{loo_pilot,datasize_sweep,matched_partition}.py`.

**⭐ E37 — MULTI-TARGET PROXY, leave-one-LLM-out (the Exp-2 thesis experiment) — DONE.** One proxy
(frozen Llama-3.2-3B + LoRA, ~26M trainable) trained on 3 targets' ALIGNED z + text, LOLO-tested on the
held-out 4th (all 4 = Llama-2/Mistral/Llama-3/DeepSeek). `z` = per-source best-TBG → Llama-2 TBG:30 frame
(E36 layers, label-free Procrustes); **per-model normalization**; **SAME questions to all sources** (the
model-invariance signal — NOT a disjoint partition). Arms z/z_q/z_q_resp/q_only/q_resp_only + **fuse =
rank-fusion(z⊕q_resp_only)**; baseline = ridge on the same aligned pooled z (label-free on target). 3 seeds.
**Results (3-seed means, Spearman): fuse 0.664 / q_resp_only 0.648 / ridge 0.591** (means across Llama-2
0.679·0.680·0.604, Mistral 0.667·0.630·0.586, Llama-3 0.659·0.622·0.607, DeepSeek 0.650·0.662·0.565).
**Findings:** (1) **label-free fusion ≥ supervised-on-sources ridge on ALL 4 by mean**; (2) **`q_resp_only`
(text, NO target hidden states) beats the ridge on all 4** — the model-agnostic pathway transfers across
every LLM swap (the thesis); (3) **z tracks CKA** (beats ridge only on high-CKA Mistral; on low-CKA DeepSeek
text dominates); (4) **late > early fusion**; (5) `q_only` 0.550. **Significance (conservative unpaired
bootstrap over 200 examples): fuse BEATS ridge on 3/4** (overlaps Llama-3), q_resp_only 2/4, z never —
**paired bootstrap PENDING** (needs ridge per-example preds recomputed). **Step-1 slice first verified the
pipeline** (proxy z → unseen Llama-3 0.586±0.007 vs ridge 0.607 vs chance 0.056). **⚠️ Per-seed data was
LOST once (missing `json.dump`) then fully recovered by a deterministic re-run** — `exp2_run.py` now saves
per-fold incrementally + per-example predictions ([[persist-results-before-done]]). **Deployable proxy**
(`--deploy`, all-4 pool) trains with ALL checkpoints (`results/deploy_checkpoints/`) + full training log
(`results/deploy_curves.json`: per-step loss/lr/grad-norm + per-epoch train/val/spearman + config).
Caveats: 3 seeds; Llama-2 native-frame ridge inflated; unique-Q coverage capped at 1440. Artifacts:
`exp2_run.py`, `exp2_step1_zarm.py`, `results/exp2_{lolo_full,lolo_foldmeans,summary}.*`,
`scratch_xllm/stage_to_data2.sh`. Full arc + per-seed table: EXPERIMENTS.md E37.

**⭐ E38 — CORRECTNESS eval of the E37 LOLO proxy (does it catch WRONG answers?) — DONE.** E37 scored
only vs the SE label; E38 re-scores the **same 200 held-out `te` rows per fold** vs `incorrect = 1`
(`correctness_eval_e37.py`, additive, trains nothing — reads E37's saved per-example preds; CPU/`se_probes`,
~25 min with `--data_dir /data2/mn1025/stage1`). **Audits passed first:** id-mapping exact (max dev
**0.000e+00** on all 4 folds), ridge rebuilt to **4 dp** (and its **per-example preds are now saved** →
closes E37's "paired bootstrap PENDING"), and the script **reproduces E31's Llama-3 column exactly**
(0.775/0.720/0.729) via a different code path. **AUROC_incorrect (Mistral/Llama-3/DeepSeek/Llama-2, MEAN):**
true 10-sample SE 0.762/0.775/0.821/0.783 (**0.785**) · SEP-single 0.721/0.720/0.740/0.611 (0.698) ·
SEP-5layer (0.721) · ridge_z (0.729) · z (0.728) · z_q_resp (0.755) · q_only (0.738) ·
**`q_resp_only` 0.796/0.767/0.844/0.797 (0.801)** · fuse (0.781). **Findings:** (1) **⭐ `q_resp_only` is
statistically ON PAR with the true 10-sample SE** — all 4 paired-bootstrap CIs include 0 — with **no
sampling, no target hidden states, no target labels**; say *on par*, **NOT** "beats" (nominally ahead,
0.801 vs 0.785, leads 3/4, but no CI excludes 0). **This UPDATES E31 finding #2** ("sampling beats
amortization"), which held for the E27/E30-era closed-form predictors but **not** for the E37 trained proxy.
(2) **Label-free proxy > supervised SEP on correctness, sig. on 3/4** (Δ +0.074\*/+0.047/+0.103\*/+0.186\*)
→ E37's headline is not an SE-scoring artifact. (3) **Multi-source training helped, apples-to-apples:** on
Llama-3's *identical* 200 rows `q_resp_only` goes **0.739 (E31 single-source ref proxy) → 0.767 (E37 3-source
LOLO)**. (4) **SE→correctness drop is smallest for text** (q_resp_only +0.062 vs true SE +0.215, SEP +0.113,
z ~+0.100). (5) **Orderings DIFFER on all 4** — SE-fidelity **over-ranks `fuse`, under-ranks `q_resp_only`**;
⇒ **for wrong-answer detection pick `q_resp_only`, not the SE-optimal `fuse`.** (6) z significantly *below*
true SE on DeepSeek (low-CKA), echoing E33. **Caveats:** **N=200/fold → wide CIs (±0.06–0.08)**, the main
limit; the **Llama-2 fold is not a clean cross-model test** (anchor = native frame, *and* its SEP is
anomalously weak at 0.611 / layer TBG:21 vs TBG:28–31 elsewhere) so its Δ-vs-SEP is inflated — the 3/4
significance rests on the clean Mistral + DeepSeek folds; 3 seeds; ~10% label noise (E32) ⇒ mild
under-estimates. **Bug fixed en route in `correctness_eval.py` (E31):** `label_free["q_resp_only"]` was
hardcoded `True` but on the **reference target (Llama-2)** that predictor IS the Llama-2-trained proxy →
now `not is_reference`, matching `aligned_z_ridge`/`rank_fusion_ensemble`. **Metadata-only — no AUROC
changes** (fit/eval ids always disjoint; re-verified n2000 ∩ fresh-n1000 = 0 for all 4). ⚠️ the committed
`correctness_eval_*.json` still carry the pre-fix flag (re-run needs GPU + `amortized_stage2`). Artifacts:
`correctness_eval_e37.py`, `results/correctness_eval_e37.json`, `correctness_e37.log`. Full arc:
EXPERIMENTS.md E38.

**⚠️ E39 — OOD (cross-DATASET) correctness: trivia-fit → squad — E38's parity with sampling does NOT
survive the shift.** `correctness_eval_ood.py`, squad n1000, **Llama-2 + Mistral only** (the only two with
squad records); squad is a real shift (mean_acc 0.236/0.228 vs trivia ~0.65; incorrect rate 0.77).
**⚠️ E37's LOLO run saved NO checkpoints**, so the proxies run are **DEPLOY** (all-4 trivia-trained →
cross-dataset only, target WAS in pool, NOT label-free — the fair peer of SEP) and **REFERENCE**
(Llama-2-only, text arms only → on **Mistral** it is **cross-LLM AND cross-dataset, fully label-free**).
**AUROC_incorrect (Llama-2 / Mistral / MEAN):** true 10-sample SE 0.784/0.774 (**0.779**) · SEP-single
0.603/0.667 (0.635) · SEP-5layer (0.648) · ridge_z (0.672) · deploy z (0.675) · deploy z_q_resp (0.696) ·
deploy q_only (0.655) · **deploy q_resp_only 0.716/0.763 (0.739)** · deploy fuse (0.731) ·
**reference q_resp_only 0.692/0.713 (0.703)** · random (0.529).
**(1) ⭐ Sampling is ROBUST to the shift, amortization is NOT.** Matched to the same 2 targets, ID→OOD:
**true SE 0.773→0.779 (flat, +0.007)** while q_resp_only 0.797→0.739, fuse 0.786→0.731, ridge_z
0.736→0.672, z 0.741→0.675, SEP 0.666→0.635. vs true SE: q_resp_only **−0.068\*** (Llama-2) / −0.011
(Mistral, includes 0) ⇒ **E38's "on par with sampling" is IN-DISTRIBUTION ONLY; this RESTORES E31's
"sampling beats amortization" out of distribution.** *(Proxy rows are DEPLOY not LOLO ⇒ indicative; the
true-SE/SEP/ridge_z rows are strictly matched.)* **(2) The proxy still beats supervised SEP OOD,
significantly on BOTH targets** (q_resp_only +0.113\*/+0.096\*; fuse +0.098\*/+0.095\*) — amortization
degrades **less than the in-model probe it replaces**. **(3) ⭐ Strict thesis test PASSES:**
`reference_q_resp_only` on Mistral (never saw Mistral, never saw squad, label-free) **0.713 vs Mistral's
own SEP 0.667, Δ +0.046 [+0.004,+0.089] excludes 0**. **(4) `q_only` COLLAPSES OOD** (0.655/0.628, on
Mistral *below* SEP) → the **response** text carries the transferable signal, not the question.
**(5) Target-in-pool ≈ +0.05** (Mistral: deploy 0.763 vs reference 0.713). **Caveats:** 2 targets; DEPLOY
rows not label-free (Finding 3 rests on the Mistral reference row); **Llama-2's SEP anomalously weak again**
(0.603, TBG:21) so its Δ-vs-SEP is inflated — **Mistral is the clean column**; base rate 0.77 ⇒ prefer
PRR/acc@coverage over AUPRC. **Second latent bug fixed:** `exp2_run.py` wrote the `checkpoint/v1` tag but
omitted `k`/`transform`, so `load_checkpoint` **KeyErrors on its own checkpoints** — fixed for future runs,
with a compat loader (`_load_exp2_ckpt`) for existing files. Artifacts: `correctness_eval_ood.py`,
`results/correctness_eval_ood.json`, `correctness_ood.log`. Full arc: EXPERIMENTS.md E39.

**⚠️ E40 — is the pooled multi-model RIDGE model-SPECIFIC, or only a question-difficulty detector?**
Asks whether, on questions where the targets genuinely disagree (SE_Llama-2 1.8 vs SE_Mistral 1.2), the
shared probe reproduces the disagreement — i.e. whether it preserves "THIS model is uncertain", not just
"this question is hard". **Answer: genuinely model-specific but THIN — 12.6% of the attainable.**
There is plenty to predict (question effect 65.7% of normalized SE variance, **model-specific residual
34.3%**; cross-model SE Spearman only 0.486–0.583; **40.1% of model-pairs differ by >0.5 nats**). Clean
result (leave-TWO-out, see below): pooled **r(dP,dY) = +0.110 [+0.027, +0.192], sign-flip p = 0.0002**,
against a matched split-half ceiling of **0.870**. **The signal lives only in the LARGE gaps** —
magnitude-weighted correlation is significant while unweighted pair-ordering accuracy is not (0.515
[0.477, 0.550]); in the LOO frame accuracy climbs 0.509 → 0.531 → 0.547 → **0.600** with gap size.
**Gated by alignment quality, not uniform:** Mistral↔Llama-3 **+0.262** (p<0.001) carries most of it,
while the low-CKA **Llama-2↔DeepSeek pair is exactly +0.001** (E30's CKA story again). **Response TEXT is
far more model-specific than the aligned hidden state** (`q_resp_only` +0.237 ≫ `z` +0.090 ≈ `ridge_z`
+0.075; ordinal only, biased frame) — the sampled answer IS the model's own output, whereas alignment
rotates hidden states into a shared frame that washes out what makes each model distinctive.

> **⚠️ METHODOLOGICAL — reuse this.** **The leave-ONE-out null is NEGATIVE, not zero.** For target T the
> probe trains on the other 3 models, so it estimates *their* SE; since the model-specific residuals sum
> to zero, `mean_{k≠T} s_k = −s_T/3`, so **a predictor that knows nothing about T is ANTI-correlated with
> it**. Proved exactly: the perfect pure-difficulty LOO predictor scores **−1.0000**. Caught only because
> the `q_only` control (question text only — identical input for every target, so it *cannot* be
> model-specific) came out −0.097 (p=0.013) instead of ~0. **Never assume chance = 0 for a LOO
> model-specificity metric.** The fix is the **leave-TWO-out** design: one ridge scores BOTH members of a
> held-out pair, so `dP = P_A − P_B` shares weights and the artifact cancels — a question-only predictor
> gives `dP = 0`, so the null really is 0. Also: the difficulty-oracle semi-partial is **biased UP** (the
> oracle is itself noisy; a noiseless difficulty predictor scores +0.27) — always print its matched null.

Caveats: N=200 questions (wide CIs; only 1 of 6 pairs individually significant, pooled result leans on
Mistral↔Llama-3); the LTO probes train on only **2** source models so +0.110 is likely a mild
UNDER-estimate; Llama-2 is the anchor (identity W) so its pairs aren't clean cross-model tests.
**Checkpoint gap found + fixed (user-caught): the whole E35/E36/E37 ridge line had NEVER saved a fitted
ridge** — `exp2_run.py --ckpt_dir` covers the proxy (`train_arm`) but not `ridge_on_z`. Now
`save_ridge_bundle()`/`load_ridge_bundle()` persist the **full inference chain** (per-model Procrustes W +
centering, scalers, label z-stats, all 10 ridges, `meta.json`) to
`stage2/runs/E40_pooled_multimodel_ridge/checkpoints/`. Artifacts: `e40_model_specificity.py`,
`e40b_lto_significance.py`, `e40c_lto_ceiling.py`, `results/e40{,b,c}_*.json`. `se_probes`, CPU, minutes,
**`--data_dir /data2/mn1025/stage1`** (3.5s vs minutes on NFS). Full arc: EXPERIMENTS.md E40.

**Cross-LLM transfer (E20) — the thesis result.** Frozen Llama-2 proxy → Llama-3-8B, 5-seed Spearman
(control = same harness on Llama-2's own 200, reproduces ID to 4 sig figs):

| arm | control (=ID) | **transfer** | retained |
|-----|---------------|--------------|----------|
| **z** (hidden only) | 0.602 | **0.056** | ~0% (chance) |
| z_q · z_q_resp | 0.590 / 0.583 | 0.116 / 0.102 | ~20% |
| **q_only** (no target LLM) | 0.494 | **0.436** | **88%** |
| **q_resp_only** (answer text) | 0.521 | **0.562** | **full** |

**Hidden states do NOT transfer** (naive PRH fails for SE across Llama-2→Llama-3; z-arms with text
bolted on stay broken, ~0.1). **Text DOES transfer** (q_only 88%, q_resp_only full) → validates the
text-reading proxy: only the model-agnostic text pathway survives a target-model swap. Tooling:
`eval_cross_llm.py` (scores any checkpoint set on another target's records, split=all); JSONs under
`runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints/cross_llm_*.json`. Fix that unblocked the Llama-3
build: `huggingface_models.py` token-boundary offset recovery (Llama-3 decodes " ?"→"?"; Llama-2 path
byte-identical; no SE/probe logic touched).

**Cross-LLM #2 (E21) — REPLICATES E20 on a different family.** Target **Mistral-7B-Instruct-v0.2**
(4096-dim, non-Llama; loads in `se_probes_llama3`, no code change), same 200 held-out ids. All 5 arms
(Llama-2 proxy → Mistral transfer, Spearman): z **0.044 (chance)**, z_q **0.116**, z_q_resp **0.102**,
q_only **0.410 (76% of the 0.540 ceiling)**, q_resp_only **0.511 (~full)**. So "text transfers,
hidden geometry doesn't" is now **two-family** (Llama-3 same-family + Mistral different-family), with a
consistent ~0.5 shared-difficulty ceiling. **Ceilings:** Llama-2↔Llama-3 SE Spearman **0.505**,
Llama-2↔Mistral **0.540** — the model-independent "question difficulty" the text arms target.

**Cross-LLM #3 (E22) — ROLE SWAP, transfer is directionally symmetric.** Trained a fresh proxy on
**Mistral n2000** (best layers TBG:31,SLT:20 via linear_ceiling_probe) and tested it on **Llama-2** —
the reverse of E21. All 5 arms — **in-dist (Mistral test-200) → Mistral→Llama-2 transfer** (Spearman):
z 0.638→**−0.002 (chance)**, z_q 0.597→**0.004**, z_q_resp 0.630→**0.037**, q_only 0.414→**0.476 (88%
of 0.540 ceiling)**, q_resp_only 0.528→**0.509**. Mirrors E21: hidden-only + hybrid arms collapse to
chance, pure-text arms transfer. "text transfers, hidden doesn't" holds regardless of direction → a
property of the model *pair*, not of Llama-2 being a special source. Checkpoints:
`runs/E22_Mistral_proxy_p1024_5arm_ckpt/` (25). New: `stage2/run.py --stage1_model_name/--stage1_dataset`
(train the proxy on any target's records).

**E23 — replication on a FRESH 1000-question held-out batch (zero overlap, proven).** No retraining:
scored the frozen REFERENCE (Llama-2) and E22 (Mistral) proxies on 1000 brand-new trivia_qa questions
(the complement of every prior build), both targets, all 5 arms. Confirms E20–E22 at 5× power (tight
std): transfer z ≈ chance both ways (0.014 / 0.031), text transfers (q_only ~0.475 = ~90% of the 0.524
ceiling, q_resp_only ~0.52); **in-dist z stays high on fresh questions (0.56 / 0.63) → the model swap,
not question novelty, kills z.** Datasets `{Llama-2-7b-chat,Mistral-7B-Instruct-v0.2}_trivia_qa_n1000`.

**⭐⭐ E24 — hidden states DO transfer after unsupervised alignment (PRH holds for SE).** The E20–E23
raw z-failure is a **basis mismatch, not incompatibility**. Ridge-level Procrustes test
(`procrustes_alignment.py`, CPU, additive; TBG only, NO SE labels in the fit): fit orthogonal W from
Mistral TBG → Llama-2 TBG on the shared 1440 train, translate Mistral's 200 test states, feed Llama-2's
frozen ridge, score vs Mistral SE → **0.545** (raw floor −0.05, Mistral skyline 0.620 → **88.8% of the
gap recovered**). Controls: mean-shift-only = floor (−0.05), random rotation = chance (0.07) → only the
LEARNED alignment recovers it. CKA 0.865 (spaces highly alignable-by-rotation). Mechanically it's a
label-free linear SE probe on Mistral (`x·Wβ_llama2`), so it sits below Mistral's supervised skyline.
**Payoff:** build an SE probe for a NEW model with no N-sample labels — just paired forward passes to
fit W, then reuse a reference probe. Caveat: needs paired hidden states (shared questions, cheap).

**⭐ E27 — alignment HELPS uncertainty estimation; label-free estimator on par with the supervised
baseline.** Mistral→Llama-2, fresh n1000, vs Mistral SE (`procrustes_e27*.py`). The aligned hidden state
adds SE info beyond the question text (E27a semi-partial +0.091, robust across dirs/eval-sets/seeds/
anchor-resamples). Best **label-free** recipe: **standardized average of aligned-z ridge + `q_resp_only`
→ Spearman 0.609 / AUROC 0.867** (no target labels), which is **on par with the actual Mistral SEP
baseline** (single-layer logistic on matched data = 0.857 AUROC; supervised-ridge proxy 0.863) and
recovers ~96% of the Spearman skyline (0.632). *(Saved official SEP 0.726 was N=400-underpowered → 0.795
at N=1000; `procrustes_e27_sep_comparison.json`.)* *(A label-FITTED 2-input
ridge combiner matches it, 0.608/0.866, but fits its weights on Mistral labels — so the AVERAGE, not the
ridge combiner, is the label-free result; `procrustes_e27_labelfree_ensemble.py`.)* Mechanistic: a linear ridge
beats the 3B proxy on aligned z (0.580 vs best arm 0.545); **late fusion (stacking) beats early fusion**
(a trained `z_resp` arm = 0.523 < pure z; adding text to z-arms hurts); the question helps only when
there's no z (`resp_only` 0.455 < `q_resp_only` 0.531). New arms `z_resp`/`resp_only` added to
`stage2/train.py`; checkpoints in `runs/E27_{zresp,resp_only}_arm/`. Full arc in EXPERIMENTS.md E27.
**Rank-fusion addendum** (`procrustes_e27_rank_fusion.py`): empirical-CDF rank average (label-free) ties
the other fusions on ID (0.608/0.866) and **TIES std-avg on squad OOD** (0.541/0.771; paired-bootstrap
Δ CI includes 0 — earlier "best OOD" was noise). **Floor control: the trivia-fit Procrustes W transfers
cross-domain** — raw Mistral-squad states (NO W) = chance (0.491 AUROC), with W = 0.743. Built
`Mistral-..._squad_n1000` for this OOD test.

**Storage:** all Stage-1 datasets + proxy checkpoints live on `/vol/bitbucket` (source of truth) AND
W&B. `push_to_wandb` defaults True (smokes excluded); dataset artifact names auto-distinct per
(model,dataset,N); back-fill datasets with `push_dataset_wandb.py`. **W&B cache is redirected off the
12 GB home quota** — `WANDB_CACHE_DIR`/`WANDB_DATA_DIR` → `/vol/bitbucket` (in `~/.bashrc` above the
`$PS1` guard). Without this, artifact staging fills home and pushes fail with `Disk quota exceeded`.

**Reference model — SAVED, 25 checkpoints** at `runs/REFERENCE_multipos_p1024_5arm_ckpt/checkpoints/`
(5 arms × 5 seeds, ~30M trainable params each, no frozen backbone; reproduces the run below to 4 dp).
TBG L22 + SLT L15, projector 1024, k=4, 5 seeds — **mean ± std:**

| arm | needs target LLM? | ID Spearman | OOD Spearman | ID AUROC | OOD AUROC |
|-----|-------------------|-------------|--------------|----------|-----------|
| **z (hidden only)** | yes (hidden states) | **0.602 ± 0.019** | 0.368 ± 0.033 | **0.807 ± 0.013** | 0.669 ± 0.014 |
| z + question | yes | 0.590 ± 0.049 | **0.402 ± 0.033** | 0.808 ± 0.025 | 0.684 ± 0.018 |
| z + question + resp | yes | 0.583 ± 0.015 | 0.398 ± 0.060 | 0.799 ± 0.012 | 0.682 ± 0.025 |
| **q_only** | **NO — nothing** | 0.494 ± 0.049 | 0.259 ± 0.047 | 0.758 ± 0.031 | 0.614 ± 0.026 |
| **q_resp_only** | answer text only | 0.521 ± 0.049 | 0.399 ± 0.073 | 0.768 ± 0.028 | 0.684 ± 0.038 |

### The three settled conclusions (details in EXPERIMENTS.md)

1. **The proxy is neither over- nor under-fitting** (E15–E17). Its 5-seed train–test gap (0.227) ≈
   optimally-tuned ridge's (0.213) — a ~0.2 gap is what *optimal* looks like at N=1440/D=8192.
   Confirmed by: `weight_decay` is a dead knob; projector-form linear=mlp (noise); the capacity curve
   is flat past width 1024; more data won't help (ridge plateaus at ~400 rows; we're already at SEP's
   2000-across-tasks data scale). The residual **−0.04 to ridge is structural** — routing z through
   soft tokens into a frozen backbone vs ridge reading all 8192 dims directly. *(An earlier
   single-seed "the proxy overfits" claim was **RETRACTED** at 5 seeds — E16.)*

2. **Negative result (single target LLM):** with hidden states available, a plain **ridge beats the
   3B proxy** (0.642 vs 0.602 ID; 0.437 vs 0.368 OOD; ridge on SLT L15 alone = 0.495 OOD), and an
   **MLP loses to ridge** at every input → the z→SE relation is **linear**; the frozen backbone has
   no nonlinear signal to add. A linear probe on hidden states ≈ **SEP**, so the z-branch re-derives
   existing work and does it worse. *(Conditional on staying in one target LLM: under the cross-LLM
   goal, ridge-on-A can't run on B at all, so there it's a baseline via alignment, not a replacement.)*

3. **⭐ Positive result / the thesis (E12/E13):** the **`q_only`** arm predicts SE **from the question
   text alone, no target-LLM forward pass** — ID Spearman **0.494** (54% of the achievable ceiling,
   82% of what the hidden state gets). A hidden-state probe cannot do this by construction. Controlled
   against TF-IDF→ridge (E13): bag-of-words gets 0.351 ID and **collapses to 0.037 = chance OOD** vs
   the 3B's 0.259 (**7× gap**) → not a surface shortcut; the 3B reads something *semantic* that
   transfers. `q_resp_only` OOD (0.399) beats z-only (0.368).

**Two ceilings (do not conflate):** label-noise ceiling ≈ **0.914 ID / 0.901 squad** (unreachable —
the SE label is a 10-sample estimate; squad's labels are as reliable as trivia's, so the OOD drop is
real transfer failure, not noisier labels). Information ceiling ≈ **0.64 ID** (the most a single
hidden state yields — the 0.64→0.91 gap is information absent from one forward pass). The proxy
recovers 66% of achievable ID.

**Provenance / do-not-cite:** `runs/..._n2000_full/` (TBG L12 — the **retracted** text-arm claims,
E6); `runs/..._n2000_TBG_L22/` (layer-only fix; JSON lost to the fixed `build_ood` mkdir bug, numbers
in `logs/tbg_L22_multiseed.log`). Reference numbers: `runs/REFERENCE_multipos_p1024_5arm_ckpt/` and
`runs/stage2_textonly_5arm_p1024/`.

---

## 📋 TO-DO (single source of truth — other docs point here)

**DONE — cross-LLM experiment #1 (E20):** frozen Llama-2 proxy → Llama-3-8B evaluated on the 200
held-out questions. **Hidden states do NOT transfer (z 0.602→0.056), text DOES (q_only 88%,
q_resp_only full).** Full result table + tooling in Current state / EXPERIMENTS.md E20. Built with
`stage1.py --only_ids`; scored with `stage2/eval_cross_llm.py`.

**DONE — cross-LLM experiment #2 (E21):** Mistral-7B-Instruct-v0.2 (different family, 4096-dim)
**replicates E20** — z 0.044 (chance), q_only 0.410 (76%), q_resp_only 0.511 (~full); ceiling 0.540.
Two-family generality established. See Current state / EXPERIMENTS.md E21.

**DONE — role swap (E22):** Mistral proxy → Llama-2 mirrors E21 (z chance, q_only 0.476, q_resp_only
0.509) → transfer is directionally symmetric. See Current state / EXPERIMENTS.md E22.

**DONE — the Procrustes alignment line (E24–E27):** label-free orthogonal map makes z readable
cross-model (weakly PRH-positive), best label-free ensemble ≈ matched Mistral SEP on AUROC, W transfers
cross-domain. See EXPERIMENTS.md E24–E27 + "Where we stand" conclusions 5–7.

**DONE — correctness-based eval (E31):** re-scored all predictors vs `incorrect` (not SE) on all 4 targets.
SE-fidelity ≠ correctness (−0.10 to −0.15 AUROC); true 10-sample SE is the best correctness detector;
label-free rank-fusion ≥ supervised SEP on correctness too; method ordering matches on 3/4 (differs on
Llama-3). `correctness_eval.py` + `correctness_eval_*.json`. See EXPERIMENTS.md E31.

**DONE — model-specificity of the pooled ridge (E40):** the ridge preserves genuine per-model uncertainty
but only 12.6% of the attainable, only on large disagreements, and only for well-aligned pairs; the
**leave-one-out null is NEGATIVE (−1.0 for a perfect difficulty predictor)** so use the leave-TWO-out
design for any such test. The multi-model ridge is now **saved** (it never had been). See EXPERIMENTS.md E40.

**Opened by E40 (new, untouched):**
- **Re-read E37's LOLO conclusions in light of the negative LOO null** — E37/E38 scored *absolute* SE
  fidelity, which the artifact does not touch, so those headlines stand; but any *model-specific* reading
  of a LOLO number is biased down. Worth one pass to check nothing in the write-up leans that way.
- **Raise the power of the E40 clean test:** N=200 held-out questions and 2-source LTO probes are the
  binding limits. A 3-source clean design needs a 5th target LLM; more questions need fresh Stage-1.
- **Does the SLM proxy preserve more model-specificity than the ridge?** [F] hints yes for `q_resp_only`
  (+0.237 vs +0.075) but only in the biased LOO frame — a clean answer needs the proxy retrained
  leave-TWO-out (GPU), which would also test whether response text is genuinely the model-specific channel.

**NOW — pick the next thrust (all open):**
- **E35 follow-ups (harden the pooling result — RESULTS ARE NOT REVIEW-VERIFIED):**
  - **Resume `/code-review`** on `amortized_ue/e35_pooling_*.py` (it was stopped before emitting findings
    for token cost) — one bug was already found+fixed, so an independent pass is warranted before trusting.
  - **Add bootstrap CIs** to the pooling deltas (currently 3–4 seeds, no CIs → "small & consistent", not
    "significant"); and **fix the α-selection asymmetry** (pooled selects α on a 3× larger val than single).
  - **✅ Re-run E35 with PROPER per-model layers (source side) — DONE 2026-08-17.**
    `e35_pooling_matched_partition_bestlayer.py` (audited clone of `matched_partition.py`; only the layer
    indexing changed): each **source at its leak-free best TBG** (Mistral 31, Llama-3 31, DeepSeek 28) →
    Llama-2 anchor (tested TBG:30 and 22). Findings: **(1) best-source lifted the hobbled models** —
    pooled@1440 Mistral 0.560→**0.594** (+0.032), Llama-3 0.584→**0.603** (+0.017); DeepSeek ~0
    (0.588→0.579/0.589, already near-best at 22). Clean isolation (best-source vs shared-22, both anchor 22):
    Mistral +0.032, Llama-3 +0.017, DeepSeek +0.001. **(2) Pooling conclusion HOLDS** — diversity effect
    (pooled−single) still ~+0.02, always positive; magnitudes corrected upward, story unchanged. **(3)
    Depth-matching hypothesis REJECTED** — anchor 30 vs 22 is a **wash** (all deltas ≤0.010, within 4-seed
    noise; 22 even marginally ahead on single-source, plausibly because Llama-2 L22 generalizes slightly
    better on test). **Chose anchor 30** (best→best internal consistency, per user), **carrying 22 as a
    cheap comparison**. ⚠️ 4 seeds, **no CIs** → "small & consistent", not "significant". JSONs
    `scratch_xllm/e35_bestlayer_matched_anchor{30,22}.json`. **Corrected Exp-2 baseline** (pooled@1440,
    anchor 30): Mistral 0.594 / Llama-3 0.603 / DeepSeek 0.579. **Still open:** re-run the loo_pilot +
    datasize_sweep with best layers too (only matched_partition was redone); add bootstrap CIs.
  - **⭐ The real untapped lever — different questions per model:** all E35 pooling used models run on the
    *same* 1440 questions (needed for alignment), so pooling adds model-diversity but **not question
    coverage**, and both single+pooled ridges are **data-saturated (~800 Q)**. Test whether giving each
    source model **distinct** questions (keeping a small shared anchor set for `W`) lets pooling actually
    improve — needs new Stage-1 generation on fresh prompts.
- **3rd-family alignment (breadth):** build Llama-3 n2000 → replicate the E24/E25 controls + the E27 SEP
  comparison on Llama-3, to check the alignment findings generalise beyond the Llama-2↔Mistral pair.
- **Anchor-count efficiency sweep for W:** how *few* paired anchors suffice to fit a good Procrustes W?
  Quantifies the only label-free cost (paired forward passes). Cheap, CPU, reuses existing data.
- **✅ Multi-target training (Exp 2) — DONE (E37).** Built `exp2_run.py` (multi-source aligned loader +
  LOLO/deploy driver, all 5 arms + late fusion) + `exp2_step1_zarm.py` (verified pipeline slice). 4-fold
  LOLO, 3-seed means: **fuse 0.664 / q_resp_only 0.648 / ridge 0.591** — label-free fusion ≥ ridge on all
  4; text beats ridge on all 4 with no target hidden states; z tracks CKA; late>early. Conservative
  bootstrap: fuse beats ridge 3/4. See Current state E37 + EXPERIMENTS.md E37 (per-seed table).
  **Ridge per-example preds now saved by E38** (`results/correctness_eval_e37.json` → `ridge_te_preds`,
  rebuild verified to 4 dp) — the SE-target paired bootstrap vs ridge is now unblocked but **not yet run**.
  The deployable all-4 proxy run (checkpoints + training curves) is DONE
  (`results/deploy_checkpoints/`, `results/deploy_curves.json`). **Deviation from plan (better):** used **same-questions
  pooling** (each Q to all sources → model-invariance signal) NOT the disjoint matched-partition, and
  anchor **TBG:30** (best→best); layers reconfirmed leak-free (Llama-2 TBG:30, Mistral/Llama-3 TBG:31,
  DeepSeek SLT:16 — E36).

**Pending / carried over:**
3. **(Partly done)** `amortized_ue/RESULTS.md` now holds the **four-model cross-LLM picture (E20–E30)**:
   reference single-LLM arms, cross-LLM transfer, alignment recovery/CKA/increment, and the full-power
   label-free-ensemble-vs-SEP table. **Still open:** fold in the exhaustive single-LLM ablation/diagnostic
   configs (TF-IDF baseline, ceilings, superseded runs) by cross-checking every `runs/` dir + E0–E19.
4. **Proxy learning curve** (`stage2/proxy_learning_curve.py --sizes 250,500,1000,1440 --seeds 3`) —
   confirm the *proxy* (not just ridge) plateaus with data. Was launched then killed by the FS outage.
5. **SEP-comparison write-up** — the honest framing (proxy ~ comparable to a SEP-style probe on the
   same data; ridge slightly beats it; the real edge is `q_only`). User will decide when to write it.
6. **⭐ Reconcile our SEP numbers vs the SEP PAPER (arXiv:2406.15927).** E27 ran the SEP *method* on our
   matched data (Mistral SEP 0.857 / Llama-2 SEP 0.795 AUROC; our label-free ensemble 0.867 on par),
   `procrustes_e27_sep_comparison.json`. **Still open:** compare these against the paper's *published*
   AUROC table for the corresponding setup, and document any residual gap (binarisation, dataset, metric
   conventions). Also root CLAUDE.md outstanding task #4.
7. **(Housekeeping)** rotate the HF token that was pasted in chat (security).

**Cancelled / resolved:** multi-layer *band* ablation (layers within a position redundant, +0.005);
the regularisation experiment (done — E16, no live dial); the "decide the thesis framing" item
(resolved — the thesis is cross-LLM transfer); the E18 checkpoint bugs (fixed).

**⚠️ Infra caveat:** `/vol/bitbucket` is a shared NFS that periodically degrades (bulk reads time out
while `ls`/writes stay fast). It has frozen jobs for hours. Test real recovery with
`time cat <records>/*.pt >/dev/null` (NOT `ls`). Node-local `/data2` (ext4, ~11T, non-NFS) is an
escape route if it stays bad. Run long jobs with a watchdog that pings on done/crash/stall.
