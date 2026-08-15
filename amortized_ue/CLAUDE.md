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

## Current state (updated 2026-08-15)

**Target LLMs (4):** Llama-2-7b-chat (reference), Mistral-7B-Instruct-v0.2, Meta-Llama-3-8B-Instruct,
and **DeepSeek-LLM-7B-Chat (E28, NEW)**. Per-target z-layers (via `linear_ceiling_probe.py`, not the
3B sweep): Llama-2 **TBG:22/SLT:15**, Mistral **TBG:31/SLT:20**, **DeepSeek TBG:28/SLT:16** (best SLT:16,
0.680; 30-layer model), **Llama-3 best SLT:31** (0.708, on n2000; `scratch_xllm/{deepseek,llama3}_layer_pick.json`).

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

**NOW — pick the next thrust (all open):**
- **3rd-family alignment (breadth):** build Llama-3 n2000 → replicate the E24/E25 controls + the E27 SEP
  comparison on Llama-3, to check the alignment findings generalise beyond the Llama-2↔Mistral pair.
- **Anchor-count efficiency sweep for W:** how *few* paired anchors suffice to fit a good Procrustes W?
  Quantifies the only label-free cost (paired forward passes). Cheap, CPU, reuses existing data.
- **Multi-target training (Exp 2):** train one proxy on Llama-2 **+** Mistral (both 4096-dim; needs a new
  multi-source Stage-2 loader), then **leave-one-out** test on unseen Llama-3. If z now transfers to
  Llama-3 → multi-target training induces a model-agnostic hidden code. Both n2000 sets exist.

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
