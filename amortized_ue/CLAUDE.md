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

## Current state (updated 2026-09-04)

**E73 (2026-09-04) — E72 for the small-tier Qwen/Gemma "set 2" (`Qwen3-8B`, `Qwen3.5-9B`,
`gemma-7b-it`, `gemma-2-9b-it`): per-model INDIVIDUAL proxy vs ridge vs SEP, ID + OOD.** Exact analog
of E72 one tier down; same 2-position input per model on **E71's `aligned_ridge` layer picks**
(Qwen3-8B 34/23, Qwen3.5-9B 31/31, gemma-7b-it 27/18, gemma-2-9b-it 41/28). New
`e73_settwo_individual.py` (adapted from E72; **per-model curve files → fixes the E72 race**).
**Result (MEAN AUROC / ρ, ID | OOD):** ridge 0.760/0.705 | 0.702/0.604 · proxy 0.747/0.691 |
0.689/0.587 · SEP 0.739/0.644 | 0.658/0.496 · true SE 0.785 | 0.738. **⭐ Reproduces E72: ridge ≥
proxy ≥ SEP everywhere, `proxy − ridge` never positive (sig-loses 4/8, ties rest)** → E15–E17 holds
for set 2. Everything < true SE (gap wider than E72's big tier). **New nuance:** E71's cross-model
*text* LOLO proxy (0.791 ID AUROC) beats the per-model *hidden-state* methods ID — the answer-text
channel is that strong for small Qwen/Gemma (E45–E53) — but per-model ridge still wins OOD and ID ρ.
Checkpoints (12 + 4 bundles) + W&B `e73_settwo_individual_ckpts:v0`. See EXPERIMENTS.md E73.

**E72 (2026-09-04) — big-tier 5×27B PER-MODEL (individual, not LOLO) supervised ceiling: proxy vs
ridge vs SEP, ID + OOD.** The complement to E65/E69/E70 (all cross-model). For each big-tier model,
train its OWN proxy / ridge / SEP on its OWN trivia n2000, all on the **identical** input
`concat([TBG:L_tbg, SLT:L_slt])` — the same per-position layers E70's `aligned_ridge` selected. New
`e72_bigtier_individual.py`; proxy = `z` arm (hidden-state-in, no text), E37/E53/E65 recipe, 3 seeds,
`h_in = 2H`. **Result (MEAN AUROC_incorrect / ρ, ID | OOD):** ridge 0.762/0.684 | 0.742/0.649 ·
proxy 0.754/0.672 | 0.732/0.628 · SEP 0.752/0.643 | 0.697/0.536 · true SE 0.760 | 0.765.
**⭐ ridge ≥ proxy ≥ SEP everywhere; `proxy − ridge` a tie on 8/10 cells, proxy never beats ridge**
→ E15–E17's "ridge beats the 3B proxy for hidden-state-in" now confirmed at 27B AND OOD (z→SE
linear; frozen backbone adds nothing). Per-model ridge ≈ true 10-sample SE for wrong-answer
detection ID (0.762 vs 0.760). proxy > SEP sig. on 3/5 OOD. Per-model supervised ≫ cross-model
transfer (OOD ρ: own ridge 0.649 vs E70 aligned 0.498 vs E65 LOLO 0.520). Checkpoints (15 `.pt` + 5
`z_bundle.pkl`) + W&B `e72_bigtier_individual_ckpts:v0`. ⚠️ gemma-2/3 per-step training curves lost
to a concurrent-write race when the folds were split across 2 GPUs (val-Spearman preserved in
`e72_train_gpu1.log`; result unaffected). See EXPERIMENTS.md E72.

## Current state (updated 2026-09-03)

**E71 (2026-09-03) — the E70 comparison for the small-tier Qwen/Gemma "set 2"** (`Qwen3-8B`,
`Qwen3.5-9B`, `gemma-7b-it`, `gemma-2-9b-it`). Built BOTH a leave-one-of-4-out `q_resp_only` proxy
(GPU, ~2 h; no clean small-tier LOLO proxy existed) AND the aligned ridge (E70's PCA(512)→Procrustes,
Qwen3-8B anchor); evaluated proxy + aligned_ridge + fuse + SEP + true SE, ID + OOD. New
`e71_settwo_lolo_aligned_ridge.py` + `results/e71_settwo_lolo_aligned_ridge.json`; checkpoints + W&B
`e71_settwo_lolo_qresp_ckpts:v0` / `e71_settwo_aligned_ridge_bundles:v0` (verified).
**Result (MEAN AUROC_incorrect / Spearman, ID / OOD):** proxy 0.791/0.697 / 0.678/0.537 · fuse
0.780/0.702 / 0.678/0.546 · aligned_ridge 0.741/0.643 / 0.637/0.425 · SEP 0.714/0.602 / 0.634/0.407 ·
true SE 0.785 / 0.738. **⭐ Set 2 reproduces the set-1 (E37/E38) ordering — proxy > aligned_ridge
(ID +0.050 AUROC mean, sig. on 3/4 folds) — and `fuse` does NOT beat the proxy** (ID fuse−proxy ≤ 0
on all 4). This is the OPPOSITE of E70's big tier (proxy≈ridge, fuse wins) and **retro-confirms the
E70 explanation**: the big-tier convergence was driven by the 3 near-identical Qwen-27B siblings
making LOLO ≈ in-distribution for the hidden-state ridge; set 2's 4 genuinely distinct models behave
like set 1. gemma-2-9b-it again the weak fold (proxy 0.698 ID, only fold true SE sig. leads);
everything loses to true SE OOD (E39/E69/E70 pattern); SEP weakest throughout. See EXPERIMENTS.md E71.

**E70 (2026-09-03) — big-tier 5×27B LOLO ALIGNED-RIDGE (label-free), ID + OOD.** Closes the other
half of the E37 thesis experiment for the big tier (E65/E69 did only the `q_resp_only` text arm).
Dimension wall (Qwen 5120 / gemma-2 4608 / gemma-3 5376, none 4096) fixed with per-model **PCA(512)
→ orthogonal Procrustes** into a **Qwen3.5-27B** anchor frame (user-chosen). Label-free: PCA + W fit
on each model's trivia-n2000 train, pooled ridge on the other 4 models' aligned z, held-out model
scored on trivia n1000 (ID) + squad n1000 (OOD); baselines reloaded per-id from E65/E69. New
`e70_bigtier_lolo_aligned_ridge.py` (CPU, ~4 min) + `results/e70_bigtier_lolo_aligned_ridge.json`.
**Result (MEAN AUROC_incorrect, ID / OOD):** `aligned_ridge` 0.748 / 0.684 · `q_resp_only` 0.747 /
0.694 · **`fuse` = rank-fusion(aligned_ridge ⊕ q_resp_only) 0.764 / 0.715** · true SE 0.760 / 0.765 ·
SEP 0.736 / 0.670. (1) aligned-ridge alone ≈ text arm alone — neither dominates; aligned-ridge
**collapses on gemma-3-OOD** (0.530, alignment fails for the near-degenerate-SE outlier). (2) ⭐
**label-free `fuse` is the best predictor** — E37's late-fusion headline replicates at 27B: best ID
number (on par with true SE, sig. > text on 3/5), OOD clearly > either arm and > SEP (sig. > text on
3/5) but **still < true 10-sample SE OOD** (consistent with E39/E69). See EXPERIMENTS.md E70.

**E69 (2026-09-03) — the squad OOD counterpart of E65 (big-tier 5×27B LOLO `q_resp_only` proxy).**
E65's flagged open item. Same 15 `E65_bigtier_lolo_qresp` checkpoints (held-out target never in the
fold's pool, trivia only), scored on each held-out model's `squad_n1000` (E55 builds, finished
2026-09-03); SEP/ridge fit on that model's own trivia n2000. New additive
`e69_bigtier_lolo_squad_ood.py` + `results/e69_bigtier_lolo_squad_ood.json`; E65 outputs untouched.
**Result (MEAN AUROC_incorrect):** proxy 0.694 · true SE 0.765 · SEP 0.670 · own-model ridge 0.735.
Proxy **loses to true 10-sample SE on all 5 folds, every CI excludes 0** (−0.046 to −0.085) — same
OOD pattern as E39/E52/E54/E68; E65-final's trivia parity with sampling does not survive the shift.
Proxy **beats supervised SEP on 3/5** (Qwen3.5 +0.043\*/Qwen3.8 +0.056\* sig., gemma-3 +0.035 n.s.),
ties Qwen3.6, nominally behind on gemma-2 (−0.021 n.s.); beats SEP on Spearman on all 5 (0.520 vs
0.435). Below the own-model ridge ceiling on all 5 (context, not label-free). The E65-final family
split holds: gemma-2 softest vs SEP, gemma-3 weakest overall. See EXPERIMENTS.md E69.

**E68 (2026-09-01) — extend the TRUE LOLO-proxy squad OOD eval (E52/E54) from Llama-2/Mistral to
Llama-3 + DeepSeek.** Same E37/E43 LOLO `q_resp_only` checkpoints (held-out target never in that
fold's training pool), E55's `squad_n1000` builds, E41 fixed SEP layers. No retraining; new additive
script + outputs (`e68_lolo_squad_llama3_deepseek.py`, `results/e68_*`), E52/E54 JSONs untouched.
**Result: identical shape to E54.** Proxy beats SEP on both metrics, both targets (Δρ Llama-3
+0.174\*/DeepSeek +0.276\*; Δauroc_inc +0.050\*/+0.119\*); loses to true 10-sample SE, a real gap not
"on par" (−0.060\*/−0.039\*, both CIs exclude 0). All 4 targets now cover this cell: proxy > SEP
always, proxy < true SE always. See EXPERIMENTS.md E68.

## Current state (updated 2026-08-29)

**E63 (2026-08-29) — leave-TWO-out cross-model test: a `q_resp_only` proxy trained on 6 target LLMs,
scored on the SE disagreement of 2 held-out ones (DeepSeek-LLM-7B-Chat vs Qwen3-8B) → response text
IS the model-specific channel, ~3.6× stronger than E40's aligned-hidden-state ridge.** Settles the
open E40 task (*"proxy retrained leave-TWO-out (GPU) … test whether response text is genuinely the
model-specific channel"*). ONE `q_resp_only` proxy (frozen Llama-3.2-3B + LoRA, no hidden states, no
alignment), pooled from Llama-2 / Mistral-v0.2 / Llama-3 / Qwen3.5-9B / gemma-7b-it / gemma-2-9b-it
trivia n2000 (SE z-scored per model; 8640/2160 pooled rows), DeepSeek + Qwen3-8B held out entirely.
Same recipe as E53 / deploy ckpt (3 seeds, batch 8 × grad-accum 4 = eff 32, proj 1024, k=4, 10 ep).
The **same** proxy scores both held-out models on their **full shared 1000-Q set** (id overlap
1000/1000 identical, asserted); `predicted_diff = proxy(DeepSeek) − proxy(Qwen3-8B)` vs
`true_diff = SE(DeepSeek) − SE(Qwen3-8B)`. **Null IS 0** (one proxy, both members; the question text
is identical for the pair so a difficulty-only predictor emits `predicted_diff = 0` — no
leave-ONE-out fold artifact; empirically `predicted_diff` mean −0.014). **Results (N=1000):**
**(a)** raw Spearman(dP,dY) **+0.399 [+0.337, +0.460]**, Pearson **+0.501 [+0.439, +0.559]**, overall
sign-agreement **0.643 [0.612, 0.676]** on 701 non-tied Qs (qnorm/E40b variant: +0.363 / +0.473 /
0.598) — vs **E40's pooled leave-TWO-out ridge r = +0.110** on the same estimand. **(b)** sign-agreement
monotone in the true gap: Q2 0.562 → Q3 0.622 → **Q4 (largest gap) 0.730** (E40's top-9% peaked at
0.600; here the top-25% is 0.730); Q1 all-tied = 0.500. **(c)** sanity — the LTO proxy (never saw
either model) has strong absolute SE-fidelity on both: DeepSeek ρ **+0.770**, Qwen3-8B ρ **+0.727**
(above E62's reference proxy on its cross-model targets, 0.58–0.68). **Conclusion: the model-specific
uncertainty signal lives in the response text, not (much) in the aligned hidden state** — the sampled
answer *is* each model's own output, whereas Procrustes alignment washes out what makes each model
distinctive (consistent with E40 #5 / E33 / E38). **Failure analysis** (`e63_lto_failure_analysis.py` → `results/e63_lto_failures.json`): of 701 real
disagreements, 407 right / 294 fail — 206 opposite-direction + **88 no-direction** (`predicted_diff`
exactly 0 because both models gave the *identical* canonical answer, a hard ceiling on a
text-only proxy, not a bug); strict hit-rate 0.581 if those 88 are misses. Failures concentrate
at small gaps (Q4 hit-rate 0.727 → Q1 0.497) and the proxy systematically **under-reads DeepSeek's
uncertainty** (158/206 opposite-direction failures have DeepSeek as the truly-more-uncertain one —
it is the unseen *family*). Caveats: one held-out pair (not E40b's 6-pair
pool), but N=1000 = 5× E40; the two models differ in base rate (SE 0.804 vs 0.561) — the qnorm
variant controls for it and still gives +0.363. Infra: `e63_gpu_swap.sh` (GPU1, interrupted
gemma-3-27b-it at 1420/2000 — resumed clean, one reload, 0 lost; first try OOM'd at a 16 GB fence,
relaunched at 30 GB) + two new safety nets: `e63_lane_safety_net.sh` (force-CONT the lane if the swap
script is `kill -9`'d / after 3 h) and `e63_gpu_bridge.sh` (hold the freed GPU1 slack above gemma-3's
budget through the ~30 s handoff — cannot OOM the reload). Results:
`results/e63_lto_{deepseek_qwen3_8b,disagreement_table,examples_curated,train_curves}.json`,
checkpoints `stage2/runs/E63_lto_6model_qresp/checkpoints/` + **W&B `stage2_ckpts_E63_lto_6model_qresp:v0`**
(verified by fetch). Full arc: EXPERIMENTS.md E63.

**E64 (2026-08-29) — is E45's gemma-2-9b-it zero-shot LOSS a base-rate artefact? → NO, it is a
real per-model transfer failure.** E45's DEPLOY proxy lost to true 10-sample SE only on
gemma-2-9b-it (AUROC_incorrect 0.722 vs 0.769, −0.047\*), which is also a base-rate outlier
(mean_acc 0.684 vs 0.42–0.56 for the other 3). New additive `e64_gemma_baserate_reanalysis.py`
(read-only over E44/E45 records, no retraining; Stage A re-runs the frozen proxy's forward pass —
**reproduces E45's AUROCs to 3 dp** — and persists per-question scores to
`results/e64_perid_preds.json` so future Qwen/Gemma reanalyses need no GPU). Three checks: (1)
restrict all 4 targets to gemma-2-9b-it's 316 wrong questions; (2a) downsample the other 3 targets'
wrong answers to match gemma-2-9b-it's *count*; (2b) fully balanced 316-wrong + 316-correct
(incorrect-rate exactly 0.5 for every target), 1000 resamples. **Result:** matching the wrong-answer
count leaves the other 3 targets' proxy-vs-SE delta **unchanged** (+0.053/+0.009/+0.078, identical
to 3 dp); the balanced 0.5/0.5 subset reproduces gemma-2-9b-it's **−0.047 in 100% of 1000
resamples** while the other 3 stay firmly positive. **Separation diagnostic:** `q_resp_only`'s
wrong-vs-right score margin for gemma-2-9b-it is compressed (**0.545** vs 0.81–0.90 for the other 3),
while **true SE's** margin for gemma-2-9b-it is completely normal (0.70 vs 0.65–0.84) — sampling on
gemma-2-9b-it's *own* distribution works fine; the deploy proxy reading its *response text* does
not. Consistent with E47 (gemma-2-9b-it SE-fidelity *rank* correlation fine at 0.674, *class
separation* not). **No E45 headline overturned** — it removes the base-rate caveat and localises the
loss. Infra: `e64_gpu_swap.sh` (E61/E62/E63 pattern, GPU0, interrupted Qwen3.8-27B — one retry
attempt + a reload, 0 records lost). Results: `results/e64_{perid_preds,gemma_baserate_reanalysis}.json`.
Full arc: EXPERIMENTS.md E64.

**⭐ E66 (2026-08-30) — swap the proxy BACKBONE (Llama-3.2-3B → Qwen2.5-3B) and re-run the thesis
experiment: the result is backbone-agnostic.** Objection tested: does E37/E38's headline hold
*because* the proxy shares Llama-family pretraining lineage with the Llama-family targets? Swapped
the frozen backbone to `Qwen/Qwen2.5-3B` (base, Apache-2.0, d_model 2048, 15.8M LoRA params), ONE
leave-one-LLM-out fold — **Mistral held out**, trained on Llama-2 + Llama-3 + DeepSeek trivia n2000,
arm `q_resp_only` (text only, no hidden states / layers / alignment), 3 seeds, recipe identical to
E37/E53/E63/E65. Eval on Mistral's fresh shared-ID trivia n1000. **Results (AUROC_incorrect /
Spearman-vs-SE):** Qwen2.5-3B proxy **0.775 / 0.647** vs the Llama-3.2-3B same-fold proxy 0.767 /
0.635 → **Δ AUROC_inc +0.008 [−0.009, +0.025], CI includes 0 (statistically identical)**; still
beats Mistral's own supervised SEP (fixed TBG:31) by **+0.061 [+0.033, +0.090]\*** (same headline as
E38); on par with / slightly edges true 10-sample SE (+0.028 [+0.000, +0.056]); both text proxies
beat the white-box own-model ridge ceiling (0.724) — SE-fidelity ≠ wrong-answer detection
(E31/E38 #4). **Conclusion: the transferable predictive content is in the question+response TEXT,
which any competent small LM extracts — not representational kinship between proxy and targets.**
**No shared code changed** — backbone swap is one `Stage2Config(proxy_model=...)` in the new script;
`train_arm` round-trips `proxy_model` via `cfg.as_dict()` into checkpoint meta, `_cfg_from_meta` /
`ProxyModel.__init__` rebuild it on load; LoRA `q/k/v/o_proj` + projector `backbone.config.hidden_size`
work unchanged for Qwen2.5. Caveats: ONE fold (not a 4-fold table), 3 seeds; N=1000 eval (CIs
±0.02–0.03). Infra: `e66_gpu_swap.sh` (E61–E65 SIGSTOP-lane borrow, GPU1, ~37 min, 0 records lost —
interrupted Qwen3.8-27B trivia n1000 resumed to 1000/1000). Qwen2.5-3B weights cached at
`/data2/mn1025/hf_cache`. Artifacts: `e66_qwen25_proxy_lolo.py`, `e66_gpu_swap.sh`,
`results/e66_qwen25_proxy_lolo{,_train_curves}.json`, checkpoints
`stage2/runs/E66_qwen25_proxy_lolo/checkpoints/Mistral-7B-Instruct-v0.2/` (3), W&B
`stage2_ckpts_E66_qwen25_proxy_lolo:v0` (run `44rn7kmf`, verified). Full arc: EXPERIMENTS.md E66.

**E65 (2026-08-30, FINAL) — 5-fold leave-one-LLM-out `q_resp_only` proxy over the 5 big-tier 27B
targets** (`Qwen3.5/3.6/3.8-27B`, `gemma-2-27b-it`, `gemma-3-27b-it`). Each fold: proxy (frozen
Llama-3.2-3B + LoRA, E53/E63 recipe, 3 seeds) trained on the **other 4 models' pooled n2000** trivia
(SE z-scored + features scaled PER model), scored seed-averaged on the held-out model's **full n1000
shared-ID trivia set (all 1000 rows)**; SEP + ridge baselines fit on that model's own n2000 tr/va →
predicted onto the disjoint n1000 (n2000 ∩ n1000 = 0, asserted). **The 15 preliminary checkpoints
are reused unchanged** (training pool doesn't depend on the eval set); `--stage eval --eval_n 1000`.
**Result — the thesis is FAMILY-DEPENDENT at the 27B tier:**
- **Qwen-27B (×3): thesis holds.** proxy AUROC_incorrect on par with true 10-sample SE (Δ CIs all
  include 0: Qwen3.5 −0.002, Qwen3.6 −0.002, Qwen3.8 −0.018) and **significantly beats supervised
  SEP on 2/3** (Qwen3.6 +0.045\*, Qwen3.8 +0.033\*; Qwen3.5 +0.018 n.s.) — label-free, no sampling,
  no target states/labels.
- **Gemma-27B (×2): does NOT extend.** **gemma-2-27b-it = significant loss to true SE (−0.056\*)** —
  same direction *and now same significance* as E45/E64's gemma-2-9b-it answer-text compression, at
  27B with proper power. **gemma-3-27b-it weak across the board** (proxy 0.666 < SEP 0.699 < ridge
  0.719, none sig.; near-degenerate SE, best_split 0.328 — every SE-derived predictor struggles).
- **Means** (AUROC_incorrect / Spearman-vs-SE): proxy **0.747 / 0.626**, true SE 0.760 / —, SEP
  0.736 / 0.607, white-box ridge 0.754 / 0.668. Proxy > SEP on ρ (4/5 folds), never sig. worse than
  SEP on AUROC any fold, still < the white-box ridge (needs target states+labels).
**⚠️ Corrects the preliminary 200-row read:** proxy mean 0.799→0.747 (now nominally *below* true SE
not above); the preliminary's only "significant" delta — gemma-3-27b-it proxy > true SE +0.123\* —
**collapses to +0.014 n.s.** (200-row true-SE AUROC 0.570 was a small-sample artefact, 0.652 at
n1000); the proxy-vs-SEP edge that only *touched* 0 at 200 rows now clears it on 2 folds. SEP/ridge
still pick very late layers (Qwen 27B ≈64 L → TBG/SLT 62–64; gemma-2-27b → 43–45; gemma-3-27b → 58);
all ridges max α=1e4. Infra: `e65_eval_n1000.sh` borrowed GPU1 gap-free (poll → SIGSTOP lane →
`e63_lane_safety_net.sh` + slack-holder `gpu_reserve --retry_secs` → kill resumable stage1 child →
`--stage eval --eval_n 1000` ~40 min 3B-inference-only → `e65_bridge.sh` handoff → SIGCONT lane; ran
21:35→22:17, rc=0, lane resumed clean, 0 records lost). Shared-code (all additive, backward-compat):
`Stage2Config.stage1_run_name`, `Stage2Data` plumbing, `arm_preds(run_name=None)`,
`gpu_reserve.py --retry_secs`, `e65_bigtier_lolo.do_eval(eval_n=...)` / `do_check(require_manifest=...)`.
Results `results/e65_bigtier_lolo_n1000.json` (FINAL; 5 folds + `_summary` + per-id preds);
`results/e65_bigtier_lolo.json` = preliminary/superseded. Checkpoints
`stage2/runs/E65_bigtier_lolo_qresp/checkpoints/<held>/` (15), **W&B
`stage2_ckpts_E65_bigtier_lolo_qresp:v0`** (run `3lg7ycm4`, verified). Full arc: EXPERIMENTS.md E65.

**E62 (2026-08-29) — `q_resp_only` ALONE (reference proxy text arm, no fusion) vs each target's OWN
supervised SEP, all 4 alignment targets.** New additive `e62_qresp_alone_vs_sep.py` (reuses
`compute_sep`/`score_block` from `se_fidelity_proxy_vs_sep.py`; `--dry_run` checks the CPU setup).
Fills the gap E27/E29/E30 left: that comparison existed cleanly only for Mistral; Llama-3/DeepSeek
had a point estimate buried in `procrustes_e30_ens_vs_qresp_*.json` with no CI vs SEP, Llama-2 had
nothing, and no target had a paired-bootstrap `(q_resp_only − SEP)` delta. (E51's
`se_fidelity_proxy_vs_sep.json` does **not** cover this — its "proxy" is the fused ensemble.) proxy
= `REFERENCE_multipos_p1024_5arm_ckpt` `q_resp_only` 5 seeds seed-averaged, one forward pass, **no
target hidden states / labels / sampling / fusion**; SEP = E41-fixed TBG layer, fit on target's OWN
n2000 train; eval = fresh trivia n1000 (0 id-overlap); paired bootstrap 10k. **Results (Spearman /
AUROC_se; Δ = q_resp_only − SEP):** Llama-2 (ID) 0.622/0.835 vs 0.523/0.779 → **Δρ +0.095\* / Δauc
+0.056\***; Mistral 0.587/0.852 vs 0.548/0.834 → Δρ +0.035 n.s.; Llama-3 0.581/0.827 vs 0.596/0.843
→ Δρ −0.014 n.s.; DeepSeek 0.683/0.857 vs 0.583/0.805 → **Δρ +0.098\* / Δauc +0.053\***.
**The label-free text-only arm is on par with or beats the matched SEP on SE-fidelity — sig. better
on Llama-2 (ID) + DeepSeek, tie on Mistral + Llama-3, never sig. worse.** Reproduces the E30
partials (Mistral/Llama-3/DeepSeek) to full precision. Consistent with E33 ("`q_resp_only` is the
deployable primitive"). Results: `results/e62_qresp_alone_vs_sep.json`. Full arc: EXPERIMENTS.md
E62. **Infra:** ran via `e62_gpu_swap.sh` — SIGSTOP the training LANE (a stopped-but-alive pid keeps
the watchdog dormant, so no need to kill the watchdog as in E61), kill+resume the resumable stage1
child, fence the freed memory with `gpu_reserve`, run, SIGCONT the lane. Cost: one lane retry
attempt + a model reload on the interrupted Qwen3.8 build, nothing lost.

**E61 (2026-08-29) — RQ1 inference-latency benchmark.** New additive `rq1_latency.py` +
`run_rq1_latency*.sh` + `resume_training_queue.sh`. Llama-2-7b-chat AND Mistral-7B-Instruct-v0.2,
each on its own 200-q held-out test split, one L40, bs=1, 10 warm-ups, `torch.cuda.synchronize()`
bracketing. **ms/question (Llama-2 / Mistral):** Block A (1 canonical fp32 gen) 242.6 / 281.0 ·
Block B (10 samples + DeBERTa clustering) **3647 / 4104** (sampling 2513/3093, clustering 1135/1012) ·
Block C (1 `q_resp_only` proxy pass, deploy ckpt, bf16 Llama-3.2-3B) **41.5 / 40.1**; batched bs=32
Block C ~299 q/s both. **Speedups: B/C bs=1 87.9× / 102.5× · end-to-end (A+B)/(A+C) 13.7× both.**
Sanity: Block B re-derived CAE/n_clusters 0.570/2.71 (Llama-2), 0.465/2.33 (Mistral) — match
stored labels. Absolute ms are L40-specific — ratios are the result; fp32 target vs bf16 proxy is the
honest as-built comparison. Results: `results/rq1_latency_{Llama-2-7b-chat,Mistral-7B-Instruct-v0.2}.json`.
Full arc: EXPERIMENTS.md E61.
**⚠️ E61-efficiency addendum (2026-09-01) — see EXPERIMENTS.md.** Audited all E61 arithmetic (reproduces
exactly). **Reporting fix:** the "~1100×/~1225× at batch 32" figure is Block-B(bs=1) ÷ batched-proxy —
an SE-step ratio pairing an un-batched baseline with a batched proxy, **NOT end-to-end**. True
end-to-end with the batched proxy = (A+B)/(A+C_batched) = **15.8× / 15.4×**. Added token counts (input
~160, generated ~3/gen, DeBERTa ~32 fwd/q, proxy 24 tok), **estimated FLOPs** (baseline ÷ proposed
~11.5×; 2·P·T model, documented), proxy peak mem 6259 MiB (bf16 3B), fwd-only ≈ tok+fwd (tokenizer adds
~0). Caveats documented: Block C pre-tokenizes outside timed region (≈0 ms impact), Block B uses 2
warm-ups not 10, fp32 targets vs bf16 proxy, Block B = as-built Stage-1 sampler not an optimal SE
sampler. New: `e61_efficiency.py`, `e61_eff_gpu_swap.sh`, `results/e61_efficiency_*.json`. **Infra lesson:** stop the training WATCHDOG (not just the lane) before
benchmarking on a shared card — it resurrects the GPU lane mid-run; a memory fence can't help (fp32
7B peaks ~33GB of the 44GB card).

**E56 (2026-08-27) — how much of SE's wrong-answer signal survives cheaper supervision?** New
standalone `supervision_signal_compare.py` (read-only, no GPU, no target-LLM calls) scores 5 signals
by AUROC vs `incorrect` on trivia_qa **n2000, full set no split**, all 4 original targets, with a
10k paired bootstrap (shared indices; `paired_bootstrap_auc`/`ci` reused from `correctness_eval.py`).
**AUROC (Llama-2 / Mistral / Llama-3 / DeepSeek):** SE-continuous 0.787/0.752/0.773/0.815 ·
**`n_clusters` 0.782/0.750/0.769/0.810** (within +0.002–0.006 of SE, sig. on 3/4, tied on Mistral —
**the entailment clustering carries nearly all the signal; the entropy formula on top adds almost
nothing**) · MC sequence entropy 0.749/0.747/**0.784**/0.782 (drop the entailment model: −0.03 on
Llama-2/DeepSeek, wash on Mistral, nominally *ahead* on Llama-3) · **SE_binary 0.723/0.685/0.709/0.727
(−0.06 to −0.09 vs continuous SE, every CI excludes 0 — binarising the label is the biggest single
loss, and SE_binary scores below even raw `n_clusters` on all 4)** · perplexity 0.629/0.684/0.565/0.610
(single canonical pass, no sampling: −0.07 to −0.21, sig. on all 4). Llama-2 SE 0.787 ≈ E31/E38's true
10-sample SE ~0.783 (consistency check). **Caveat: full-set AUROC, no train/test split — the ranking
of signals + the deltas are the result, not the absolute numbers.** Artifacts:
`supervision_signal_compare.py`, `results/supervision_signal_compare_{llama2,mistral,llama3,deepseek}_trivia.json`.
Full arc: EXPERIMENTS.md E56.

**🔄 E55 (2026-08-26, IN PROGRESS) — data-readiness, not a result: DeepSeek/Llama-3 squad builds
(done) + a "nothink" regeneration of all 5 Qwen targets (running).** Two gaps closed/closing: (1)
squad correctness studies (E39/E52/E54) were stuck at 2 targets — DeepSeek + Llama-3 now have
`squad_n1000` too (1000/1000 each, same question selection as Llama-2/Mistral's), bringing squad
coverage to 8/14 targets (4 original + Qwen3-8B/Qwen3.5-9B/gemma-7b-it/gemma-2-9b-it). (2) Qwen's
`<think>` generation (E44's Qwen3.8-27B 65/1000-in-40h stall, Qwen3.5-9B's ~5-6% non-convergent
tail) is now disabled outright via `_DISABLE_THINKING_MODELS` in `huggingface_models.py`
(`apply_chat_template(..., enable_thinking=False)` — this pipeline never called
`apply_chat_template` before, so E44's `tolerate_thinking` fix never actually reached Qwen's real
switch) + a `skip_special_tokens=False` fix for the token-count desync that caused (confirmed live
on Qwen3.5-9B/Qwen3.6-27B). Writes to new `_nothink`-suffixed dirs, old `_full` dirs untouched — all
E44-E54 results stay reproducible. **Small tier (Qwen3-8B, Qwen3.5-9B) fully done** (trivia
n1000/n2000 + squad n1000, 6/6 builds verified). **Big tier in progress** via a dual-GPU
work-stealing queue (`lane_a_gpu0.sh`/`lane_b_gpu1.sh` + `watchdog_lanes.sh` for crash recovery) —
Qwen3.5-27B trivia n1000 done, n2000 running; Qwen3.6-27B n1000 running; Qwen3.6-27B n2000 +
Qwen3.8-27B n1000/n2000 still queued. **Live fencing bug caught+fixed:** a one-shot GPU-memory hold
computed before the small-tier phase went negative when free memory dipped below the big-tier
budget, silently skipping protection for the whole lane — fixed with per-phase dynamic re-fencing.
**All completed builds already auto-pushed to W&B, verified via `wandb.Api()`** (`push_to_wandb`
defaults `True`, none of these scripts opt out) — no manual push needed. Full arc + status table:
EXPERIMENTS.md E55. **Do not cite `_nothink` numbers as final until the big-tier queue drains** —
check record counts on disk before use.

**E54 (2026-08-25) — the TRUE LOLO proxy's CORRECTNESS (not just SE-fidelity) on squad OOD, the
last open cell of {model-seen/unseen} × {trivia/squad} × {SE-fidelity/correctness} for the 2
targets with squad data.** E39 ran squad correctness but had to substitute the DEPLOY proxy
(target's own trivia data WAS in its pool) since E37's LOLO run had no checkpoints yet; E52 later
scored the true LOLO proxy (checkpoints now exist) on squad but only against SE, never
`incorrect`. New additive script `correctness_eval_lolo_squad.py` closes it: same LOLO
`q_resp_only` checkpoints (trained on the OTHER 3 targets, zero exposure to this target OR squad),
scored against actual wrong answers. **Result: LOLO still beats SEP and ridge on both targets**
(AUROC_incorrect Llama-2 0.729 vs SEP 0.621/ridge 0.641, Δ vs SEP +0.108\*; Mistral 0.735 vs SEP
0.669/ridge 0.703, Δ vs SEP +0.066\*) — but unlike E38's in-distribution result, **the gap to true
10-sample SE here is real, not "on par"** (Δ −0.055\*/−0.039\*, both CIs exclude 0), matching E39's
general OOD finding now confirmed for the true LOLO proxy specifically. **Infra: hit the same NFS
stall twice while launching** (once from forgetting `--trivia_dir`, once because squad itself had
never been staged off NFS) — fixed by parallel-copying both targets' squad n1000 records to
`/data2` (`find | xargs -P32 cp`, <15s, vs a bulk read that itself timed out at 30s on the then-
degraded NFS) at the user's explicit request; squad now has a local copy for these 2 models,
closing part of [[use-data2-not-nfs]]'s trivia-only limitation. Full arc: EXPERIMENTS.md E54.
Artifacts: `correctness_eval_lolo_squad.py`, `results/correctness_eval_lolo_squad.json`,
`/data2/mn1025/stage1/{Llama-2-7b-chat,Mistral-7B-Instruct-v0.2}_squad_n1000_full/`.

**⭐ E53 (2026-08-25) — reverse-E45: a proxy trained ONLY on the 4 Qwen/Gemma small-tier models,
zero-shot on Llama-2/Mistral — beats SEP on both metrics, ties true SE on correctness, never saw
either target.** E45 went original-4 → Qwen/Gemma; this is the reverse direction. Pooled
`q_resp_only` (text only, deploy-style, no held-out) from Qwen3-8B/Qwen3.5-9B/gemma-7b-it/
gemma-2-9b-it's n2000 train/val, trained ONE proxy, scored zero-shot on Llama-2 + Mistral's fresh
n1000 (zero exposure to either target in any form). **Result:** proxy beats SEP on SE-fidelity
(Spearman 0.632/0.634 vs SEP 0.523/0.548) and correctness (AUROC_inc 0.748/0.746 vs SEP
0.681/0.714) on both targets, and is **statistically on par with true 10-sample SE on correctness**
on both (Δ vs true SE: −0.013/−0.000, both CIs include 0) — stronger than E45's own direction (1/4
targets there lost to true SE). **Ridge context (full-access ceiling, NOT a fair opponent — cannot
run zero-shot by construction):** the zero-access proxy's Spearman (0.632) actually **exceeds**
Llama-2's own ridge ceiling (0.585) and ties Mistral's (0.634 vs 0.632/0.647) — the access-vs-no-
access gap is smaller than expected, not proof the proxy beats ridge in general (E8-E10's "ridge >
3B proxy given full access" stands). **Two infra fixes along the way, both real:** (1) batch_size=32
(the established recipe) OOM'd — Qwen3.5-9B's `<think>` traces occasionally hit the full
`max_seq_len=256`, and a T=256, B=32 forward pass needs far more memory than any prior `q_resp_only`
run ever actually processed. Fixed with **gradient accumulation** (added to `exp2_run.train_arm` via
`cfg.grad_accum`, additive, byte-identical at the default) rather than just training at a smaller
batch and hoping it's equivalent — mathematically exact reproduction of the batch=32 recipe (no
batchnorm in `ProxyModel`), verified on a toy model to float32 precision. (2) A co-tenant raced into
`build_bigtier_n2000_gpu0_resume.sh`'s unfenced "poll-then-launch" loop mid-load and OOM'd it —
patched with `gpu_reserve.py` fencing (a tool this project already built after an earlier, identical
failure mode, but this script had never adopted it) + restored the dropped `Qwen3.5-27B` to the
queue. **Also built (after the SEP-Spearman numbers were correctly challenged as looking low): a
canonical, non-hand-typed SEP reference** — `build_sep_reference.py` → `results/sep_reference_values.json`
(8 targets × 3 settings, extracted from `se_fidelity_proxy_vs_sep.json`). The "too low" concern
resolved to a metric mismatch, not a bug: SEP is a single-layer LOGISTIC classifier (AUROC-native);
its Spearman is a repurposed use of a classifier probability against a full continuous-ranking
question it was never optimized for — mechanically why it reads lower than `ridge` (a proper
regressor for the same task). Independently re-verified from scratch (CPU-only `compute_sep`),
matched to 4 dp. `e53_full_comparison.py` consolidates true SE / SEP / ridge-context / proxy (both
metrics, both targets) into ONE file, replacing what were briefly three separate scripts/outputs.
Full arc: EXPERIMENTS.md E53. Artifacts: `e53_{train_qwengemma_deploy,eval_on_llama2_mistral,
full_comparison}.py`, `build_sep_reference.py`, `results/{e53_*,sep_reference_values}.json`,
`stage2/runs/E53_qwengemma_deploy_qresp/checkpoints/`.

**E52 (2026-08-24) — the LOLO proxy (zero exposure to this target) tested on squad OOD, closing
the one setting/target combination E51 left untested.** E51 had `lolo` (LOLO proxy, but only ever
on trivia_qa) and `squad` (squad OOD, but with the DEPLOY proxy, which includes the target in its
training pool). Never combined: LOLO proxy × squad — simultaneously cross-LLM (never this target)
and cross-dataset (never squad), the hardest transfer regime available. New `lolo_squad` setting in
`se_fidelity_proxy_vs_sep.py` (+ `arm_preds_per_seed_prefixed` helper, needed because the LOLO
checkpoint dir holds all 4 folds' files together and the existing glob would silently mix targets).
Llama-2 + Mistral only (the 2 targets with squad records), reusing the E37/E43 LOLO checkpoints and
the E41 fixed-layer SEP recipe, no retraining. **Result: proxy beats SEP on both metrics, both
targets, every CI excludes 0** — Llama-2 Δρ **+0.378** [+0.32,+0.44] (the largest single delta
recorded anywhere in E51/E52), Mistral Δρ **+0.123** [+0.07,+0.18] (smaller than DEPLOY's squad row,
+0.168, consistent with E42's "in-pool data narrows the shift penalty" but not a controlled
ablation of that). Per-seed spread is the widest seen in the project (least-informed regime).
Paused/resumed a live GPU1 Stage-1 build (`Qwen3.6-27B`, SIGTERM+relaunch, confirmed resumed from
498→499 records) to get GPU access, with explicit user go-ahead — same pattern as E51's infra note.
Full arc: EXPERIMENTS.md E52. Artifacts: `se_fidelity_proxy_vs_sep.py` (`lolo_squad` setting),
`results/se_fidelity_proxy_vs_sep.json`, `logs/lolo_squad_eval.log`,
`build_bigtier_n2000_gpu1_resume_qwen36.sh`.

**⭐ E51 (2026-08-23) — the direct proxy-vs-SEP SE-fidelity head-to-head, across every regime
built so far: proxy wins.** No prior script had put `q_resp_only` and SEP side-by-side against the
SAME continuous SE label on the SAME held-out rows with a paired-bootstrap CI on the delta (E37/E47
scored each alone; E38/E39/E45 compared them on *correctness*, not SE). New additive script
(`se_fidelity_proxy_vs_sep.py`) does this across **14 target/setting combinations** spanning every
regime the project has data for: **LOLO** trivia (4 targets, proxy = E37/E43 saved per-seed preds,
CPU-only), **squad OOD** (Llama-2+Mistral, DEPLOY proxy), **fresh trivia n1000** (all 4 training
models — confirmed Llama-3 now HAS a genuine disjoint fresh n1000, correcting a stale note in
`correctness_eval.py`'s TARGETS dict), and **Qwen/Gemma zero-shot** (DEPLOY proxy vs a *fair*
target-specific SEP fit on that model's own n2000 train / evaluated on its disjoint n1000 eval —
0 id-overlap confirmed for all 4). SEP uses the E41 fixed layer where established
(`exp2_run.BEST_TBG`), else leak-free val-selection (Qwen/Gemma, no CV layer picked yet — flagged).
**Result: proxy beats SEP on Spearman in 13/14 settings (all CIs exclude 0, all positive; ties on
Mistral-LOLO) and on AUROC-vs-SE in 10/14 (4 CIs include 0, none negative) — it never loses.**
Largest margins are exactly where SEP is weakest: squad OOD (Δρ +0.352 Llama-2, +0.168 Mistral —
SEP collapses to 0.236 under the dataset shift) and LOLO-Llama-2 (+0.266, SEP's known outlier
layer per E41). Smallest-but-still-positive margins are where SEP is already strong
(Mistral-LOLO ρ 0.599; Qwen3.5-9B ρ 0.700, the best SEP score in either family). Per-seed scores
always sit 0.02-0.07 below the ensemble — reported separately in the JSON, never collapsed into
the ensemble number. New helper `arm_preds_per_seed` (copy of
`procrustes_e27_rank_fusion.arm_preds` that keeps every seed's prediction instead of only the
mean) is now the standard way to get per-seed proxy predictions for any checkpoint set. **Infra:**
both GPUs were saturated with live big-tier Stage-1 builds when the GPU-dependent settings needed
to run; with user go-ahead, paused the least-progressed job (`gemma-3-27b-it`, 132/2000, SIGTERM —
resumable) to free a GPU, then relaunched it identically after (confirmed it resumed from 132, not
scratch). Full arc + all 14 rows: EXPERIMENTS.md E51. Artifacts: `se_fidelity_proxy_vs_sep.py`,
`results/se_fidelity_proxy_vs_sep.json`.

**⭐ E49 (2026-08-22) — think-leak recheck: E45-E48's Qwen3.5-9B numbers need NO correction.**
Precisely counted the "hardest tail" flagged in E44: **58/1000 (5.8%)** trivia_qa eval records
have `canonical_response` literally `"<think>"`/`"<think>\n\n</think>"` (budget exhausted, no real
answer). User asked directly whether E45-E48 needed to exclude these. Re-scored all 4 predictors
(true SE, `q_only`, `q_resp_only`, frozen backbone) on all-1000 vs clean-942: **removing them makes
EVERY number worse** (−0.02 to −0.04 uniformly), the opposite of the initial "confidently wrong"
hypothesis — checked directly, these 58 rows have **mean true SE 2.00** (vs clean rows' 0.97), so
they're the *easiest* wrong-answer cases (high uncertainty, clearly wrong), not inflators. **No
correction needed to any E45-E48 result.** Likely mechanism: the entailment model can't judge two
`"<think>"` fragments as equivalent, so they scatter into many small clusters (high entropy)
instead of one (low entropy) as first guessed. Artifacts: `e49_qwen35_9b_think_leak_check.py`.

**Infra fixed alongside E49 (found while investigating it):**
- **Stale manifest metadata for `Qwen3.5-9B_trivia_qa_n1000_full`** (`meta.mean_accuracy` etc.
  stuck at an old 46-record partial value from an earlier manifest rebuild) — fixed by
  recomputing from the 1000 records already on disk; **no `.pt` data or any E45-E48 result was
  ever wrong**, this was purely a stale summary-stat display bug.
- **`squad_v2` dataset loading was broken under `se_probes_v5`** (newer `huggingface_hub` rejects
  the legacy unnamespaced `"squad_v2"` short form) — fixed in
  `semantic_uncertainty/uncertainty/data/data_utils.py` (`"squad_v2"` → `"rajpurkar/squad_v2"`,
  verified byte-identical under both the old and new env before applying; stopped and asked first
  per this repo's `semantic_uncertainty/uncertainty/` rule).
- **Squad n1000 OOD test data now built for all 4 small-tier targets** (`Qwen3-8B`, `Qwen3.5-9B`,
  `gemma-7b-it`, `gemma-2-9b-it`, all 1000/1000) — same recipe as the existing Llama-2/Mistral
  squad sets (no `--only_ids`, default seed reproduces the identical question selection), built on
  GPU0 in parallel with the big-tier n1000 queue on GPU1. Qwen3.5-9B's squad build took 9.2×
  longer than its trivia_qa build (squad is harder on average, triggers its long-`<think>`
  behavior far more often) — the other 3 targets were only 1.2-1.4× slower.
- **Big-tier queue parallelized across both GPUs**: `gemma-2-27b-it`/`gemma-3-27b-it` pulled out
  of the sequential GPU1 queue into an independent GPU0 lane
  (`build_gemma_bigtier_gpu0.sh`, single-GPU — GPU1 is occupied so dual-GPU isn't available
  right now anyway, also sidesteps that untested risk), starting the moment GPU0 frees up.
  Roughly halves the big-tier queue's total wall time.

**Data-generation status at session end (2026-08-22 ~08:00), for a fresh session to pick up:**
small-tier trivia_qa n2000 (training data) and squad n1000 (OOD test data) are **both fully done**
for all 4 small-tier targets. Big-tier n1000 (eval data): GPU1 running Qwen3.5-27B (~620/1000,
then Qwen3.6-27B resume from 93/1000, then Qwen3.8-27B fresh); GPU0 finishing the last squad build
then auto-starting `gemma-2-27b-it` → `gemma-3-27b-it`. `gemma-4-31B-it` still excluded (broken,
undiagnosed). **⚠️ Nothing from E46-E49 or this infra work is committed to git yet** — check
`git status` in `amortized_ue/` before assuming any of it is on `origin/main`.

**⭐ E46-E48 (2026-08-21) — three follow-up checks on the E45 zero-shot result, all positive.**
(1) **E46 — does `q_resp_only` distinguish CROSS-model disagreement**, not just within-model
correctness (E45 never tested this)? All 6 pairs among the 4 Qwen/Gemma targets: SE-gap
correlation +0.23 to +0.42 (all CIs exclude 0), pairwise accuracy on divergent rows **69-81%** —
substantially stronger than E40's analogous test on the original 4 models (51.5%, not
significant); no LOO-null correction needed here since none of the 4 new targets were in the
deploy proxy's training set (every pair is symmetric). (2) **E47 — SE-FIDELITY (not correctness)**:
Spearman rho(`q_resp_only`, true SE) = **0.67-0.75 on all 4 targets**, at or above the proxy's own
training-family benchmark (E37: ~0.648). gemma-2-9b-it's SE-fidelity (0.674) is on par with
gemma-7b-it's (0.670) despite gemma-2-9b-it being E45's one correctness-detection *loss* — direct
target-specific confirmation of E31's SE-fidelity≠correctness finding. Root cause of the missing
decode scale clarified (not just worked around): `exp2_run.py` intentionally saves an identity
transform because targets are z-scored PER SOURCE MODEL before pooling — there is no single
absolute SE scale for a pooled proxy to save, by design, not an oversight; **`position`/`layer`
are still unfixed at the save site** though, a real open gap. (3) **E48 — does the LoRA training
add anything beyond the frozen backbone's pretrained knowledge?** Built a p_true-style zero-shot
baseline (same `meta-llama/Llama-3.2-3B`, NO training, few-shot True/False prompt). Untrained
baseline is well above chance (AUROC 0.67-0.80) but the **trained proxy beats it on every target,
every metric, no exceptions** (AUROC +0.05 to +0.08; **SE-fidelity rho +0.24 to +0.28, remarkably
uniform across all 4 targets**) — training clearly adds a real, graded uncertainty-calibration
skill beyond the backbone's coarse pretrained correctness sense, and that skill transfers to
unseen model families. Full arc + all tables: `EXPERIMENTS.md` E46/E47/E48. Artifacts:
`e46_qwen_gemma_pairwise_disagreement.py` (+`e46_examples.py`),
`e47_qwen_gemma_se_fidelity.py` (+`e47_examples.py`), `e48_frozen_backbone_baseline.py`.

**⭐ E45 (2026-08-21) — zero-shot flagship-proxy transfer to Qwen/Gemma: mixed but real.** Scored
the DEPLOY proxy (trained by pooling Llama-2/Mistral/Llama-3/DeepSeek n2000, text arms only) on
the 4 E44 small-tier targets with **zero retraining/calibration** — the sharpest generalization
test run so far, since Qwen/Gemma are genuinely different vendors/architectures the proxy never
saw. **AUROC_incorrect, `q_resp_only` vs true 10-sample SE:** Qwen3-8B **0.840 vs 0.787 (beats,
+0.053\*)**, gemma-7b-it **0.848 vs 0.771 (beats, +0.078\*)**, Qwen3.5-9B 0.818 vs 0.810 (on par),
gemma-2-9b-it **0.722 vs 0.769 (loses, −0.047\*)**. `q_resp_only` never loses to `q_only` on any
target. **2/4 beat true SE outright (stronger than any same-family result in E38), 1/4 on par,
1/4 a genuine loss** — gemma-2-9b-it is also an accuracy/incorrect-rate outlier vs the other 3
(0.684/0.316 vs 0.42-0.56/0.44-0.58), flagged as a hypothesis for the weaker signal, not
confirmed. **Three latent bugs found+fixed en route** (all generalize beyond this run): (1)
`arm_preds`'s checkpoint glob didn't match `deploy_checkpoints`' `deploy_`-prefixed filenames,
silently `IndexError`'d; (2) `deploy_checkpoints` (exp2_run.py-saved, same era as E39's
`k`/`transform` bug) also omit `position`/`layer` in meta — `load_checkpoint` itself (not just
`correctness_eval_ood.py`'s narrower compat shim) now falls back to the stored config; (3)
`Qwen3.5-9B`'s manifest was stale at 46/1000 (a resume run overwrote instead of merging) — all
1000 `.pt` files were fine, manifest mechanically rebuilt from disk. **Also: `amortized_stage2`
lives on NFS and a degraded window made it unusable (multi-minute import stalls, confirmed via
`rpc_wait_bit_killable`)** — fixed properly with a new `/data2` venv `amortized_stage2_v5`
(versions pinned to match live: transformers 4.52.4/peft 0.19.1/accelerate 1.14.0; **torch could
NOT stay at 2.1.1** — `bitsandbytes` transitively pulls torch 2.13.0+cu130 regardless, verified
checkpoint-load-compatible before trusting results) + `deploy_checkpoints` copied to `/data2`.
z-arm not run (no Procrustes alignment fit for Qwen/Gemma yet — natural next step). Full arc:
`EXPERIMENTS.md` E45. Artifacts: `e45_qwen_gemma_zeroshot.py`, `results/e45_qwen_gemma_zeroshot.json`.

**⭐ E44 (2026-08-20/21) — Qwen + Gemma families added as new cross-LLM targets.** Extends the
target-LLM set from 4 to (eventually) 14: 5 Qwen (`Qwen3-8B`, `Qwen3.5-9B`, `Qwen3.5-27B`,
`Qwen3.6-27B`, `Qwen3.8-27B`) + 5 Gemma (`gemma-7b-it`, `gemma-2-9b-it`, `gemma-2-27b-it`,
`gemma-3-27b-it`, `gemma-4-31B-it`). All released within the last ~7 months of this Aug-2026
session (Qwen3.5/3.6/3.8 and Gemma 4 postdate every existing conda env's `transformers`).

**Code changes (`huggingface_models.py`, `utils.py` — both blocks-execution, no SE/probe logic touched):**
- New `qwen` load branch (`Qwen/{model_name}`, never gated) + `init_model()` dispatch whitelist
  widened to `qwen`/`gemma` (exact same pattern as the existing `deepseek` entry).
- `gemma` branch now redirects gated checkpoints (`gemma-7b-it`, `gemma-2-9b-it`,
  `gemma-2-27b-it`, `gemma-3-27b-it`) to the ungated `unsloth` mirror; `gemma-4-*` loads
  directly from `google/` (not gated).
- **Two NEW model-scoped generation fixes**, both gated by exact model-name tuples/dicts near
  the top of `huggingface_models.py` (`_LEADING_WHITESPACE_MODELS`,
  `_EXTRA_TOKEN_BUDGET_MODELS`) — **off (byte-identical) for every model not explicitly listed**:
  1. `StoppingCriteriaSub.tolerate_thinking`: Qwen3.5-9B/Qwen3.8-27B (reasoning models) open
     completions with a blank line then often a multi-line `<think>...</think>` block; the
     pipeline's default stop-on-`\n` rule fires on the leading blank line, or (once that's
     skipped) on the first newline *inside* the still-open think block — captured "answer" ends
     up empty or literally `"<think>"`. Fix: suppress the stop match entirely while a `<think>`
     tag is open (more opens than closes seen so far), only resume normal stopping after
     `</think>`. **Raising `model_max_new_tokens` alone does NOT fix this** — the stop fires
     within the first few tokens regardless of budget; don't waste time on that lever again.
  2. `model_max_new_tokens` override (250 vs default 50) for the same two models — a safety
     ceiling so a long-but-finishing reasoning block has room to also emit the answer. With fix
     #1 in place this is genuinely just a ceiling: for the *hardest* residual questions the
     model still won't finish thinking even at 250 (see below) — that's accepted as legitimate
     "model couldn't converge" signal, not corrupted data, since fix #1 already guarantees no
     more truncated-empty/mid-think garbage.
  - **Debugging trap hit twice: `write_manifest()` runs ONCE at the end of the whole batch**,
    while `save_record()` writes each `.pt` incrementally. Killing a run partway through and then
    reading `manifest.json` shows **stale** data even though the individual `.pt` files are
    already correct — always let a build finish (or read `.pt` files directly) before judging
    whether a fix worked.

**Dataset builds (all on `--only_ids /data2/mn1025/stage1_meta/shared_n1000_ids.txt`, the SAME
1000 ids as the existing Llama-2/Mistral/Llama-3/DeepSeek `*_n1000_full` datasets — every
model's records line up question-for-question):**

| Model | Status | Notes |
|---|---|---|
| Qwen3-8B | ✅ done, 1000/1000 verified | clean |
| Qwen3.5-9B | ✅ done, 1000/1000 verified | needed both fixes above; 46 records are the hardest tail (some never finish thinking even at 250 tokens — kept as-is per user decision, "accept it") |
| gemma-7b-it | ✅ done, 1000/1000 verified | clean, via unsloth mirror |
| gemma-2-9b-it | ✅ done, 1000/1000 verified | clean, via unsloth mirror |
| Qwen3.5-27B | 🔄 in progress at session end | needed both fixes; single-GPU+CPU-offload only (see dual-GPU bug below) |
| Qwen3.6-27B | 🔄 in progress at session end | single-GPU+CPU-offload |
| Qwen3.8-27B | ⏳ queued at session end | single-GPU+CPU-offload |
| gemma-2-27b-it | ⏳ queued, will attempt dual-GPU | different arch than Qwen, untested at scale |
| gemma-3-27b-it | ⏳ queued, will attempt dual-GPU | different arch than Qwen, untested at scale |
| gemma-4-31B-it | ❌ **broken, not attempted at n1000** | degenerate output (echoes the last few-shot answer verbatim, or `"the la la la..."` gibberish) even in a clean 3-question smoke test — needs real diagnosis, not a quick fix. Do not build until this is understood. |

Check current status: `ls /data2/mn1025/stage1_meta/`, record counts via
`ls /data2/mn1025/stage1/<Model>_trivia_qa_n1000_full/records | wc -l`, and
`amortized_ue/logs/<Model>_trivia_qa_n1000*.log` / `amortized_ue/build_big_tier_n1000.sh`'s
driver output for what's still running. The two build scripts
(`build_small_tier_n1000.sh` DONE, `build_big_tier_n1000.sh` still running at session end) are
resumable — records save incrementally, `--overwrite` not passed means already-built records
are skipped, so re-running a script (or a single model's command from inside it) after any crash
just continues.

**⚠️ Dual-GPU sharding is reproducibly BROKEN for the Qwen3.5+ hybrid (Gated-DeltaNet + Gated-
Attention) architecture** — `device_map='auto'` across both GPUs crashes every time
(`torch.multinomial`: "probability tensor contains inf/nan", CUDA device-side assert), fast and
deterministic, reproduced on both Qwen3.5-9B (forced 2-GPU split via a tight `max_memory` cap)
and the real Qwen3.5-27B/3.6-27B builds. **Root cause (diagnosed, not fixed):** the device map is
clean at layer granularity (no single layer split mid-way) — the crash is specifically at the
boundary between a Gated-Attention layer on one GPU and a Gated-DeltaNet layer on the other.
DeltaNet layers carry recurrent state across generation steps; `accelerate`'s generic multi-GPU
dispatch hooks evidently don't correctly re-sync that state when the layer's input activation
arrives from a different GPU than the layer's home device. This is almost certainly an upstream
`transformers`/`accelerate` bug (the architecture is ~1 week old at time of writing) — **not
something to patch locally**; fixing it for real would mean patching vendored `transformers`
modeling code, out of proportion to the payoff. **Workaround (in use): single-GPU + CPU offload**
— proven reliable in every build, just slower (~45-75s/record vs a much faster from-GPU pace for
the small models). Gemma models are a different, untested-at-scale architecture — worth trying
dual-GPU for those specifically before assuming the same bug applies.

**New env `se_probes_v5`** (`/data2/mn1025/conda_envs/se_probes_v5`) — a plain **`venv`, not a
conda env** (bootstrapped via `/data/sv/miniconda3/bin/python3 -m venv`, entirely bypassing
conda's NFS-hosted `pkgs_dirs`), `transformers==5.15.1`, `torch==2.13.0+cu130`. Built this way
specifically because the NFS mount was in one of its documented degraded windows and even
`pip freeze`/`conda create --clone` on the existing NFS-hosted envs were hanging — this venv's
packages install fresh from PyPI (network-bound, not NFS-bound) and its own imports never touch
`/vol/bitbucket`. Use it for all 10 new-family models (covers every architecture: `qwen3`,
`qwen3_5`, `gemma`, `gemma2`, `gemma3`, `gemma4`). `HF_HOME=/data2/mn1025/hf_cache` (also off
NFS). The shared question-id file lives at `/data2/mn1025/stage1_meta/shared_n1000_ids.txt`
(1000 ids, extracted from the existing `Llama-2-7b-chat_trivia_qa_n1000_full` manifest — reuse
this file for any future new-model build so everything stays id-aligned).

**⚠️ `E43` (LOLO retrain, prior session) also completed successfully during this session** —
all 4 folds done, checkpoints saved to `amortized_ue/results/exp2_lolo_full_ckpt.json` +
`amortized_ue/stage2/runs/E37_LOLO_ckpt/checkpoints/`. Not yet written up as a formal
EXPERIMENTS.md entry — do that before citing its numbers anywhere.

## Current state (updated 2026-08-18) — pre-E44, still valid for the original 4 targets

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
- **Does the SLM proxy preserve more model-specificity than the ridge? — ✅ DONE (E63): YES, ~3.6×.**
  A `q_resp_only` proxy trained leave-TWO-out on 6 LLMs reproduces the DeepSeek-vs-Qwen3-8B SE
  disagreement at Spearman +0.399 / sign-agreement 0.643 (N=1000), vs E40's aligned-ridge +0.110 on
  the same estimand — in a clean null-is-0 design. Response text is the model-specific channel. See
  EXPERIMENTS.md E63.

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

**Opened by E55 (new, untouched):** DeepSeek + Llama-3 now have squad_n1000 — E39/E52/E54's
squad correctness studies could be extended from 2 targets (Llama-2, Mistral) to 4. Once the
Qwen `_nothink` big-tier queue finishes, the whole Qwen/Gemma small-tier E45-E54 line could in
principle be re-run on cleaner (non-thinking-contaminated) data — not yet decided whether the
gain is worth the compute; ask before repointing any existing analysis at `_nothink` dirs.

**✅ Closed by E65-final (2026-08-30):** re-ran `e65_bigtier_lolo.py --stage eval --eval_n 1000` on
the n1000 shared-ID trivia sets (all 5 big-tier models, all 1000 Qs/fold). **Thesis is
family-dependent at 27B:** holds for Qwen-27B (proxy on par with true SE, sig. beats SEP on 2/3),
fails for gemma-2-27b-it (sig. loss to true SE −0.056\*, matching E45/E64), gemma-3-27b-it weak all
round (degenerate SE). Proxy/true-SE/SEP/ridge means AUROC 0.747 / 0.760 / 0.736 / 0.754, ρ 0.626 /
— / 0.607 / 0.668. Preliminary 200-row numbers superseded (its gemma-3 "+0.123\*" was an artefact →
+0.014 n.s.). Results `results/e65_bigtier_lolo_n1000.json`. Full write-up: EXPERIMENTS.md E65.
**Still open (optional):** an E41-style CV layer pick for the big tier (SEP currently val-selected).
**✅ E65-OOD done (E69, 2026-09-03):** proxy < true SE on all 5 folds (CIs exclude 0), > SEP on 3/5,
< own-model ridge on all 5; E65-final family split holds. `results/e69_bigtier_lolo_squad_ood.json`.
**✅ Aligned-ridge LOLO done (E70, 2026-09-03):** PCA(512)→Procrustes into a Qwen3.5-27B anchor;
aligned_ridge ≈ q_resp_only alone, label-free `fuse` of the two is best (ID 0.764 ≈ true SE, OOD
0.715 > either arm & > SEP, still < true SE OOD). `results/e70_bigtier_lolo_aligned_ridge.json`.

**✅ E71 DONE (2026-09-03) — the E70 comparison for the small-tier Qwen/Gemma set 2.** LOLO
`q_resp_only` proxy + aligned_ridge + fuse + SEP + true SE, ID + OOD. **Set 2 reproduces set 1**
(proxy > aligned_ridge; fuse doesn't help) — opposite of E70's big tier, confirming that E70's
convergence was a near-identical-Qwen-27B-sibling artefact. `results/e71_settwo_lolo_aligned_ridge.json`,
W&B `e71_settwo_lolo_qresp_ckpts:v0` + `e71_settwo_aligned_ridge_bundles:v0`.

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
