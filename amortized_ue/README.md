# Amortized UE — Stage 1 (dataset) + Stage 2 (SLM proxy)

MSc project: **amortized uncertainty estimation** — train a small model to predict a large LLM's
**semantic entropy** in a single forward pass, avoiding the multi-sample cost at inference. The
long-term goal is a proxy that **transfers across target LLMs** (the thesis; motivated by the
Platonic Representation Hypothesis).

> 📓 **Authoritative docs:**
> - **`CLAUDE.md`** (this folder) — current state, how to run, locked design, and the **single
>   to-do list**.
> - **`../EXPERIMENTS.md`** — the chronological log of every experiment (E0→E20), what changed,
>   what came out, and what was **retracted**.
>
> This README is a short overview; the two files above are the source of truth.

## Two stages

- **Stage 1 (dataset):** for one target LLM + QA dataset, build one **self-contained, id-keyed
  record per prompt** — canonical low-temp answer, TBG/SLT hidden states (all layers), N=10 high-temp
  samples, and a **continuous** `cluster_assignment_entropy` label. Reuses SEP's logic read-only.
- **Stage 2 (proxy):** a frozen **Llama-3.2-3B** reads `[k soft tokens] (+ [text]) + [REG]` in one
  forward pass and regresses the SE label. Only a projector, LoRA adapters, REG token and head train.

## Record schema (`stage1-v1`)

```
id, question, context, reference
canonical: response, accuracy, token_log_likelihoods, hidden_states{TBG,SLT: [L+1,1,H]}
samples:   [{response, token_log_likelihoods, semantic_id}, ...]   # N high-temp
labels:    cluster_assignment_entropy (CONTINUOUS), semantic_ids, n_clusters, n_samples
meta:      { model, dataset, temperatures, entailment settings, git_commit, ... }
```
Joined **by id** (not list position). `TBG` = last input token; `SLT` = 2nd-last generated token.
⚠️ SEP's own stored TBG/SLT keys are *inverted* vs position — see `CLAUDE.md`.

## Environments

| env | transformers | for |
|-----|-------------|-----|
| `se_probes` | 4.35.2 (baseline) | Stage-1 for Llama-2 + the diagnostics |
| `amortized_stage2` | 4.52.4 | Stage-2 proxy training (Llama-3.2-3B) |
| `se_probes_llama3` | 4.44.2 | Stage-1 for Llama-3 |

## Usage

**Stage 1** (`se_probes` env for Llama-2, `se_probes_llama3` for Llama-3; resumable):
```bash
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3
python -m amortized_ue.stage1 --model_name Llama-2-7b-chat --dataset trivia_qa --num_samples 2000
bash amortized_ue/smoke_llama3.sh          # Llama-3 Stage-1 smoke
```

**Diagnostics first** (`se_probes` env, no GPU): `linear_ceiling_probe.py` (ridge baseline + the
correct way to pick the layer), `label_noise_ceiling.py` (achievable ceiling).

**Stage 2** (`amortized_stage2` env; checkpoints save by default; do NOT use the built-in 3B layer
sweep — pick the layer with `linear_ceiling_probe.py`):
```bash
python -m amortized_ue.stage2.run \
  --ood --ood_dataset squad --ood_num_samples 1000 \
  --seeds 5 --reuse_selection \
  --arms z,z_q,z_q_resp,q_only,q_resp_only \
  --z_inputs TBG:22,SLT:15 --selected_k 4 --projector_hidden_dim 1024 \
  --run_name REFERENCE_multipos_p1024_5arm_ckpt
python -m amortized_ue.stage2.run --eval --eval_datasets squad:1000    # reload ckpts, no retrain
```

## Headline result (reference model, saved; Llama-2-7b-chat / trivia_qa → squad OOD)

TBG L22 + SLT L15, projector 1024, k=4, 5 seeds. **Spearman primary; AUROC secondary.**

| arm | needs target LLM? | ID Spearman | OOD Spearman | ID AUROC |
|-----|-------------------|-------------|--------------|----------|
| **z (hidden only)** | yes | **0.602** | 0.368 | **0.807** |
| z + question | yes | 0.590 | **0.402** | 0.808 |
| z + question + resp | yes | 0.583 | 0.398 | 0.799 |
| **q_only** | **NO — nothing** | 0.494 | 0.259 | 0.758 |
| **q_resp_only** | answer text only | 0.521 | 0.399 | 0.768 |

**Three conclusions** (details in `../EXPERIMENTS.md`):
1. The proxy is **neither over- nor under-fitting** — its −0.04 gap to ridge is structural.
2. **Negative result** (single target LLM): a plain **ridge beats the proxy** (0.642 vs 0.602 ID),
   and an MLP loses to ridge — the z→SE relation is linear, so the frozen backbone adds nothing. A
   linear probe on hidden states ≈ **SEP**.
3. **⭐ Positive result / the thesis:** **`q_only`** predicts SE **from the question alone, with no
   target-LLM forward pass** (0.494, 54% of the achievable ceiling) — something a hidden-state probe
   cannot do; a bag-of-words baseline collapses to chance OOD (0.037) while the 3B holds (0.259).

## What's next

**Cross-LLM transfer** — evaluate the frozen Llama-2-trained proxy on **Llama-3-8B** (all 5 arms;
the z-arm transfer is the PRH test). The Llama-3 Stage-1 env is built and validated. See the to-do
list in **`CLAUDE.md`**.
