# RESULTS — amortized UE, the four-model cross-LLM picture (E20→E30)

**Thesis:** you can get *supervised-quality* uncertainty estimates for a **new target LLM with no
labels from that LLM**, by (a) reusing a frozen text-reading proxy and (b) a label-free orthogonal
alignment of its hidden states onto a reference model. This document is the consolidated results
record for the cross-LLM line. The blow-by-blow history (including retractions) is in
`../EXPERIMENTS.md` (E0→E30); current-state / how-to-run is in `CLAUDE.md`.

Reference model throughout = **Llama-2-7b-chat**. Task = trivia_qa short-form; SE label =
continuous `cluster_assignment_entropy`. Metrics: Spearman (ρ, continuous SE) and AUROC (binarised at
`best_split`). "Label-free" = uses **no SE labels from the target** (the alignment map and the ensemble
weights use none; the proxy was trained only on Llama-2 data).

---

## The four target LLMs

| model | role | family vs Llama-2 | dims | layers | best z-layer (ridge) |
|---|---|---|---|---|---|
| **Llama-2-7b-chat** | reference / anchor | — | 4096 | 32 | TBG:22 / SLT:15 |
| **Mistral-7B-Instruct-v0.2** | target | different lineage | 4096 | 32 | TBG:31 / SLT:20 |
| **Meta-Llama-3-8B-Instruct** | target | **same family (Meta Llama)** | 4096 | 32 | SLT:31 |
| **deepseek-llm-7b-chat** | target | different lineage | 4096 | 30 | SLT:16 / TBG:28 |

Datasets (all on `/vol/bitbucket` + W&B `amortized_ue_stage1`): each target has trivia_qa **n2000**
(seed-10 selection, the fit set) and the disjoint **fresh n1000** (E23 held-out; the N=1000 eval),
**except Llama-3** which has only n2000 (→ eval at N=200, its one residual limit).

---

## A. Single-LLM proxy (reference, Llama-2) — the starting point

Frozen Llama-3.2-3B proxy, 5 arms × 5 seeds (`runs/REFERENCE_multipos_p1024_5arm_ckpt`), z at
TBG:22+SLT:15. ID = trivia_qa, OOD = squad.

| arm | needs target LLM? | ID ρ | OOD ρ | ID AUROC | OOD AUROC |
|---|---|---|---|---|---|
| **z** (hidden only) | yes | 0.602 | 0.368 | 0.807 | 0.669 |
| z + question | yes | 0.590 | 0.402 | 0.808 | 0.684 |
| z + question + response | yes | 0.583 | 0.398 | 0.799 | 0.682 |
| **q_only** | **no — nothing** | 0.494 | 0.259 | 0.758 | 0.614 |
| **q_resp_only** | answer text only | 0.521 | 0.399 | 0.768 | 0.684 |

Context: a plain ridge on hidden states ≈ SEP and slightly beats the 3B proxy (0.642 vs 0.602 ID) —
the z-branch re-derives SEP. The novel arm is **`q_only`**: SE from the **question alone, no target
forward pass** (0.494, 54 % of the achievable ceiling; a hidden-state probe cannot do this).

---

## B. Cross-LLM transfer WITHOUT alignment (E20–E23) — text transfers, raw z does not

Frozen Llama-2 proxy scored on another target's questions (Spearman; "ceiling" = model-independent
shared-difficulty ceiling for the pair):

| target | raw **z** | **q_only** | **q_resp_only** | ceiling |
|---|---|---|---|---|
| Llama-3-8B | 0.056 (chance) | 0.436 (88 % of ceil) | 0.562 (full) | 0.505 |
| Mistral-7B | 0.044 (chance) | 0.410 (76 %) | 0.511 (~full) | 0.540 |
| Mistral→Llama-2 (reverse) | −0.002 (chance) | 0.476 (88 %) | 0.509 | 0.540 |
| DeepSeek-7B | ≈ chance (E30 floor) | — | — | — |

**Raw hidden states do NOT transfer** (z ≈ chance on any model swap, both directions, 2 families,
replicated on a fresh 1000-Q batch). **Text transfers** (q_only 76–90 %, q_resp_only ~full) — the
model-agnostic pathway. This is the core argument for a text-reading proxy.

---

## C. Hidden states DO transfer after label-free alignment (E24–E30) — but it tracks CKA, not family

A label-free orthogonal **Procrustes map** `W` (source→reference, fit on paired hidden states of shared
questions, **no SE labels**) lets the reference ridge read the target's SE. Full-power master table
(`procrustes_e30_master_table.json`; DeepSeek/Mistral N=1000, Llama-3 N=200):

| pair (predict source SE) | N | recovery | **CKA** | model-specific increment [95 % CI] | sig |
|---|---|---|---|---|---|
| Mistral→Llama-2 | 1000 | 92.8 % | 0.80 | **+0.032 [+0.001, +0.063]** | **yes** |
| DeepSeek→Llama-2 | 1000 | 94.7 % | **0.25** | +0.009 [−0.028, +0.044] | no |
| Llama-2→DeepSeek | 1000 | 94.1 % | **0.27** | +0.006 [−0.032, +0.044] | no |
| Llama-3→Llama-2 | 200 | 91.8 % | **0.87** | +0.069 [−0.004, +0.143] | no |
| Llama-2→Llama-3 | 200 | 94.6 % | **0.87** | −0.023 [−0.107, +0.056] | no |

- **Recovery is high (~92–95 %) for every pair** — but it is dominated by **shared question-difficulty**
  (the reference's *own* states already predict the target's SE), so it does **not** discriminate.
- **The discriminator is CKA (rotational alignability): Llama-3 0.87 > Mistral 0.80 ≫ DeepSeek 0.25.**
  Family is at best a *weak* predictor — Llama-3 (same family) is highest, but Mistral (different) is
  close, and **DeepSeek (different) is a striking low-CKA outlier despite matching 4096 dims**.
- **The genuine model-specific increment tracks CKA, not family:** Mistral +0.032 (significant at
  N=1000); **DeepSeek ~0 with a tight CI** → its hidden geometry carries almost no model-specific SE
  beyond shared difficulty. (Llama-3's increment is unresolved — N=200 only.)

---

## D. Label-free uncertainty estimation ≥ supervised SEP — on all three targets (E30, full power)

The payoff. Estimator = **rank-fusion of {aligned-z ridge, `q_resp_only`}**, using **no target SE
labels**, vs the target's **own supervised SEP** (single-layer logistic on binarised SE):

| target | label-free ensemble AUROC / ρ | supervised SEP AUROC | Δ(ens − SEP) AUROC [95 % CI] | Δ ρ [95 % CI] |
|---|---|---|---|---|
| Mistral (N=1000) | 0.866 / 0.608 | 0.832 | **+0.035 [+0.006, +0.063]** ✅ | +0.068 [+0.024, +0.109] ✅ |
| DeepSeek (N=1000) | 0.869 / 0.711 | 0.805 | **+0.065 [+0.041, +0.088]** ✅ | +0.128 [+0.092, +0.167] ✅ |
| Llama-3 (N=200) | 0.892 / 0.672 | 0.839 | +0.054 [−0.003, +0.115] (≈) | +0.156 [+0.069, +0.258] ✅ |

**The label-free ensemble matches or beats each model's own supervised SEP** (AUROC Δ excludes 0 for
Mistral & DeepSeek; ρ excludes 0 for all three). It is robust even for DeepSeek, whose hidden geometry
barely transfers (CKA 0.25): the **text arm carries it**, and aligned-z still adds via shared difficulty.

---

## Settled conclusions

1. **Text transfers, raw hidden states do not** (B) — replicated two families, both directions, fresh batch.
2. **Alignment recovers the hidden-state pathway label-free** (C), but the *genuine* model-specific
   component is small and **tracks representational compatibility (CKA), not family lineage**; DeepSeek
   is the low-CKA outlier with ~0 increment.
3. **Label-free UE ≈ or > supervised SEP on every target, with no target labels** (D) — the thesis,
   validated at full power on 3 targets + the reference.
4. Single-LLM negatives still hold: with hidden states available, a plain ridge ≈ SEP and slightly beats
   the proxy; the real edge is `q_only` (SE from the question alone).

## Limitations (honest)

- **Llama-3 is N=200** (no fresh n1000) → its increment CI can't be resolved; the same-family increment
  question stays open. Everything else is N=1000.
- **CKA vs family** is established on 3 targets (1 same-family, 2 different) — a broader panel would
  strengthen "family is only a weak predictor".
- The label-free alignment needs **paired anchor forward passes** (shared questions) to fit `W` — cheap,
  but not sample-free. The text arm needs nothing.

## Provenance

- Master table: `procrustes_e30_master_table.json` (+ `.py`); E29 preliminary (n1000/N=100):
  `procrustes_e29_*.json`. Per-pair alignment: `procrustes_e30_{deepseek_to_llama2,llama2_to_deepseek,
  llama3_to_llama2,llama2_to_llama3}*.json` + `procrustes_e25_mistral_to_llama2.json`. Ensembles:
  `procrustes_e30_ensemble_sep_*.json`. Reference arms: `runs/REFERENCE_multipos_p1024_5arm_ckpt`.
- Datasets (local + W&B, verified by fetch): `{Llama-2,Mistral,deepseek-llm-7b-chat}_trivia_qa_n2000`
  + fresh `n1000`; `Meta-Llama-3-8B-Instruct_trivia_qa_n2000`.
- Tooling: `procrustes_alignment.py` (E24/E25, `--position/--source_layer/--target_layer`),
  `procrustes_e30_ensemble_sep.py`, `gpu_reserve.py` + `build_n2000_waiter.sh` (GPU fencing).

*Scope note: this file is the cross-LLM (E20–E30) results record. It does not re-enumerate every
superseded single-LLM config/ablation — those live in `../EXPERIMENTS.md`.*
