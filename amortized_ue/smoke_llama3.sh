#!/bin/bash
# Llama-3-8B Stage-1 SMOKE TEST — de-risks generating a cross-LLM Stage-1 dataset with a
# different-family, 4096-dim target (so all 5 proxy arms could later transfer to it).
#
# WHY a smoke test: Stage-1 normally runs in `se_probes` (transformers 4.35.2), which CANNOT
# load Llama-3's tokenizer (predates Llama-3; support landed in 4.40). We instead run it in
# `amortized_stage2` (transformers 4.52.4), which DOES load Llama-3. Unknowns to verify:
#   1. the whole Stage-1 pipeline (SEP HuggingfaceModel.predict, DeBERTa entailment, load_ds)
#      imports+runs under 4.52.4 (mostly version-stable APIs, but untested here);
#   2. Llama-3's stop tokens / EOS terminate generations correctly (the code appends
#      tokenizer.eos_token generically at huggingface_models.py:235, and the '\n'/'Question:'
#      stops usually catch short-form QA — but CONFIRM by reading the answers);
#   3. answer quality is sane with the SEP (Llama-2-style) brief prompt.
#
# CODE FIX already applied (blocks-execution): huggingface_models.py loading branch now
# accepts '8b' (was '7b'|'13b' only -> raised ValueError for Llama-3-8B). No SE/probe logic touched.
#
# PREREQUISITES (both currently UNMET):
#   - Llama-3-8B weights downloaded (~16GB). The HF cache entry is a 512-byte STUB right now.
#     First run will auto-download from meta-llama/Meta-Llama-3-8B-Instruct (needs gated access +
#     a healthy filesystem — do NOT start this on the degraded NFS or it will wedge).
#   - /vol/bitbucket responsive (real content read completes in seconds, not `ls` — see
#     amortized_ue/CLAUDE.md infra note).
#
# WHAT TO INSPECT after it runs: the printed record's `canonical.response` and the high-temp
# `samples` — do the answers stop cleanly (no run-on / no leaked <|eot_id|> / no repetition)?
# If yes -> clear to build the real Llama-3-8B Stage-1 dataset. If run-on/garbled -> add Llama-3's
# stop tokens / chat prompt format before building.

set -e
cd "$(dirname "$0")/.."                                   # repo root
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate amortized_stage2                          # transformers 4.52.4 — loads Llama-3
export HF_HOME=/vol/bitbucket/mn1025/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONUNBUFFERED=1

echo ">>> Llama-3-8B Stage-1 smoke: 3 prompts, trivia_qa, env=amortized_stage2"
python -m amortized_ue.stage1 --smoke --smoke_num_samples 3 \
  --model_name Meta-Llama-3-8B-Instruct --dataset trivia_qa

echo
echo ">>> INSPECT the printed answers above: do they stop cleanly and look sane?"
echo "    yes -> build the real dataset;  run-on/garbled -> fix Llama-3 stop-tokens/prompt first."
