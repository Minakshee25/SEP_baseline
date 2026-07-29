#!/bin/bash
# Cross-LLM #2 (E21) Stage-1 build: Mistral-7B-Instruct-v0.2 on the 200 held-out
# (Llama-2 test-split) questions -- same ids as Llama-3 (E20), leakage-free & directly
# comparable. Env: se_probes_llama3 (transformers 4.44 loads Mistral; the old 4.35.2
# tokenizer failure is gone). Mistral is 4096-dim = Llama-2 -> all 5 arms transfer.
set -euo pipefail
cd "$(dirname "$0")/.."                                     # repo root
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes_llama3
export HF_HOME=/vol/bitbucket/mn1025/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}      # fp32 Mistral-7B needs ~29GB free
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

python -m amortized_ue.stage1 \
  --model_name Mistral-7B-Instruct-v0.2 --dataset trivia_qa \
  --num_samples 200 --selection_num_samples 2000 \
  --only_ids scratch_xllm/llama2_test_ids.txt
