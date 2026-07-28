#!/bin/bash
# Cross-LLM (E20) Stage-1 build: Llama-3-8B on the 200 held-out (Llama-2 test-split) questions.
# Same questions the proxy never trained on -> leakage-free, directly comparable to the ID numbers.
# Env: se_probes_llama3 (transformers 4.44 -- loads Llama-3 AND DeBERTa's .bin; see cross-llm-llama3-env).
set -euo pipefail
cd "$(dirname "$0")/.."                                     # repo root
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes_llama3
export HF_HOME=/vol/bitbucket/mn1025/hf_cache
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}      # fp32 Llama-3-8B needs ~32GB free
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export PYTHONUNBUFFERED=1

python -m amortized_ue.stage1 \
  --model_name Meta-Llama-3-8B-Instruct --dataset trivia_qa \
  --num_samples 200 --selection_num_samples 2000 \
  --only_ids scratch_xllm/llama2_test_ids.txt
