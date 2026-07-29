#!/bin/bash
# Experiment 1 (E22) TRAINING set: Mistral-7B-Instruct-v0.2 Stage-1 on the full 2000
# trivia_qa questions (same seed as Llama-2 n2000 -> identical questions + 1440/360/200
# split). Resumable: the 200 held-out test records were copied in from n200 and are skipped.
# Auto-pushes to W&B at the end (push_to_wandb defaults True now). Env: se_probes_llama3.
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
  --num_samples 2000
