#!/bin/bash
# LoRA + Prompt calibration-tuning training run, using the authors' own entrypoint
# (experiments/calibration_tune.py) unmodified except for the NousResearch Llama-2 redirect in
# llm/models/llama2.py (see that file's comment). Trains ONLY the calibration-query LoRA head
# on our precomputed (question, answer, correctness-label) triples -- compute_lm_loss skips
# generation entirely whenever the "query_label" CSV column is present (see
# calibration_tuning_baseline/scripts/build_offline_dataset.py's docstring).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> bash run_calibration_tune.sh llama2:7b-chat  sep-Llama-2-7b-chat
#   CUDA_VISIBLE_DEVICES=<gpu> bash run_calibration_tune.sh mistral:7b-instruct sep-Mistral-7B-Instruct-v0.2
set -euo pipefail

MODEL_NAME="${1:?model-name, e.g. llama2:7b-chat or mistral:7b-instruct}"
DATASET_NAME="${2:?offline dataset name, e.g. sep-Llama-2-7b-chat}"
BATCH_SIZE="${3:-4}"

REPO=/data2/mn1025/calibration_tuning_baseline/calibration-tuning
DATA_DIR=/data2/mn1025/calibration_tuning_baseline/data
LOG_DIR=/data2/mn1025/calibration_tuning_baseline/checkpoints/${DATASET_NAME}

source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate /data2/mn1025/conda_envs/calibration_tuning
export HF_HOME=/data2/mn1025/hf_cache
export TRITON_CACHE_DIR=/data2/mn1025/triton_cache

cd "$REPO"
torchrun --nnodes=1 --nproc_per_node=1 experiments/calibration_tune.py \
    --dataset="offline:${DATASET_NAME}" \
    --data-dir="$DATA_DIR" \
    --prompt-style=oe \
    --model-name="$MODEL_NAME" \
    --log-dir="$LOG_DIR" \
    --int8=False \
    --lora-rank=8 --lora-alpha=32 --lora-dropout=0.1 \
    --batch-size="$BATCH_SIZE" \
    --lr=1e-4 \
    --kl-decay=1.0 \
    --max-steps=5000
