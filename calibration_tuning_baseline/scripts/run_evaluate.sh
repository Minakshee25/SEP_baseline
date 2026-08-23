#!/bin/bash
# Score the trained calibration-query LoRA adapter on our held-out test split, using the authors'
# own experiments/evaluate.py --mode=query entrypoint unmodified. Since our offline CSV's test
# split has "output" (precomputed answer) and "query_label" (our own accuracy>=0.5 label) set for
# every row, no generation and no re-grading happens -- this only runs the calibration-query
# forward pass and saves raw per-example logits to:
#   <log-dir>/metrics/offline:<dataset-name>/test/query_data.bin   ({"q_logits":[N,2], "q_labels":[N]})
# which amortized_ue/correctness_eval.py --calibration_tuning_log_dir <log-dir> then reads to add
# a "calibration_tuning" row to correctness_eval_<model>.json (uncertainty = 1 - softmax(q_logits)[:,1]).
#
# Usage:
#   CUDA_VISIBLE_DEVICES=<gpu> bash run_evaluate.sh llama2:7b-chat sep-Llama-2-7b-chat [checkpoint-step]
# If checkpoint-step is omitted, uses the LAST checkpoint written; pass e.g. "1000" to pick a
# specific one (e.g. the best-val-loss checkpoint rather than the final step, if training overfit).
set -euo pipefail

MODEL_NAME="${1:?model-name, e.g. llama2:7b-chat or mistral:7b-instruct}"
DATASET_NAME="${2:?offline dataset name, e.g. sep-Llama-2-7b-chat}"
CHECKPOINT_STEP="${3:-}"

REPO=/data2/mn1025/calibration_tuning_baseline/calibration-tuning
DATA_DIR=/data2/mn1025/calibration_tuning_baseline/data
TRAIN_LOG_DIR=/data2/mn1025/calibration_tuning_baseline/checkpoints/${DATASET_NAME}
EVAL_LOG_DIR=/data2/mn1025/calibration_tuning_baseline/logs/eval_${DATASET_NAME}

source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate /data2/mn1025/conda_envs/calibration_tuning
export HF_HOME=/data2/mn1025/hf_cache
export TRITON_CACHE_DIR=/data2/mn1025/triton_cache

if [ -n "$CHECKPOINT_STEP" ]; then
    QUERY_PEFT_DIR="${TRAIN_LOG_DIR}/checkpoint-${CHECKPOINT_STEP}"
    if [ ! -d "$QUERY_PEFT_DIR" ]; then
        echo "No such checkpoint dir: ${QUERY_PEFT_DIR}" >&2
        exit 1
    fi
else
    # find the last checkpoint-<step> dir written by training
    QUERY_PEFT_DIR=$(ls -d "${TRAIN_LOG_DIR}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1)
    if [ -z "$QUERY_PEFT_DIR" ]; then
        echo "No checkpoint found under ${TRAIN_LOG_DIR} -- run run_calibration_tune.sh first." >&2
        exit 1
    fi
fi
echo "Using checkpoint: $QUERY_PEFT_DIR"

cd "$REPO"
python experiments/evaluate.py \
    --dataset="offline:${DATASET_NAME}" \
    --data-dir="$DATA_DIR" \
    --prompt-style=oe \
    --model-name="$MODEL_NAME" \
    --query-peft-dir="$QUERY_PEFT_DIR" \
    --mode=query \
    --log-dir="$EVAL_LOG_DIR" \
    --batch-size=8

echo "Wrote ${EVAL_LOG_DIR}/metrics/offline:${DATASET_NAME}/test/query_data.bin"
