#!/bin/bash
# gemma-3-27b-it n2000 TRAINING-data build, pinned to GPU0. GPU0 went idle once
# gemma-3-27b-it's n1000 eval build finished (2026-08-23), so this runs immediately and in
# PARALLEL with build_bigtier_n2000_gpu1.sh's GPU1 queue (Qwen3.5-27B -> Qwen3.6-27B ->
# gemma-2-27b-it) rather than queued behind it -- same recipe (no --only_ids, no
# --selection_num_samples -- default seed picks a fresh 2000-question set for training).

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting gemma-3-27b-it (n2000, GPU0) ==="
python -m amortized_ue.stage1 \
  --model_name gemma-3-27b-it --dataset trivia_qa \
  --num_samples 2000 \
  --output_dir /data2/mn1025/stage1 \
  > amortized_ue/logs/gemma-3-27b-it_trivia_qa_n2000.log 2>&1
status=$?
if [ $status -eq 0 ]; then
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] gemma-3-27b-it DONE (exit 0) ==="
else
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] gemma-3-27b-it FAILED (exit $status) -- see amortized_ue/logs/gemma-3-27b-it_trivia_qa_n2000.log ==="
fi

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] gemma-3-27b-it n2000 GPU0 build finished ==="
