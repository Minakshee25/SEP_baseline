#!/bin/bash
# One-off reordering of build_big_tier_n1000.sh's GPU1 queue: Qwen3.5-27B was interrupted
# by a CUDA OOM at 877/999 (another process was briefly sharing GPU1). Original queue script
# (PID 871720) was killed while on Qwen3.6-27B (602/1000 done, saved incrementally) so
# Qwen3.5-27B could finish first on an exclusive GPU1. This script: finishes Qwen3.5-27B,
# then resumes Qwen3.6-27B from where it left off, then Qwen3.8-27B fresh -- same recipe as
# build_big_tier_n1000.sh, just Qwen3.5-27B moved to the front. Resumable throughout
# (--overwrite not passed -> existing records skipped).

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
mkdir -p amortized_ue/logs
MODELS=(Qwen3.5-27B Qwen3.6-27B Qwen3.8-27B)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (GPU1, single-GPU+offload) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples 1000 \
    --output_dir /data2/mn1025/stage1 \
    > "amortized_ue/logs/${m}_trivia_qa_n1000_resumeq35first.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n1000_resumeq35first.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] GPU1 resume queue finished ==="
