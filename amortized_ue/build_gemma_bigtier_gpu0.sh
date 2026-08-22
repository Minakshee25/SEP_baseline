#!/bin/bash
# gemma-2-27b-it + gemma-3-27b-it n1000 build, pinned to GPU0, single-GPU+CPU-offload.
# Runs INDEPENDENTLY of build_big_tier_n1000.sh's GPU1 queue (Qwen3.5-27B -> Qwen3.6-27B ->
# Qwen3.8-27B), which won't reach these two gemma models for many hours -- pulling them out here
# lets GPU0 start on them immediately instead of sitting idle. Single-GPU (not the originally
# planned dual-GPU attempt) because GPU1 is fully occupied by the Qwen queue right now, so dual-GPU
# isn't available as an option anyway; also sidesteps the untested dual-GPU risk for this new
# architecture entirely. Same shared-id recipe as every other big-tier build.

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
mkdir -p amortized_ue/logs
MODELS=(gemma-2-27b-it gemma-3-27b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (GPU0, single-GPU+offload) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples 1000 \
    --output_dir /data2/mn1025/stage1 \
    > "amortized_ue/logs/${m}_trivia_qa_n1000_gpu0.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n1000_gpu0.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] GPU0 gemma big-tier queue finished ==="
