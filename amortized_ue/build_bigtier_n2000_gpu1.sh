#!/bin/bash
# n2000 TRAINING-data build for the big-tier (27B+) cross-LLM targets, mirroring
# build_small_tier_n2000_gpu0.sh/build_small_tier_n2000_gpu1.sh's recipe (no --only_ids,
# no --selection_num_samples -- default seed picks a fresh 2000-question set for training,
# distinct from the shared n1000 eval-id set used by build_big_tier_n1000.sh).
#
# Queued here: Qwen3.5-27B, Qwen3.6-27B, gemma-2-27b-it (n1000 eval data done for all 3).
# gemma-3-27b-it's n1000 finished 2026-08-23 too, but once it did GPU0 went idle, so its
# n2000 build was moved to build_gemma3_n2000_gpu0.sh instead of queuing behind 2 more
# models here -- runs in parallel with this GPU1 queue rather than after it.
# Qwen3.8-27B is paused (killed mid-n1000-build, 65/1000 saved, resumable) because its
# reasoning traces make it ~29x slower than its siblings -- deliberately left out of both
# the n1000 and n2000 queues until that's addressed; see EXPERIMENTS.md / CLAUDE.md.
#
# GPU1 + CPU offload, same as build_gpu1_resume_qwen35_first.sh (dual-GPU sharding
# crashes the Qwen3.5+/3.6/3.8 hybrid Gated-DeltaNet architecture; gemma-2-27b-it hasn't
# been tested at dual-GPU scale so stays on the proven single-GPU path here too).

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs
MODELS=(Qwen3.5-27B Qwen3.6-27B gemma-2-27b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (n2000, GPU1) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --num_samples 2000 \
    --output_dir /data2/mn1025/stage1 \
    > "amortized_ue/logs/${m}_trivia_qa_n2000.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n2000.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] big-tier n2000 GPU1 queue finished ==="
