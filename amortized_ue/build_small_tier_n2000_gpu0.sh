#!/bin/bash
# n2000 TRAINING-data build (lane 1/2, GPU0) for 2 of the 4 small-tier targets.
# Mirrors the exact n2000 recipe already used for Llama-2/Mistral/Llama-3/DeepSeek:
# NO --only_ids => default random_seed=10, num_few_shot=5 selection, which is the
# SAME 2000-question pool as those existing *_n2000_full builds (verified 0 overlap
# with shared_n1000_ids.txt, the eval set already built for these 4 models).
#
# Paired with build_small_tier_n2000_gpu1.sh to run both lanes in parallel across
# the two free GPUs (big-tier n1000 builds were stopped/requeued to make room).

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs
MODELS=(Qwen3-8B gemma-7b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (n2000, GPU0) ==="
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

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] GPU0 small-tier n2000 lane finished ==="
