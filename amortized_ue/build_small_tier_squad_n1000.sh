#!/bin/bash
# squad n1000 OOD build for the 4 small-tier Qwen/Gemma targets, on GPU0 (free -- big-tier n1000
# queue owns GPU1 via single-GPU+CPU-offload, see build_big_tier_n1000.sh). Runs in parallel with
# zero contention.
#
# NO --only_ids: default random_seed=10, num_few_shot=5 reproduces the EXACT same squad selection
# already used for Llama-2-7b-chat_squad_n1000_full / Mistral-7B-Instruct-v0.2_squad_n1000_full
# (verified: those manifests show random_seed=10, num_few_shot=5, no only_ids) -- so this build
# lines up question-for-question with the existing squad OOD sets.

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs
MODELS=(Qwen3-8B Qwen3.5-9B gemma-7b-it gemma-2-9b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (squad n1000, GPU0) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset squad \
    --num_samples 1000 \
    --output_dir /data2/mn1025/stage1 \
    > "amortized_ue/logs/${m}_squad_n1000.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_squad_n1000.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] small-tier squad n1000 queue finished ==="
