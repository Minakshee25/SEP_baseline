#!/bin/bash
# Unattended sequential n1000 build queue for the 5 big-tier (27B+) new cross-LLM targets:
# Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B, gemma-2-27b-it, gemma-3-27b-it. All already
# smoke-tested OK. gemma-4-31B-it excluded (degenerate output, needs real diagnosis).
#
# Same shared 1000 question ids as the small-tier batch and the existing Llama-2/Mistral/
# Llama-3/DeepSeek targets -- every model's records line up question-for-question.
#
# GPU (per-model, see 2026-08-21 finding): dual-GPU device_map sharding reproducibly
# crashes the Qwen3.5+/3.6/3.8 hybrid (Gated-DeltaNet) architecture (torch.multinomial
# "probability tensor contains inf/nan", CUDA device-side assert -- 2/2 crashed fast,
# identical signature) -- those 3 stay pinned to GPU1 + CPU offload (proven reliable in
# smoke tests, just slower). gemma-2-27b-it/gemma-3-27b-it are a different architecture,
# untested at dual-GPU scale -- try both-GPUs sharding for them (no offload, much faster
# if it works).
# HF cache + records both on /data2 (non-NFS).
#
# Each model's full stdout/stderr goes to its own log under amortized_ue/logs/; a failure
# in one model does NOT stop the queue -- it logs the failure and moves to the next model.

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
mkdir -p amortized_ue/logs

# model:gpu_mode  (gpu_mode = "single" pins GPU1+offload, "dual" leaves both GPUs visible)
MODELS=(
  "Qwen3.5-27B:single"
  "Qwen3.6-27B:single"
  "Qwen3.8-27B:single"
  "gemma-2-27b-it:dual"
  "gemma-3-27b-it:dual"
)

for entry in "${MODELS[@]}"; do
  m="${entry%%:*}"
  gpu_mode="${entry##*:}"
  if [ "$gpu_mode" = "single" ]; then
    export CUDA_VISIBLE_DEVICES=1
  else
    unset CUDA_VISIBLE_DEVICES
  fi
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (gpu_mode=$gpu_mode) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples 1000 \
    --output_dir /data2/mn1025/stage1 \
    > "amortized_ue/logs/${m}_trivia_qa_n1000.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n1000.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] big-tier queue finished ==="
