#!/bin/bash
# Unattended sequential n1000 build queue for the 4 small-tier (~7-9B) new cross-LLM
# targets: Qwen3-8B, Qwen3.5-9B, gemma-7b-it, gemma-2-9b-it. All already smoke-tested OK.
#
# Runs on the SAME shared 1000 question ids as the existing Llama-2/Mistral/Llama-3/DeepSeek
# "fresh n1000" batch (amortized_ue/data/stage1/*_trivia_qa_n1000_full), extracted from that
# batch's own manifest -- so every model's records line up question-for-question, matching the
# project's existing cross-model alignment convention. --selection_num_samples 3074 replicates
# the exact seed-driven pool that batch was drawn from (see that manifest's config).
#
# GPU: pinned to GPU1 only, one model at a time (sequential, not parallel) to avoid
# self-contention and to avoid GPU0 (held by the E43 job, a separate legitimate run).
# HF cache + records both on /data2 (non-NFS) -- see amortized_ue/CLAUDE.md infra note.
#
# Each model's full stdout/stderr goes to its own log under amortized_ue/logs/; a failure in
# one model does NOT stop the queue -- it logs the failure and moves to the next model.

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
mkdir -p amortized_ue/logs

MODELS=(Qwen3-8B Qwen3.5-9B gemma-7b-it gemma-2-9b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m ==="
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

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] small-tier queue finished ==="
