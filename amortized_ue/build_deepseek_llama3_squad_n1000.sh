#!/bin/bash
# squad n1000 OOD build for DeepSeek + Llama-3 -- the 2 targets missing squad records (only
# Llama-2/Mistral had them before). GPU0, sequential (fp32 7-8B load needs ~35-37GB, doesn't fit
# alongside another big model). Written straight to /data2 (not NFS) from the start.
#
# NO --only_ids: default random_seed=10, num_few_shot=5 reproduces the EXACT same squad selection
# already used for Llama-2-7b-chat_squad_n1000_full / Mistral-7B-Instruct-v0.2_squad_n1000_full
# (verified convention, see build_small_tier_squad_n1000.sh) -- so this lines up question-for-
# question with the existing squad OOD sets.
#
# Env: se_probes_llama3 (transformers 4.44 -- loads both DeepSeek and Llama-3; mirrors
# smoke_deepseek.sh / smoke_llama3.sh). HF cache stays on the NFS path both models are already
# fully cached under (verified present) -- only bulk .pt-record reads hit the documented NFS
# degradation, not HF checkpoint loads, so this is left as-is rather than pre-copying weights.

cd "$(dirname "$0")/.."                                   # repo root
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes_llama3
export HF_HOME=/vol/bitbucket/mn1025/hf_cache OPENAI_API_KEY=placeholder
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

mkdir -p amortized_ue/logs
MODELS=(deepseek-llm-7b-chat Meta-Llama-3-8B-Instruct)

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

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] DeepSeek+Llama-3 squad n1000 queue finished ==="
