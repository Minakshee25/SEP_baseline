#!/bin/bash
# Resume build_bigtier_n2000_gpu1.sh's queue after it was paused (SIGTERM) to free GPU1 for the
# LOLO-proxy-on-squad eval (se_fidelity_proxy_vs_sep.py --only lolo_squad). Qwen3.5-27B already
# finished/moved to its own GPU0 resume script before the pause, so this queue picks up at
# Qwen3.6-27B (paused at 498/2000, resumable -- overwrite=False skips existing records) and then
# continues to gemma-2-27b-it, exactly mirroring the original queue's remaining order.

cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=1
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs
MODELS=(Qwen3.6-27B gemma-2-27b-it)

for m in "${MODELS[@]}"; do
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (n2000, GPU1) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --num_samples 2000 \
    --output_dir /data2/mn1025/stage1 \
    >> "amortized_ue/logs/${m}_trivia_qa_n2000.log" 2>&1
  status=$?
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n2000.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] big-tier n2000 GPU1 queue (resumed) finished ==="
