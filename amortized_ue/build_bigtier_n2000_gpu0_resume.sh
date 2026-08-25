#!/bin/bash
# Resume the big-tier n2000 builds on GPU0, WITH memory fencing so a co-tenant's job cannot land
# in the gap between "GPU0 has enough free memory" and "our model has finished loading" and OOM us
# mid-load (which is exactly what happened to Qwen3.5-27B on 2026-08-25: the wait loop saw 43GB
# free and started loading, but another user's job (sh2419, router-tuning.py) started concurrently
# and grabbed memory during our ~44s load window -> OOM 44s in, at exit 1, before a single new
# record was written). Mirrors amortized_ue/build_n2000_waiter.sh's holder pattern (gpu_reserve.py):
# once enough free memory is confirmed, immediately fence the REMAINDER with a dummy allocation
# before our real job starts loading, so a racing co-tenant OOMs on the fence instead of on us.
#
# Queue (in order): Qwen3.5-27B (stalled at 1179/2000 -- was silently dropped by the unfenced
# version of this script after its OOM, since the old per-model loop moved on rather than
# retrying), gemma-3-27b-it (1334/2000), Qwen3.8-27B (not started). Builds are resumable
# (overwrite=False skips existing .pt records), so re-issuing the same command per model
# continues where it left off.

set -uo pipefail
cd "$(dirname "$0")/.."                                   # repo root
NEED_MIB=${NEED_MIB:-40000}                                # min free MiB to launch AND this job's budget
SAFETY=${SAFETY:-400}                                       # MiB left unfenced as headroom
POLL=${POLL:-60}
GPU=0

source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs
MODELS=(Qwen3.5-27B gemma-3-27b-it Qwen3.8-27B)

HOLDER_PID=""
cleanup(){ [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM                                  # never leave a holder fencing the GPU

wait_and_fence() {
  local label="$1"
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S') waiting for GPU$GPU >= ${NEED_MIB}MiB free (before $label)"
  local FREE HOLD
  while true; do
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
    [ "$FREE" -ge "$NEED_MIB" ] && break
    sleep $POLL
  done
  HOLD=$(( FREE - NEED_MIB - SAFETY ))
  if [ "$HOLD" -gt 512 ]; then
    python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
    HOLDER_PID=$!
    echo ">>> $(date '+%Y-%m-%d %H:%M:%S') GPU$GPU has ${FREE}MiB free -- fencing ${HOLD}MiB (holder pid $HOLDER_PID), starting $label"
    sleep 4                                                  # let the holder grab its allocation first
  else
    HOLDER_PID=""
    echo ">>> $(date '+%Y-%m-%d %H:%M:%S') GPU$GPU has ${FREE}MiB free (~= budget, no slack to fence) -- starting $label"
  fi
}

for m in "${MODELS[@]}"; do
  wait_and_fence "$m (n2000, GPU$GPU)"
  echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] starting $m (n2000, GPU$GPU) ==="
  python -m amortized_ue.stage1 \
    --model_name "$m" --dataset trivia_qa \
    --num_samples 2000 \
    --output_dir /data2/mn1025/stage1 \
    >> "amortized_ue/logs/${m}_trivia_qa_n2000_gpu0resume.log" 2>&1
  status=$?
  [ -n "$HOLDER_PID" ] && { kill "$HOLDER_PID" 2>/dev/null || true; HOLDER_PID=""; }
  if [ $status -eq 0 ]; then
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m DONE (exit 0) ==="
  else
    echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] $m FAILED (exit $status) -- see amortized_ue/logs/${m}_trivia_qa_n2000_gpu0resume.log ==="
  fi
done

echo "=== [$(date '+%Y-%m-%d %H:%M:%S')] big-tier n2000 GPU0 resume queue (fenced) finished ==="
