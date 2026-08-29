#!/bin/bash
# run_rq1_latency.sh -- RQ1 latency benchmark on GPU0. SINGLE-SHOT.
#
# Assumes it is handed an ALREADY-FREE, watchdog-free GPU0 (the caller stops the
# training watchdog + GPU0 lane + Qwen job first, and restarts the queue after --
# see the session notes). This script only: verifies GPU0 is clean, runs Block A+B
# (se_probes) then Block C (amortized_stage2_v5), one attempt each, writes the JSON.
# NO GPU memory fence -- fp32 Llama-2-7b-chat peaks near ~41GB of the 44GB card.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=amortized_ue/logs/rq1_latency_run.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') RQ1 latency run (fenceless single-shot) starting ==="

DATA_DIR=/data2/mn1025/stage1
OUT=amortized_ue/results/rq1_latency_Llama-2-7b-chat.json

wait_gpu0_below() {   # <mib> <timeout_s>
  local target="$1" deadline=$(( $(date +%s) + ${2:-120} )) used
  while :; do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
    [ "$used" -le "$target" ] && { echo "  GPU0 used=${used} MiB"; return 0; }
    [ "$(date +%s)" -ge "$deadline" ] && { echo "  GPU0 still ${used} MiB (timeout)"; return 1; }
    sleep 5
  done
}

run_one() {   # <label> <blocks> <extra args...>
  local label="$1" barg="$2"; shift 2
  echo "--- $(date '+%F %T') $label ---"
  CUDA_VISIBLE_DEVICES=0 python -m amortized_ue.rq1_latency \
    --target Llama-2-7b-chat --dataset trivia_qa --num_samples 2000 \
    --data_dir "$DATA_DIR" --blocks "$barg" --warmup 10 --out "$OUT" "$@"
  local rc=$?
  echo "$label rc=$rc"
  return $rc
}

wait_gpu0_below 3000 60 || { echo "!!! GPU0 not clean at start -- aborting"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Block A + B (se_probes / Llama-2-7b-chat) ------------------------------
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes
export OPENAI_API_KEY=placeholder HF_HOME=/vol/bitbucket/mn1025/hf_cache
run_one "Block A+B" A,B; AB_RC=$?
conda deactivate
pkill -9 -f "rq1_latency --target Llama-2" 2>/dev/null || true
wait_gpu0_below 3000 90 || echo "  WARNING: GPU0 not fully released before Block C"

# ---- Block C (amortized_stage2_v5 / Llama-3.2-3B proxy) --------------------
source /data2/mn1025/conda_envs/amortized_stage2_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
run_one "Block C" C \
  --deploy_ckpt amortized_ue/results/deploy_checkpoints/deploy_q_resp_only_seed0.pt; C_RC=$?
deactivate 2>/dev/null || true
pkill -9 -f "rq1_latency --target Llama-2" 2>/dev/null || true

echo "=== $(date '+%F %T') RQ1 latency run done (AB_RC=${AB_RC} C_RC=${C_RC}) ==="
