#!/bin/bash
# run_rq1_latency_mistral.sh -- RQ1 latency benchmark for Mistral-7B-Instruct-v0.2,
# GPU0, single-shot. Same protocol as run_rq1_latency.sh; Blocks A+B use the
# se_probes_llama3 env (Mistral-v0.2 loads there, no code change).
#
# Assumes GPU0 is ALREADY FREE + watchdog-free (caller stops the watchdog + GPU0
# lane + job first, restarts the queue after).
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=amortized_ue/logs/rq1_latency_run.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') RQ1 latency run (Mistral, fenceless single-shot) starting ==="

DATA_DIR=/data2/mn1025/stage1
OUT=amortized_ue/results/rq1_latency_Mistral-7B-Instruct-v0.2.json
TARGET=Mistral-7B-Instruct-v0.2

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
    --target "$TARGET" --dataset trivia_qa --num_samples 2000 \
    --data_dir "$DATA_DIR" --blocks "$barg" --warmup 10 --out "$OUT" "$@"
  local rc=$?
  echo "$label rc=$rc"
  return $rc
}

wait_gpu0_below 3000 60 || { echo "!!! GPU0 not clean at start -- aborting"; exit 1; }
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Block A + B (se_probes_llama3 / Mistral-7B-Instruct-v0.2) --------------
source /data/sv/miniconda3/etc/profile.d/conda.sh
conda activate se_probes_llama3
export OPENAI_API_KEY=placeholder HF_HOME=/data2/mn1025/hf_cache
run_one "Block A+B (Mistral)" A,B; AB_RC=$?
conda deactivate
pkill -9 -f "rq1_latency --target Mistral" 2>/dev/null || true
wait_gpu0_below 3000 90 || echo "  WARNING: GPU0 not fully released before Block C"

# ---- Block C (amortized_stage2_v5 / Llama-3.2-3B proxy) --------------------
source /data2/mn1025/conda_envs/amortized_stage2_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
run_one "Block C (Mistral)" C \
  --deploy_ckpt amortized_ue/results/deploy_checkpoints/deploy_q_resp_only_seed0.pt; C_RC=$?
deactivate 2>/dev/null || true
pkill -9 -f "rq1_latency --target Mistral" 2>/dev/null || true

echo "=== $(date '+%F %T') RQ1 latency run (Mistral) done (AB_RC=${AB_RC} C_RC=${C_RC}) ==="
