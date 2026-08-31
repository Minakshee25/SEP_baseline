#!/bin/bash
# cross_llm_5arm_gpu_swap.sh — RACE-FREE GPU borrow on GPU 1 for the canonical eval-only
# raw cross-LLM transfer eval (amortized_ue.cross_llm_5arm_eval). Inference only, no training.
# ZERO NFS: /data2 venv + /data2 HF cache + /data2 checkpoints + /data2 data.
# Same SIGSTOP-the-lane pattern as e62/e63/eval_5arm_gpu_swap.sh.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data2/mn1025/hf_cache
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

GPU=1
LANE_PID="${LANE_PID:?export LANE_PID}"
STAGE1_PID="${STAGE1_PID:?export STAGE1_PID}"
RUN_DIR=/data2/mn1025/stage1/Qwen3.6-27B_squad_n1000_nothink
BUDGET_MIB="${BUDGET_MIB:-13000}"
TIMEOUT="${TIMEOUT:-4200}"
LOG=amortized_ue/cross_llm_5arm_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] cross-LLM 5-arm eval GPU-swap starting (GPU $GPU, interrupting Qwen3.6-27B squad)"

ps -p "$LANE_PID"   -o cmd= | grep -q "training_lane.sh 1" || { echo "ABORT: LANE_PID $LANE_PID not 'training_lane.sh 1'"; exit 1; }
ps -p "$STAGE1_PID" -o cmd= | grep -q "Qwen3.6-27B"        || { echo "ABORT: STAGE1_PID $STAGE1_PID not the Qwen3.6-27B stage1"; exit 1; }

MYHOLD=""
RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: drop fence + SIGCONT lane $LANE_PID"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null || true
  echo "[$(date)] lane continued -- will re-fence and resume Qwen3.6-27B ($(ls $RUN_DIR/records 2>/dev/null | wc -l)/1000)"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

kill -STOP "$LANE_PID"; echo "[$(date)] SIGSTOP lane $LANE_PID"; sleep 1

BEFORE=$(ls "$RUN_DIR/records" 2>/dev/null | wc -l)
kill -TERM "$STAGE1_PID" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$STAGE1_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$STAGE1_PID" 2>/dev/null || true
sleep 5
echo "[$(date)] Qwen3.6-27B stage1 killed (records ${BEFORE}/1000 preserved). GPU$GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU

FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
HOLD=$(( FREE - BUDGET_MIB ))
if [ "$HOLD" -gt 512 ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
  MYHOLD=$!
  sleep 6
  echo "[$(date)] fence pid $MYHOLD holding ~${HOLD}MiB. GPU$GPU free now:"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU
else
  echo "[$(date)] WARNING: only ${FREE}MiB free -- running unfenced"
fi

echo "[$(date)] launching cross-LLM eval (timeout ${TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $TIMEOUT $PY -m amortized_ue.cross_llm_5arm_eval \
  --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] eval exited rc=$RC"

resume
echo "[$(date)] === cross-LLM 5-arm eval GPU-swap complete (rc=$RC) ==="
exit $RC
