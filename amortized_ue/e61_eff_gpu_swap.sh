#!/bin/bash
# e61_eff_gpu_swap.sh -- RACE-FREE GPU borrow on GPU 1 for the E68 LOLO-squad eval
# (amortized_ue.e61_efficiency --stage proxy --data_dir /data2/mn1025/stage1). Inference only, no training.
# Same pattern as eval_5arm_gpu_swap.sh / e63_gpu_swap.sh.
#
#   1. SIGSTOP the GPU1 training-lane script (a stopped pid still passes kill -0,
#      so training_watchdog_adopt.sh stays dormant).
#   2. SIGTERM->SIGKILL the gemma-2-27b-it stage1 child -> frees ~41 GB.
#      stage1 is fully resumable (skips existing record .pt files).
#   3. Fence the freed memory, leaving ~BUDGET_MIB for the eval.
#   4. Run the eval (hard cap TIMEOUT).
#   5. Drop the fence, SIGCONT the lane -> it reaps the killed child, re-fences,
#      RESUMES gemma-2-27b-it from its last saved record.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/vol/bitbucket/mn1025/conda_envs/amortized_stage2/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=1
LANE_PID="${LANE_PID:?export LANE_PID}"
STAGE1_PID="${STAGE1_PID:?export STAGE1_PID}"
RUN_DIR=/data2/mn1025/stage1/gemma-2-27b-it_squad_n1000_full
BUDGET_MIB="${BUDGET_MIB:-14000}"
TIMEOUT="${TIMEOUT:-1800}"
LOG=amortized_ue/e61_eff_gpu_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E61-eff proxy GPU-swap starting (GPU $GPU, interrupting gemma-2-27b-it squad)"

ps -p "$LANE_PID"   -o cmd= | grep -q "training_lane.sh 1" || { echo "ABORT: LANE_PID $LANE_PID not 'training_lane.sh 1'"; exit 1; }
ps -p "$STAGE1_PID" -o cmd= | grep -q "gemma-2-27b-it"     || { echo "ABORT: STAGE1_PID $STAGE1_PID not the gemma-2-27b-it stage1"; exit 1; }

MYHOLD=""
RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: drop fence + SIGCONT lane $LANE_PID"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null || true
  echo "[$(date)] lane continued -- will re-fence and resume gemma-2-27b-it ($(ls $RUN_DIR/records 2>/dev/null | wc -l)/1000)"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

kill -STOP "$LANE_PID"; echo "[$(date)] SIGSTOP lane $LANE_PID"; sleep 1

BEFORE=$(ls "$RUN_DIR/records" 2>/dev/null | wc -l)
kill -TERM "$STAGE1_PID" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$STAGE1_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$STAGE1_PID" 2>/dev/null || true
sleep 5
echo "[$(date)] gemma-2-27b-it stage1 killed (records ${BEFORE}/1000 preserved). GPU$GPU free:"
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

echo "[$(date)] launching E61-eff proxy (timeout ${TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $TIMEOUT $PY -m amortized_ue.e61_efficiency --stage proxy --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] eval exited rc=$RC"

resume
echo "[$(date)] === E61-eff proxy GPU-swap complete (rc=$RC) ==="
exit $RC
