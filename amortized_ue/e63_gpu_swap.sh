#!/bin/bash
# e63_gpu_swap.sh — RACE-FREE GPU borrow on GPU 1 for the E63 leave-two-out proxy
# (train 3 seeds + eval). Same pattern as e62_gpu_swap.sh / E61.
#
# Sequence:
#   1. SIGSTOP the GPU1 training-lane script -> it cannot retry / re-fence / claim.
#      (watchdog only restarts a DEAD pid; a stopped pid still passes kill -0, so
#       the watchdog stays dormant throughout — confirmed E61/E62.)
#   2. SIGTERM (then SIGKILL) the gemma-3-27b-it stage1 child -> frees ~40 GB.
#      stage1 is fully resumable: it skips existing record .pt files.
#   3. Immediately fence the freed memory, leaving ~16 GB for E63.
#   4. Run E63 --stage all (train then eval), hard 2.5-h cap.
#   5. Drop the fence, SIGCONT the lane -> it reaps the killed child, sees the
#      gemma-3 job incomplete, re-fences, RESUMES it from its last saved record.
#
# Cost to gemma-3-27b-it: one model reload + <=2 in-flight records + one of the
# lane's 3 retry attempts for that job. GPU 0 / Qwen3.8-27B is untouched.
#
# Safety: `resume` is idempotent and runs from EXIT + INT/TERM traps.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/vol/bitbucket/mn1025/conda_envs/amortized_stage2/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=1
LANE_PID="${LANE_PID:?export LANE_PID (bash amortized_ue/training_lane.sh 1)}"
STAGE1_PID="${STAGE1_PID:?export STAGE1_PID (gemma-3-27b-it stage1 on GPU1)}"
RUN_DIR=/data2/mn1025/stage1/gemma-3-27b-it_trivia_qa_n2000_full
E63_BUDGET_MIB="${E63_BUDGET_MIB:-30000}"   # 3B LoRA training forward+backward OOM'd at 16000
E63_TIMEOUT="${E63_TIMEOUT:-9000}"
LOG=amortized_ue/e63_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E63 GPU-swap starting (GPU $GPU, interrupting gemma-3-27b-it)"

ps -p "$LANE_PID"   -o cmd= | grep -q "training_lane.sh 1" || { echo "ABORT: LANE_PID $LANE_PID is not 'training_lane.sh 1'"; exit 1; }
ps -p "$STAGE1_PID" -o cmd= | grep -q "gemma-3-27b-it"      || { echo "ABORT: STAGE1_PID $STAGE1_PID is not the gemma-3-27b-it stage1"; exit 1; }

MYHOLD=""
RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: drop fence + SIGCONT lane $LANE_PID"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null || true
  echo "[$(date)] lane continued -- it will re-fence and resume gemma-3-27b-it from records ($(ls $RUN_DIR/records 2>/dev/null | wc -l)/2000)"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

# 1. freeze the lane
kill -STOP "$LANE_PID"
echo "[$(date)] SIGSTOP lane $LANE_PID (watchdog dormant: pid still exists)"
sleep 1

# 2. kill the in-flight stage1 (resumable from record .pt files)
BEFORE=$(ls "$RUN_DIR/records" 2>/dev/null | wc -l)
kill -TERM "$STAGE1_PID" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$STAGE1_PID" 2>/dev/null || break; sleep 1; done
kill -KILL "$STAGE1_PID" 2>/dev/null || true
sleep 5
echo "[$(date)] gemma-3 stage1 killed (records ${BEFORE}/2000 preserved). GPU$GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU

# 3. fence the freed memory immediately
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
HOLD=$(( FREE - E63_BUDGET_MIB ))
if [ "$HOLD" -gt 512 ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
  MYHOLD=$!
  sleep 6
  echo "[$(date)] fence pid $MYHOLD holding ~${HOLD}MiB. GPU$GPU free now:"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU
else
  echo "[$(date)] WARNING: only ${FREE}MiB free -- running E63 unfenced"
fi

# 4. run E63 (train 3 seeds + eval)
echo "[$(date)] launching E63 --stage all (timeout ${E63_TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $E63_TIMEOUT $PY -m amortized_ue.e63_lto_deepseek_qwen3_8b \
  --stage all --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] E63 exited rc=$RC"

# 5. resume (also via trap)
resume
echo "[$(date)] === E63 GPU-swap complete (rc=$RC) ==="
exit $RC
