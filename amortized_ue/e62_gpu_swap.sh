#!/bin/bash
# e62_gpu_swap.sh — RACE-FREE GPU borrow on GPU 1 (Qwen3.8-27B lane).
#
# Both 27B builds are crawling under GPU contention; waiting for one to finish
# isn't viable. Instead: cleanly interrupt GPU1's Qwen3.8 (stage1 is fully
# resumable -- it skips existing record .pt files), borrow the GPU for E62,
# then let the lane resume it.
#
# Sequence:
#   1. SIGSTOP the GPU1 lane script  -> it cannot retry / re-fence / claim.
#      (watchdog only restarts a DEAD pid; a stopped pid still passes kill -0,
#       so the watchdog stays dormant throughout.)
#   2. SIGTERM (then SIGKILL) the Qwen3.8 stage1 child -> frees ~39 GB.
#   3. Immediately fence the freed memory, leaving ~10.5 GB for E62 -> closes
#      the window against any external job in ~5 s.
#   4. Run E62 (hard 40-min cap).
#   5. Drop the fence, SIGCONT the lane -> it reaps the killed child, sees the
#      job incomplete, re-fences, and RESUMES Qwen3.8 from its last saved record.
#
# Cost to Qwen3.8: one model reload (~a few min) + <=2 in-flight records +
# one of the lane's 3 retry attempts for that job. Nothing else is touched;
# GPU 0 / gemma-2 keeps running.
#
# Safety: `resume` is idempotent and runs from EXIT + INT/TERM traps, so the
# lane is ALWAYS continued -- even if E62 crashes, times out, or this script
# is killed.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/vol/bitbucket/mn1025/conda_envs/amortized_stage2/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU=1
LANE_PID=2056752          # bash amortized_ue/training_lane.sh 1
STAGE1_PID=2057015        # Qwen3.8-27B stage1 on GPU1
RUN_DIR=/data2/mn1025/stage1/Qwen3.8-27B_trivia_qa_n2000_nothink
E62_BUDGET_MIB=10500
E62_TIMEOUT=2400
LOG=amortized_ue/e62_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E62 GPU-swap starting (GPU $GPU, interrupting Qwen3.8)"

ps -p $LANE_PID   -o cmd= | grep -q "training_lane.sh 1" || { echo "ABORT: LANE_PID $LANE_PID is not 'training_lane.sh 1'"; exit 1; }
ps -p $STAGE1_PID -o cmd= | grep -q "Qwen3.8-27B"        || { echo "ABORT: STAGE1_PID $STAGE1_PID is not the Qwen3.8 stage1"; exit 1; }

MYHOLD=""
RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: drop fence + SIGCONT lane $LANE_PID"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT $LANE_PID 2>/dev/null || true
  echo "[$(date)] lane continued -- it will re-fence and resume Qwen3.8 from records ($(ls $RUN_DIR/records 2>/dev/null | wc -l)/2000)"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

# 1. freeze the lane
kill -STOP $LANE_PID
echo "[$(date)] SIGSTOP lane $LANE_PID (watchdog dormant: pid still exists)"
sleep 1

# 2. kill the in-flight stage1 (resumable from record .pt files)
BEFORE=$(ls $RUN_DIR/records 2>/dev/null | wc -l)
kill -TERM $STAGE1_PID 2>/dev/null || true
for i in $(seq 1 60); do kill -0 $STAGE1_PID 2>/dev/null || break; sleep 1; done
kill -KILL $STAGE1_PID 2>/dev/null || true
sleep 5
echo "[$(date)] Qwen3.8 stage1 killed (records ${BEFORE}/2000 preserved). GPU$GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU

# 3. fence the freed memory immediately
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
HOLD=$(( FREE - E62_BUDGET_MIB ))
if [ "$HOLD" -gt 512 ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 --hold_mib $HOLD --parent_pid $$ &
  MYHOLD=$!
  sleep 6
  echo "[$(date)] fence pid $MYHOLD holding ~${HOLD}MiB. GPU$GPU free now:"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU
else
  echo "[$(date)] WARNING: only ${FREE}MiB free -- running E62 unfenced"
fi

# 4. run E62
echo "[$(date)] launching E62 (timeout ${E62_TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $E62_TIMEOUT $PY -m amortized_ue.e62_qresp_alone_vs_sep \
  --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] E62 exited rc=$RC"

# 5. resume (also via trap)
resume
echo "[$(date)] === E62 GPU-swap complete (rc=$RC) ==="
exit $RC
