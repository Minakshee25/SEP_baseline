#!/bin/bash
# e64_gpu_swap.sh -- RACE-FREE GPU borrow on GPU 0 to run E64 Stage A (the DEPLOY-proxy
# zero-shot forward pass that recovers E45's per-question q_only/q_resp_only scores).
# Same proven pattern as e62_gpu_swap.sh / e63_gpu_swap.sh / E61.
#
# Sequence:
#   1. SIGSTOP the GPU0 training-lane script -> it cannot retry / re-fence / claim.
#      (watchdog only restarts a DEAD pid; a stopped pid still passes kill -0, so
#       the watchdog stays dormant throughout -- confirmed E61/E62/E63.)
#   2. SIGTERM (then SIGKILL) the Qwen3.8-27B stage1 child -> frees ~39 GB.
#      stage1 is fully resumable: it skips existing record .pt files.
#   3. Immediately fence the freed memory, leaving a generous budget for E64.
#   4. Run E64 --stage preds (inference only; deterministic; reproduces E45's AUROCs).
#   5. Drop the fence, SIGCONT the lane -> it reaps the killed child, sees the
#      Qwen3.8-27B job incomplete, re-fences, RESUMES it from its last saved record.
#
# Cost to Qwen3.8-27B: one model reload + <=2 in-flight records + one of the lane's
# 3 retry attempts for that job. GPU 1 / E63 is untouched.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python   # NFS-free venv (see E45)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder PYTHONUNBUFFERED=1

GPU=0
LANE_PID="${LANE_PID:?export LANE_PID (bash amortized_ue/training_lane.sh 0)}"
STAGE1_PID="${STAGE1_PID:?export STAGE1_PID (Qwen3.8-27B stage1 on GPU0)}"
RUN_DIR=/data2/mn1025/stage1/Qwen3.8-27B_trivia_qa_n2000_nothink
E64_BUDGET_MIB="${E64_BUDGET_MIB:-15000}"
E64_TIMEOUT="${E64_TIMEOUT:-5400}"
LOG=amortized_ue/e64_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E64 GPU-swap starting (GPU $GPU, interrupting Qwen3.8-27B)"

ps -p "$LANE_PID"   -o cmd= | grep -q "training_lane.sh 0" || { echo "ABORT: LANE_PID $LANE_PID is not 'training_lane.sh 0'"; exit 1; }
ps -p "$STAGE1_PID" -o cmd= | grep -q "Qwen3.8-27B"        || { echo "ABORT: STAGE1_PID $STAGE1_PID is not the Qwen3.8-27B stage1"; exit 1; }

MYHOLD=""
RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: drop fence + SIGCONT lane $LANE_PID"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null || true
  echo "[$(date)] lane continued -- it will re-fence and resume Qwen3.8-27B from records ($(ls $RUN_DIR/records 2>/dev/null | wc -l)/2000)"
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
echo "[$(date)] Qwen3.8-27B stage1 killed (records ${BEFORE}/2000 preserved). GPU$GPU free:"
nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU

# 3. fence the freed memory immediately
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
HOLD=$(( FREE - E64_BUDGET_MIB ))
if [ "$HOLD" -gt 512 ]; then
  CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
  MYHOLD=$!
  sleep 6
  echo "[$(date)] fence pid $MYHOLD holding ~${HOLD}MiB. GPU$GPU free now:"
  nvidia-smi --query-gpu=memory.free --format=csv,noheader -i $GPU
else
  echo "[$(date)] WARNING: only ${FREE}MiB free -- running E64 unfenced"
fi

# 4. run E64 Stage A (per-id proxy predictions only)
echo "[$(date)] launching E64 --stage preds (timeout ${E64_TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $E64_TIMEOUT $PY -m amortized_ue.e64_gemma_baserate_reanalysis --stage preds
RC=$?
echo "[$(date)] E64 stage-preds exited rc=$RC"

# 5. resume (also via trap)
resume
echo "[$(date)] === E64 GPU-swap complete (rc=$RC) ==="
exit $RC
