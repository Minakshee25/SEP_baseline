#!/bin/bash
# e66_gpu_swap.sh -- RACE-FREE GPU borrow on GPU 1 for the E66 backbone-swap proxy
# (Qwen2.5-3B q_resp_only LOLO, Mistral held out: train 3 seeds + eval + push).
# Same proven pattern as e63_gpu_swap.sh / e65_run.sh (E61-E65, zero data loss).
#
# Sequence:
#   1. SIGSTOP the GPU1 training-lane script -> it cannot retry / re-fence / claim.
#      (watchdog only restarts a DEAD pid; a stopped pid still passes kill -0, so the
#       watchdog stays dormant -- confirmed E61-E65.)
#   2. Start a retry-grab slack fence: it spins holding (CARD - E66_FREE) MiB, which
#      only becomes grabbable once the lane's child dies -> gap-free.
#   3. kill -9 the GPU1 stage1 child (fully resumable: it skips existing record .pt).
#   4. Run E66 --stage all (train 3 seeds then eval), hard cap.
#   5. On rc=0, push the 3 checkpoints to W&B (se_probes_v5, CPU).
#   6. Drop the fence, arm an exit-bridge (holds only the slack ABOVE the 27B working
#      set so the reload can't OOM), SIGCONT the lane -> it reaps the killed child,
#      re-fences, RESUMES the interrupted build from its last saved record.
#
# Cost to the interrupted GPU1 build: one model reload + <=2 in-flight records +
# one of the lane's retry attempts. GPU 0's lane is untouched.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
PY_WANDB=/data2/mn1025/conda_envs/se_probes_v5/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data2/mn1025/hf_cache

GPU=1
CARD_MIB="${CARD_MIB:-46068}"
E66_FREE_MIB="${E66_FREE_MIB:-31000}"       # leave this free on GPU1 for the E66 python
E66_TIMEOUT="${E66_TIMEOUT:-9000}"          # 2.5 h hard cap (1 fold x 3 seeds ~ 1.5-2 h)
LANEJOB_MIB="${LANEJOB_MIB:-44000}"         # leave this free for the 27B reload on the way out
LOG=amortized_ue/e66_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E66 GPU-swap starting (GPU $GPU)"

LANE_PID=$(pgrep -f "training_lane.sh $GPU" | head -1 || true)
[ -z "$LANE_PID" ] && { echo "ABORT: no 'training_lane.sh $GPU' running"; exit 1; }
ps -p "$LANE_PID" -o cmd= | grep -q "training_lane.sh $GPU" || { echo "ABORT: LANE_PID $LANE_PID mismatch"; exit 1; }
echo "[$(date)] GPU$GPU lane = pid $LANE_PID"

MYHOLD="" ; SNPID="" ; RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: arm exit-bridge + drop fence + SIGCONT lane $LANE_PID"
  LANEJOB_MIB=$LANEJOB_MIB GEMMA3_BUDGET_MIB=$LANEJOB_MIB FLOOR_MIB=6000 \
    nohup bash amortized_ue/e63_gpu_bridge.sh "$$" >> amortized_ue/e66_bridge.log 2>&1 & disown
  echo "[$(date)] exit-bridge pid $!"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null && echo "[$(date)] SIGCONT lane $LANE_PID" || echo "[$(date)] SIGCONT failed (lane gone?)"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

# 1. freeze the lane
kill -STOP "$LANE_PID"
echo "[$(date)] SIGSTOP lane $LANE_PID"
sleep 1

# belt-and-braces: force-resume the lane if THIS script is kill -9'd (no trap runs)
DEADLINE_S=$(( E66_TIMEOUT + 3600 )) nohup bash amortized_ue/e63_lane_safety_net.sh "$LANE_PID" "$$" "" \
  >> amortized_ue/e66_lane_safety_net.log 2>&1 & disown
SNPID=$!
echo "[$(date)] lane safety-net pid $SNPID"

# 2. retry-grab the slack fence BEFORE killing the child (gap-free)
HOLD=$(( CARD_MIB - E66_FREE_MIB ))
CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 \
  --hold_mib "$HOLD" --retry_secs 180 --parent_pid $$ &
MYHOLD=$!
echo "[$(date)] slack fence pid $MYHOLD (target hold ${HOLD}MiB, retry 180s)"
sleep 1

# 3. kill the GPU1 stage1 child (resumable from its record .pt files)
KILLED=""
for p in $(pgrep -f "amortized_ue.stage1"); do
  if grep -qxz "CUDA_VISIBLE_DEVICES=$GPU" "/proc/$p/environ" 2>/dev/null; then
    what=$(ps -o cmd= -p "$p" | grep -oE "model_name [^ ]+ --dataset [^ ]+.*--run_name [^ ]+" || echo "?")
    echo "[$(date)] kill -9 stage1 child $p  ($what)"
    kill -9 "$p" 2>/dev/null && KILLED="$p"
  fi
done
[ -z "$KILLED" ] && echo "[$(date)] WARNING: found no GPU$GPU stage1 child to kill"
sleep 10
echo "[$(date)] GPU$GPU now: used/free = $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i $GPU)"

# 4. run E66 (train 3 seeds + eval)
echo "[$(date)] launching E66 --stage all (timeout ${E66_TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $E66_TIMEOUT $PY -m amortized_ue.e66_qwen25_proxy_lolo \
  --stage all --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] E66 --stage all exited rc=$RC"

# 5. push checkpoints (CPU, off-GPU) -- only if train+eval succeeded
if [ $RC -eq 0 ]; then
  echo "[$(date)] pushing checkpoints to W&B"
  $PY_WANDB -m amortized_ue.e66_qwen25_proxy_lolo --stage push_wandb || echo "[$(date)] push failed (ckpts local)"
fi

# 6. resume (also via traps)
resume
echo "[$(date)] === E66 GPU-swap complete (rc=$RC) ==="
exit $RC
