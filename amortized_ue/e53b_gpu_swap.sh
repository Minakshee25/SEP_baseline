#!/bin/bash
# e53b_gpu_swap.sh -- RACE-FREE GPU borrow on GPU 1 for the E53b eval
# (Qwen/Gemma proxy zero-shot on Llama-2/Mistral SQuAD -- pure 3B inference, no training).
# Same proven pattern as e66_gpu_swap.sh (E61-E66, zero data loss); just runs the eval and
# does no training / no W&B push.
set -uo pipefail
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/data2/mn1025/hf_cache

GPU=1
CARD_MIB="${CARD_MIB:-46068}"
FREE_MIB="${FREE_MIB:-24000}"          # leave free on GPU1 for the eval python (3B bf16 inference)
TIMEOUT="${TIMEOUT:-2400}"             # 40 min hard cap (2 targets x 3 seeds inference)
LANEJOB_MIB="${LANEJOB_MIB:-44000}"    # leave free for the 27B reload on the way out
LOG=amortized_ue/e53b_swap.log

exec >>"$LOG" 2>&1
echo "======================================================================"
echo "[$(date)] E53b GPU-swap starting (GPU $GPU)"

LANE_PID=$(pgrep -f "training_lane.sh $GPU" | head -1 || true)
[ -z "$LANE_PID" ] && { echo "ABORT: no 'training_lane.sh $GPU' running"; exit 1; }
ps -p "$LANE_PID" -o cmd= | grep -q "training_lane.sh $GPU" || { echo "ABORT: LANE_PID mismatch"; exit 1; }
echo "[$(date)] GPU$GPU lane = pid $LANE_PID"

MYHOLD="" ; SNPID="" ; RESUMED=0
resume() {
  [ "$RESUMED" = 1 ] && return; RESUMED=1
  echo "[$(date)] RESUME: arm exit-bridge + drop fence + SIGCONT lane $LANE_PID"
  LANEJOB_MIB=$LANEJOB_MIB GEMMA3_BUDGET_MIB=$LANEJOB_MIB FLOOR_MIB=6000 \
    nohup bash amortized_ue/e63_gpu_bridge.sh "$$" >> amortized_ue/e53b_bridge.log 2>&1 & disown
  echo "[$(date)] exit-bridge pid $!"
  [ -n "$MYHOLD" ] && kill "$MYHOLD" 2>/dev/null || true
  kill -CONT "$LANE_PID" 2>/dev/null && echo "[$(date)] SIGCONT lane $LANE_PID" || echo "[$(date)] SIGCONT failed"
}
trap 'resume; exit 130' INT TERM
trap resume EXIT

kill -STOP "$LANE_PID"; echo "[$(date)] SIGSTOP lane $LANE_PID"; sleep 1

DEADLINE_S=$(( TIMEOUT + 3600 )) nohup bash amortized_ue/e63_lane_safety_net.sh "$LANE_PID" "$$" "" \
  >> amortized_ue/e53b_lane_safety_net.log 2>&1 & disown
SNPID=$!; echo "[$(date)] lane safety-net pid $SNPID"

HOLD=$(( CARD_MIB - FREE_MIB ))
CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 \
  --hold_mib "$HOLD" --retry_secs 180 --parent_pid $$ &
MYHOLD=$!
echo "[$(date)] slack fence pid $MYHOLD (hold ${HOLD}MiB)"; sleep 1

KILLED=""
for p in $(pgrep -f "amortized_ue.stage1"); do
  if grep -qxz "CUDA_VISIBLE_DEVICES=$GPU" "/proc/$p/environ" 2>/dev/null; then
    echo "[$(date)] kill -9 stage1 child $p"
    kill -9 "$p" 2>/dev/null && KILLED="$p"
  fi
done
[ -z "$KILLED" ] && echo "[$(date)] WARNING: no GPU$GPU stage1 child found"
sleep 10
echo "[$(date)] GPU$GPU used/free = $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i $GPU)"

echo "[$(date)] launching E53b eval (timeout ${TIMEOUT}s)..."
CUDA_VISIBLE_DEVICES=$GPU timeout $TIMEOUT $PY -m amortized_ue.e53b_eval_on_llama2_mistral_squad \
  --data_dir /data2/mn1025/stage1
RC=$?
echo "[$(date)] E53b eval exited rc=$RC"

resume
echo "[$(date)] === E53b GPU-swap complete (rc=$RC) ==="
exit $RC
