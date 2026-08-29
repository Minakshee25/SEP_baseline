#!/bin/bash
# e63_gpu_bridge.sh -- close the ~1-2 min co-tenant window on GPU1 between E63 exiting
# and gemma-3-27b-it's model reload re-occupying the card.
#
# The instant the E63 python exits, this holds ONLY the free memory ABOVE gemma-3's
# full working set (so it can NEVER OOM the resuming build), re-evaluating every 5 s,
# and releases as soon as gemma-3 has taken the card (free < FLOOR) or after a cap.
#
#   nohup bash amortized_ue/e63_gpu_bridge.sh <E63_PY_PID> & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/vol/bitbucket/mn1025/conda_envs/amortized_stage2/bin/python

E63_PY_PID="${1:?usage: e63_gpu_bridge.sh <E63_python_pid>}"
GPU=1
GEMMA3_BUDGET_MIB="${GEMMA3_BUDGET_MIB:-43000}"   # leave >= this free for the 27B reload
FLOOR_MIB="${FLOOR_MIB:-6000}"                    # free below this => card is taken, we're done
MAX_BRIDGE_S="${MAX_BRIDGE_S:-360}"
LOG=amortized_ue/e63_gpu_bridge.log
log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

log "=== bridge armed: waiting for E63 python $E63_PY_PID to exit ==="
while kill -0 "$E63_PY_PID" 2>/dev/null; do sleep 5; done
log "E63 python $E63_PY_PID exited -- bridging GPU$GPU now"

HOLD_PID=""
drop(){ [ -n "$HOLD_PID" ] && kill "$HOLD_PID" 2>/dev/null && log "released bridge fence $HOLD_PID" || true; HOLD_PID=""; }
trap 'drop; exit 0' INT TERM EXIT

START=$(date +%s)
while [ $(( $(date +%s) - START )) -lt "$MAX_BRIDGE_S" ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  if [ "${free:-0}" -lt "$FLOOR_MIB" ]; then
    log "GPU$GPU free ${free}MiB < ${FLOOR_MIB} -- card is occupied (gemma-3 back), bridge done"
    break
  fi
  if [ -z "$HOLD_PID" ] || ! kill -0 "$HOLD_PID" 2>/dev/null; then
    hold=$(( free - GEMMA3_BUDGET_MIB ))
    if [ "$hold" -gt 1500 ]; then
      CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 \
        --hold_mib "$hold" --parent_pid $$ --poll 2 &
      HOLD_PID=$!
      log "bridge fence pid $HOLD_PID holding ~${hold}MiB (free ${free}MiB, gemma-3 budget ${GEMMA3_BUDGET_MIB})"
      sleep 4
    fi
  fi
  sleep 5
done
drop
trap - INT TERM EXIT
log "=== bridge complete ==="
