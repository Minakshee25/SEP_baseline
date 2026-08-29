#!/bin/bash
# e63_lane_safety_net.sh -- belt-and-braces guarantee that the SIGSTOPed GPU1 training
# lane gets SIGCONT'd even if e63_gpu_swap.sh is SIGKILL'd (a kill -9 can't run its
# EXIT/TERM trap, which is the only path that normally resumes the lane).
#
# Does NOTHING in the normal case: e63_gpu_swap.sh's own trap resumes the lane within
# seconds of E63 exiting, this script sees the lane is no longer stopped, and quits.
# It only acts if the lane is STILL stopped after the swap script is gone, or after a
# hard deadline.
#
# Idempotent and harmless to run alongside the swap script.
#   nohup bash amortized_ue/e63_lane_safety_net.sh <LANE_PID> <SWAP_PID> <FENCE_PID> & disown
set -uo pipefail
cd "$(dirname "$0")/.."

LANE_PID="${1:?usage: e63_lane_safety_net.sh <LANE_PID> <SWAP_PID> [FENCE_PID]}"
SWAP_PID="${2:?need the e63_gpu_swap.sh pid}"
FENCE_PID="${3:-}"
DEADLINE_S="${DEADLINE_S:-10800}"          # 3 h hard cap
LOG=amortized_ue/e63_lane_safety_net.log
START=$(date +%s)

log(){ echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

is_stopped(){ [ "$(ps -o state= -p "$1" 2>/dev/null | tr -d ' ')" = "T" ]; }
alive(){ kill -0 "$1" 2>/dev/null; }

resume_lane(){
  log "RESUMING lane $LANE_PID (reason: $1)"
  [ -n "$FENCE_PID" ] && alive "$FENCE_PID" && { kill "$FENCE_PID" 2>/dev/null && log "killed orphan fence $FENCE_PID"; }
  # also sweep any gpu_reserve whose parent is the (dead) swap script
  for p in $(pgrep -f "gpu_reserve.*--parent_pid $SWAP_PID" 2>/dev/null); do
    kill "$p" 2>/dev/null && log "killed orphan fence $p"
  done
  kill -CONT "$LANE_PID" 2>/dev/null && log "SIGCONT sent to lane $LANE_PID" || log "SIGCONT failed (lane gone?)"
  sleep 3
  log "lane $LANE_PID state now: $(ps -o state= -p "$LANE_PID" 2>/dev/null | tr -d ' ' || echo GONE)"
}

log "=== safety net armed: lane=$LANE_PID swap=$SWAP_PID fence=${FENCE_PID:-none} deadline=${DEADLINE_S}s ==="

while true; do
  if ! alive "$LANE_PID"; then
    log "lane $LANE_PID no longer exists -- nothing to resume, exiting"; exit 0
  fi
  if ! is_stopped "$LANE_PID"; then
    log "lane $LANE_PID is running (not stopped) -- normal resume happened, exiting"; exit 0
  fi
  # lane is stopped. is that still expected?
  if ! alive "$SWAP_PID"; then
    resume_lane "swap script $SWAP_PID gone but lane still stopped -> trap failed"
    exit 0
  fi
  if [ $(( $(date +%s) - START )) -ge "$DEADLINE_S" ]; then
    resume_lane "hard deadline ${DEADLINE_S}s reached, lane still stopped"
    exit 0
  fi
  sleep 20
done
