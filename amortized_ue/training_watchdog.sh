#!/bin/bash
# training_watchdog.sh -- keeps the two training_lane.sh instances (GPU0, GPU1)
# alive across crashes (a co-tenant OOM-ing one of them is the expected failure).
# Launches both lanes itself, then supervises. Capped restarts so a real bug
# surfaces as an alert instead of a silent crash-loop.
#
# Launch detached:
#   nohup bash amortized_ue/training_watchdog.sh > amortized_ue/logs/training_watchdog.out 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p amortized_ue/logs
WDLOG=amortized_ue/logs/training_watchdog.log
MAX_RESTARTS=6
MIN_FREE_MIB=20000
log(){ echo "$(date '+%F %T') $*" | tee -a "$WDLOG"; }

launch(){ # launch <gpu> -> echoes pid
  local gpu="$1"
  nohup bash amortized_ue/training_lane.sh "$gpu" \
    >> "amortized_ue/logs/training_gpu${gpu}_driver.log" 2>&1 &
  echo $!
}

supervise(){
  local gpu="$1" pid="$2" restarts=0
  local dl="amortized_ue/logs/training_gpu${gpu}_driver.log"
  log "[gpu$gpu] supervising pid $pid"
  while true; do
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    if grep -q "this lane is done" "$dl" 2>/dev/null; then
      log "[gpu$gpu] clean finish -- watchdog stopping for this lane"; return
    fi
    restarts=$((restarts+1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      log "[gpu$gpu] !!! ALERT: crashed $restarts times, giving up -- needs a human. See $dl"; return
    fi
    log "[gpu$gpu] crash (pid $pid gone, no done-marker). Restart $restarts/$MAX_RESTARTS."
    while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu")" -lt "$MIN_FREE_MIB" ]; do
      log "[gpu$gpu] waiting for >=${MIN_FREE_MIB}MiB free before relaunch..."; sleep 15
    done
    pid=$(launch "$gpu")
    log "[gpu$gpu] relaunched as pid $pid"
  done
}

log "=== training watchdog starting ==="
PID0=$(launch 0); log "[gpu0] launched pid $PID0"
PID1=$(launch 1); log "[gpu1] launched pid $PID1"
supervise 0 "$PID0" &
supervise 1 "$PID1" &
wait
log "=== training watchdog exiting (both lanes reached a terminal state) ==="
