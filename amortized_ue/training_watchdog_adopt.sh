#!/bin/bash
# training_watchdog_adopt.sh <pid_gpu0> <pid_gpu1> -- same as training_watchdog.sh
# but SUPERVISES two already-running training_lane.sh processes instead of
# launching them (used after a manual restart_lane_inplace.sh pass so the
# watchdog doesn't spawn duplicate lanes).
#
#   nohup bash amortized_ue/training_watchdog_adopt.sh <PID0> <PID1> \
#     > amortized_ue/logs/training_watchdog.out 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p amortized_ue/logs
WDLOG=amortized_ue/logs/training_watchdog.log
MAX_RESTARTS=6
MIN_FREE_MIB=20000
log(){ echo "$(date '+%F %T') $*" | tee -a "$WDLOG"; }

launch(){ local gpu="$1"
  nohup bash amortized_ue/training_lane.sh "$gpu" \
    >> "amortized_ue/logs/training_gpu${gpu}_driver.log" 2>&1 &
  echo $!
}

supervise(){
  local gpu="$1" pid="$2" restarts=0
  local dl="amortized_ue/logs/training_gpu${gpu}_driver.log"
  log "[gpu$gpu] adopting + supervising pid $pid"
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

PID0="${1:?usage: training_watchdog_adopt.sh <pid_gpu0> <pid_gpu1>}"
PID1="${2:?usage: training_watchdog_adopt.sh <pid_gpu0> <pid_gpu1>}"
log "=== training watchdog (adopt) starting: gpu0=$PID0 gpu1=$PID1 ==="
kill -0 "$PID0" 2>/dev/null || { log "ABORT: pid $PID0 (gpu0) not alive"; exit 1; }
kill -0 "$PID1" 2>/dev/null || { log "ABORT: pid $PID1 (gpu1) not alive"; exit 1; }
supervise 0 "$PID0" &
supervise 1 "$PID1" &
wait
log "=== training watchdog (adopt) exiting (both lanes reached a terminal state) ==="
