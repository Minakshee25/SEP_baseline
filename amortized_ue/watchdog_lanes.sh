#!/bin/bash
# watchdog_lanes.sh -- keeps lane_a_gpu0.sh (GPU0) and lane_b_gpu1.sh (GPU1) alive across
# crashes (e.g. a neighbouring job OOM-ing one of them). Attaches to the two lane processes
# that are ALREADY running (found via pgrep at startup) and supervises them from there.
#
# Why not just re-run the lane script blindly on crash: a big-tier job's claim (mkdir under
# $CLAIMS_DIR) is never released by the lane script itself on failure, so a naive relaunch
# would see "already claimed" and skip exactly the job that crashed, leaving it stuck
# forever. This watchdog finds the in-flight job from the lane's own log (the last
# "starting X" line with no later "X exited rc=" line) and removes its claim before
# relaunching, so the resumed lane (or the other lane) can pick it back up. stage1.py itself
# already skips completed *records* on disk, so the resumed job continues, it doesn't restart.
#
# Capped at MAX_RESTARTS per lane so a REAL bug (not a transient OOM) surfaces as a logged
# alert instead of crash-looping silently forever -- same rationale as the rc-capture fix
# already in both lane scripts (a silent crash-loop is exactly what burned 5 jobs before that
# fix landed).
#
# Scope: watches the small-tier + shared big-tier queue phase. The final gemma-2-27b-it
# resume tail is treated as "done, stop watching" whether it succeeds or not -- restart
# coverage for that last bonus phase is intentionally out of scope here.
#
# Self-contained -- launch via `nohup ... & disown` so it survives independently of any
# session, same pattern as resume_lane_a_tomorrow_noon.sh.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p amortized_ue/logs
CLAIMS_DIR=/data2/mn1025/stage1_meta/nothink_bigtier_claims
MAX_RESTARTS=5
MIN_FREE_MIB=20000   # conservative floor before relaunching -- at least small-tier headroom
WDLOG=amortized_ue/logs/watchdog.log

log(){ echo "$(date '+%Y-%m-%d %H:%M:%S') $1" >> "$WDLOG"; }

release_stuck_claim() {
  # $1 = lane log file. Finds the last "starting X" line; if no later "X exited rc=" line
  # follows it in the same log, X was in-flight when the process died -- release its claim
  # (if any; small-tier jobs have none, rmdir on a missing dir is a harmless no-op).
  local log_file="$1"
  local last_start_line line_no run_name
  last_start_line=$(grep -n " starting " "$log_file" 2>/dev/null | tail -1)
  [ -z "$last_start_line" ] && return
  line_no=${last_start_line%%:*}
  run_name=$(echo "$last_start_line" | sed 's/.* starting //')
  [ -z "$run_name" ] && return
  if ! tail -n +"$line_no" "$log_file" | grep -q "$run_name exited rc="; then
    local claim="$CLAIMS_DIR/${run_name}.claimed"
    if [ -d "$claim" ]; then
      rmdir "$claim" 2>/dev/null && log "[watchdog] released stuck claim: $run_name"
    else
      log "[watchdog] in-flight job was $run_name (no claim to release, small-tier or already unclaimed)"
    fi
  fi
}

wait_for_mem() {
  local gpu_idx="$1"
  while true; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$gpu_idx")
    [ "$free" -ge "$MIN_FREE_MIB" ] && break
    sleep 10
  done
}

clean_finish() {
  # $1 = lane log file. True if the lane reached a real terminal state (not a crash).
  grep -qE "FULL SHARED QUEUE COMPLETE|is handling the gemma resume. Done.|gemma-2-27b-it n2000 resume exited rc=" "$1" 2>/dev/null
}

supervise() {
  local lane_name="$1" script="$2" gpu_idx="$3" pid="$4" log_file="$5"
  local restarts=0
  log "[watchdog:$lane_name] attached to pid $pid, watching $log_file"
  while true; do
    while kill -0 "$pid" 2>/dev/null; do sleep 20; done
    if clean_finish "$log_file"; then
      log "[watchdog:$lane_name] finished cleanly. Watchdog stopping for this lane."
      return
    fi
    restarts=$((restarts+1))
    if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
      log "[watchdog:$lane_name] !!! ALERT: crashed $restarts times, giving up -- needs a human. See $log_file"
      return
    fi
    log "[watchdog:$lane_name] crash detected (pid $pid died, no clean-finish marker). Restart attempt $restarts/$MAX_RESTARTS."
    release_stuck_claim "$log_file"
    wait_for_mem "$gpu_idx"
    nohup bash "$script" > "amortized_ue/logs/${lane_name}_restart${restarts}.log" 2>&1 &
    pid=$!
    log_file="amortized_ue/logs/${lane_name}_restart${restarts}.log"
    log "[watchdog:$lane_name] relaunched as pid $pid, now watching $log_file"
  done
}

PID_A=$(pgrep -f "bash amortized_ue/lane_a_gpu0.sh" | head -1)
PID_B=$(pgrep -f "bash amortized_ue/lane_b_gpu1.sh" | head -1)
if [ -z "$PID_A" ] || [ -z "$PID_B" ]; then
  log "[watchdog] ERROR: could not find both lane pids at startup (A='$PID_A' B='$PID_B'). Exiting."
  exit 1
fi

supervise "lane_a" amortized_ue/lane_a_gpu0.sh 0 "$PID_A" amortized_ue/logs/lane_a_driver.log &
supervise "lane_b" amortized_ue/lane_b_gpu1.sh 1 "$PID_B" amortized_ue/logs/lane_b_driver_restart.log &
wait
