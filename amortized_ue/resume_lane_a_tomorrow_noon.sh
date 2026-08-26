#!/bin/bash
# Waits until tomorrow 12:00, then waits for GPU0 to actually have enough free memory
# (safety margin in case the team runs a bit over noon), then resumes lane_a_gpu0.sh.
# Fully self-contained -- meant to be launched once via `nohup ... & disown` so it survives
# independently of any Claude Code session or SSH connection, same pattern as the lanes
# themselves (see lane_a_gpu0.sh / lane_b_gpu1.sh headers).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p amortized_ue/logs
LOG=amortized_ue/logs/resume_lane_a_waiter.log
NEED_MIB=24000   # matches lane_a_gpu0.sh's SMALL_BUDGET

TARGET_EPOCH=$(date -d "tomorrow 13:00" +%s)
echo "$(date '+%Y-%m-%d %H:%M:%S') [resume-waiter] waiting until $(date -d @$TARGET_EPOCH '+%Y-%m-%d %H:%M:%S') before even checking GPU0" >> "$LOG"

while [ "$(date +%s)" -lt "$TARGET_EPOCH" ]; do
  sleep 60
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [resume-waiter] target time reached, now waiting for GPU0 to have >=${NEED_MIB}MiB free..." >> "$LOG"
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  [ "$FREE" -ge "$NEED_MIB" ] && break
  sleep 30
done

echo "$(date '+%Y-%m-%d %H:%M:%S') [resume-waiter] GPU0 has ${FREE}MiB free -- launching lane_a_gpu0.sh" >> "$LOG"
nohup bash amortized_ue/lane_a_gpu0.sh > amortized_ue/logs/lane_a_driver.log 2>&1 &
disown
echo "$(date '+%Y-%m-%d %H:%M:%S') [resume-waiter] lane A relaunched, pid $!. resume-waiter done." >> "$LOG"
