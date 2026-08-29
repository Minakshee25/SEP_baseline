#!/bin/bash
# Full clean restart of the supervised training queue. Kills any stray lane/job/
# reserve, clears claims, starts the watchdog (which starts a lane per GPU).
# Every Stage-1 job resumes from disk (records save incrementally).
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== $(date '+%F %T') resume_training_queue ==="
pkill -9 -f "amortized_ue/training_watchdog.sh" 2>/dev/null || true
pkill -9 -f "amortized_ue/training_lane.sh" 2>/dev/null || true
pkill -f "amortized_ue.stage1" 2>/dev/null || true
pkill -f "amortized_ue.gpu_reserve" 2>/dev/null || true
sleep 6
pkill -9 -f "amortized_ue.stage1" 2>/dev/null || true
pkill -9 -f "amortized_ue.gpu_reserve" 2>/dev/null || true
sleep 3
rm -rf /data2/mn1025/stage1_meta/training_n2000_claims
mkdir -p /data2/mn1025/stage1_meta/training_n2000_claims
nohup bash amortized_ue/training_watchdog.sh > amortized_ue/logs/training_watchdog.out 2>&1 &
disown
sleep 12
echo "--- procs ---"
pgrep -af "training_watchdog|training_lane|amortized_ue.stage1" | grep -v "bin/bash -c" || echo "  (NONE - CHECK)"
nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "=== $(date '+%F %T') resume done ==="
