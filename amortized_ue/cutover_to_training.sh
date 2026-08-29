#!/bin/bash
# cutover_to_training.sh -- ONE-SHOT switch from the old nothink_bigtier lanes to
# the training-data-only queue (see training_lane.sh header).
#
# Kills, in order: the old watchdog, both old lanes + their stage1 children, and
# all stray gpu_reserve holders. Then launches training_watchdog.sh which starts
# a training_lane.sh on each GPU. Records already on disk are untouched and every
# job resumes from where it stopped.
#
# Run from repo root:  bash amortized_ue/cutover_to_training.sh
set -uo pipefail
cd "$(dirname "$0")/.."
echo "=== $(date '+%F %T') cutover: old jobs BEFORE ==="
pgrep -af "lane_a_gpu0.sh|lane_b_gpu1.sh|watchdog_lanes.sh|amortized_ue.stage1|amortized_ue.gpu_reserve" || true

echo "--- killing old watchdog ---"
pkill -f "watchdog_lanes.sh" 2>/dev/null || true
sleep 2
echo "--- killing old lanes + their stage1 children + reserve holders ---"
pkill -f "amortized_ue/lane_a_gpu0.sh" 2>/dev/null || true
pkill -f "amortized_ue/lane_b_gpu1.sh" 2>/dev/null || true
pkill -f "amortized_ue.stage1"        2>/dev/null || true
pkill -f "amortized_ue.gpu_reserve"   2>/dev/null || true
sleep 5
# escalate anything that ignored SIGTERM
pkill -9 -f "amortized_ue.stage1"      2>/dev/null || true
pkill -9 -f "amortized_ue.gpu_reserve" 2>/dev/null || true
pkill -9 -f "amortized_ue/lane_[ab]_gpu" 2>/dev/null || true
sleep 3

echo "--- clearing stale training claims (fresh start) ---"
rm -rf /data2/mn1025/stage1_meta/training_n2000_claims
mkdir -p /data2/mn1025/stage1_meta/training_n2000_claims

echo "=== GPU state after kill ==="
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv
echo "=== leftover procs (should be empty) ==="
pgrep -af "lane_a_gpu0.sh|lane_b_gpu1.sh|watchdog_lanes.sh|amortized_ue.stage1|amortized_ue.gpu_reserve" || echo "  (none)"

echo "--- launching training watchdog (starts a lane on each GPU) ---"
nohup bash amortized_ue/training_watchdog.sh > amortized_ue/logs/training_watchdog.out 2>&1 &
disown
sleep 8
echo "=== new procs ==="
pgrep -af "training_watchdog.sh|training_lane.sh|amortized_ue.stage1" || true
echo "=== $(date '+%F %T') cutover done. tail -f amortized_ue/logs/training_gpu{0,1}_driver.log ==="
