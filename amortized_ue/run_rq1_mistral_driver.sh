#!/bin/bash
# One-shot: free GPU0 (stop watchdog + GPU0 lane + the GPU0 Stage-1 job), run the
# Mistral RQ1 latency benchmark, then clean-restart the training queue. GPU1's
# Stage-1 job runs untouched throughout; it only blips at the final restart.
set -uo pipefail
cd "$(dirname "$0")/.."
LOG=amortized_ue/logs/rq1_mistral_driver.log
exec > >(tee -a "$LOG") 2>&1
echo "=== $(date '+%F %T') mistral driver starting ==="

trap 'echo "=== $(date) trap: resuming queue ==="; bash amortized_ue/resume_training_queue.sh' EXIT

GPU0_JOB=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i 0 \
  | xargs -r -n1 ps -o cmd= -p 2>/dev/null | grep -oE 'run_name [A-Za-z0-9._-]+' | awk '{print $2}' | head -1)
echo "GPU0 Stage-1 job detected: ${GPU0_JOB:-<none>}"

echo "--- stopping watchdog + GPU0 lane + GPU0 job ---"
pkill -9 -f "amortized_ue/training_watchdog.sh" 2>/dev/null || true
for p in $(pgrep -f "amortized_ue/training_lane.sh 0"); do kill -9 "$p" 2>/dev/null || true; done
[ -n "${GPU0_JOB:-}" ] && pkill -f "run_name ${GPU0_JOB}" 2>/dev/null || true
sleep 8
[ -n "${GPU0_JOB:-}" ] && pkill -9 -f "run_name ${GPU0_JOB}" 2>/dev/null || true
pkill -9 -f "gpu_reserve --device 0 --hold_mib 656" 2>/dev/null || true
[ -n "${GPU0_JOB:-}" ] && rm -rf "/data2/mn1025/stage1_meta/training_n2000_claims/${GPU0_JOB}.claimed"
sleep 4
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

echo "--- running Mistral benchmark ---"
bash amortized_ue/run_rq1_latency_mistral.sh
echo "=== $(date '+%F %T') mistral benchmark returned rc=$? ==="
# trap resumes the queue
