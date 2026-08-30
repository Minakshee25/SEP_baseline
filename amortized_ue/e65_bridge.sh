#!/bin/bash
# e65_bridge.sh -- close the co-tenant window on GPU 1 between E65 releasing the
# card and the training lane's 27B stage1 build reloading onto it (~30s+). Holds
# everything ABOVE the lane job's working set, re-evaluating every few seconds,
# and exits once the lane has retaken the card (or after a hard cap).
#   nohup bash amortized_ue/e65_bridge.sh > amortized_ue/logs/e65_bridge.out 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
GPU=1
LANEJOB_MIB="${LANEJOB_MIB:-44000}"   # leave >= this free for the 27B reload
CAP_S="${CAP_S:-420}"
say(){ echo ">>> $(date '+%F %T') [e65-bridge] $*"; }

say "armed on GPU$GPU (hold above ${LANEJOB_MIB}MiB until the lane retakes the card)"
end=$(( $(date +%s) + CAP_S ))
while [ "$(date +%s)" -lt "$end" ]; do
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  if [ "${free:-0}" -lt 6000 ]; then say "GPU$GPU free ${free}MiB -- lane has retaken it, done"; exit 0; fi
  h=$(( free - LANEJOB_MIB ))
  if [ "$h" -gt 1500 ]; then
    CUDA_VISIBLE_DEVICES=$GPU timeout 30 $PY -m amortized_ue.gpu_reserve \
      --device 0 --hold_mib "$h" --parent_pid $$ --poll 2 2>/dev/null || true
  else
    sleep 3
  fi
done
say "cap reached -- exiting"
