#!/bin/bash
# Retry-until-done GPU waiter for the Mistral n2000 TRAINING build (E22). Polls for a GPU
# with enough free memory for fp32 Mistral-7B, retries on OOM (a slot can vanish mid-load),
# and stops once all 2000 records exist. Resumable, so retries continue from disk.
set -uo pipefail
cd "$(dirname "$0")/.."                                     # repo root
NEED_MIB=${NEED_MIB:-33000}
POLL=${POLL:-15}
TARGET=${TARGET:-2000}
RECDIR=amortized_ue/data/stage1/Mistral-7B-Instruct-v0.2_trivia_qa_n2000_full/records

have() { ls "$RECDIR"/*.pt 2>/dev/null | wc -l; }

echo ">>> $(date '+%H:%M:%S') mistral-n2000 waiter up: need>=${NEED_MIB}MiB, poll=${POLL}s, target=${TARGET}, have=$(have)"
attempt=0
while [ "$(have)" -lt "$TARGET" ]; do
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -v need="$NEED_MIB" '$2>=need {print $1; exit}')
  if [ -z "${PICK:-}" ]; then sleep "$POLL"; continue; fi
  attempt=$((attempt+1))
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt on GPU $PICK (have=$(have)/${TARGET})"
  CUDA_VISIBLE_DEVICES="$PICK" bash amortized_ue/build_mistral_n2000.sh
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt exited rc=$?, have=$(have)/${TARGET}"
  [ "$(have)" -lt "$TARGET" ] && sleep "$POLL"
done
echo ">>> $(date '+%H:%M:%S') BUILD COMPLETE: have=$(have)/${TARGET} records"
