#!/bin/bash
# Grab a GPU for the fp32 Llama-3-8B cross-LLM (E20) Stage-1 build, RETRYING until all
# TARGET records exist. Shared GPUs are contended: a slot can vanish in the ~12s the model
# takes to load (OOM), so we do NOT exec-once -- we loop, re-attempting whenever a GPU has
# enough free memory, until the build is actually complete. The build is resumable
# (overwrite=False), so every retry continues from the records already on disk.
set -uo pipefail
cd "$(dirname "$0")/.."                                     # repo root
NEED_MIB=${NEED_MIB:-33000}                                 # fp32 Llama-3-8B ~32GB + margin
POLL=${POLL:-15}
TARGET=${TARGET:-200}
RECDIR=amortized_ue/data/stage1/Meta-Llama-3-8B-Instruct_trivia_qa_n200_full/records

have() { ls "$RECDIR"/*.pt 2>/dev/null | wc -l; }

echo ">>> $(date '+%H:%M:%S') hardened waiter up: need>=${NEED_MIB}MiB, poll=${POLL}s, target=${TARGET}, have=$(have)"
attempt=0
while [ "$(have)" -lt "$TARGET" ]; do
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -v need="$NEED_MIB" '$2>=need {print $1; exit}')
  if [ -z "${PICK:-}" ]; then
    sleep "$POLL"; continue
  fi
  attempt=$((attempt+1))
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt on GPU $PICK (have=$(have)/${TARGET})"
  CUDA_VISIBLE_DEVICES="$PICK" bash amortized_ue/build_llama3_xllm200.sh
  rc=$?
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt exited rc=$rc, have=$(have)/${TARGET}"
  # rc!=0 (e.g. lost the memory race -> OOM) just loops back and waits for another slot.
  [ "$(have)" -lt "$TARGET" ] && sleep "$POLL"
done
echo ">>> $(date '+%H:%M:%S') BUILD COMPLETE: have=$(have)/${TARGET} records"
