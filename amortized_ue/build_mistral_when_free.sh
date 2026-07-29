#!/bin/bash
# Grab a GPU for the fp32 Mistral-7B cross-LLM (E21) Stage-1 build, RETRYING until all
# TARGET records exist (a slot can vanish during the ~12s model load -> OOM; the build is
# resumable so each retry continues from disk). Same pattern as build_llama3_when_free.sh.
set -uo pipefail
cd "$(dirname "$0")/.."                                     # repo root
NEED_MIB=${NEED_MIB:-33000}                                 # fp32 Mistral-7B ~29GB + margin
POLL=${POLL:-15}
TARGET=${TARGET:-200}
RECDIR=amortized_ue/data/stage1/Mistral-7B-Instruct-v0.2_trivia_qa_n200_full/records

have() { ls "$RECDIR"/*.pt 2>/dev/null | wc -l; }

echo ">>> $(date '+%H:%M:%S') mistral waiter up: need>=${NEED_MIB}MiB, poll=${POLL}s, target=${TARGET}, have=$(have)"
attempt=0
while [ "$(have)" -lt "$TARGET" ]; do
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
         | awk -v need="$NEED_MIB" '$2>=need {print $1; exit}')
  if [ -z "${PICK:-}" ]; then
    sleep "$POLL"; continue
  fi
  attempt=$((attempt+1))
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt on GPU $PICK (have=$(have)/${TARGET})"
  CUDA_VISIBLE_DEVICES="$PICK" bash amortized_ue/build_mistral_xllm200.sh
  rc=$?
  echo ">>> $(date '+%H:%M:%S') attempt #$attempt exited rc=$rc, have=$(have)/${TARGET}"
  [ "$(have)" -lt "$TARGET" ] && sleep "$POLL"
done
echo ">>> $(date '+%H:%M:%S') BUILD COMPLETE: have=$(have)/${TARGET} records"
