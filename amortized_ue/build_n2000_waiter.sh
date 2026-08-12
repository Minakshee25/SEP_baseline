#!/bin/bash
# Resumable retry-waiter for a Stage-1 n2000 build on a SHARED GPU cluster, WITH memory fencing so
# co-tenants cannot grab the slack and OOM us mid-run (which killed the Llama-3 n2000 build once).
# Per attempt: (1) wait for a GPU with >= NEED_MIB free; (2) launch a gpu_reserve holder that grabs
# (free - NEED_MIB - SAFETY) MiB on that GPU, fencing the leftover; (3) run the build on the same GPU;
# (4) kill the holder when the attempt exits. Build is resumable, so each stretch saves records.
# The holder self-releases if this waiter dies (--parent_pid), so it can never orphan-block the GPU.
# Args: MODEL ENV [NEED_MIB]
#   MODEL = deepseek-llm-7b-chat | Meta-Llama-3-8B-Instruct | ...
#   ENV   = se_probes_llama3 | se_probes
#   NEED_MIB = min free MiB to launch AND the job's memory budget (default 37000; fp32 7-8B peak ~35GB)
set -uo pipefail
cd "$(dirname "$0")/.."                                     # repo root
MODEL="$1"; ENV="$2"; NEED_MIB="${3:-37000}"; POLL=${POLL:-5}; TARGET=${TARGET:-2000}; SAFETY=${SAFETY:-400}
RECDIR="amortized_ue/data/stage1/${MODEL}_trivia_qa_n2000_full/records"
have(){ ls "$RECDIR"/*.pt 2>/dev/null | wc -l; }

source /data/sv/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/vol/bitbucket/mn1025/hf_cache OPENAI_API_KEY=placeholder
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1
export WANDB_CACHE_DIR=/vol/bitbucket/mn1025/wandb_cache WANDB_DATA_DIR=/vol/bitbucket/mn1025/wandb_data

HOLDER_PID=""
cleanup(){ [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM                                  # never leave a holder fencing the GPU

echo ">>> $(date +%H:%M:%S) fenced n2000 waiter: model=$MODEL env=$ENV need=${NEED_MIB}MiB target=$TARGET have=$(have)"
while [ "$(have)" -lt "$TARGET" ]; do
  PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$NEED_MIB '$2>=n{print $1;exit}')
  [ -z "${PICK:-}" ] && { sleep $POLL; continue; }
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$PICK")
  HOLD=$(( FREE - NEED_MIB - SAFETY ))
  ( conda activate "$ENV"
    if [ "$HOLD" -gt 512 ]; then
      CUDA_VISIBLE_DEVICES="$PICK" python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
      HOLDER_PID=$!
      echo ">>> $(date +%H:%M:%S) fencing GPU $PICK: hold ${HOLD}MiB (free ${FREE}, budget ${NEED_MIB}), holder pid $HOLDER_PID"
      sleep 4
    else
      echo ">>> $(date +%H:%M:%S) GPU $PICK free ${FREE} ~= budget; no slack to fence"
    fi
    echo ">>> $(date +%H:%M:%S) attempt on GPU $PICK (have=$(have)/$TARGET)"
    CUDA_VISIBLE_DEVICES="$PICK" python -m amortized_ue.stage1 --model_name "$MODEL" --dataset trivia_qa --num_samples "$TARGET"
    rc=$?
    [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true
    echo ">>> $(date +%H:%M:%S) attempt exited rc=$rc, have=$(have)/$TARGET" )
  HOLDER_PID=""
  [ "$(have)" -lt "$TARGET" ] && sleep $POLL
done
echo ">>> $(date +%H:%M:%S) n2000 BUILD COMPLETE model=$MODEL have=$(have)/$TARGET"
