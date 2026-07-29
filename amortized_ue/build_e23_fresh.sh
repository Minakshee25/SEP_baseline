#!/bin/bash
# E23: build a fresh, held-out trivia_qa batch (1000 ids in NO existing build) for one target.
# Retry-until-done GPU waiter. Args: MODEL ENV [GPU]
#   MODEL = Llama-2-7b-chat | Mistral-7B-Instruct-v0.2
#   ENV   = se_probes (Llama-2, faithful to the reference) | se_probes_llama3 (Mistral)
#   GPU   = optional: pin to this GPU index; omit to auto-pick any GPU with >=33GB free.
# Same Stage-1 procedure as n2000/n200: --selection_num_samples 3074 makes the seed-10
# selection cover the whole val set so --only_ids can pick the fresh (complement) ids.
set -uo pipefail
cd "$(dirname "$0")/.."                                     # repo root
MODEL="$1"; ENV="$2"; PIN="${3:-}"
NEED_MIB=${NEED_MIB:-33000}; POLL=${POLL:-15}; TARGET=${TARGET:-1000}
RECDIR="amortized_ue/data/stage1/${MODEL}_trivia_qa_n1000_full/records"
have(){ ls "$RECDIR"/*.pt 2>/dev/null | wc -l; }

source /data/sv/miniconda3/etc/profile.d/conda.sh
export HF_HOME=/vol/bitbucket/mn1025/hf_cache OPENAI_API_KEY=placeholder
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python PYTHONUNBUFFERED=1
export WANDB_CACHE_DIR=/vol/bitbucket/mn1025/wandb_cache WANDB_DATA_DIR=/vol/bitbucket/mn1025/wandb_data
echo ">>> $(date +%H:%M:%S) E23 waiter: model=$MODEL env=$ENV pin='${PIN:-auto}' target=$TARGET have=$(have)"
while [ "$(have)" -lt "$TARGET" ]; do
  if [ -n "$PIN" ]; then
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$PIN")
    PICK=""; [ "$FREE" -ge "$NEED_MIB" ] && PICK="$PIN"
  else
    PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n=$NEED_MIB '$2>=n{print $1;exit}')
  fi
  [ -z "${PICK:-}" ] && { sleep $POLL; continue; }
  echo ">>> $(date +%H:%M:%S) attempt on GPU $PICK (have=$(have)/$TARGET)"
  ( conda activate "$ENV"
    CUDA_VISIBLE_DEVICES="$PICK" python -m amortized_ue.stage1 \
      --model_name "$MODEL" --dataset trivia_qa \
      --num_samples 1000 --selection_num_samples 3074 --only_ids scratch_xllm/e23_fresh_ids.txt )
  echo ">>> $(date +%H:%M:%S) attempt exited rc=$?, have=$(have)/$TARGET"
  [ "$(have)" -lt "$TARGET" ] && sleep $POLL
done
echo ">>> $(date +%H:%M:%S) E23 BUILD COMPLETE model=$MODEL have=$(have)/$TARGET"
