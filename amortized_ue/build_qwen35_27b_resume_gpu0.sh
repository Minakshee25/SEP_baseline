#!/bin/bash
# Resume the Qwen3.5-27B n2000 build that OOM'd on GPU1 at 1072/2000 records (2026-08-24 03:34).
# Build is resumable (overwrite=False skips existing .pt records), so re-issuing the identical
# command continues from 1072 rather than restarting. Queued on GPU0 (not GPU1, where it failed)
# because GPU0 has nothing queued after gemma-3-27b-it's n2000 build finishes, whereas GPU1 already
# has gemma-2-27b-it queued behind Qwen3.6-27B -- this avoids delaying that queue and avoids
# relaunching straight into the same tight-memory GPU1 that caused the OOM.
# Waits for >= NEED_MIB free on GPU0 (i.e. for gemma-3-27b-it to finish) before starting.

set -uo pipefail
cd "$(dirname "$0")/.."                                   # repo root
NEED_MIB=${NEED_MIB:-40000}
POLL=${POLL:-60}
GPU=0

source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONUNBUFFERED=1

mkdir -p amortized_ue/logs

echo ">>> $(date '+%Y-%m-%d %H:%M:%S') waiting for GPU$GPU >= ${NEED_MIB}MiB free (resuming Qwen3.5-27B from 1072/2000)"
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i $GPU)
  [ "$FREE" -ge "$NEED_MIB" ] && break
  sleep $POLL
done
echo ">>> $(date '+%Y-%m-%d %H:%M:%S') GPU$GPU has ${FREE}MiB free -- starting resume"

python -m amortized_ue.stage1 \
  --model_name Qwen3.5-27B --dataset trivia_qa \
  --num_samples 2000 \
  --output_dir /data2/mn1025/stage1 \
  > amortized_ue/logs/Qwen3.5-27B_trivia_qa_n2000_resume.log 2>&1
status=$?
if [ $status -eq 0 ]; then
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S') Qwen3.5-27B n2000 RESUME DONE (exit 0)"
else
  echo ">>> $(date '+%Y-%m-%d %H:%M:%S') Qwen3.5-27B n2000 RESUME FAILED (exit $status) -- see amortized_ue/logs/Qwen3.5-27B_trivia_qa_n2000_resume.log"
fi
