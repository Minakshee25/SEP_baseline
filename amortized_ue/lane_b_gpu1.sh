#!/bin/bash
# LANE B (GPU1) -- priority sequence, GPU1 half. Runs in parallel with lane_a_gpu0.sh.
# Order: (0) the gemma-2-27b-it n2000 job already running on GPU1 (unrelated to this task --
# Gemma has no thinking mode) is paused (SIGTERM, resumable) by the caller right before this
# script starts -> (1) Qwen3.5-9B nothink (trivia n1000, n2000, squad n1000) -> (2) drain the
# SHARED big-tier work-stealing queue (see lane_a_gpu0.sh for the full rationale) -> (3) once
# the whole shared queue is empty, whichever lane gets there resumes gemma-2-27b-it n2000.
#
# REBALANCING: the 6 big-tier jobs (Qwen3.5-27B/3.6-27B/3.8-27B x {n1000,n2000}) live in a
# SHARED file ($JOBS_FILE); both lanes race to claim each line via atomic `mkdir`. Whichever
# lane finishes its small-tier work first naturally picks up more of the shared queue.
#
# Fencing: DYNAMIC per-phase gpu_reserve.py hold, resized via refence() whenever the job size
# changes (small-tier ~SMALL_BUDGET vs big-tier ~BIG_BUDGET). blocks-execution fix (mn1025,
# 2026-08): the original version (a) waited for the FULL BIG_BUDGET to be free before starting
# even the small 9B job (unnecessary delay) and (b) computed one static hold at the start --
# see lane_a_gpu0.sh for the live bug this caused there (free < BIG_BUDGET at fence time =
# silently zero protection for the whole small-tier phase). Fixed identically here.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
JOBS_FILE=/data2/mn1025/stage1_meta/nothink_bigtier_jobs.txt
CLAIMS_DIR=/data2/mn1025/stage1_meta/nothink_bigtier_claims
OUT=/data2/mn1025/stage1
SMALL_BUDGET=24000   # ~8-9B model in bf16 + activations (bumped from 20000: live OOM showed
                      # Qwen3.5-9B's real peak was 19.36GiB + a transient spike, 20000's margin
                      # was too thin -- our OWN fence starved our OWN job.)
BIG_BUDGET=44000     # 27B single-GPU+CPU-offload budget (bumped from 42000, same reasoning)
SAFETY=800           # bumped from 400 -- PyTorch allocator fragmentation ate into the nominal margin
LANE_TAG="lane B"
mkdir -p amortized_ue/logs "$CLAIMS_DIR"

echo ">>> $(date +%H:%M:%S) [$LANE_TAG] waiting for GPU1 memory to clear after gemma pause..."
while true; do
  FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  [ "$FREE" -ge "$SMALL_BUDGET" ] && break
  sleep 2
done
echo ">>> $(date +%H:%M:%S) [$LANE_TAG] GPU1 clear (${FREE}MiB free). Fencing for the rest of this lane."

HOLDER_PID=""
cleanup(){ [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

refence(){ # refence <budget_mib> -- release old hold (if any), re-fence sized to current free memory
  local budget="$1"
  if [ -n "$HOLDER_PID" ]; then
    kill "$HOLDER_PID" 2>/dev/null || true
    wait "$HOLDER_PID" 2>/dev/null || true
    HOLDER_PID=""
  fi
  local free hold
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 1)
  hold=$(( free - budget - SAFETY ))
  if [ "$hold" -gt 512 ]; then
    python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$hold" --parent_pid $$ &
    HOLDER_PID=$!
    echo ">>> $(date +%H:%M:%S) [$LANE_TAG] fencing GPU1: hold ${hold}MiB (free ${free}, budget ${budget}), holder pid $HOLDER_PID"
    sleep 3
  else
    echo "!!! $(date +%H:%M:%S) [$LANE_TAG] WARNING: free ${free}MiB is at/below budget ${budget}MiB -- no slack to fence, job may be memory-tight"
  fi
}

refence "$SMALL_BUDGET"

run(){ # run <run_name> <target_count> <extra stage1 args...>
  # blocks-execution (mn1025, 2026-08): see the matching comment in lane_a_gpu0.sh -- the
  # original "$(date...)...rc=$?" pattern clobbered $? with date's own exit status, which
  # is exactly how the pre-fix chat-template crash burned through 5 jobs undetected,
  # logged as rc=0 the whole time. Capture rc immediately and abort the lane (rather than
  # log-and-continue) if the job didn't reach its target record count.
  local run_name="$1" target="$2"; shift 2
  echo ">>> $(date +%H:%M:%S) [$LANE_TAG] starting $run_name"
  python -m amortized_ue.stage1 "$@" --output_dir "$OUT" --run_name "$run_name" \
    > "amortized_ue/logs/${run_name}.log" 2>&1
  local rc=$?
  local have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
  echo ">>> $(date +%H:%M:%S) [$LANE_TAG] $run_name exited rc=$rc, records=${have}/${target}"
  if [ "$rc" -ne 0 ] || [ "$have" -lt "$target" ]; then
    echo "!!! $(date +%H:%M:%S) [$LANE_TAG] ABORTING LANE: $run_name did not complete (rc=$rc, ${have}/${target}). See amortized_ue/logs/${run_name}.log"
    exit 1
  fi
}

# --- this lane's own small-tier queue (not shared -- Qwen3.5-9B is only ever run here) -----
run Qwen3.5-9B_trivia_qa_n1000_nothink 1000 \
  --model_name Qwen3.5-9B --dataset trivia_qa --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples 1000
run Qwen3.5-9B_trivia_qa_n2000_nothink 2000 \
  --model_name Qwen3.5-9B --dataset trivia_qa --num_samples 2000
run Qwen3.5-9B_squad_n1000_nothink 1000 \
  --model_name Qwen3.5-9B --dataset squad --num_samples 1000

# --- borrowed from lane A (paused for the team's GPU0 need until 1pm tomorrow) -------------
# blocks-execution (mn1025, 2026-08): user wants ALL small-tier data (both Qwen3-8B and
# Qwen3.5-9B) done early rather than letting Qwen3-8B's squad job sit idle on a paused GPU0
# for 15+ hours. Picks up exactly where lane A left off (418/1000, resumable). Lane A's own
# script still lists this same job when it resumes tomorrow -- it'll just skip-scan through
# already-completed records (fast, no re-generation) since stage1.py skips existing records.
run Qwen3-8B_squad_n1000_nothink 1000 \
  --model_name Qwen3-8B --dataset squad --num_samples 1000

# --- shared big-tier work-stealing queue ----------------------------------------------------
refence "$BIG_BUDGET"
echo ">>> $(date +%H:%M:%S) [$LANE_TAG] small-tier done, draining shared big-tier queue: $JOBS_FILE"
while IFS='|' read -r run_name target model_name dataset only_ids num_samples; do
  [ -z "$run_name" ] && continue
  claim="$CLAIMS_DIR/${run_name}.claimed"
  mkdir "$claim" 2>/dev/null || { echo ">>> $(date +%H:%M:%S) [$LANE_TAG] $run_name already claimed, skipping"; continue; }
  have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
  if [ "$have" -ge "$target" ]; then
    echo ">>> $(date +%H:%M:%S) [$LANE_TAG] $run_name already complete (${have}/${target}), skipping"
    continue
  fi
  if [ "$only_ids" = "yes" ]; then
    run "$run_name" "$target" --model_name "$model_name" --dataset "$dataset" \
      --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples "$num_samples"
  else
    run "$run_name" "$target" --model_name "$model_name" --dataset "$dataset" --num_samples "$num_samples"
  fi
done < "$JOBS_FILE"
echo ">>> $(date +%H:%M:%S) [$LANE_TAG] this lane's contribution to the shared queue is done."

# --- wait for the OTHER lane's contribution too, then exactly one lane resumes gemma --------
echo ">>> $(date +%H:%M:%S) [$LANE_TAG] waiting for the full shared queue (both lanes) to finish..."
while IFS='|' read -r run_name target model_name dataset only_ids num_samples; do
  [ -z "$run_name" ] && continue
  while [ "$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)" -lt "$target" ]; do sleep 15; done
done < "$JOBS_FILE"
echo "=== $(date '+%Y-%m-%d %H:%M:%S') [$LANE_TAG] FULL SHARED QUEUE COMPLETE ==="

cleanup
trap - EXIT INT TERM
if mkdir "$CLAIMS_DIR/GEMMA_RESUME.claimed" 2>/dev/null; then
  echo ">>> $(date +%H:%M:%S) [$LANE_TAG] resuming gemma-2-27b-it n2000 (paused at 969/2000)"
  python -m amortized_ue.stage1 --model_name gemma-2-27b-it --dataset trivia_qa --num_samples 2000 \
    --output_dir "$OUT" > "amortized_ue/logs/gemma-2-27b-it_trivia_qa_n2000_resume.log" 2>&1
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') [$LANE_TAG] gemma-2-27b-it n2000 resume exited rc=$? ==="
else
  echo ">>> $(date +%H:%M:%S) [$LANE_TAG] other lane is handling the gemma resume. Done."
fi
