#!/bin/bash
# LANE A (GPU0) -- priority sequence, GPU0 half.
# Order: (0) wait for the ALREADY-RUNNING build_deepseek_llama3_squad_n1000.sh (pid 178474,
# started before this session's work -- deepseek squad, then Llama-3 squad, both on GPU0) to
# fully finish, untouched -> (1) Qwen3-8B nothink (trivia n1000, n2000, squad n1000) ->
# (2) drain the SHARED big-tier work-stealing queue (see below) -> (3) once the whole shared
# queue is empty, whichever lane gets there resumes gemma-2-27b-it n2000.
#
# REBALANCING (added after the first version left 2 of the 3 big-tier 27B models statically
# pinned to lane A and only 1 to lane B, making lane A's queue ~2x lane B's): the 6 big-tier
# jobs (Qwen3.5-27B/3.6-27B/3.8-27B x {n1000,n2000}) now live in a SHARED file
# ($JOBS_FILE), and both lanes race to claim each line via `mkdir` (atomic on POSIX -- exactly
# one caller wins). A lane that loses a claim just moves to the next line instantly, so
# whichever lane finishes its small-tier work first naturally picks up more of the shared
# queue -- no static per-lane assignment, no idle GPU while the other lane still has backlog.
#
# Fencing: DYNAMIC per-phase gpu_reserve.py hold, resized via refence() whenever the job size
# changes (small-tier ~SMALL_BUDGET vs big-tier ~BIG_BUDGET), not one static hold computed once
# at the start. blocks-execution fix (mn1025, 2026-08): the original single-hold version
# computed HOLD = free_at_start - BIG_BUDGET - SAFETY ONCE, before the small-tier phase even
# began; if free memory at that instant was BELOW BIG_BUDGET (as happened live: GPU0 had
# ~38.5GB free right after the squad job exited, under the 42GB big-tier budget), HOLD went
# negative and the `if HOLD > 512` check silently skipped fencing ENTIRELY for the whole lane,
# leaving 20GB+ of genuinely free memory completely unprotected the whole small-tier phase --
# caught live via a direct user question, fixed with an immediate stopgap fence + this rewrite.
set -uo pipefail
cd "$(dirname "$0")/.."
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
JOBS_FILE=/data2/mn1025/stage1_meta/nothink_bigtier_jobs.txt
CLAIMS_DIR=/data2/mn1025/stage1_meta/nothink_bigtier_claims
OUT=/data2/mn1025/stage1
SMALL_BUDGET=24000   # ~8-9B model in bf16 + activations (bumped from 20000: live OOM showed
                      # Qwen3.5-9B's real peak was 19.36GiB + a transient spike, 20000's margin
                      # was too thin -- our OWN fence starved our OWN job. See lane B's crash log.)
BIG_BUDGET=44000     # 27B single-GPU+CPU-offload budget (bumped from 42000, same reasoning)
SAFETY=800           # bumped from 400 -- PyTorch allocator fragmentation ate into the nominal margin
SQUAD_WRAPPER_PID=178474   # build_deepseek_llama3_squad_n1000.sh -- covers BOTH deepseek + llama-3
LANE_TAG="lane A"
mkdir -p amortized_ue/logs "$CLAIMS_DIR"

echo ">>> $(date +%H:%M:%S) [$LANE_TAG] waiting for deepseek+llama-3 squad wrapper (pid $SQUAD_WRAPPER_PID) to finish..."
while kill -0 "$SQUAD_WRAPPER_PID" 2>/dev/null; do sleep 2; done
echo ">>> $(date +%H:%M:%S) [$LANE_TAG] deepseek+llama-3 squad done. Fencing GPU0 for the rest of this lane."

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
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i 0)
  hold=$(( free - budget - SAFETY ))
  if [ "$hold" -gt 512 ]; then
    python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$hold" --parent_pid $$ &
    HOLDER_PID=$!
    echo ">>> $(date +%H:%M:%S) [$LANE_TAG] fencing GPU0: hold ${hold}MiB (free ${free}, budget ${budget}), holder pid $HOLDER_PID"
    sleep 3
  else
    echo "!!! $(date +%H:%M:%S) [$LANE_TAG] WARNING: free ${free}MiB is at/below budget ${budget}MiB -- no slack to fence, job may be memory-tight"
  fi
}

refence "$SMALL_BUDGET"

run(){ # run <run_name> <target_count> <extra stage1 args...>
  # blocks-execution (mn1025, 2026-08): the original version read rc via
  # "echo ...$(date...)...rc=$?" -- the $(date) command SUBSTITUTION runs first and
  # clobbers $? to date's own exit status (0) before the echo's rc=$? is evaluated, so
  # every failure silently logged as rc=0. This is exactly how the pre-fix chat-template
  # bug crash-looped through 5 jobs undetected. Capture rc IMMEDIATELY, and abort the
  # whole lane (not just log-and-continue) if the job didn't reach its target record
  # count -- do not let a second silent failure burn through the rest of the queue.
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

# --- this lane's own small-tier queue (not shared -- Qwen3-8B is only ever run here) -------
run Qwen3-8B_trivia_qa_n1000_nothink 1000 \
  --model_name Qwen3-8B --dataset trivia_qa --only_ids "$IDS_FILE" --selection_num_samples 3074 --num_samples 1000
run Qwen3-8B_trivia_qa_n2000_nothink 2000 \
  --model_name Qwen3-8B --dataset trivia_qa --num_samples 2000
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
