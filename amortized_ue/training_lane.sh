#!/bin/bash
# training_lane.sh <gpu_idx>
# -----------------------------------------------------------------------------
# Reprioritization 2026-08-27 (user request): "first generate training dataset
# of all models" -- every target LLM must have a COMPLETE trivia_qa n2000 build
# (the proxy-TRAINING dataset) before any more n1000 eval builds run.
#
# Supersedes lane_a_gpu0.sh / lane_b_gpu1.sh for the duration of this queue.
# The old shared-queue lanes had drifted: Lane B's `while read` fd desynced when
# nothink_bigtier_jobs.txt was edited in place mid-run, so it silently skipped
# both 27B n2000 lines and serialized the remainder onto one GPU. This is a
# clean restart with a training-only job list.
#
# One instance per physical GPU; both instances race to claim each job line via
# atomic `mkdir` on $CLAIMS_DIR. Records are the source of truth -- stage1.py
# skips existing .pt files, so every job here RESUMES (Qwen3.5-27B ~880/2000,
# gemma-2 969/2000, gemma-3 1387/2000, the two fresh Qwen 0/2000).
#
# Env: se_probes_v5 venv (/data2, off NFS). GPU fenced with gpu_reserve.py sized
# to current free memory so a co-tenant can't OOM the build mid-flight (this is
# exactly what killed gemma-3's n2000 build on Aug 25).
# -----------------------------------------------------------------------------
set -uo pipefail
cd "$(dirname "$0")/.."
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

GPU="${1:?usage: training_lane.sh <gpu_idx>}"
export CUDA_VISIBLE_DEVICES="$GPU"
LANE_TAG="gpu$GPU"

JOBS_FILE=/data2/mn1025/stage1_meta/training_n2000_jobs.txt
CLAIMS_DIR=/data2/mn1025/stage1_meta/training_n2000_claims
OUT=/data2/mn1025/stage1
IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
BIG_BUDGET=44000     # 27B single-GPU + CPU offload working set
SAFETY=800
MAX_ATTEMPTS=3       # per lane, per job; then release for the other lane / give up

mkdir -p amortized_ue/logs "$CLAIMS_DIR"
log(){ echo ">>> $(date '+%m-%d %H:%M:%S') [$LANE_TAG] $*"; }

HOLDER_PID=""
cleanup(){ [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

fence(){
  if [ -n "$HOLDER_PID" ]; then kill "$HOLDER_PID" 2>/dev/null || true; wait "$HOLDER_PID" 2>/dev/null || true; HOLDER_PID=""; fi
  local free hold
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
  hold=$(( free - BIG_BUDGET - SAFETY ))
  if [ "$hold" -gt 512 ]; then
    python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$hold" --parent_pid $$ &
    HOLDER_PID=$!
    log "fenced GPU$GPU: hold ${hold}MiB (free ${free}MiB, budget ${BIG_BUDGET})"
    sleep 3
  else
    log "WARNING: free ${free}MiB <= budget ${BIG_BUDGET}MiB -- running unfenced"
  fi
}

# Release any stale claim left by a previous incarnation of THIS lane (crash /
# watchdog restart). Only this GPU's own claims, and only if still incomplete.
for c in "$CLAIMS_DIR"/*.claimed; do
  [ -d "$c" ] || continue
  [ -f "$c/owner" ] || continue
  grep -q "^gpu$GPU " "$c/owner" || continue
  rn=$(basename "$c" .claimed)
  IFS='|' read -r _ tgt _ _ _ _ < <(grep "^${rn}|" "$JOBS_FILE")
  have=$(ls "$OUT/$rn/records" 2>/dev/null | wc -l)
  if [ "${have:-0}" -lt "${tgt:-999999}" ]; then
    rmdir "$c" 2>/dev/null && log "released stale own claim: $rn ($have/$tgt)" || rm -rf "$c"
  fi
done

run_job(){
  local run_name="$1" target="$2" model_name="$3" dataset="$4" only_ids="$5" num_samples="$6"
  local jlog="amortized_ue/logs/${run_name}.log"
  local extra=(); [ "$only_ids" = "yes" ] && extra=(--only_ids "$IDS_FILE" --selection_num_samples 3074)
  local attempt=1 have
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
    if [ "${have:-0}" -ge "$target" ]; then log "$run_name complete ($have/$target)"; return 0; fi
    log "starting $run_name (attempt $attempt/$MAX_ATTEMPTS, have ${have:-0}/$target)"
    fence
    python -m amortized_ue.stage1 --model_name "$model_name" --dataset "$dataset" \
      --num_samples "$num_samples" "${extra[@]}" \
      --output_dir "$OUT" --run_name "$run_name" >> "$jlog" 2>&1
    local rc=$?
    have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
    log "$run_name attempt $attempt exited rc=$rc, records=${have:-0}/$target"
    if [ "${have:-0}" -ge "$target" ]; then return 0; fi
    attempt=$((attempt+1)); sleep 15
  done
  log "!!! $run_name INCOMPLETE after $MAX_ATTEMPTS attempts (${have:-0}/$target)"
  return 1
}

log "draining training queue: $JOBS_FILE"
progress=1
while [ "$progress" = 1 ]; do
  progress=0
  while IFS='|' read -r run_name target model_name dataset only_ids num_samples; do
    [ -z "${run_name:-}" ] && continue
    have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
    [ "${have:-0}" -ge "$target" ] && continue
    [ -e "$CLAIMS_DIR/${run_name}.failed.gpu$GPU" ] && continue   # this lane already gave up on it
    claim="$CLAIMS_DIR/${run_name}.claimed"
    mkdir "$claim" 2>/dev/null || continue
    echo "gpu$GPU pid $$ $(date '+%F %T')" > "$claim/owner"
    progress=1
    if run_job "$run_name" "$target" "$model_name" "$dataset" "$only_ids" "$num_samples"; then
      : # keep the claim -- job done
    else
      touch "$CLAIMS_DIR/${run_name}.failed.gpu$GPU"
      rmdir "$claim" 2>/dev/null || rm -rf "$claim"   # let the other lane try
      sleep 60
    fi
  done < "$JOBS_FILE"
done

cleanup; trap - EXIT INT TERM
log "=== this lane is done (no more claimable training jobs) ==="
# Status summary for the driver log.
while IFS='|' read -r run_name target _ _ _ _; do
  [ -z "${run_name:-}" ] && continue
  have=$(ls "$OUT/${run_name}/records" 2>/dev/null | wc -l)
  log "  ${have:-0}/${target}  ${run_name}"
done < "$JOBS_FILE"
