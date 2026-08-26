#!/bin/bash
# Regenerate Stage-1 data for all 5 Qwen targets (the only ones with a real thinking mode --
# no Gemma target has one, checked offline against all 4 Gemma tokenizers) with thinking
# HARD-DISABLED via huggingface_models.py's new _DISABLE_THINKING_MODELS path
# (apply_chat_template(..., enable_thinking=False), see that file for the full rationale).
#
# Motivation: Qwen3.8-27B stalled at 65/1000 records over >40h burning generation time on
# <think> content even under the SEP baseline's raw completion prompting (which never reaches
# Qwen's official enable_thinking switch); Qwen3.5-9B's manifest history separately shows a
# ~5-6% "never finishes thinking" tail. Disabling thinking should fix both speed and the
# residual contamination in one change.
#
# Writes to NEW run_names (suffix _nothink) -- the existing *_full dirs (used by E44-E54) are
# left completely untouched, so every existing result stays reproducible against what's on
# disk. This is a deliberate, non-destructive choice (see amortized_ue/CLAUDE.md and this
# session's discussion) -- do not repoint existing analysis scripts at the new dirs without
# checking with the user first.
#
# 12 builds total:
#   small tier (Qwen3-8B, Qwen3.5-9B):   trivia n1000 (shared ids) + trivia n2000 + squad n1000
#   big tier   (Qwen3.5-27B/3.6-27B/3.8-27B): trivia n1000 (shared ids) + trivia n2000
# (big tier never had a squad build before this, so none is added here -- matches prior scope.)
#
# GPU: waits for a free slot (>= NEED_MIB) on either GPU, fences the leftover with
# gpu_reserve.py (same pattern as build_n2000_waiter.sh) so a co-tenant can't OOM us mid-run,
# then runs ONE model at a time (sequential, not parallel -- Qwen3.5+ hybrid arch reproducibly
# crashes under dual-GPU device_map sharding, see amortized_ue/CLAUDE.md). Resumable: stage1.py
# skips any record whose file already exists (no --overwrite passed), so killing/restarting
# this script just continues each build from wherever it left off.
#
# Usage: bash amortized_ue/build_qwen_nothink_regen.sh [NEED_MIB_SMALL] [NEED_MIB_BIG]

set -uo pipefail
cd "$(dirname "$0")/.."                                   # repo root
source /data2/mn1025/conda_envs/se_probes_v5/bin/activate
export HF_HOME=/data2/mn1025/hf_cache OPENAI_API_KEY=placeholder
export HF_XET_HIGH_PERFORMANCE=1
export PYTHONUNBUFFERED=1

IDS_FILE=/data2/mn1025/stage1_meta/shared_n1000_ids.txt
OUT=/data2/mn1025/stage1
mkdir -p amortized_ue/logs

NEED_SMALL="${1:-20000}"   # ~8-9B bf16 + activations
NEED_BIG="${2:-42000}"     # 27B single-GPU+CPU-offload budget (matches build_big_tier_n1000.sh's live footprint)
POLL=10
SAFETY=400

have(){ ls "/data2/mn1025/stage1/$1/records" 2>/dev/null | wc -l; }

HOLDER_PID=""
cleanup(){ [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

wait_for_gpu_and_run(){
  # $1 = NEED_MIB, $2 = target record count, $3 = run_name (for progress log), rest = stage1 args
  local need="$1" target="$2" run_name="$3"; shift 3
  local n=$(have "$run_name")
  while [ "$n" -lt "$target" ]; do
    PICK=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -v n="$need" '$2>=n{print $1;exit}')
    if [ -z "${PICK:-}" ]; then
      echo ">>> $(date +%H:%M:%S) [$run_name] no GPU with >=${need}MiB free yet (have $n/$target); waiting..."
      sleep "$POLL"; continue
    fi
    FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$PICK")
    HOLD=$(( FREE - need - SAFETY ))
    (
      if [ "$HOLD" -gt 512 ]; then
        CUDA_VISIBLE_DEVICES="$PICK" python -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
        HOLDER_PID=$!
        echo ">>> $(date +%H:%M:%S) [$run_name] fencing GPU $PICK: hold ${HOLD}MiB (free ${FREE}, budget ${need}), holder pid $HOLDER_PID"
        sleep 4
      fi
      echo ">>> $(date +%H:%M:%S) [$run_name] launching on GPU $PICK (have=$n/$target)"
      CUDA_VISIBLE_DEVICES="$PICK" python -m amortized_ue.stage1 "$@" \
        > "amortized_ue/logs/${run_name}.log" 2>&1
      rc=$?
      [ -n "$HOLDER_PID" ] && kill "$HOLDER_PID" 2>/dev/null || true
      echo ">>> $(date +%H:%M:%S) [$run_name] attempt exited rc=$rc"
    )
    HOLDER_PID=""
    n=$(have "$run_name")
    [ "$n" -lt "$target" ] && sleep "$POLL"
  done
  echo "=== $(date '+%Y-%m-%d %H:%M:%S') [$run_name] COMPLETE $n/$target ==="
}

# --- small tier: Qwen3-8B, Qwen3.5-9B ---------------------------------------------------
for m in Qwen3-8B Qwen3.5-9B; do
  run="${m}_trivia_qa_n1000_nothink"
  wait_for_gpu_and_run "$NEED_SMALL" 1000 "$run" \
    --model_name "$m" --dataset trivia_qa --only_ids "$IDS_FILE" \
    --selection_num_samples 3074 --num_samples 1000 --output_dir "$OUT" --run_name "$run"

  run="${m}_trivia_qa_n2000_nothink"
  wait_for_gpu_and_run "$NEED_SMALL" 2000 "$run" \
    --model_name "$m" --dataset trivia_qa --num_samples 2000 --output_dir "$OUT" --run_name "$run"

  run="${m}_squad_n1000_nothink"
  wait_for_gpu_and_run "$NEED_SMALL" 1000 "$run" \
    --model_name "$m" --dataset squad --num_samples 1000 --output_dir "$OUT" --run_name "$run"
done

# --- big tier: Qwen3.5-27B, Qwen3.6-27B, Qwen3.8-27B ------------------------------------
for m in Qwen3.5-27B Qwen3.6-27B Qwen3.8-27B; do
  run="${m}_trivia_qa_n1000_nothink"
  wait_for_gpu_and_run "$NEED_BIG" 1000 "$run" \
    --model_name "$m" --dataset trivia_qa --only_ids "$IDS_FILE" \
    --selection_num_samples 3074 --num_samples 1000 --output_dir "$OUT" --run_name "$run"

  run="${m}_trivia_qa_n2000_nothink"
  wait_for_gpu_and_run "$NEED_BIG" 2000 "$run" \
    --model_name "$m" --dataset trivia_qa --num_samples 2000 --output_dir "$OUT" --run_name "$run"
done

echo "=== $(date '+%Y-%m-%d %H:%M:%S') Qwen no-think regen queue FULLY COMPLETE ==="
