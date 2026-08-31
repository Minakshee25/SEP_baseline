#!/bin/bash
# e65_eval_n1000.sh -- THE CORRECT E65 eval. Waits for all 5 big-tier n1000 shared-ID
# trivia sets (Qwen3.8-27B is the last one generating), then borrows GPU 1 from its
# training lane, runs `e65_bigtier_lolo --stage eval --eval_n 1000` (checkpoints already
# exist -- NO retraining), and hands the card back so data generation resumes.
#
# Same gap-free borrow as e65_run.sh: a slack-holder retry-grabs the freed memory the
# instant the lane's resumable stage1 child is killed, an exit-bridge holds it from the
# moment eval finishes until the lane's 27B build reclaims it. GPU 0's lane is untouched.
#
#   nohup bash amortized_ue/e65_eval_n1000.sh > amortized_ue/logs/e65_eval_n1000.out 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
GPU=1
CARD_MIB=46068
E65_FREE_MIB=20000       # the eval proxy is a 3B inference pass -- far smaller than training; 20G is ample headroom
LANEJOB_MIB=44000        # leave this free for the lane's 27B reload on the way out
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
say(){ echo ">>> $(date '+%F %T') $*"; }

LANE="" SLACK=""
handback(){
  kill "${SLACK:-}" 2>/dev/null || true
  [ -z "${LANE:-}" ] && return
  LANEJOB_MIB=$LANEJOB_MIB nohup bash amortized_ue/e65_bridge.sh >> amortized_ue/logs/e65_bridge.out 2>&1 & disown
  say "exit-bridge armed (pid $!) -- GPU$GPU stays fenced until the lane reclaims it"
  kill -CONT "$LANE" 2>/dev/null && say "SIGCONT lane $LANE (re-fences + resumes its job)"
}
trap handback EXIT

# 1. wait until all 5 n1000 shared-ID eval sets are on disk
until $PY -m amortized_ue.e65_bigtier_lolo --stage check_eval --eval_n 1000 --data_dir /data2/mn1025/stage1; do
  say "n1000 eval sets not ready -- sleep 10m"; sleep 600
done
say "all 5 big-tier n1000 shared-ID eval sets ready"

# 2. borrow GPU 1 without a gap
LANE=$(pgrep -f "training_lane.sh $GPU" | head -1 || true)
if [ -n "$LANE" ]; then
  say "SIGSTOP lane $LANE"; kill -STOP "$LANE"
  DEADLINE_S=14400 nohup bash amortized_ue/e63_lane_safety_net.sh "$LANE" $$ "" \
    >> amortized_ue/logs/e65_lane_safety_net.out 2>&1 & disown
  say "lane safety-net armed (pid $!)"
fi
CUDA_VISIBLE_DEVICES=$GPU $PY -m amortized_ue.gpu_reserve --device 0 \
  --hold_mib $(( CARD_MIB - E65_FREE_MIB )) --retry_secs 90 --parent_pid $$ & SLACK=$!
sleep 1
for p in $(pgrep -f amortized_ue.stage1); do
  grep -qxz "CUDA_VISIBLE_DEVICES=$GPU" "/proc/$p/environ" 2>/dev/null && { say "kill -9 stage1 child $p (resumable)"; kill -9 "$p"; }
done
sleep 6
say "GPU$GPU now: $(nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader -i $GPU)"

# 3. eval only -- 5 folds x 3 seeds, reuses the 15 existing checkpoints, ~30-45 min
CUDA_VISIBLE_DEVICES=$GPU timeout 7200 \
  $PY -m amortized_ue.e65_bigtier_lolo --stage eval --eval_n 1000 --data_dir /data2/mn1025/stage1
RC=$?
say "e65 --stage eval --eval_n 1000 exited rc=$RC"
say "results -> amortized_ue/results/e65_bigtier_lolo_n1000.json"
say "done (rc=$RC)"       # handback() (EXIT trap) drops the slack-holder, arms the bridge, SIGCONTs the lane
exit $RC
