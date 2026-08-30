#!/bin/bash
# e65_run.sh -- wait for the 5 big-tier n2000 training sets, then train the E65
# LOLO proxy on GPU 1 borrowed from its training lane. GPU 1 is never left
# unfenced: a slack-holder retry-grabs the memory the instant the lane's child
# is killed (~0.2s), and an exit-bridge holds it from the moment E65 finishes
# until the lane's build reclaims the card. GPU 0's lane is untouched.
#
#   nohup bash amortized_ue/e65_run.sh > amortized_ue/logs/e65_run.out 2>&1 & disown
set -uo pipefail
cd "$(dirname "$0")/.."
PY=/data2/mn1025/conda_envs/amortized_stage2_v5/bin/python
PY_WANDB=/data2/mn1025/conda_envs/se_probes_v5/bin/python
GPU=1
CARD_MIB=46068
E65_FREE_MIB=31000       # leave this free on GPU1 for the E65 python
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

# 1. wait until all 5 datasets are on disk
until $PY -m amortized_ue.e65_bigtier_lolo --stage check --data_dir /data2/mn1025/stage1; do
  say "not ready -- sleep 10m"; sleep 600
done
say "all 5 big-tier n2000 datasets ready"

# 2. borrow GPU 1 without a gap
LANE=$(pgrep -f "training_lane.sh $GPU" | head -1 || true)
if [ -n "$LANE" ]; then
  say "SIGSTOP lane $LANE"; kill -STOP "$LANE"
  # belt-and-braces: force-resume the lane + sweep orphan fences if THIS script is
  # kill -9'd (no EXIT trap) or hangs. Does nothing if handback() resumes normally.
  DEADLINE_S=36000 nohup bash amortized_ue/e63_lane_safety_net.sh "$LANE" $$ "" \
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

# 3. train 5 folds + eval, then push checkpoints
CUDA_VISIBLE_DEVICES=$GPU timeout 28800 \
  $PY -m amortized_ue.e65_bigtier_lolo --stage all --data_dir /data2/mn1025/stage1
RC=$?
say "e65 --stage all exited rc=$RC"
[ $RC -eq 0 ] && { say "push to W&B"; $PY_WANDB -m amortized_ue.e65_bigtier_lolo --stage push_wandb || say "push failed (ckpts local)"; }
say "done (rc=$RC)"       # handback() (EXIT trap) drops the slack-holder, arms the bridge, SIGCONTs the lane
exit $RC
