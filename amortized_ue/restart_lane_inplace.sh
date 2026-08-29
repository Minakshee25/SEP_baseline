#!/bin/bash
# restart_lane_inplace.sh <gpu> -- restart ONE training_lane.sh so it picks up an
# edited run_job(), without a co-tenant grabbing the card during the model reload.
#
# The replacement job is the SAME job (stage1 resumes from its record .pt files);
# a 27B build needs ~all of the card, so there is no room to fence the reload
# itself -- instead we hold the freed memory with gpu_reserve from the instant the
# old stage1 dies until the new lane's stage1 is actually loading, then drop it.
#
# Assumes the training_watchdog is ALREADY DEAD (kill it first, relaunch the
# adopt-watchdog after both GPUs are done). One GPU at a time.
#
#   bash amortized_ue/restart_lane_inplace.sh 1
#   bash amortized_ue/restart_lane_inplace.sh 0
set -uo pipefail
cd "$(dirname "$0")/.."
GPU="${1:?usage: restart_lane_inplace.sh <gpu>}"
PY=/data2/mn1025/conda_envs/se_probes_v5/bin/python
DRIVER="amortized_ue/logs/training_gpu${GPU}_driver.log"
GPU_UUID=$(nvidia-smi --query-gpu=index,gpu_uuid --format=csv,noheader | awk -F', ' -v g="$GPU" '$1==g{print $2}')
log(){ echo ">>> $(date '+%m-%d %H:%M:%S') [restart gpu$GPU] $*"; }

OLD_LANE=$(pgrep -f "training_lane.sh ${GPU}\$" || true)
[ -z "$OLD_LANE" ] && { log "ABORT: no 'training_lane.sh $GPU' process"; exit 1; }
# stage1 child on THIS gpu (match by the card it has memory on)
OLD_STAGE1=$(nvidia-smi --query-compute-apps=gpu_uuid,pid,used_memory --format=csv,noheader,nounits \
  | awk -F', ' -v u="$GPU_UUID" '$1==u && $3+0>5000 {print $2}')
OLD_RESERVE=$(pgrep -f "gpu_reserve .*parent_pid ${OLD_LANE}\$" || true)
log "old lane=$OLD_LANE  stage1=${OLD_STAGE1:-none}  reserve=${OLD_RESERVE:-none}  uuid=$GPU_UUID"
[ -z "$OLD_STAGE1" ] && { log "ABORT: could not identify the stage1 child on gpu$GPU"; exit 1; }

RUN_NAME=$(ps -o cmd= -p "$OLD_STAGE1" | grep -oE 'run_name [^ ]+' | awk '{print $2}')
RUN_DIR="/data2/mn1025/stage1/${RUN_NAME}"
before=$(ls "$RUN_DIR/records" 2>/dev/null | wc -l)
log "job=$RUN_NAME  records before=$before"

BRIDGE=""
NEW_LANE=""
cleanup(){
  [ -n "$BRIDGE" ] && kill "$BRIDGE" 2>/dev/null || true
  # if we were interrupted before launching the replacement, un-freeze the old
  # lane so it isn't left stopped (stage1 may already be dead -> it will just
  # retry the job, now with the fixed run_job once it is itself restarted).
  [ -z "$NEW_LANE" ] && kill -CONT "$OLD_LANE" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

# 1. freeze the old lane so it can't retry/re-fence/claim once its child dies
kill -STOP "$OLD_LANE"; log "SIGSTOP old lane $OLD_LANE"

# 2. kill the in-flight stage1 (resumable)
kill -TERM "$OLD_STAGE1" 2>/dev/null || true
for i in $(seq 1 60); do kill -0 "$OLD_STAGE1" 2>/dev/null || break; sleep 1; done
kill -KILL "$OLD_STAGE1" 2>/dev/null || true
sleep 4
FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")
log "stage1 killed; gpu$GPU free=${FREE}MiB (records still $(ls "$RUN_DIR/records" 2>/dev/null | wc -l))"

# 3. bridge-fence the freed slack IMMEDIATELY (hold all but ~1.5GB)
HOLD=$(( FREE - 1500 ))
if [ "$HOLD" -gt 512 ]; then
  CUDA_VISIBLE_DEVICES="$GPU" "$PY" -m amortized_ue.gpu_reserve --device 0 --hold_mib "$HOLD" --parent_pid $$ &
  BRIDGE=$!
  sleep 5
  log "bridge pid $BRIDGE holding ~${HOLD}MiB; gpu$GPU free now $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU")MiB"
else
  log "WARNING: only ${FREE}MiB free -- no bridge"
fi

# 4. kill the old lane + its stale reserve holder
kill -KILL "$OLD_LANE" 2>/dev/null || true
[ -n "$OLD_RESERVE" ] && kill "$OLD_RESERVE" 2>/dev/null || true
log "old lane $OLD_LANE killed"

# 5. launch the new lane
MARK=">>> RESTART $(date '+%s') <<<"
echo "$MARK" >> "$DRIVER"
nohup bash amortized_ue/training_lane.sh "$GPU" >> "$DRIVER" 2>&1 &
NEW_LANE=$!
log "new lane pid $NEW_LANE -- waiting for it to reach stage1 launch..."

# 6. wait until the new lane is past fence() and about to run stage1, then drop bridge
ok=0
for i in $(seq 1 120); do
  if awk -v m="$MARK" 'f && /starting .*have .*stalls/ {print; exit} $0==m{f=1}' "$DRIVER" | grep -q .; then
    ok=1; break
  fi
  kill -0 "$NEW_LANE" 2>/dev/null || { log "ABORT: new lane died early"; exit 1; }
  sleep 2
done
[ "$ok" = 1 ] && log "new lane reached run_job:" && tail -n 3 "$DRIVER"
sleep 3                       # let fence() log its 'unfenced' line and exec python
kill "$BRIDGE" 2>/dev/null || true; BRIDGE=""
log "bridge dropped; new stage1 can now allocate"

# 7. confirm the reload + resume
for i in $(seq 1 90); do
  now=$(ls "$RUN_DIR/records" 2>/dev/null | wc -l)
  used=$(nvidia-smi --query-compute-apps=gpu_uuid,used_memory --format=csv,noheader,nounits | awk -F', ' -v u="$GPU_UUID" '$1==u{s+=$2}END{print s+0}')
  if [ "$used" -gt 15000 ]; then log "stage1 back on gpu$GPU (${used}MiB used, records ${now}/$(grep "^${RUN_NAME}|" /data2/mn1025/stage1_meta/training_n2000_jobs.txt | cut -d'|' -f2)); OK"; exit 0; fi
  sleep 4
done
log "WARNING: stage1 not clearly back after ~6min -- check $DRIVER"
exit 0
