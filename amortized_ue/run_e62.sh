#!/bin/bash
# E62 — direct run (user supplies a free GPU via CUDA_VISIBLE_DEVICES).
#   CUDA_VISIBLE_DEVICES=<id> bash amortized_ue/run_e62.sh
set -eu
cd /vol/bitbucket/mn1025/individual_project/semantic-entropy-probes
PY=/vol/bitbucket/mn1025/conda_envs/amortized_stage2/bin/python
: "${CUDA_VISIBLE_DEVICES:?set CUDA_VISIBLE_DEVICES to a free GPU id first}"
echo "[$(date)] E62 on GPU $CUDA_VISIBLE_DEVICES"
$PY -m amortized_ue.e62_qresp_alone_vs_sep --data_dir /data2/mn1025/stage1 2>&1 | tee amortized_ue/e62.log
echo "[$(date)] E62 done"
