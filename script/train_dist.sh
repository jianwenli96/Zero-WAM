#!/usr/bin/env bash

set -euo pipefail
umask 007

# 在集群的每个节点执行本脚本；也可显式设置以下标准分布式变量。
NNODES=${NNODES:-${MA_NUM_HOSTS:-}}
NODE_RANK=${NODE_RANK:-${VC_TASK_INDEX:-}}
worker_hosts=${VC_WORKER_HOSTS:-}
MASTER_ADDR=${MASTER_ADDR:-${worker_hosts%%,*}}
NGPU=${NGPU:-${MA_NUM_GPUS:-}}
MASTER_PORT=${MASTER_PORT:-29501}

PYTHON_BIN=${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}
CONFIG_NAME=${CONFIG_NAME:-robotwin_train}
DATASETS=${DATASETS:-robotwin:1.0,agibot:1.0,robocoin:1.0,robomind:1.0,interna1:1.0,oxe:1.0}

export MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-scratch}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/robotwin_train_dist}"
export HUMAN_GEN_ROOT="${HUMAN_GEN_ROOT:-/mnt/sfs_turbo/public/datasets/HumanGen}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ascend 集群通信设置，与参考集群脚本保持一致，允许外部覆盖。
export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-/mnt/sfs_turbo/public/caches/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NGPU}" \
    --tee 3 \
    -m wan_va.train --config-name "${CONFIG_NAME}" --datasets "${DATASETS}" "$@"
