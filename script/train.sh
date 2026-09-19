#!/usr/bin/bash

set -x

umask 007

NGPU=${NGPU:-"8"}
PYTHON_BIN=${PYTHON_BIN:-"/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python"}
MASTER_PORT=${MASTER_PORT:-"29501"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-"0"}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train"} # MCP is enabled for all training configs
DATASETS=${DATASETS:-"robotwin:1.0,agibot:1.0"}

# 将模型路径替换为实际位置
export MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-scratch}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/robotwin_train}"
export HUMAN_GEN_ROOT="${HUMAN_GEN_ROOT:-/mnt/sfs_turbo/public/datasets/HumanGen}"

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}
config_name=${CONFIG_NAME}
datasets=${DATASETS}

cd "$(dirname "${BASH_SOURCE[0]}")/.."
export HF_HOME="${HF_HOME:-/mnt/sfs_turbo/public/caches/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

## cmd setting
export ASCEND_LAUNCH_BLOCKING=1
export TOKENIZERS_PARALLELISM=false
PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nproc_per_node=${num_gpu} \
    --local-ranks-filter=${log_rank} \
    --master_port ${master_port} \
    --tee 3 \
    -m wan_va.train --config-name ${config_name} --datasets "${datasets}" "$@"
