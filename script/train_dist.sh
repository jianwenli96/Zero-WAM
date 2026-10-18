#!/usr/bin/env bash

set -euo pipefail
umask 007

zero_wam_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN=${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}
# Run once on every node; unset NGPU uses the platform allocation or visible NPUs.
source "${zero_wam_repo_root}/script/distributed_env.sh"

CONFIG_NAME=${CONFIG_NAME:-aloha_train}
if [[ "${CONFIG_NAME}" == "aloha_train" ]]; then
    # 30% ALOHA; remaining 70% distributed equally across HumanGen collections.
    DEFAULT_MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
    DATASETS=${DATASETS:-aloha:18,robotwin:7,agibot:7,robocoin:7,robomind:7,interna1:7,oxe:7}
else
    DEFAULT_MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-scratch
    DATASETS=${DATASETS:-robotwin:1,agibot:1,robocoin:1,robomind:1,interna1:1,oxe:1}
fi
export ALOHA_PREPARED_ROOT=${ALOHA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3}

export MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-${zero_wam_repo_root}/outputs/aloha_humangen_train_dist}"
export HUMAN_GEN_ROOT="${HUMAN_GEN_ROOT:-/mnt/sfs_turbo/public/datasets/HumanGen}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Ascend 集群通信设置，与参考集群脚本保持一致，允许外部覆盖。
export HCCL_DEBUG="${HCCL_DEBUG:-INFO}"
export HCCL_EXEC_TIMEOUT="${HCCL_EXEC_TIMEOUT:-7200}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-7200}"
export HCCL_ASYNC_ERROR_HANDLING="${HCCL_ASYNC_ERROR_HANDLING:-0}"
export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"

cd "${zero_wam_repo_root}"
export HF_HOME="${HF_HOME:-/mnt/sfs_turbo/public/caches/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

if [[ ",${DATASETS}," == *,aloha:* ]]; then
    "${PYTHON_BIN}" script/check_aloha.py --root "${ALOHA_PREPARED_ROOT}" >&2
fi

exec "${PYTHON_BIN}" -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --master_port="${MASTER_PORT}" \
    --nproc_per_node="${NGPU}" \
    --tee 3 \
    -m wan_va.train --config-name "${CONFIG_NAME}" --datasets "${DATASETS}" "$@"
