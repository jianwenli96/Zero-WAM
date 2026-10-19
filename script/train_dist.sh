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
# Cluster images may load an older system libstdc++ before torch_npu's
# sqlite3 -> ICU dependency. Pin only this runtime, not the entire env/lib
# directory (which could override Ascend's other shared libraries).
python_executable=$(command -v -- "${PYTHON_BIN}")
python_env_root=$(dirname "$(dirname "$(readlink -f "${python_executable}")")")
python_cxx_runtime="${python_env_root}/lib/libstdc++.so.6"
if [[ -f "${python_cxx_runtime}" ]]; then
    export LD_PRELOAD="${python_cxx_runtime}${LD_PRELOAD:+:${LD_PRELOAD}}"
    echo "Using Python C++ runtime: ${python_cxx_runtime}" >&2
fi
CONFIG_NAME=${CONFIG_NAME:-piperpika_train}
if [[ "${CONFIG_NAME}" == "piperpika_train" ]]; then
    DEFAULT_MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
    # 20% PiperPika, 80% real HumanGen. Robotwin can be reintroduced explicitly.
    DATASETS=${DATASETS:-piperpika:5,agibot:4,robocoin:4,robomind:4,interna1:4,oxe:4}
elif [[ "${CONFIG_NAME}" == "aloha_train" ]]; then
    # 30% ALOHA; remaining 70% distributed equally across HumanGen collections.
    DEFAULT_MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
    DATASETS=${DATASETS:-aloha:18,robotwin:7,agibot:7,robocoin:7,robomind:7,interna1:7,oxe:7}
else
    DEFAULT_MODEL_PATH=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-scratch
    DATASETS=${DATASETS:-robotwin:1,agibot:1,robocoin:1,robomind:1,interna1:1,oxe:1}
fi
export PIPERPIKA_SOURCE_ROOT=${PIPERPIKA_SOURCE_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16}
export PIPERPIKA_PREPARED_ROOT=${PIPERPIKA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2}
export ALOHA_PREPARED_ROOT=${ALOHA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3}

export MODEL_PATH="${MODEL_PATH:-${DEFAULT_MODEL_PATH}}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/${CONFIG_NAME}_dist}"
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

# Fail early instead of starting an accidental single-node training run.
for value in NNODES NGPU; do
    if ! [[ "${!value}" =~ ^[1-9][0-9]*$ ]]; then
        echo "$value must be set (or supplied by cluster MA_* variables)" >&2; exit 1
    fi
done
if ! [[ "$NODE_RANK" =~ ^[0-9]+$ ]] || (( NODE_RANK >= NNODES )) || [[ -z "$MASTER_ADDR" ]]; then
    echo 'Set NODE_RANK and MASTER_ADDR, or cluster VC_* variables' >&2; exit 1
fi
if [[ ",${DATASETS}," == *,piperpika:* ]]; then
    if [[ "${PREPARE_DATA:-1}" == "1" ]]; then
        # Global rank 0 writes metadata, all 64 ranks encode disjoint jobs, then validate.
        "${PYTHON_BIN}" -m torch.distributed.run \
            --nnodes="$NNODES" --node_rank="$NODE_RANK" --master_addr="$MASTER_ADDR" \
            --master_port="${PREPARE_MASTER_PORT:-$((MASTER_PORT + 1))}" --nproc_per_node="$NGPU" \
            --tee 3 script/prepare_piperpika.py distributed >&2
    else
        "${PYTHON_BIN}" script/prepare_piperpika.py check >&2
    fi
fi

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
