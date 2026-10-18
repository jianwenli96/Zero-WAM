#!/usr/bin/env bash
# Source after setting PYTHON_BIN. Resolve once per node, before torchrun.

zero_wam_uint() {
    local value="${!1}"
    if [[ ! "$value" =~ ^[0-9]{1,9}$ ]]; then
        printf '%s must be an integer, got: %s\n' "$1" "$value" >&2
        return 1
    fi
    value=$((10#$value))
    if (( value < $2 )); then
        printf '%s must be >= %s, got: %s\n' "$1" "$2" "$value" >&2
        return 1
    fi
    printf -v "$1" '%s' "$value"
}

zero_wam_env_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
zero_wam_worker_hosts="${VC_WORKER_HOSTS:-}"
if [[ "$zero_wam_worker_hosts" =~ (^,|,,|,$) ]]; then
    printf 'VC_WORKER_HOSTS must contain nonempty comma-separated hosts\n' >&2
    return 1
fi
IFS=',' read -r -a zero_wam_host_list <<< "$zero_wam_worker_hosts"

NNODES="${NNODES:-${MA_NUM_HOSTS:-${VC_WORKER_NUM:-}}}"
if [[ -z "$NNODES" ]]; then
    NNODES="${#zero_wam_host_list[@]}"
    if [[ "$NNODES" == 0 ]]; then NNODES=1; fi
fi
zero_wam_uint NNODES 1

NODE_RANK="${NODE_RANK:-${VC_TASK_INDEX:-}}"
if [[ -z "$NODE_RANK" ]]; then
    if [[ "$NNODES" == 1 ]]; then
        NODE_RANK=0
    else
        printf 'Multi-node training requires NODE_RANK or VC_TASK_INDEX on each node\n' >&2
        return 1
    fi
fi
zero_wam_uint NODE_RANK 0
if (( NODE_RANK >= NNODES )); then
    printf 'NODE_RANK=%s must be smaller than NNODES=%s\n' "$NODE_RANK" "$NNODES" >&2
    return 1
fi

MASTER_ADDR="${MASTER_ADDR:-${zero_wam_worker_hosts%%,*}}"
if [[ -z "$MASTER_ADDR" ]]; then
    if [[ "$NNODES" == 1 ]]; then
        MASTER_ADDR=127.0.0.1
    else
        printf 'Multi-node training requires MASTER_ADDR or VC_WORKER_HOSTS\n' >&2
        return 1
    fi
fi
MASTER_PORT="${MASTER_PORT:-${MA_TORCHRUN_MASTER_PORT:-29501}}"
zero_wam_uint MASTER_PORT 1
if (( MASTER_PORT > 65535 )); then
    printf 'MASTER_PORT must be <= 65535\n' >&2
    return 1
fi

NGPU="${NGPU:-auto}"
if [[ "$NGPU" != auto ]]; then
    zero_wam_device_source=NGPU
elif [[ ! -v ASCEND_RT_VISIBLE_DEVICES && -n "${MA_NUM_GPUS:-}" && "${MA_NUM_GPUS}" != 0 ]]; then
    # ModelArts' allocation is authoritative unless visibility is restricted.
    NGPU="$MA_NUM_GPUS"
    zero_wam_device_source=MA_NUM_GPUS
else
    if ! NGPU="$("$PYTHON_BIN" "$zero_wam_env_dir/detect_visible_npus.py")"; then
        printf 'Visible NPU detection failed; check the Ascend environment and device access\n' >&2
        return 1
    fi
    zero_wam_device_source=torch_npu.npu.device_count
fi
zero_wam_uint NGPU 1

export NNODES NODE_RANK MASTER_ADDR MASTER_PORT NGPU
printf 'Training topology: nodes=%s node_rank=%s devices_per_node=%s world_size=%s device_count_source=%s\n' \
    "$NNODES" "$NODE_RANK" "$NGPU" "$((NNODES * NGPU))" "$zero_wam_device_source" >&2
