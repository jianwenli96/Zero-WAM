#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON_BIN=${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}
ALOHA_PREPARED_ROOT=${ALOHA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3}
ALOHA_SOURCE_ROOT=${ALOHA_SOURCE_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_icl_data}
DEVICE_TYPE=${DEVICE_TYPE:-npu}
NGPU=${NGPU:-8}
if ! [[ "$NGPU" =~ ^[1-9][0-9]*$ ]]; then echo "NGPU must be positive" >&2; exit 1; fi
"$PYTHON_BIN" script/prepare_aloha.py prepare --source "$ALOHA_SOURCE_ROOT" --output "$ALOHA_PREPARED_ROOT"
pids=()
for ((shard=0; shard<NGPU; shard++)); do
    "$PYTHON_BIN" script/prepare_aloha.py encode --output "$ALOHA_PREPARED_ROOT" \
        --num-shards "$NGPU" --shard-index "$shard" --device "$DEVICE_TYPE:$shard" &
    pids+=("$!")
done
status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
if ((status)); then exit "$status"; fi
"$PYTHON_BIN" script/check_aloha.py --root "$ALOHA_PREPARED_ROOT" --deep --load-samples
