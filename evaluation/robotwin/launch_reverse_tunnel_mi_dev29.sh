#!/usr/bin/env bash
# Run inside mi_dev_29's ljw_dev container. Server and client use START_PORT=30556.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
export GPU_SSH_HOST=${GPU_SSH_HOST:-ubuntu@175.27.240.116}
export START_PORT=${START_PORT:-30556}
export REMOTE_START_PORT=${REMOTE_START_PORT:-30556}
export SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-${HOME}/.ssh/zero_wam_gpu_tunnel_dev29}
export TUNNEL_STATE_DIR=${TUNNEL_STATE_DIR:-${SCRIPT_DIR}/../../evals/logs/tunnel_mi_dev29}
if (($# == 0)); then
  set -- start
fi
exec bash "${SCRIPT_DIR}/launch_reverse_tunnel.sh" "$@"
