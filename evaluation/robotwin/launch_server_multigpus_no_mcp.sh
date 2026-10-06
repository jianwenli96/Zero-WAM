#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

# Point MODEL_PATH at this experiment's model root (containing transformer/).
: "${MODEL_PATH:?Set MODEL_PATH to the no_mcp experiment model root}"
export SAVE_ROOT="${SAVE_ROOT:-${ZERO_WAM_ROOT}/evals/visualization/ablation_no_mcp}"
export LOG_ROOT="${LOG_ROOT:-${ZERO_WAM_ROOT}/evals/logs/ablation_no_mcp}"
export TORCHINDUCTOR_CACHE_ROOT="${TORCHINDUCTOR_CACHE_ROOT:-/tmp/zero_wam_torchinductor/ablation_no_mcp}"

exec bash "${SCRIPT_DIR}/launch_server_multigpus.sh" "$@" --disable-mcp
