#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

# Point MODEL_PATH at this experiment's model root (containing transformer/).
: "${MODEL_PATH:?Set MODEL_PATH to the text_only experiment model root}"
export SAVE_ROOT="${SAVE_ROOT:-${ZERO_WAM_ROOT}/visualization/ablation_text_only}"
export LOG_ROOT="${LOG_ROOT:-${ZERO_WAM_ROOT}/logs/ablation_text_only}"
export TORCHINDUCTOR_CACHE_ROOT="${TORCHINDUCTOR_CACHE_ROOT:-/tmp/zero_wam_torchinductor/ablation_text_only}"

exec bash "${SCRIPT_DIR}/launch_server_multigpus.sh" "$@" --disable-human-video
