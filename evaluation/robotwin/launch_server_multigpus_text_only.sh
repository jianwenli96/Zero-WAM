#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

# Point MODEL_PATH at this experiment's model root (containing transformer/).
: "${MODEL_PATH:?Set MODEL_PATH to the text_only experiment model root}"
export SAVE_ROOT="${SAVE_ROOT:-${ZERO_WAM_ROOT}/evals/visualization/ablation_text_only}"
export LOG_ROOT="${LOG_ROOT:-${ZERO_WAM_ROOT}/evals/logs/ablation_text_only}"
export TORCHINDUCTOR_CACHE_ROOT="${TORCHINDUCTOR_CACHE_ROOT:-/tmp/zero_wam_torchinductor/ablation_text_only}"

# Pin target text on the server: legacy clients explicitly send -1.
export TARGET_TEXT_CFG="${TARGET_TEXT_CFG:-1}"

exec bash "${SCRIPT_DIR}/launch_server_multigpus.sh" "$@" --disable-human-video --target-text-cfg "${TARGET_TEXT_CFG}"
