#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-${SCRIPT_DIR}/../outputs/ablation_text_only}"
exec bash "${SCRIPT_DIR}/train_dist.sh" "$@" --disable-human-video
