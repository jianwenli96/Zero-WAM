#!/usr/bin/env bash
# Source before launching mentor's transfer_to_npu training/server entrypoints.
ZERO_WAM_CANN_ENV=${CANN_ENV_PATH:-/usr/local/Ascend/cann-9.1.0/set_env.sh}
if [[ ! -f "$ZERO_WAM_CANN_ENV" ]]; then
  echo "CANN environment not found: $ZERO_WAM_CANN_ENV; set CANN_ENV_PATH" >&2
  return 1 2>/dev/null || exit 1
fi
source "$ZERO_WAM_CANN_ENV"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PYTORCH_NPU_ALLOC_CONF=${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd):${PYTHONPATH:-}"
# ASCEND_RT_VISIBLE_DEVICES must identify the devices allocated to this run.
