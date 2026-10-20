#!/usr/bin/env bash
# Recorded-image timing experiment; no robot or simulator is connected.
# Defaults to two idle visible NPUs. Override with --device-count/--device_count
# (2-4), or pin exact runtime IDs with --devices 0 3. No torchrun/FSDP is used.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${MODEL_PATH:?Set MODEL_PATH to the complete A=32, stride=3 PiperPika model root}"
PYTHON_BIN=${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}
source script/python_runtime.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1
exec "$PYTHON_BIN" -m evaluation.piperpika.rtc_pipeline_probe --model "$MODEL_PATH" "$@"
