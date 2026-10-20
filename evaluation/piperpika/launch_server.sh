#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${MODEL_PATH:?Set MODEL_PATH to the model root containing transformer, vae, tokenizer and text_encoder}"
PYTHON_BIN=${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}
source script/python_runtime.sh
export TOKENIZERS_PARALLELISM=false
exec "$PYTHON_BIN" -m torch.distributed.run --nproc_per_node="${NGPU:-1}" \
    --master_port="${MASTER_PORT:-29611}" -m evaluation.piperpika.server \
    --model "$MODEL_PATH" --port "${PORT:-29536}" "$@"
