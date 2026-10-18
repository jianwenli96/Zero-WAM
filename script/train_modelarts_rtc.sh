#!/usr/bin/env bash
# Run once per ModelArts training container; train_dist.sh owns torchrun.
set -eo pipefail
umask 007

rtc_repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
rtc_ascend_env="${ASCEND_ENV_FILE:-/usr/local/Ascend/ascend-toolkit/set_env.sh}"
if [[ ! -r "$rtc_ascend_env" ]]; then
    printf 'Ascend environment file not readable: %s\n' "$rtc_ascend_env" >&2
    exit 1
fi
source "$rtc_ascend_env"
set -u

cd "$rtc_repo_root"
export PYTHONPATH="$rtc_repo_root:${PYTHONPATH:-}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"

export PYTHON_BIN="${PYTHON_BIN:-/mnt/sfs_turbo/public/apps/miniforge3/envs/lingbot-vggt/bin/python}"
export ALOHA_PREPARED_ROOT="${ALOHA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3}"
export MODEL_PATH="${MODEL_PATH:-/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/aloha_step6000_validation/model}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-/mnt/sfs_turbo/guzhai/modelarts_outputs/aloha_rtc_a32_stride3}"
export HF_HOME="${HF_HOME:-$ZERO_WAM_SAVE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ZERO_WAM_SAVE_ROOT/.cache}"
export CONFIG_NAME="${CONFIG_NAME:-aloha_train}"
export DATASETS="${DATASETS:-aloha:1}"

export MASTER_PORT="${MASTER_PORT:-${MA_TORCHRUN_MASTER_PORT:-29648}}"
# Keep ASCEND_RT_VISIBLE_DEVICES exactly as supplied by the platform.
# train_dist.sh resolves nodes, rank, and every node's visible NPU count.

if [[ ! -x "$PYTHON_BIN" ]]; then
    printf 'Python not executable; mount the shared environment or set PYTHON_BIN: %s\n' "$PYTHON_BIN" >&2
    exit 1
fi
if [[ ! -r "$MODEL_PATH/transformer/config.json" ]]; then
    printf 'Model not readable; check all symlink targets and mounts: %s\n' "$MODEL_PATH" >&2
    exit 1
fi
if [[ ! -r "$ALOHA_PREPARED_ROOT/preparation.json" ]]; then
    printf 'Prepared ALOHA data not readable: %s\n' "$ALOHA_PREPARED_ROOT" >&2
    exit 1
fi
mkdir -p "$ZERO_WAM_SAVE_ROOT" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
if [[ ! -w "$ZERO_WAM_SAVE_ROOT" ]]; then
    printf 'Training output is not writable by uid %s: %s\n' "$(id -u)" "$ZERO_WAM_SAVE_ROOT" >&2
    exit 1
fi
printf 'ModelArts RTC training output: %s\n' "$ZERO_WAM_SAVE_ROOT"

exec bash "$rtc_repo_root/script/train_dist.sh" \
    --action-stride 3 --actions-per-latent 32 \
    --length-bucket-steps 0 --learning-rate 1e-5 \
    --batch-size 1 --gradient-accumulation-steps 1 \
    --num-steps 5000 --save-interval 2500 \
    --init-worker 2 --load-worker 2 --disable-wandb "$@"
