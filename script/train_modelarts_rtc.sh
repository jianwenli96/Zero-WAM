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
export CONFIG_NAME="${CONFIG_NAME:-aloha_train}"
rtc_sampling_args=()
case "$CONFIG_NAME" in
    aloha_train)
        rtc_default_model=/mnt/sfs_turbo/lijianwen/Codes/Zero-WAM/outputs/aloha_step6000_validation/model
        rtc_default_datasets=aloha:1
        rtc_default_lr=1e-5
        rtc_default_save_interval=2500
        ;;
    piperpika_train)
        rtc_sampling_args=(--action-sampling-datasets piperpika)
        rtc_default_model=/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain
        rtc_default_datasets=piperpika:5,agibot:4,robocoin:4,robomind:4,interna1:4,oxe:4
        rtc_default_lr=2e-6
        rtc_default_save_interval=500
        ;;
    *) printf 'RTC training entry supports aloha_train or piperpika_train, got: %s\n' "$CONFIG_NAME" >&2; exit 1 ;;
esac
export ALOHA_PREPARED_ROOT="${ALOHA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3}"
export PIPERPIKA_PREPARED_ROOT="${PIPERPIKA_PREPARED_ROOT:-/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2}"
export MODEL_PATH="${MODEL_PATH:-$rtc_default_model}"
export ZERO_WAM_SAVE_ROOT="${ZERO_WAM_SAVE_ROOT:-/mnt/sfs_turbo/guzhai/modelarts_outputs/${CONFIG_NAME%_train}_rtc_a32_stride3}"
export HF_HOME="${HF_HOME:-$ZERO_WAM_SAVE_ROOT/.cache/huggingface}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$ZERO_WAM_SAVE_ROOT/.cache}"
export DATASETS="${DATASETS:-$rtc_default_datasets}"
if [[ "$DATASETS" == *,* ]]; then rtc_bucket_steps=8; else rtc_bucket_steps=0; fi

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
if [[ "$CONFIG_NAME" == piperpika_train ]]; then
    # PREPARE_DATA=1 may create metadata in train_dist.sh; check mode requires it now.
    if [[ "${PREPARE_DATA:-1}" != 1 && ! -r "$PIPERPIKA_PREPARED_ROOT/preparation.json" ]]; then
        printf 'Prepared PiperPika data not readable: %s\n' "$PIPERPIKA_PREPARED_ROOT" >&2; exit 1
    fi
elif [[ ! -r "$ALOHA_PREPARED_ROOT/preparation.json" ]]; then
    printf 'Prepared ALOHA data not readable: %s\n' "$ALOHA_PREPARED_ROOT" >&2; exit 1
fi
mkdir -p "$ZERO_WAM_SAVE_ROOT" "$HF_DATASETS_CACHE" "$XDG_CACHE_HOME"
if [[ ! -w "$ZERO_WAM_SAVE_ROOT" ]]; then
    printf 'Training output is not writable by uid %s: %s\n' "$(id -u)" "$ZERO_WAM_SAVE_ROOT" >&2
    exit 1
fi
printf 'ModelArts RTC training output: %s\n' "$ZERO_WAM_SAVE_ROOT"

exec bash "$rtc_repo_root/script/train_dist.sh" \
    --action-stride 3 --actions-per-latent 32 \
    --length-bucket-steps "$rtc_bucket_steps" --learning-rate "$rtc_default_lr" \
    --batch-size 1 --gradient-accumulation-steps 1 \
    --num-steps 5000 --save-interval "$rtc_default_save_interval" \
    --init-worker 2 --load-worker 2 --disable-wandb "${rtc_sampling_args[@]}" "$@"
