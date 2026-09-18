#!/usr/bin/env bash
# A short pilot by default. Explicit device allocation is required to launch.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv/bin/python"}
export MODEL_PATH=${MODEL_PATH:-"$ROOT/checkpoints/zero-wam-wan-init-fp32-seed42"}
HUMANGEN_ROOT=${HUMANGEN_ROOT:-"$ROOT/data/HumanGen"}
SAMPLING_CONFIG=${SAMPLING_CONFIG:-"$ROOT/wan_va/configs/humangen_robotwin_sampling.json"}
SAMPLING_ARGS=(--config "$SAMPLING_CONFIG")
if [[ -v DATASETS ]]; then
  SAMPLING_ARGS+=(--datasets "$DATASETS")
fi
DATASETS=$("$PYTHON_BIN" script/resolve_training_sampling.py "${SAMPLING_ARGS[@]}")
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-"$ROOT/train_out/wan-humangen-robotwin-4to1-sqrt-seed42-pilot"}
export HF_HOME=${HF_HOME:-"$ROOT/outputs/hf-cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
MODE=${1:---dry-run}
if [[ "$MODE" != --dry-run && "$MODE" != --check-only && "$MODE" != --run ]]; then
  echo 'Usage: bash script/train_humangen_wan_npu.sh [--dry-run|--check-only|--run]' >&2
  exit 2
fi
for file in "$MODEL_PATH/transformer/config.json" "$MODEL_PATH/initialization.json" \
            "$HUMANGEN_ROOT/external-preparation.json"; do
  test -f "$file" || { echo "Missing $file; follow docs/humangen-wan-training.md" >&2; exit 1; }
done
if [[ "$DATASETS" == *robotwin:* ]]; then
  for file in "$HUMANGEN_ROOT/preparation.json" "$HUMANGEN_ROOT/icl_configs/ICL_config_robotwin_train.json"; do
    test -f "$file" || { echo "缺少 $file；请先准备 RoboTwin 数据" >&2; exit 1; }
  done
fi
if [[ "$MODE" == --check-only ]]; then
  "$PYTHON_BIN" script/resolve_training_sampling.py "${SAMPLING_ARGS[@]}" --output-format json
  "$PYTHON_BIN" script/check_humangen_training.py --root "$HUMANGEN_ROOT"
  if [[ "$DATASETS" == *robotwin:* ]]; then
    "$PYTHON_BIN" script/check_robotwin_training.py --root "$HUMANGEN_ROOT"
  fi
  exit
fi
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CARDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
  NPROC_PER_NODE=${NPROC_PER_NODE:-${#CARDS[@]}}
  [[ "$NPROC_PER_NODE" -eq "${#CARDS[@]}" ]] || { echo 'NPROC_PER_NODE must match assigned visible devices' >&2; exit 1; }
else
  NPROC_PER_NODE=${NPROC_PER_NODE:-6}
fi
TRAIN_MODULE=wan_va.train
if [[ "${TRAIN_DIAGNOSTICS:-0}" == 1 ]]; then
  TRAIN_MODULE=script.benchmark_npu_training
fi
COMMAND=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE"
  --master_port "${MASTER_PORT:-29617}" --tee 3 -m "$TRAIN_MODULE"
  --config-name agibot_train --datasets "$DATASETS"
  --model-path "$MODEL_PATH" --human-gen-root "$HUMANGEN_ROOT"
  --save-root "$ZERO_WAM_SAVE_ROOT" --seed "${TRAIN_SEED:-42}"
  --num-steps "${NUM_STEPS:-10}" --save-interval "${SAVE_INTERVAL:-10}"
  --batch-size 1 --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --init-worker "${INIT_WORKERS:-4}" --load-worker "${LOAD_WORKERS:-2}"
  --learning-rate "${LEARNING_RATE:-0.0001}" --drop-icl 0.1 --droptext-target 0.4
  --disable-wandb)
# The empirical profile is calibrated only for the eight-card configuration.
# Explicit 'off' preserves uncropped training (or a manual MAX_TRAIN_FRAMES cap).
CAPACITY_PROFILE=${SEQUENCE_CAPACITY_PROFILE:-}
if [[ ! -v SEQUENCE_CAPACITY_PROFILE && "$NPROC_PER_NODE" -eq 8 ]]; then
  CAPACITY_PROFILE="$ROOT/wan_va/configs/sequence_capacity_8npu.json"
fi
if [[ -n "$CAPACITY_PROFILE" && "$CAPACITY_PROFILE" != off ]]; then
  COMMAND+=(--sequence-capacity-profile "$CAPACITY_PROFILE")
fi
if [[ -n "${MAX_TRAIN_FRAMES:-}" ]]; then
  COMMAND+=(--max-train-frames "$MAX_TRAIN_FRAMES")
fi
if [[ -n "${LENGTH_BUCKET_STEPS:-}" ]]; then
  COMMAND+=(--length-bucket-steps "$LENGTH_BUCKET_STEPS")
fi
if [[ -n "${FSDP_GRANULARITY:-}" ]]; then
  COMMAND+=(--fsdp-granularity "$FSDP_GRANULARITY")
fi
if [[ "$MODE" == --dry-run ]]; then
  "$PYTHON_BIN" script/resolve_training_sampling.py "${SAMPLING_ARGS[@]}" --output-format json
  printf 'Device allocation: %s\n' "${ASCEND_RT_VISIBLE_DEVICES:-not assigned; required for --run}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit
fi
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the physical NPU cards allocated to this run}"
test -f "$HUMANGEN_ROOT/external-validation.json" || { echo 'Run --check-only first' >&2; exit 1; }
if [[ "$DATASETS" == *robotwin:* ]]; then
  test -f "$HUMANGEN_ROOT/validation.json" || { echo 'Check RoboTwin data first' >&2; exit 1; }
fi
# Existing run directories may hold valuable checkpoints and loss curves.
if [[ -e "$ZERO_WAM_SAVE_ROOT" ]]; then
  echo "Refusing to reuse run directory: $ZERO_WAM_SAVE_ROOT" >&2
  exit 1
fi
# CANN's environment scripts are not nounset-safe.
set +u
source "$ROOT/setup_npu_env.sh"
set -u
mkdir -p "$ZERO_WAM_SAVE_ROOT"
"$PYTHON_BIN" script/resolve_training_sampling.py "${SAMPLING_ARGS[@]}" --output-format json > "$ZERO_WAM_SAVE_ROOT/sampling.json"
printf '%q ' "${COMMAND[@]}" > "$ZERO_WAM_SAVE_ROOT/command.txt"
printf '\n' >> "$ZERO_WAM_SAVE_ROOT/command.txt"
cp "$HUMANGEN_ROOT/external-preparation.json" "$ZERO_WAM_SAVE_ROOT/data-preparation.json"
cp "$HUMANGEN_ROOT/external-validation.json" "$ZERO_WAM_SAVE_ROOT/data-validation.json"
if [[ "$DATASETS" == *robotwin:* ]]; then
  cp "$HUMANGEN_ROOT/preparation.json" "$ZERO_WAM_SAVE_ROOT/robotwin-preparation.json"
  cp "$HUMANGEN_ROOT/validation.json" "$ZERO_WAM_SAVE_ROOT/robotwin-validation.json"
fi
cp "$MODEL_PATH/initialization.json" "$ZERO_WAM_SAVE_ROOT/model-initialization.json"
git rev-parse HEAD > "$ZERO_WAM_SAVE_ROOT/code-head.txt"
git diff HEAD > "$ZERO_WAM_SAVE_ROOT/code-changes.patch"
tar --exclude=__pycache__ -czf "$ZERO_WAM_SAVE_ROOT/source.tar.gz" wan_va script
"${COMMAND[@]}" 2>&1 | tee "$ZERO_WAM_SAVE_ROOT/train.log"
