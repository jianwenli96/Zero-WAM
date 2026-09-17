#!/usr/bin/env bash
# A short pilot by default. Explicit device allocation is required to launch.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"
PYTHON_BIN=${PYTHON_BIN:-"$ROOT/.venv/bin/python"}
export MODEL_PATH=${MODEL_PATH:-"$ROOT/checkpoints/zero-wam-wan-init-fp32-seed42"}
HUMANGEN_ROOT=${HUMANGEN_ROOT:-"$ROOT/data/HumanGen"}
export ZERO_WAM_SAVE_ROOT=${ZERO_WAM_SAVE_ROOT:-"$ROOT/train_out/wan-robotwin-icl-seed42-pilot"}
export HF_HOME=${HF_HOME:-"$ROOT/outputs/hf-cache"}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-"$HF_HOME/datasets"}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
MODE=${1:---dry-run}
if [[ "$MODE" != --dry-run && "$MODE" != --check-only && "$MODE" != --run ]]; then
  echo 'Usage: bash script/train_robotwin_wan_npu.sh [--dry-run|--check-only|--run]' >&2
  exit 2
fi
for file in "$MODEL_PATH/transformer/config.json" "$MODEL_PATH/initialization.json" \
            "$HUMANGEN_ROOT/preparation.json" "$HUMANGEN_ROOT/icl_configs/ICL_config_robotwin_train.json"; do
  test -f "$file" || { echo "Missing $file; follow docs/robotwin-wan-training.md" >&2; exit 1; }
done
if [[ "$MODE" == --check-only ]]; then
  "$PYTHON_BIN" script/check_robotwin_training.py --root "$HUMANGEN_ROOT"
  exit
fi
if [[ -n "${ASCEND_RT_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a CARDS <<< "$ASCEND_RT_VISIBLE_DEVICES"
  NPROC_PER_NODE=${NPROC_PER_NODE:-${#CARDS[@]}}
  [[ "$NPROC_PER_NODE" -eq "${#CARDS[@]}" ]] || { echo 'NPROC_PER_NODE must match assigned visible devices' >&2; exit 1; }
else
  NPROC_PER_NODE=${NPROC_PER_NODE:-8}
fi
COMMAND=("$PYTHON_BIN" -m torch.distributed.run --nproc_per_node "$NPROC_PER_NODE"
  --master_port "${MASTER_PORT:-29617}" --tee 3 -m wan_va.train
  --config-name robotwin_train --datasets robotwin:1.0
  --model-path "$MODEL_PATH" --dataset-path "$HUMANGEN_ROOT/robotwin_data"
  --icl-manifest-path "$HUMANGEN_ROOT/icl_configs/ICL_config_robotwin_train.json"
  --human-latent-path "$HUMANGEN_ROOT/human_latents/robotwin"
  --save-root "$ZERO_WAM_SAVE_ROOT" --seed "${TRAIN_SEED:-42}"
  --num-steps "${NUM_STEPS:-10}" --save-interval "${SAVE_INTERVAL:-10}"
  --batch-size 1 --gradient-accumulation-steps "${GRAD_ACCUM:-1}"
  --init-worker 1 --load-worker "${LOAD_WORKERS:-2}"
  --learning-rate "${LEARNING_RATE:-0.0001}" --drop-icl 0.1 --droptext-target 0.4
  --disable-wandb)
if [[ "$MODE" == --dry-run ]]; then
  printf 'Device allocation: %s\n' "${ASCEND_RT_VISIBLE_DEVICES:-not assigned; required for --run}"
  printf '%q ' "${COMMAND[@]}"
  printf '\n'
  exit
fi
: "${ASCEND_RT_VISIBLE_DEVICES:?Set the physical NPU cards allocated to this run}"
test -f "$HUMANGEN_ROOT/validation.json" || { echo 'Run --check-only first' >&2; exit 1; }
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
printf '%q ' "${COMMAND[@]}" > "$ZERO_WAM_SAVE_ROOT/command.txt"
printf '\n' >> "$ZERO_WAM_SAVE_ROOT/command.txt"
cp "$HUMANGEN_ROOT/preparation.json" "$ZERO_WAM_SAVE_ROOT/data-preparation.json"
cp "$HUMANGEN_ROOT/validation.json" "$ZERO_WAM_SAVE_ROOT/data-validation.json"
cp "$MODEL_PATH/initialization.json" "$ZERO_WAM_SAVE_ROOT/model-initialization.json"
git rev-parse HEAD > "$ZERO_WAM_SAVE_ROOT/code-head.txt"
git diff HEAD > "$ZERO_WAM_SAVE_ROOT/code-changes.patch"
tar --exclude=__pycache__ -czf "$ZERO_WAM_SAVE_ROOT/source.tar.gz" wan_va script
"${COMMAND[@]}" 2>&1 | tee "$ZERO_WAM_SAVE_ROOT/train.log"
