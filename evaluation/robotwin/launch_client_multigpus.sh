#!/usr/bin/env bash
set -euo pipefail

export LD_LIBRARY_PATH="/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
ZERO_WAM_ROOT=$(cd "${SCRIPT_DIR}/../.." && pwd -P)

export ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-/home/ubuntu/a-glj-ws/RoboTwin}"  # Set this to the root of the Robotwin repository

SAVE_ROOT=${1:-${ZERO_WAM_ROOT}/results}
SEED=${SEED:-0}
TEST_NUM=${TEST_NUM:-100}
HOST=${HOST:-127.0.0.1}
START_PORT=${START_PORT:-29556}
TARGET_TEXT_CFG=${TARGET_TEXT_CFG:--1}
ICL_CFG=${ICL_CFG:-5}
ICL_SEED=${ICL_SEED:-${SEED}}
ICL_HUMAN_VIDEO_MAP=${ICL_HUMAN_VIDEO_MAP:-${SCRIPT_DIR}/robotwin_icl_human_videos.py}
LOG_ROOT=${LOG_ROOT:-${ZERO_WAM_ROOT}/evals/logs}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-30}

echo "Checking policy servers at ${HOST}, ports $((START_PORT + 1))-$((START_PORT + 7))"
python "${SCRIPT_DIR}/check_servers.py" --host "${HOST}" \
  --start-port "${START_PORT}" --timeout "${SERVER_START_TIMEOUT}"
# Diagnose connectivity without loading RoboTwin or starting an evaluation.
if [[ "${CHECK_ONLY:-0}" == 1 ]]; then
  exit 0
fi

if [[ "${SAVE_ROOT}" != /* ]]; then
  SAVE_ROOT="${ZERO_WAM_ROOT}/${SAVE_ROOT#./}"
fi

# Seven concurrent simulations distributed over the two remote GPUs.
GPU_DEVICES=${GPU_DEVICES:-0,1}
if [[ ! "$GPU_DEVICES" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "GPU_DEVICES must be comma-separated device indices" >&2
  exit 2
fi
IFS=',' read -r -a DEVICES <<< "$GPU_DEVICES"

# Seven unique tasks match the seven servers on NPU devices 1..7.
# START_PORT is the same base on both machines; ports are START_PORT+1..+7.
TASKS=(
  stack_blocks_three
  place_object_scale
  stamp_seal
  open_microwave
  move_stapler_pad
  place_bread_basket
  place_empty_cup
)

export PYTHONPATH="${ZERO_WAM_ROOT}:${ROBOTWIN_ROOT}:${PYTHONPATH:-}"
mkdir -p "${LOG_ROOT}"
PID_FILE="${LOG_ROOT}/client_pids.txt"
: > "${PID_FILE}"
BATCH_TIME=$(date +%Y%m%d_%H%M%S)
CLIENT_PIDS=()

cleanup() {
  trap - EXIT INT TERM
  if ((${#CLIENT_PIDS[@]})); then
    kill "${CLIENT_PIDS[@]}" 2>/dev/null || true
    wait "${CLIENT_PIDS[@]}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

cd "${ROBOTWIN_ROOT}"
for i in "${!TASKS[@]}"; do
  task_name=${TASKS[$i]}
  port=$((START_PORT + i + 1))
  task_save_root=${SAVE_ROOT}
  log_file="${LOG_ROOT}/client_${i}_${task_name}_${BATCH_TIME}.log"
  CUDA_VISIBLE_DEVICES=${DEVICES[$((i % ${#DEVICES[@]}))]} \
  PYTHONWARNINGS=ignore::UserWarning \
  python -m evaluation.robotwin.eval_policy_client_openpi \
    --config "${ROBOTWIN_ROOT}/policy/ACT/deploy_policy.yml" \
    --host "${HOST}" \
    --port "${port}" \
    --save_root "${task_save_root}" \
    --video_guidance_scale "${TARGET_TEXT_CFG}" \
    --action_guidance_scale 1 \
    --icl_guidance_scale "${ICL_CFG}" \
    --icl_human_video_map "${ICL_HUMAN_VIDEO_MAP}" \
    --icl_seed "${ICL_SEED}" \
    --test_num "${TEST_NUM}" \
    --overrides \
    --task_name "${task_name}" \
    --task_config demo_clean \
    --train_config_name 0 \
    --model_name 0 \
    --ckpt_setting 0 \
    --seed "${SEED}" \
    --policy_name ACT > "${log_file}" 2>&1 &
  CLIENT_PIDS+=("$!")
  echo "$!" >> "${PID_FILE}"
done

echo "Robotwin clients started; PIDs: ${PID_FILE}"
status=0
for pid in "${CLIENT_PIDS[@]}"; do
  wait "${pid}" || status=$?
done
exit "${status}"
