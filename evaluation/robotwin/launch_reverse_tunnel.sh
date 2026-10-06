#!/usr/bin/env bash
# Run beside the inference server, inside its container if applicable.
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
SCRIPT_PATH="${SCRIPT_DIR}/launch_reverse_tunnel.sh"
TUNNEL_STATE_DIR=${TUNNEL_STATE_DIR:-${SCRIPT_DIR}/../../evals/logs/tunnel}
mkdir -p "$TUNNEL_STATE_DIR"
PID_FILE="${TUNNEL_STATE_DIR}/supervisor.pid"
LOG_FILE="${TUNNEL_STATE_DIR}/tunnel.log"

running() {
  [[ -f "$PID_FILE" ]] || return 1
  read -r supervisor_pid < "$PID_FILE"
  [[ "$supervisor_pid" =~ ^[0-9]+$ ]] || return 1
  kill -0 "$supervisor_pid" 2>/dev/null || return 1
  # Never signal an unrelated process after a stale PID has been reused.
  [[ -r "/proc/${supervisor_pid}/cmdline" ]] || return 1
  tr '\0' '\n' < "/proc/${supervisor_pid}/cmdline" | grep -Fxq "$SCRIPT_PATH"
}

case "${1:-foreground}" in
  status)
    if running; then
      echo "Tunnel supervisor running: PID ${supervisor_pid} (may be reconnecting). Log: ${LOG_FILE}"
    else
      echo "Tunnel supervisor stopped."
      exit 1
    fi
    exit 0 ;;
  stop)
    if running; then
      kill -TERM "$supervisor_pid"
      echo "Stop requested for tunnel supervisor ${supervisor_pid}."
    else
      echo "Tunnel supervisor already stopped."
    fi
    exit 0 ;;
  start|foreground|--supervise) ;;
  *) echo "Usage: $0 [start|stop|status|foreground]" >&2; exit 2 ;;
esac

: "${GPU_SSH_HOST:?Set GPU_SSH_HOST to user@gpu-machine}"
START_PORT=${START_PORT:-29556}
REMOTE_START_PORT=${REMOTE_START_PORT:-${START_PORT}}
SSH_PORT=${SSH_PORT:-22}
SSH_IDENTITY_FILE=${SSH_IDENTITY_FILE:-${HOME}/.ssh/zero_wam_gpu_tunnel}
RECONNECT_DELAY=${RECONNECT_DELAY:-5}
for base in "$START_PORT" "$REMOTE_START_PORT"; do
  if [[ ! "$base" =~ ^[1-9][0-9]*$ ]] || ((base > 65528)); then
    echo "Port bases must be integers between 1 and 65528, without leading zeros." >&2
    exit 2
  fi
done
if [[ ! "$RECONNECT_DELAY" =~ ^[1-9][0-9]*$ ]]; then
  echo "RECONNECT_DELAY must be a positive integer." >&2
  exit 2
fi

if [[ "${1:-foreground}" == start ]]; then
  if running; then
    echo "Tunnel supervisor already running: ${supervisor_pid}"
    exit 0
  fi
  # Authentication is noninteractive in the supervisor. Install a key first.
  nohup setsid bash "$SCRIPT_PATH" --supervise >> "$LOG_FILE" 2>&1 < /dev/null &
  new_pid=$!
  sleep 1
  if ! kill -0 "$new_pid" 2>/dev/null; then
    echo "Supervisor failed to start; see ${LOG_FILE}" >&2
    exit 1
  fi
  echo "Tunnel supervisor started: ${new_pid}. Log: ${LOG_FILE}"
  exit 0
fi

forwards=()
for i in {1..7}; do
  forwards+=(-R "127.0.0.1:$((REMOTE_START_PORT + i)):127.0.0.1:$((START_PORT + i))")
done
ssh_args=(-N -T -p "$SSH_PORT" -o ExitOnForwardFailure=yes
  -o ServerAliveInterval=15 -o ServerAliveCountMax=3 -o ConnectTimeout=10
  -o ConnectionAttempts=1 -o StrictHostKeyChecking=yes)
if [[ -f "$SSH_IDENTITY_FILE" ]]; then
  ssh_args+=(-i "$SSH_IDENTITY_FILE" -o IdentitiesOnly=yes)
fi
if [[ "${1:-foreground}" == foreground ]]; then
  echo "On the GPU machine use HOST=127.0.0.1 START_PORT=${REMOTE_START_PORT}."
  exec ssh "${ssh_args[@]}" "${forwards[@]}" "$GPU_SSH_HOST"
fi

# Hold an exclusive lock for the supervisor lifetime; children must not inherit it.
exec 9> "${TUNNEL_STATE_DIR}/supervisor.lock"
flock -n 9 || { echo "Another tunnel supervisor is running." >&2; exit 1; }
echo "$$" > "$PID_FILE"
child_pid=""
cleanup() {
  trap - EXIT INT TERM
  if [[ -n "$child_pid" ]]; then
    kill "$child_pid" 2>/dev/null || true
    wait "$child_pid" 2>/dev/null || true
  fi
  rm -f "$PID_FILE"
}
trap cleanup EXIT
trap 'exit 0' INT TERM
while true; do
  echo "$(date -Is) Connecting to ${GPU_SSH_HOST}; forwarding ports $((REMOTE_START_PORT + 1))-$((REMOTE_START_PORT + 7))."
  ssh "${ssh_args[@]}" -o BatchMode=yes "${forwards[@]}" "$GPU_SSH_HOST" 9>&- &
  child_pid=$!
  status=0
  wait "$child_pid" || status=$?
  child_pid=""
  echo "$(date -Is) SSH exited (${status}); reconnecting in ${RECONNECT_DELAY}s."
  sleep "$RECONNECT_DELAY" 9>&- &
  child_pid=$!
  wait "$child_pid" || true
  child_pid=""
done
