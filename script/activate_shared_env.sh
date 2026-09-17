#!/usr/bin/env bash
# Source this file; executing it cannot change the parent shell environment.
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  echo '请使用 source script/activate_shared_env.sh' >&2
  exit 1
fi
_zw_shared_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ ! -x "$_zw_shared_root/.venv/bin/python" ]]; then
  echo '共享 Python 环境不可用，请检查共享目录的挂载路径。' >&2
  unset _zw_shared_root
  return 1
fi
source "$_zw_shared_root/.venv/bin/activate"
unset _zw_shared_root
