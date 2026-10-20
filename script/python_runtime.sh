#!/usr/bin/env bash
# Source before importing torch_npu, including visible-device discovery.
zero_wam_python_executable="$(command -v -- "$PYTHON_BIN")"
zero_wam_python_root="$(dirname "$(dirname "$(readlink -f "$zero_wam_python_executable")")")"
zero_wam_cxx_runtime="$zero_wam_python_root/lib/libstdc++.so.6"
if [[ -f "$zero_wam_cxx_runtime" ]]; then
    zero_wam_cxx_runtime="$(readlink -f "$zero_wam_cxx_runtime")"
    case ":${LD_PRELOAD:-}:" in
        *":$zero_wam_cxx_runtime:"*) ;;
        *) export LD_PRELOAD="$zero_wam_cxx_runtime${LD_PRELOAD:+:$LD_PRELOAD}" ;;
    esac
    printf 'Using Python C++ runtime: %s\n' "$zero_wam_cxx_runtime" >&2
fi
