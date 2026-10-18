"""Print the NPU count visible to this process, without loading a model."""
import contextlib
import sys


def main():
    try:
        # Keep stdout machine-readable even if the plugin prints on import.
        with contextlib.redirect_stdout(sys.stderr):
            import torch_npu
            count = torch_npu.npu.device_count()
    except Exception as error:
        print(f'Cannot detect visible NPUs: {error}', file=sys.stderr)
        return 1
    print(count)
    return 0


if __name__ == '__main__':
    sys.exit(main())
