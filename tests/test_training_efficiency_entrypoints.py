import json
import os
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('script', ['train.sh', 'train_dist.sh'])
def test_launchers_only_forward_explicit_efficiency_arguments(tmp_path, script):
    executable = tmp_path / 'capture-python'
    executable.write_text('#!/usr/bin/env python3\nimport json,sys\nprint(json.dumps(sys.argv[1:]))\n')
    executable.chmod(0o755)
    env = dict(os.environ, PYTHON_BIN=str(executable), NNODES='2', NODE_RANK='1',
               MASTER_ADDR='host', NGPU='8', LENGTH_BUCKET_STEPS='8',
               MAX_TRAIN_FRAMES='32')
    flags = ['--length-bucket-steps', '--max-train-frames']
    def launch(extra):
        result = subprocess.run(['bash', str(ROOT/'script'/script), *extra], env=env,
                                capture_output=True, text=True, check=True)
        return json.loads(result.stdout)
    args = launch([])
    assert all(flag not in args for flag in flags)
    extra = ['--length-bucket-steps', '3',
             '--max-train-frames', '16', '--save-root', '/tmp/output with spaces']
    assert launch(extra)[-len(extra):] == extra
