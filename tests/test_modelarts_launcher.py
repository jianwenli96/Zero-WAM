"""Exercise the actual launchers without creating workers or allocating NPUs."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def launch(tmp_path):
    sdk = tmp_path/'fake Ascend env.sh'
    sdk.write_text('export PYTHONPATH="/tmp/ascend_sdk_package:${PYTHONPATH:-}"\n')
    model = tmp_path/'model with spaces'
    (model/'transformer').mkdir(parents=True)
    (model/'transformer/config.json').write_text('{}')
    data = tmp_path/'prepared data'
    data.mkdir()
    (data/'preparation.json').write_text('{}')
    recorder = tmp_path/'python recorder'
    recorder.write_text(
        f'#!{sys.executable}\n'
        'import json,os,sys\n'
        'if sys.argv[1].endswith("detect_visible_npus.py"):\n'
        '    print(os.environ.get("MOCK_VISIBLE_NPUS","8"))\n'
        '    sys.exit(int(os.environ.get("MOCK_PROBE_EXIT","0")))\n'
        'keys=["NNODES","NODE_RANK","MASTER_ADDR","MASTER_PORT","NGPU",'
        '"ASCEND_RT_VISIBLE_DEVICES","PYTHONPATH"]\n'
        'with open(os.environ["MOCK_RECORD"],"a") as f:\n'
        '    f.write(json.dumps(dict(argv=sys.argv[1:],env={k:os.environ.get(k) for k in keys}))+"\\n")\n'
    )
    recorder.chmod(0o755)
    record = tmp_path/'calls.jsonl'

    def run(variables=None, args=(), script='train_modelarts_rtc.sh'):
        env = os.environ.copy()
        for key in ('NNODES', 'NODE_RANK', 'MASTER_ADDR', 'MASTER_PORT', 'NGPU',
                    'MA_NUM_HOSTS', 'MA_NUM_GPUS', 'VC_TASK_INDEX', 'VC_WORKER_NUM',
                    'VC_WORKER_HOSTS', 'MA_TORCHRUN_MASTER_PORT',
                    'ASCEND_RT_VISIBLE_DEVICES', 'CONFIG_NAME', 'DATASETS',
                    'HF_HOME', 'HF_DATASETS_CACHE', 'XDG_CACHE_HOME'):
            env.pop(key, None)
        env.update(ASCEND_ENV_FILE=str(sdk), PYTHON_BIN=str(recorder),
                   MODEL_PATH=str(model), ALOHA_PREPARED_ROOT=str(data),
                   ZERO_WAM_SAVE_ROOT=str(tmp_path/'output with spaces'),
                   MOCK_RECORD=str(record), CONFIG_NAME='aloha_train', DATASETS='aloha:1')
        env.update(variables or {})
        result = subprocess.run(['bash', str(ROOT/'script'/script), *args],
                                env=env, cwd='/tmp', text=True, capture_output=True)
        calls = [json.loads(line) for line in record.read_text().splitlines()] if record.exists() else []
        return result, calls

    return run


@pytest.mark.parametrize('script', ['train_modelarts_rtc.sh', 'train_dist.sh'])
@pytest.mark.parametrize('variables,expected', [
    ({}, ['--nnodes=1', '--node_rank=0', '--nproc_per_node=8', '--master_addr=127.0.0.1']),
    ({'MA_NUM_GPUS': '4'}, ['--nproc_per_node=4']),
    ({'MA_NUM_GPUS': '0', 'MOCK_VISIBLE_NPUS': '6'}, ['--nproc_per_node=6']),
    ({'ASCEND_RT_VISIBLE_DEVICES': '1,5,6,7', 'MA_NUM_GPUS': '8', 'MOCK_VISIBLE_NPUS': '4'}, ['--nproc_per_node=4']),
    ({'NGPU': '2', 'MA_NUM_GPUS': '8'}, ['--nproc_per_node=2']),
    ({'NGPU': 'auto', 'MOCK_VISIBLE_NPUS': '6'}, ['--nproc_per_node=6']),
    ({'MA_NUM_HOSTS': '2', 'MA_NUM_GPUS': '8', 'VC_TASK_INDEX': '1',
      'VC_WORKER_HOSTS': 'worker-0,worker-1'},
     ['--nnodes=2', '--node_rank=1', '--nproc_per_node=8', '--master_addr=worker-0']),
    ({'VC_WORKER_NUM': '2', 'VC_TASK_INDEX': '1', 'VC_WORKER_HOSTS': 'worker-0,worker-1'},
     ['--nnodes=2', '--node_rank=1', '--master_addr=worker-0']),
    ({'VC_TASK_INDEX': '1', 'VC_WORKER_HOSTS': 'worker-0,worker-1'},
     ['--nnodes=2', '--node_rank=1', '--master_addr=worker-0']),
    ({'NNODES': '2', 'NODE_RANK': '1', 'MASTER_ADDR': '10.0.0.1', 'NGPU': '8'},
     ['--nnodes=2', '--node_rank=1', '--master_addr=10.0.0.1', '--nproc_per_node=8']),
    ({'NNODES': '1', 'NODE_RANK': '0', 'MASTER_ADDR': 'localhost', 'NGPU': '2',
      'MA_NUM_HOSTS': '2', 'MA_NUM_GPUS': '8', 'VC_TASK_INDEX': '1',
      'VC_WORKER_HOSTS': 'worker-0,worker-1'},
     ['--nnodes=1', '--node_rank=0', '--master_addr=localhost', '--nproc_per_node=2']),
    ({'MA_TORCHRUN_MASTER_PORT': '29876'}, ['--master_port=29876']),
])
def test_topology_and_device_resolution(launch, script, variables, expected):
    result, calls = launch(variables, script=script)
    assert result.returncode == 0, result.stderr
    assert len(calls) == 2
    assert calls[0]['argv'][0] == 'script/check_aloha.py'
    assert calls[1]['argv'][:2] == ['-m', 'torch.distributed.run']
    assert all(value in calls[1]['argv'] for value in expected)
    if 'ASCEND_RT_VISIBLE_DEVICES' in variables:
        assert calls[1]['env']['ASCEND_RT_VISIBLE_DEVICES'] == variables['ASCEND_RT_VISIBLE_DEVICES']
    if script == 'train_modelarts_rtc.sh':
        assert calls[1]['env']['PYTHONPATH'].startswith(str(ROOT)+':/tmp/ascend_sdk_package:')


@pytest.mark.parametrize('variables,error', [
    ({'NNODES': '2'}, 'requires NODE_RANK'),
    ({'NNODES': '2', 'NODE_RANK': '1'}, 'requires MASTER_ADDR'),
    ({'NNODES': '2', 'NODE_RANK': '2'}, 'must be smaller'),
    ({'NNODES': 'zero'}, 'NNODES must be an integer'),
    ({'NGPU': '0'}, 'NGPU must be >= 1'),
    ({'NGPU': 'bad'}, 'NGPU must be an integer'),
    ({'MA_NUM_GPUS': 'bad'}, 'NGPU must be an integer'),
    ({'MOCK_VISIBLE_NPUS': '0'}, 'NGPU must be >= 1'),
    ({'MOCK_PROBE_EXIT': '1'}, 'Visible NPU detection failed'),
    ({'MASTER_PORT': '70000'}, 'MASTER_PORT must be <= 65535'),
    ({'VC_WORKER_HOSTS': 'worker-0,,worker-2'}, 'nonempty comma-separated hosts'),
])
def test_invalid_topology_fails_before_launching_workers(launch, variables, error):
    result, calls = launch(variables)
    assert result.returncode != 0
    assert error in result.stderr
    assert not calls


def test_training_overrides_and_default_rtc_parameters(launch):
    extra = ['--num-steps', '10', '--save-interval', '10']
    result, calls = launch({'NGPU': '4'}, args=extra)
    assert result.returncode == 0, result.stderr
    args = calls[1]['argv']
    assert args[-len(extra):] == extra
    assert args[args.index('--action-stride')+1] == '3'
    assert args[args.index('--actions-per-latent')+1] == '32'


def test_npu_probe_uses_runtime_count_without_model_imports(tmp_path):
    (tmp_path/'torch_npu.py').write_text(
        'from types import SimpleNamespace\n'
        'print("plugin startup message")\n'
        'npu=SimpleNamespace(device_count=lambda: 3)\n')
    env = dict(os.environ, PYTHONPATH=str(tmp_path))
    result = subprocess.run([sys.executable, str(ROOT/'script/detect_visible_npus.py')],
                            env=env, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout == '3\n'
    assert 'plugin startup message' in result.stderr
