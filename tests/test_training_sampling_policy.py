import bisect
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'script'))
from resolve_training_sampling import EXTERNAL_SOURCES, resolve
from wan_va.dataset.dataset_mixture import DistributedDatasetMixtureSampler, parse_dataset_mixture


def test_default_policy_group_ratio_and_square_root_weights():
    report = resolve()
    p = report['source_probabilities']
    assert sum(p.values()) == pytest.approx(1)
    assert p['robotwin'] == pytest.approx(0.2)
    assert sum(p[n] for n in EXTERNAL_SOURCES) == pytest.approx(0.8)
    assert p['agibot'] / p['interna1'] == pytest.approx(math.sqrt(3354 / 261))
    parsed = dict(parse_dataset_mixture(report['datasets'], (*EXTERNAL_SOURCES, 'robotwin')))
    assert parsed == p


def test_six_rank_sampler_realizes_policy_with_actual_source_lengths():
    report = resolve()
    lengths = [6659, 11230, 5165, 6102, 6082, 2149]
    weights = list(report['source_probabilities'].values())
    ends = []
    for length in lengths:
        ends.append(length + (ends[-1] if ends else 0))
    counts = [0] * len(lengths)
    kwargs = dict(dataset_lengths=lengths, dataset_weights=weights, seed=42, epoch_size=120000)
    rank_streams = [list(DistributedDatasetMixtureSampler(**kwargs, num_replicas=6, rank=rank)) for rank in range(6)]
    merged = [value for row in zip(*rank_streams) for value in row]
    assert merged == list(DistributedDatasetMixtureSampler(**kwargs))
    for value in merged:
        counts[bisect.bisect_right(ends, value)] += 1
    for count, expected in zip(counts, weights):
        assert count / len(merged) == pytest.approx(expected, abs=0.005)


@pytest.mark.parametrize('value', ['', 'robotwin:0', 'agibot:nan', 'agibot:inf', 'unknown:1', 'agibot:1,agibot:2'])
def test_invalid_override_rejected(value):
    with pytest.raises(ValueError):
        resolve(datasets=value)


def test_custom_override_is_explicit_and_normalized():
    report = resolve(datasets='agibot:2,robotwin:1')
    assert report['policy'] == 'custom_DATASETS'
    assert report['source_probabilities'] == pytest.approx({'agibot': 2/3, 'robotwin': 1/3})


def test_launcher_uses_resolved_policy_without_loading_npu(tmp_path):
    root = Path(__file__).parents[1]
    model, data = tmp_path / 'model', tmp_path / 'data'
    for file in [model / 'transformer/config.json', model / 'initialization.json',
                 data / 'external-preparation.json', data / 'preparation.json',
                 data / 'icl_configs/ICL_config_robotwin_train.json']:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    env = dict(os.environ, MODEL_PATH=str(model), HUMANGEN_ROOT=str(data), PYTHON_BIN=sys.executable)
    for key in ('DATASETS', 'SAMPLING_CONFIG', 'ASCEND_RT_VISIBLE_DEVICES', 'NPROC_PER_NODE',
                'INIT_WORKERS', 'LENGTH_BUCKET_STEPS', 'SEQUENCE_CAPACITY_PROFILE',
                'FSDP_GRANULARITY', 'PYTORCH_NPU_ALLOC_CONF', 'FSDP_ASYNC_UNSHARD'):
        env.pop(key, None)
    result = subprocess.run(['bash', str(root / 'script/train_humangen_wan_npu.sh'), '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    report, _ = json.JSONDecoder().raw_decode(result.stdout)
    assert report['source_probabilities'] == resolve()['source_probabilities']
    assert '--nproc_per_node 8' in result.stdout
    assert '--init-worker 1' in result.stdout
    assert '--length-bucket-steps 10' in result.stdout
    assert '--sequence-capacity-profile' in result.stdout
    assert '--fsdp-granularity sublayer' in result.stdout
    assert '--fsdp-async-unshard' in result.stdout
    assert '--human-gen-root' in result.stdout
    assert 'NPU allocator: expandable_segments:False' in result.stdout
    assert not (tmp_path / 'data/validation.json').exists()
    env['PYTORCH_NPU_ALLOC_CONF'] = 'expandable_segments:True'
    env['FSDP_ASYNC_UNSHARD'] = '0'
    overridden = subprocess.run(
        ['bash', str(root / 'script/train_humangen_wan_npu.sh'), '--dry-run'],
        env=env, capture_output=True, text=True, check=True)
    assert 'NPU allocator: expandable_segments:True' in overridden.stdout
    assert '--no-fsdp-async-unshard' in overridden.stdout


@pytest.mark.parametrize('cards,override,enabled', [(8, None, True), (8, 'off', False), (7, None, False)])
def test_launcher_capacity_profile_only_defaults_to_eight_cards(tmp_path, cards, override, enabled):
    root = Path(__file__).parents[1]
    model, data = tmp_path / 'model', tmp_path / 'data'
    for file in [model / 'transformer/config.json', model / 'initialization.json',
                 data / 'external-preparation.json']:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    env = dict(os.environ, MODEL_PATH=str(model), HUMANGEN_ROOT=str(data),
               PYTHON_BIN=sys.executable, DATASETS='agibot:1',
               NPROC_PER_NODE=str(cards), ASCEND_RT_VISIBLE_DEVICES=','.join(map(str, range(cards))))
    env.pop('SEQUENCE_CAPACITY_PROFILE', None)
    if override is not None:
        env['SEQUENCE_CAPACITY_PROFILE'] = override
    result = subprocess.run(['bash', str(root / 'script/train_humangen_wan_npu.sh'), '--dry-run'],
                            env=env, capture_output=True, text=True, check=True)
    assert ('--sequence-capacity-profile' in result.stdout) is enabled
    if enabled:
        assert 'sequence_capacity_8npu.json' in result.stdout


@pytest.mark.parametrize('datasets,override,expected', [
    ('agibot:1', None, 0),
    ('agibot:1,interna1:1', None, 10),
    ('agibot:1,interna1:1', '0', 0),
    ('agibot:1,interna1:1', '20', 20),
])
def test_launcher_bucketing_matches_source_mode_and_explicit_override(
        tmp_path, datasets, override, expected):
    root = Path(__file__).parents[1]
    model, data = tmp_path / 'model', tmp_path / 'data'
    for file in [model / 'transformer/config.json', model / 'initialization.json',
                 data / 'external-preparation.json']:
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_text('{}')
    env = dict(os.environ, MODEL_PATH=str(model), HUMANGEN_ROOT=str(data),
               PYTHON_BIN=sys.executable, DATASETS=datasets)
    for key in ('LENGTH_BUCKET_STEPS', 'ASCEND_RT_VISIBLE_DEVICES', 'NPROC_PER_NODE',
                'SAMPLING_CONFIG'):
        env.pop(key, None)
    if override is not None:
        env['LENGTH_BUCKET_STEPS'] = override
    result = subprocess.run(
        ['bash', str(root / 'script/train_humangen_wan_npu.sh'), '--dry-run'],
        env=env, capture_output=True, text=True, check=True)
    assert f'--length-bucket-steps {expected}' in result.stdout
