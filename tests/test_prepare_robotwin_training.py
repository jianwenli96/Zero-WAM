import importlib.util
import json
import os
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location('prepare_robotwin', Path(__file__).parents[1] / 'script/prepare_robotwin_training.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def make_source(root):
    robot = root / 'robotwin_data'
    (robot / 'meta').mkdir(parents=True)
    (robot / 'meta/action_stats.json').write_text('{}')
    samples = []
    for name in sorted(module.HELD_OUT) + [f'train_{i:02d}' for i in range(43)]:
        task = robot / (name + '-variant')
        (task / 'meta').mkdir(parents=True)
        (task / 'meta/info.json').write_text(json.dumps(dict(chunks_size=1000, data_path='data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet')))
        (task / 'meta/episodes.jsonl').write_text(json.dumps(dict(episode_index=0, length=10, action_config=[dict(start_frame=0, end_frame=10)])) + '\n')
        parquet = task / 'data/chunk-000/episode_000000.parquet'
        parquet.parent.mkdir(parents=True)
        parquet.touch()
        for camera in module.CAMERAS:
            latent = task / 'latents/chunk-000' / camera / 'episode_000000_0_10.pth'
            latent.parent.mkdir(parents=True)
            latent.touch()
        human = root / 'human_latents/robotwin/run_test/samples' / name / 'video.pth'
        human.parent.mkdir(parents=True)
        human.touch()
        samples.append(dict(status_info=dict(video_rel_path=f'{task.name}/videos/chunk-000/cam/episode_000000.mp4'), human_video_path=f'prefix/run_test/samples/{name}/video.mp4'))
    (root / 'icl_configs').mkdir()
    (root / 'icl_configs/ICL_config_robotwin.json').write_text(json.dumps(dict(samples=samples)))


def test_training_manifest_isolated_and_local_cache_root(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    make_source(source)
    report = module.prepare(source, output)
    assert module.prepare(source, output) == report
    assert report['training_pairs'] == 43
    manifest = json.loads((output / 'icl_configs/ICL_config_robotwin_train.json').read_text())
    assert all(s['status_info']['video_rel_path'].startswith('train_') for s in manifest['samples'])
    # os.walk (used by the training loader) must discover all metadata without followlinks.
    infos = [Path(parent) / 'info.json' for parent, _, files in os.walk(output / 'robotwin_data') if 'info.json' in files]
    assert len(infos) == 50
    local = output / 'robotwin_data/train_00-variant'
    (local / '.cache').mkdir()
    assert not (source / 'robotwin_data/train_00-variant/.cache').exists()


def test_missing_camera_fails_before_creating_view(tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'output'
    make_source(source)
    next((source / 'robotwin_data/train_00-variant/latents').rglob('*.pth')).unlink()
    with pytest.raises(FileNotFoundError):
        module.prepare(source, output)
    assert not output.exists()
