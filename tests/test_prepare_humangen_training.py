import importlib.util
import json
from pathlib import Path
import pytest


@pytest.fixture
def prep(monkeypatch):
    scripts = Path(__file__).parents[1] / 'script'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('prepare_external', scripts / 'prepare_humangen_training.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_source(root):
    group = root / 'agibot_data/robot_a'
    repo = group / 'task_a'
    (repo / 'meta').mkdir(parents=True)
    (group / 'meta').mkdir()
    (group / 'meta/action_transform.yaml').write_text('images:\n  - camera:\n      origin_keys: camera\nnorm_stats: action_stats.json\n')
    (group / 'meta/action_stats.json').write_text(json.dumps({'method': 'abs'}))
    (repo / 'meta/info.json').write_text(json.dumps({'chunks_size': 1000, 'data_path': 'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet'}))
    sample = {'human_video_path': 'prefix/run_test/samples/one/video.mp4', 'status_info': {'video_rel_path': 'task_a/videos/chunk-000/camera/episode_000019_40_50.mp4'}}
    segment = {'start_frame': 0, 'end_frame': 10, 'source_episode_index': 19, 'source_frame_start': 40, 'source_frame_end': 50, 'source_video_rel_path': sample['status_info']['video_rel_path']}
    # Two released segments can legitimately reuse one human demonstration.
    (repo / 'meta/episodes.jsonl').write_text(json.dumps({'episode_index': 0, 'length': 10, 'action_config': [segment, segment]}) + '\n')
    parquet = repo / 'data/chunk-000/episode_000000.parquet'
    parquet.parent.mkdir(parents=True)
    parquet.touch()
    latent = repo / 'latents/chunk-000/camera/episode_000019_40_50.pth'
    latent.parent.mkdir(parents=True)
    latent.touch()
    human = root / 'human_latents/agibot/run_test/samples/one/video.pth'
    human.parent.mkdir(parents=True)
    human.touch()
    (root / 'icl_configs').mkdir()
    (root / 'icl_configs/ICL_config_agibot.json').write_text(json.dumps({'samples': [sample]}))
    return repo, latent


def test_nested_view_preserves_schema_and_source_interval(prep, tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'view'
    repo, _ = fixture_source(source)
    report = prep.prepare(source, output, ('agibot',))
    assert prep.prepare(source, output, ('agibot',)) == report
    data = report['sources']['agibot']
    assert data['segments'] == 2 and data['unique_human_videos'] == 1
    assert data['action_schema_groups'] == 1
    local = output / 'agibot_data/robot_a/task_a'
    assert prep.discover_repos(output / 'agibot_data') == [local]
    assert prep.action_metadata(local)[2] == ['camera']
    (local / '.cache').mkdir()
    assert not (repo / '.cache').exists()


def test_missing_source_interval_is_not_silently_filtered(prep, tmp_path):
    source, output = tmp_path / 'source', tmp_path / 'view'
    _, latent = fixture_source(source)
    latent.unlink()
    with pytest.raises(FileNotFoundError):
        prep.prepare(source, output, ('agibot',))
    assert not (output / 'external-preparation.json').exists()
