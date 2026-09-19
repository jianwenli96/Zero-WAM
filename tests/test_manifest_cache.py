import json
import os

from wan_va.dataset.icl_lerobot_latent_dataset import (
    _load_icl_manifest_cached, load_icl_manifest,
)


def test_manifest_cache_shares_parsing_and_refreshes(tmp_path, monkeypatch):
    _load_icl_manifest_cached.cache_clear()
    path = tmp_path / 'manifest.json'
    sample = dict(robot_video_path='task/videos/chunk-000/camera/episode_000001.mp4',
                  human_video_path='run_demo/a.mp4')
    path.write_text(json.dumps(dict(samples=[sample])))
    calls = []
    original_load = json.load
    def tracked_load(*args, **kwargs):
        calls.append(1)
        return original_load(*args, **kwargs)
    monkeypatch.setattr(json, 'load', tracked_load)
    first = load_icl_manifest(path)
    assert load_icl_manifest(path) is first
    alias = tmp_path / 'alias.json'
    alias.symlink_to(path)
    assert load_icl_manifest(alias) is first
    assert len(calls) == 1
    previous = path.stat()
    sample['human_video_path'] = 'run_demo/b.mp4'  # same size, new timestamp
    path.write_text(json.dumps(dict(samples=[sample])))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns + 1_000_000))
    second = load_icl_manifest(path)
    assert second is not first
    assert next(iter(second[0].values()))['human_video_path'] == 'run_demo/b.mp4'
    previous = path.stat()
    sample['human_video_path'] = 'run_demo/longer.mp4'
    path.write_text(json.dumps(dict(samples=[sample])))
    os.utime(path, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    assert load_icl_manifest(path) is not second  # size also invalidates
    assert len(calls) == 3
    assert _load_icl_manifest_cached.cache_info().maxsize == 8
    _load_icl_manifest_cached.cache_clear()
