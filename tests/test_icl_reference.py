from pathlib import Path

import pytest

from evaluation.robotwin.icl_reference import demonstration_id, resolve_icl_request


DEMO = 'run/samples/task/generated_video_kling-v3'


def test_client_id_does_not_require_local_data():
    assert demonstration_id('/missing/client/HumanGen/human_data/robotwin/' + DEMO + '.mp4') == DEMO


def test_server_uses_its_own_root_and_prefers_latent(tmp_path, monkeypatch):
    monkeypatch.setenv('HUMANGEN_ROOT', str(tmp_path))
    monkeypatch.delenv('ICL_LATENT_ROOT', raising=False)
    monkeypatch.delenv('ICL_VIDEO_ROOT', raising=False)
    latent = tmp_path / 'human_latents/robotwin' / (DEMO + '.pth')
    latent.parent.mkdir(parents=True)
    latent.touch()
    assert resolve_icl_request({'icl_demo_id': DEMO}) == ('', str(latent))
    latent.unlink()
    video = tmp_path / 'human_data/robotwin' / (DEMO + '.mp4')
    video.parent.mkdir(parents=True)
    video.touch()
    assert resolve_icl_request({'icl_demo_id': DEMO}) == (str(video), '')
    video.unlink()
    with pytest.raises(FileNotFoundError, match='not found on server'):
        resolve_icl_request({'icl_demo_id': DEMO})


def test_server_latent_override(tmp_path, monkeypatch):
    monkeypatch.setenv('ICL_LATENT_ROOT', str(tmp_path))
    (tmp_path / 'demo.pth').touch()
    assert resolve_icl_request({'icl_demo_id': 'demo'}) == ('', str(tmp_path / 'demo.pth'))


def test_text_only_and_disabled_icl_do_not_resolve_paths():
    assert resolve_icl_request({'icl_demo_id': '../invalid'}, enabled=False) == ('', '')
    assert resolve_icl_request({'icl_demo_id': '../invalid', 'use_icl': False}) == ('', '')


def test_legacy_paths_remain_supported():
    assert resolve_icl_request({'icl_video_path': '/v.mp4', 'icl_latent_path': '/l.pth'}) == ('/v.mp4', '/l.pth')


@pytest.mark.parametrize('value', ['', '/absolute', '../escape', 'a/../../escape', '.', None, 'a\\b'])
def test_invalid_ids(value):
    with pytest.raises(ValueError):
        resolve_icl_request({'icl_demo_id': value})
