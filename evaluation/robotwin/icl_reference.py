"""Portable demonstration IDs and server-local HumanGen path resolution."""

import os
from pathlib import Path, PurePosixPath


def demonstration_id(video_path):
    """Convert a configured video path to an ID without touching client storage."""
    parts = PurePosixPath(video_path).parts
    if 'human_data' in parts:
        parts = parts[parts.index('human_data') + 1:]
        if parts and parts[0] == 'robotwin':
            parts = parts[1:]
        value = str(PurePosixPath(*parts).with_suffix(''))
    else:
        value = str(PurePosixPath(video_path).with_suffix(''))
    return str(_relative_id(value))


def _relative_id(value):
    if not isinstance(value, str) or not value or '\\' in value:
        raise ValueError('icl_demo_id must be a nonempty relative POSIX path')
    path = PurePosixPath(value)
    if path.is_absolute() or '..' in path.parts or not path.parts:
        raise ValueError(f'Invalid icl_demo_id: {value!r}')
    return path


def resolve_icl_request(obs, *, enabled=True):
    """Resolve new IDs on the server; retain explicit-path legacy requests."""
    if not enabled or not obs.get('use_icl', True):
        return '', ''
    if 'icl_demo_id' not in obs:
        return obs.get('icl_video_path', ''), obs.get('icl_latent_path', '')
    demo = _relative_id(obs['icl_demo_id'])
    root = Path(os.environ.get('HUMANGEN_ROOT', str(
        Path(__file__).resolve().parents[2] / 'data' / 'HumanGen')))
    latent_root = Path(os.environ.get('ICL_LATENT_ROOT', str(root / 'human_latents' / 'robotwin')))
    video_root = Path(os.environ.get('ICL_VIDEO_ROOT', str(root / 'human_data' / 'robotwin')))
    # IDs omit the extension; append it to preserve dots in demonstration names.
    latent = latent_root / (str(demo) + '.pth')
    video = video_root / (str(demo) + '.mp4')
    if latent.is_file():
        return '', str(latent)
    if video.is_file():
        return str(video), ''
    raise FileNotFoundError(
        f'ICL demo {str(demo)!r} not found on server: latent={latent}, video={video}. '
        'Set HUMANGEN_ROOT or ICL_LATENT_ROOT in the server environment.')
