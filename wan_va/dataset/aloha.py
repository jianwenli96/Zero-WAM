"""ALOHA absolute-action contract, shared by preparation and open-loop evaluation."""
import json
from pathlib import Path
import numpy as np

CAMERAS = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist']
ACTION_NAMES = ([f'{side}_{axis}' for side in ('left', 'right')
                 for axis in ('x', 'y', 'z', 'qx', 'qy', 'qz', 'qw')]
                + [f'{side}_joint_{i}' for side in ('left', 'right') for i in range(6)]
                + ['left_gripper', 'right_gripper'])
CHANNELS = list(range(26)) + [28, 29]
FEATURES = [('hand.position', 0, 14), ('arm.position', 14, 26),
            ('effector.position', 26, 28)]


def validate_actions(actions):
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 28 or not np.isfinite(actions).all():
        raise ValueError('Expected finite [T,28] ALOHA absolute actions')
    for start in (3, 10):
        norm = np.linalg.norm(actions[:, start:start + 4], axis=1)
        if not np.allclose(norm, 1, atol=0.02):
            raise ValueError('Expected unit xyzw quaternions')
    return actions


def canonical_actions(actions):
    """Unit xyzw, deterministic first sign, then hemisphere-continuous per episode.

    Must run on the entire episode before slicing. Physical rotations and all
    other absolute channels remain unchanged; q and -q represent the same pose.
    """
    result = validate_actions(actions).copy()
    if not len(result):
        raise ValueError('Empty episode')
    for start in (3, 10):
        q = result[:, start:start+4]
        q /= np.linalg.norm(q, axis=1, keepdims=True)
        first_sign = 1 if q[0, np.argmax(np.abs(q[0]))] >= 0 else -1
        signs = np.r_[first_sign, np.where((q[1:]*q[:-1]).sum(-1) < 0, -1, 1)]
        q *= np.cumprod(signs)[:, None]
    return result


def action_metadata(actions):
    actions = validate_actions(actions)
    transform = {'norm_stats': 'action_stats.json', 'states': [], 'actions': [],
                 'images': [{f'observation.image.{i}': {'origin_keys': key}}
                            for i, key in enumerate(CAMERAS)]}
    stats = {'method': 'abs', 'window_size': 16, 'norm_stats': {},
             'representation': 'absolute; original units; continuous unit xyzw per episode; no state inferred',
             'action_contract': 'aloha_absolute_28_continuous_v2'}
    for name, start, end in FEATURES:
        spec = {'origin_keys': [{'action': {'start': start, 'end': end}}],
                'format': 'xyzqxyzq' if name == 'hand.position' else 'joint', 'absolute_value': True, 'use_absolute': True}
        # The generic loader requires a state entry, but use_absolute=True never
        # reads it as a physical reference. This is an alias, not measured state.
        transform['states'].append({f'observation.state.{name}': spec.copy()})
        transform['actions'].append({f'action.{name}': spec})
        values = actions[:, start:end]
        stats['norm_stats'][f'action.{name}'] = {
            'q01': np.quantile(values, .01, axis=0).tolist(),
            'q99': np.quantile(values, .99, axis=0).tolist(),
            'mean': values.mean(0).tolist(), 'std': values.std(0).tolist()}
    return transform, stats


def model_quantiles(stats_path):
    stats = json.loads(Path(stats_path).read_text())
    result = {}
    for q in ('q01', 'q99'):
        values = stats['norm_stats']
        result[q] = (values['action.hand.position'][q]
                     + values['action.arm.position'][q] + [0., 0.]
                     + values['action.effector.position'][q])
    return result


def segments(length, sampled_frames=129, stride=4):
    """Disjoint action ranges; consecutive clips share only the boundary RGB."""
    span = (sampled_frames - 1) * stride
    if sampled_frames < 5 or (sampled_frames - 1) % 4 or stride != 4:
        raise ValueError('Require 4k+1 RGB frames and stride=4 (16 actions/latent)')
    for start in range(0, length - 16, span):
        end = min(start + span, ((length - 1 - start) // 16) * 16 + start)
        if end > start:
            yield start, end + 1, list(range(start, end + 1, stride))


def capture_action_contract(root):
    root = Path(root)
    return {'stats': json.loads((root/'train/breakfast/meta/action_stats.json').read_text()),
            'preparation': json.loads((root/'preparation.json').read_text()),
            'cameras': CAMERAS, 'channels': CHANNELS}


def verify_checkpoint_contract(model_root, prepared_root, allow_unbound=False):
    model_root = Path(model_root)
    current = capture_action_contract(prepared_root)
    candidates = [model_root/'aloha_action_contract.json',
                  (model_root/'transformer').resolve().parent/'aloha_action_contract.json']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        if not allow_unbound:
            raise ValueError('Model has no ALOHA action contract. For an explicitly unbound base/legacy model, use --allow-unbound-model; this does not validate training statistics.')
        return current, False
    saved = json.loads(path.read_text())
    if saved != current:
        raise ValueError(f'Checkpoint action statistics/preparation differ from the selected dataset: {path}')
    return saved, True
