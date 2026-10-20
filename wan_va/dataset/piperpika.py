"""Piper/Pika absolute joint + gripper contract, preserving pretrained slots."""
import json
from pathlib import Path
import numpy as np
from .aloha import CAMERAS

CHANNELS = list(range(14, 26)) + [28, 29]
CONTRACT = 'piperpika_absolute_joint12_gripper2_v1'


def joint_metadata(actions):
    actions = np.asarray(actions)
    if actions.ndim != 2 or actions.shape[1] != 28 or not np.isfinite(actions[:, 14:]).all():
        raise ValueError('Expected finite 12 joint + 2 gripper channels in 28D source actions')
    transform = {'norm_stats': 'action_stats.json', 'states': [], 'actions': [],
                 'images': [{f'observation.image.{i}': {'origin_keys': cam}} for i, cam in enumerate(CAMERAS)]}
    stats = {'method': 'abs', 'window_size': 16, 'action_contract': CONTRACT,
             'representation': 'absolute joints and grippers; original recorded units; no EEF', 'norm_stats': {}}
    for name, start, end in [('arm.position', 14, 26), ('effector.position', 26, 28)]:
        spec = {'origin_keys': [{'action': {'start': start, 'end': end}}],
                'format': 'joint', 'absolute_value': True, 'use_absolute': True}
        # Required loader alias; absolute processing does not use this as measured state.
        transform['states'].append({f'observation.state.{name}': spec.copy()})
        transform['actions'].append({f'action.{name}': spec})
        v = actions[:, start:end]
        stats['norm_stats'][f'action.{name}'] = {
            'q01': np.quantile(v, .01, axis=0).tolist(), 'q99': np.quantile(v, .99, axis=0).tolist(),
            'mean': v.mean(0).tolist(), 'std': v.std(0).tolist()}
    return transform, stats


def model_quantiles(path):
    stats = json.loads(Path(path).read_text())
    if stats['action_contract'] != CONTRACT:
        raise ValueError('Wrong joint action contract')
    return {q: [0.] * 14 + stats['norm_stats']['action.arm.position'][q] + [0.] * 2
            + stats['norm_stats']['action.effector.position'][q] for q in ('q01', 'q99')}


def capture_action_contract(root):
    root = Path(root)
    return {'stats': json.loads((root/'train/piperpika/meta/action_stats.json').read_text()),
            'preparation': json.loads((root/'preparation.json').read_text()),
            'cameras': CAMERAS, 'channels': CHANNELS}


def verify_checkpoint_contract(model_root, prepared_root, allow_unbound=False):
    model_root = Path(model_root)
    current = capture_action_contract(prepared_root)
    candidates = [model_root/'piperpika_action_contract.json',
                  (model_root/'transformer').resolve().parent/'piperpika_action_contract.json']
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        if not allow_unbound:
            raise ValueError('Missing PiperPika joint contract; use --allow-unbound-model only for base model diagnostics')
        return current, False
    if json.loads(path.read_text()) != current:
        raise ValueError('Checkpoint and prepared joint statistics/split differ')
    return current, True
