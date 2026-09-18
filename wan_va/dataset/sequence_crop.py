"""Aligned random robot windows bounded by an empirical training capacity profile."""
import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class SequenceCapacity:
    name: str
    patch_size: tuple
    robot_token_margin: float
    passing_points: tuple
    training: dict
    model: dict

    @classmethod
    def load(cls, path):
        payload = json.loads(Path(path).read_text())
        if payload.get('version') != 1:
            raise ValueError('Unsupported sequence capacity profile version')
        patch = tuple(payload['patch_size'])
        if len(patch) != 3 or patch[0] != 1 or min(patch) <= 0:
            raise ValueError('Sequence capacity profiles require temporal patch size 1')
        margin = float(payload['robot_token_margin'])
        if not math.isfinite(margin) or not 0 <= margin < 1:
            raise ValueError('robot_token_margin must be in [0, 1)')
        points = tuple((int(p['robot_tokens']), int(p['human_tokens']))
                       for p in payload['passing_points'])
        if not points or any(r <= 0 or h <= 0 for r, h in points):
            raise ValueError('Capacity profile must contain positive passing token points')
        return cls(payload['name'], patch, margin, points,
                   payload['training'], payload['model'])

    def validate_training(self, config):
        defaults = {'gradient_accumulation_steps': 1, 'fsdp_granularity': 'sublayer'}
        for key, expected in self.training.items():
            actual = getattr(config, key, defaults.get(key))
            if key == 'param_dtype':
                actual = str(actual)
            if actual != expected:
                raise ValueError(f'Capacity profile {self.name}: {key} must be '
                                 f'{expected!r}, got {actual!r}')
        if tuple(config.patch_size) != self.patch_size:
            raise ValueError('Capacity profile patch_size does not match training')

    def validate_model(self, model_config):
        for key, expected in self.model.items():
            if model_config.get(key) != expected:
                raise ValueError(f'Capacity profile {self.name}: model {key} must be '
                                 f'{expected!r}, got {model_config.get(key)!r}')

    def frame_limit(self, video_tokens_per_frame, action_tokens_per_frame, human_tokens):
        if min(video_tokens_per_frame, action_tokens_per_frame) <= 0 or human_tokens < 0:
            raise ValueError('Invalid video/action/human token counts')
        eligible = [r for r, h in self.passing_points if human_tokens <= h]
        if not eligible:
            raise ValueError(f'Full human condition ({human_tokens} tokens) exceeds '
                             f'capacity profile {self.name}; robot-only cropping '
                             'cannot provide a validated budget')
        budget = math.floor(max(eligible) * (1 - self.robot_token_margin))
        frames = budget // (video_tokens_per_frame + action_tokens_per_frame)
        if frames < 1:
            raise ValueError('Capacity budget cannot fit even one robot latent frame')
        return frames

    def batch_frame_limit(self, batch):
        robot, actions, human = (batch[k] for k in ('latents', 'actions', 'icl_latents'))
        if any(t.ndim != 5 or t.shape[0] != 1 for t in (robot, actions, human)):
            raise ValueError('Capacity crop requires [1, C, F, H, W] tensors')
        _, ph, pw = self.patch_size
        for tensor in (robot, human):
            if tensor.shape[-2] % ph or tensor.shape[-1] % pw:
                raise ValueError('Latent spatial dimensions must align with patch_size')
        video = (robot.shape[-2] // ph) * (robot.shape[-1] // pw)
        action = actions.shape[-2] * actions.shape[-1]
        human_tokens = human.shape[2] * (human.shape[-2] // ph) * (human.shape[-1] // pw)
        return self.frame_limit(video, action, human_tokens)


def crop_training_batch(batch, max_frames=None, *, capacity=None, generator=None):
    """Sample a fresh contiguous robot window on CPU, preserving full ICL/text.

    The video, already-aligned action histories and action masks share one
    latent-frame window. The frame cap and capacity profile compose by minimum.
    """
    if max_frames is None and capacity is None:
        return batch
    if max_frames is not None and max_frames <= 0:
        raise ValueError('max_train_frames must be positive')
    frames = batch['latents'].shape[2]
    for key in ('actions', 'actions_mask'):
        if batch[key].shape[2] != frames:
            raise ValueError(f'{key} does not align with robot latent frames')
    limit = max_frames if max_frames is not None else frames
    if capacity is not None:
        limit = min(limit, capacity.batch_frame_limit(batch))
    if frames <= limit:
        return batch
    start = torch.randint(frames - limit + 1, (1,), generator=generator).item()
    cropped = dict(batch)
    for key in ('latents', 'actions', 'actions_mask'):
        cropped[key] = batch[key][:, :, start:start + limit].contiguous()
    cropped['_window_crop'] = dict(original_frames=frames, retained_frames=limit,
                                   start=start, end=start + limit,
                                   profile=capacity.name if capacity else None)
    return cropped
