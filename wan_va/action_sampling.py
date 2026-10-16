"""Joint latent/action time resampling with a fixed number of actions per latent."""
import json
from pathlib import Path
import torch

ACTION_SAMPLING_CONFIG_KEY = '_zero_wam_action_sampling'


def validate_action_stride(stride):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f'action_stride must be a positive integer, got {stride!r}')
    return stride


def action_sampling_metadata(stride):
    stride = validate_action_stride(stride)
    return dict(version=2, action_stride=stride, latent_stride=stride,
                layout='aligned_time_blocks', actions_per_latent_changed=False,
                vae_encoding_changed=False)


def resample_training_batch(batch, stride=1):
    """Retain z0,zS,z2S,... and group A actions spaced S native ticks apart.

    Dense inputs prepend one all-zero A-action history block. The new history
    stays A tokens; each future A-token block spans S times the original time.
    Existing VAE latents are selected, never reinterpreted or averaged. Online
    inference must encode native-rate RGB and apply the same latent selection.
    """
    stride = validate_action_stride(stride)
    if stride == 1:
        return batch
    if '_action_stride' in batch:
        if batch['_action_stride'] != stride:
            raise ValueError('Cannot apply different action strides to an already resampled batch')
        return batch
    frames = batch['latents'].shape[2]
    actions = batch['actions']
    if actions.ndim != 5 or actions.shape[-1] != 1 or actions.shape[2] != frames:
        raise ValueError('Expected aligned native video and [B,C,F,A,1] actions')
    if batch['actions_mask'].shape != actions.shape:
        raise ValueError('Action mask does not match the native actions')
    if frames <= stride:
        raise ValueError(f'Sample needs at least {stride+1} native latent frames for action_stride={stride}')
    a = actions.shape[3]
    indices = torch.arange(0, frames, stride, device=actions.device)
    starts = a + torch.arange(indices.numel()-1, device=actions.device)*stride*a
    future = starts[:, None] + torch.arange(a, device=actions.device)[None]*stride
    action_indices = torch.cat([torch.arange(a, device=actions.device), future.flatten()])
    result = dict(batch)
    result['latents'] = batch['latents'].index_select(2, indices).contiguous()
    for key in ('actions', 'actions_mask'):
        result[key] = batch[key].flatten(2).index_select(2, action_indices).reshape(
            actions.shape[0], actions.shape[1], indices.numel(), a, 1).contiguous()
    result['_action_stride'] = stride
    return result


def inference_action_stride(model_root, requested=None):
    """Resolve the checkpoint timebase; legacy weights allow explicit timing probes."""
    path = Path(model_root).expanduser()/'transformer'/'config.json'
    metadata = json.loads(path.read_text()).get(ACTION_SAMPLING_CONFIG_KEY) if path.is_file() else None
    if metadata is None:
        return validate_action_stride(1 if requested is None else requested), False
    if not isinstance(metadata, dict) or metadata.get('version') not in (1, 2):
        raise ValueError(f'Unsupported action sampling metadata in {path}')
    saved = validate_action_stride(metadata.get('action_stride'))
    if metadata['version'] == 1 and saved != 1:
        raise ValueError('This checkpoint uses the obsolete packed_native_grid action thinning; '
                         'it does not implement longer action chunks. Retrain with the version-2 time resampling.')
    if requested is not None and validate_action_stride(requested) != saved:
        raise ValueError(f'action_stride={requested} conflicts with checkpoint action_stride={saved}')
    return saved, True
