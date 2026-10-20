"""Joint action/latent time sampling with a configurable action block length."""
import json
from pathlib import Path
import torch

ACTION_SAMPLING_CONFIG_KEY = '_zero_wam_action_sampling'


def validate_action_stride(stride):
    if isinstance(stride, bool) or not isinstance(stride, int) or stride < 1:
        raise ValueError(f'action_stride must be a positive integer, got {stride!r}')
    return stride


def latent_stride_for(stride, actions_per_latent=None, native_actions_per_latent=16):
    stride = validate_action_stride(stride)
    if actions_per_latent is None:
        return stride
    for name, value in [('actions_per_latent', actions_per_latent),
                        ('native_actions_per_latent', native_actions_per_latent)]:
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    span = actions_per_latent*stride
    if span % native_actions_per_latent:
        raise ValueError('Action block must span an integer number of native latent intervals: '
                         f'{actions_per_latent}*{stride}/{native_actions_per_latent}')
    return span//native_actions_per_latent


def sampling_datasets(value):
    """None means all selected sources; otherwise scope time changes by name."""
    if value is None or value == 'all':
        return None
    names = value.split(',') if isinstance(value, str) else list(value)
    names = [str(name).strip() for name in names]
    if not names or any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError('action_sampling_datasets requires unique nonempty dataset names')
    return names


def sampling_for_dataset(config, name):
    selected = sampling_datasets(getattr(config, 'action_sampling_datasets', None))
    if selected is not None and name not in selected:
        return 1, None
    return getattr(config, 'action_stride', 1), getattr(config, 'actions_per_latent', None)


def action_sampling_metadata(stride, actions_per_latent=None, native_actions_per_latent=16, *, datasets=None):
    latent_stride = latent_stride_for(stride, actions_per_latent, native_actions_per_latent)
    result = dict(version=2, action_stride=stride, latent_stride=latent_stride,
                  layout='aligned_time_blocks', actions_per_latent_changed=False,
                  vae_encoding_changed=False)
    if actions_per_latent is not None:
        result.update(version=3, actions_per_latent=actions_per_latent,
                      native_actions_per_latent=native_actions_per_latent,
                      actions_per_latent_changed=actions_per_latent != native_actions_per_latent)
    selected = sampling_datasets(datasets)
    if selected is not None:
        result['datasets'] = selected
    return result


def resample_training_batch(batch, stride=1, actions_per_latent=None):
    """Select native latents and pair each with A targets spaced stride ticks apart.

    Native A0=16, target A=32, stride=3 selects z0,z6,z12,... and action
    blocks [a0,a3,...,a93], [a96,a99,...,a189], ... . PAD expands to A zeros.
    The original native-rate VAE encoding is reused unchanged.
    """
    stride = validate_action_stride(stride)
    if stride == 1 and actions_per_latent is None:
        return batch
    actions = batch['actions']
    a = actions.shape[3] if actions_per_latent is None else actions_per_latent
    if '_action_stride' in batch:
        if batch['_action_stride'] != stride or batch.get('_actions_per_latent', actions.shape[3]) != a:
            raise ValueError('Cannot apply different action strides or block lengths to an already resampled batch')
        return batch
    frames = batch['latents'].shape[2]
    if actions.ndim != 5 or actions.shape[-1] != 1 or actions.shape[2] != frames:
        raise ValueError('Expected aligned native video and [B,C,F,A,1] actions')
    if batch['actions_mask'].shape != actions.shape:
        raise ValueError('Action mask does not match the native actions')
    native_a = actions.shape[3]
    latent_stride = latent_stride_for(stride, a, native_a)
    if frames <= latent_stride:
        raise ValueError(f'Sample needs at least {latent_stride+1} native latent frames for action_stride={stride}, actions_per_latent={a}')
    indices = torch.arange(0, frames, latent_stride, device=actions.device)
    starts = native_a + torch.arange(indices.numel()-1, device=actions.device)*stride*a
    future = starts[:, None] + torch.arange(a, device=actions.device)[None]*stride
    # Initial normalized PAD has no physical action history. Repeat its mask
    # to the requested length; retain the existing all-zero PAD values.
    padding = torch.arange(a, device=actions.device) % native_a
    action_indices = torch.cat([padding, future.flatten()])
    result = dict(batch)
    result['latents'] = batch['latents'].index_select(2, indices).contiguous()
    for key in ('actions', 'actions_mask'):
        result[key] = batch[key].flatten(2).index_select(2, action_indices).reshape(
            actions.shape[0], actions.shape[1], indices.numel(), a, 1).contiguous()
    result['_action_stride'] = stride
    result['_actions_per_latent'] = a
    return result


def _sampling_metadata(model_root):
    path = Path(model_root).expanduser()/'transformer'/'config.json'
    metadata = json.loads(path.read_text()).get(ACTION_SAMPLING_CONFIG_KEY) if path.is_file() else None
    if metadata is None:
        return None
    if not isinstance(metadata, dict) or metadata.get('version') not in (1, 2, 3):
        raise ValueError(f'Unsupported action sampling metadata in {path}')
    saved = validate_action_stride(metadata.get('action_stride'))
    if metadata['version'] == 1 and saved != 1:
        raise ValueError('This checkpoint uses the obsolete packed_native_grid action thinning; '
                         'it does not implement longer action chunks. Retrain with the version-2 time resampling.')
    if metadata['version'] == 3:
        expected = latent_stride_for(saved, metadata.get('actions_per_latent'), metadata.get('native_actions_per_latent'))
        if metadata.get('actions_per_latent') is None or metadata.get('latent_stride') != expected:
            raise ValueError('Checkpoint action block and latent stride contract is inconsistent')
    return metadata


def validate_inference_sampling_dataset(model_root, name):
    metadata = _sampling_metadata(model_root)
    if metadata and metadata.get('datasets') is not None and name not in metadata['datasets']:
        raise ValueError(f'Checkpoint time layout targets {metadata["datasets"]}, not {name}')


def inference_action_stride(model_root, requested=None):
    """Resolve the checkpoint timebase; legacy weights allow explicit timing probes."""
    metadata = _sampling_metadata(model_root)
    if metadata is None:
        return validate_action_stride(1 if requested is None else requested), False
    saved = metadata['action_stride']
    if requested is not None and validate_action_stride(requested) != saved:
        raise ValueError(f'action_stride={requested} conflicts with checkpoint action_stride={saved}')
    return saved, True


def inference_actions_per_latent(model_root, requested=None, native_actions_per_latent=16):
    metadata = _sampling_metadata(model_root)
    saved = native_actions_per_latent
    if metadata is not None and metadata['version'] == 3:
        if metadata['native_actions_per_latent'] != native_actions_per_latent:
            raise ValueError('Checkpoint native action timing conflicts with the configured feedback timebase')
        saved = metadata['actions_per_latent']
    if requested is not None:
        latent_stride_for(1, requested, requested)  # validate the integer independently
        if metadata is not None and requested != saved:
            raise ValueError(f'actions_per_latent={requested} conflicts with checkpoint actions_per_latent={saved}')
        saved = requested
    return saved, metadata is not None
