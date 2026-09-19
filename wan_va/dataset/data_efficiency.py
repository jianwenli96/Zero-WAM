"""Training sample cost estimates and aligned random robot window cropping.

Read PyTorch archive metadata without deserializing tensor storage. Costs are
only scheduling estimates: human ICL dropout and action layout can vary. They
never filter data or alter the selected sample's tensors.
"""
import io
import pickle
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import torch


class _MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) == ('collections', 'OrderedDict'):
            return OrderedDict
        return _discard_tensor

    def persistent_load(self, pid):
        return None


def _discard_tensor(*args, **kwargs):
    return None


@lru_cache(maxsize=131072)
def _latent_shape(path):
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        payload = _MetadataOnly(io.BytesIO(archive.read(name))).load()
    shape = tuple(int(payload[k]) for k in
                  ('latent_num_frames', 'latent_height', 'latent_width'))
    if min(shape) <= 0:
        raise ValueError(f'Invalid latent shape in {path}: {shape}')
    return shape


def training_sample_costs(dataset, max_frames=None, patch_size=(1, 2, 2), workers=12):
    """Return costs in the flattened dataset's exact index order.

    Call on rank zero and broadcast once. No model tensors, Arrow action data,
    or video latent tensors are loaded. Avoid a stale disk cache: metadata is
    read anew for each invocation, then deduplicated within this invocation.
    """
    _latent_shape.cache_clear()
    pf, ph, pw = patch_size

    def cost(item):
        task, meta = item
        robot_shapes = [_latent_shape(str(task._latent_file(meta, camera)))
                        for camera in task.used_video_keys]
        frames = robot_shapes[0][0]
        if any(s[0] != frames for s in robot_shapes):
            raise ValueError(f'Camera lengths disagree for {task.root}: {meta}')
        frames = min(frames, max_frames) if max_frames else frames
        video_per_frame = sum((h // ph) * (w // pw) for _, h, w in robot_shapes)
        sample = task._lookup_icl_sample(meta)
        hf, hh, hw = _latent_shape(str(task._human_latent_candidate(sample)))
        human = (hf // pf) * (hh // ph) * (hw // pw)
        action_per_frame = 16
        # Scheduling approximation; actual action layout varies by dataset.
        video = (frames // pf) * video_per_frame
        return 2 * (video + frames * action_per_frame) + human

    items = ((task, meta) for task in dataset._datasets for meta in task.new_metas)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(cost, items))
    finally:
        _latent_shape.cache_clear()


def crop_training_batch(batch, max_frames=None, *, generator=None):
    """Sample a fresh contiguous robot window on CPU, preserving full ICL/text.

    The video, already-aligned action histories and action masks share one latent-frame window. The configured frame cap applies only to the robot window.
    """
    if max_frames is None:
        return batch
    if max_frames <= 0:
        raise ValueError('max_train_frames must be positive')
    frames = batch['latents'].shape[2]
    for key in ('actions', 'actions_mask'):
        if batch[key].shape[2] != frames:
            raise ValueError(f'{key} does not align with robot latent frames')
    limit = max_frames
    if frames <= limit:
        return batch
    start = torch.randint(frames - limit + 1, (1,), generator=generator).item()
    cropped = dict(batch)
    for key in ('latents', 'actions', 'actions_mask'):
        cropped[key] = batch[key][:, :, start:start + limit].contiguous()
    cropped['_window_crop'] = dict(original_frames=frames, retained_frames=limit,
                                   start=start, end=start + limit)
    return cropped
