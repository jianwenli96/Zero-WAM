"""CPU-only latent shape reads for cost-aware distributed sample grouping.

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


@lru_cache(maxsize=131072)
def _action_tokens_per_frame(path):
    """The normal action adapter groups four frame_ids strides per latent frame."""
    with zipfile.ZipFile(path) as archive:
        name = next(n for n in archive.namelist() if n.endswith('/data.pkl'))
        payload = _MetadataOnly(io.BytesIO(archive.read(name))).load()
    ids = payload.get('frame_ids')
    if ids is None or len(ids) < 2:
        raise ValueError(f'Missing action alignment frame_ids in {path}')
    tokens = (int(ids[1]) - int(ids[0])) * 4
    if tokens <= 0:
        raise ValueError(f'Invalid action alignment frame_ids in {path}')
    return tokens


def training_sample_costs(dataset, max_frames=None, patch_size=(1, 2, 2), workers=12,
                          capacity=None):
    """Return costs in the flattened dataset's exact index order.

    Call on rank zero and broadcast once. No model tensors, Arrow action data,
    or video latent tensors are loaded. Avoid a stale disk cache: metadata is
    read anew for each invocation, then deduplicated within this invocation.
    """
    _latent_shape.cache_clear()
    _action_tokens_per_frame.cache_clear()
    pf, ph, pw = patch_size
    if capacity is not None and tuple(patch_size) != capacity.patch_size:
        raise ValueError('Sample cost patch_size does not match capacity profile')

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
        if capacity is not None:
            action_per_frame = _action_tokens_per_frame(str(
                task._latent_file(meta, task.used_video_keys[0])))
            frames = min(frames, capacity.frame_limit(
                video_per_frame, action_per_frame, human))
        # Preserve the legacy scheduling approximation when capacity crop is off.
        video = (frames // pf) * video_per_frame
        return 2 * (video + frames * action_per_frame) + human

    items = ((task, meta) for task in dataset._datasets for meta in task.new_metas)
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(cost, items))
    finally:
        _latent_shape.cache_clear()
        _action_tokens_per_frame.cache_clear()
