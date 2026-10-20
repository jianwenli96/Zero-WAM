"""Experimental multi-NPU, late-bound RTC with recorded PiperPika RGB.

Separate from the production single-request RTC protocol. No robot is connected.
Every process owns a complete frozen model, causal VAE state and history cache.
Video proposals retain their observation time; action RTC binds the controller's
latest committed plan after video generation. Only one action decode may bind
and commit at a time. During timed playback, future RGB is never supplied before
its playback timestamp. Offline warmup contexts are discarded before playback.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import multiprocessing as mp
import os
from pathlib import Path
import queue
import threading
import time
import traceback

import numpy as np

from evaluation.piperpika.rtc_devices import MIN_REPLICA_FREE_MIB, resolve_devices

CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')
DEFAULT_ROOT = '/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2'


@dataclass(frozen=True)
class Timing:
    control_hz: float = 30.
    action_stride: int = 3
    actions_per_latent: int = 32
    frames: int = 4
    receding_actions: int = 32
    observation_stride: int = 4

    def validate(self):
        if (self.control_hz != 30 or self.action_stride != 3
                or self.actions_per_latent != 32 or self.frames != 4
                or self.observation_stride != 4):
            raise ValueError('This probe validates F=4, A=32, stride=3 at 30 Hz only')
        if (isinstance(self.receding_actions, bool)
                or not isinstance(self.receding_actions, int)
                or not 0 < self.receding_actions < self.horizon):
            raise ValueError('Need an integer receding interval between 1 and H-1 targets')
        return self

    @property
    def horizon(self):
        return self.frames * self.actions_per_latent

    @property
    def period_ticks(self):
        return self.receding_actions * self.action_stride

    @property
    def period(self):
        return self.period_ticks / self.control_hz

    @property
    def action_hz(self):
        return self.control_hz / self.action_stride

    @property
    def history_ticks(self):
        # One trained latent/action interval stays 3.2 seconds regardless of s.
        return self.actions_per_latent * self.action_stride

    def protected_steps(self, elapsed, decode_budget, margin=.05):
        # Protect the right interpolation anchor as well as elapsed commands.
        return math.ceil((elapsed + decode_budget + margin) * self.action_hz) + 1


def observed_snapshot(start, issued, images, timing, history_intervals=2):
    """A fresh observed window ending exactly at the proposal's action origin.

    Moving the window is deliberate: every pair of adjacent clean video
    anchors still spans 32 executed targets / 96 native ticks. No partial
    interval is padded or stretched, and no speculative actions enter history.
    This rebuilds a truncated context, so it is not output-equivalent to the
    incremental causal KV/VAE stream.
    """
    if (isinstance(start, bool) or not isinstance(start, int) or start < 0
            or history_intervals not in (1, 2)):
        raise ValueError('Need a nonnegative action origin and 1 or 2 history intervals')
    tick = start * timing.action_stride
    if len(issued) != tick:
        raise ValueError('Snapshot requires the exact already-issued action prefix')
    count = min(history_intervals, start // timing.actions_per_latent)
    origin = start-count*timing.actions_per_latent
    chunks = []
    for action_start in range(origin, start, timing.actions_per_latent):
        native = action_start*timing.action_stride
        end = native+timing.history_ticks
        chunks.append(dict(start=action_start,
            actions=np.stack([issued[t] for t in range(native, end, timing.action_stride)]),
            images=[images[t] for t in range(native+timing.observation_stride,
                                             end+1, timing.observation_stride)]))
    return dict(origin=origin, start=start, initial_image=images[origin*timing.action_stride],
                history=chunks, anchor_ticks=[(origin+i*timing.actions_per_latent)*timing.action_stride
                                               for i in range(count+1)])


class PlanLedger:
    """Absolute 10-Hz targets, separate from the immutable issued 30-Hz log."""

    def __init__(self, initial, timing):
        self.timing = timing
        self.plan = self._actions(initial)
        if len(self.plan) != timing.horizon:
            raise ValueError('Bootstrap plan has the wrong horizon')
        self.version = 0
        self.issued = []
        self.commits = []

    @staticmethod
    def _actions(value):
        value = np.array(value, dtype=np.float32, copy=True)
        if value.ndim != 2 or value.shape[1] != 14 or not np.isfinite(value).all():
            raise ValueError('Need finite PiperPika [T,14] absolute actions')
        return value

    def issue(self, tick):
        if tick != len(self.issued):
            raise ValueError('Controller ticks must be consecutive')
        left, fraction = divmod(tick, self.timing.action_stride)
        if left >= len(self.plan):
            raise RuntimeError('Committed action coverage exhausted')
        value = self.plan[left].copy()
        if fraction:
            if left + 1 >= len(self.plan):
                raise RuntimeError('Missing the next interpolation anchor')
            weight = fraction / self.timing.action_stride
            value[:12] = value[:12] * (1-weight) + self.plan[left+1, :12] * weight
        # Grippers are zero-order held, matching the production client.
        self.issued.append(value)
        return value

    def bind(self, start, frozen):
        if start < 0 or start >= len(self.plan):
            raise RuntimeError('Candidate observation is outside committed coverage')
        overlap = min(self.timing.horizon, len(self.plan)-start)
        if not 0 <= frozen <= overlap:
            raise RuntimeError(f'Frozen prefix d={frozen} exceeds overlap={overlap}')
        target = np.zeros((self.timing.horizon, 14), dtype=np.float32)
        target[:overlap] = self.plan[start:start+overlap]
        return dict(reference=target, overlap=overlap, frozen=frozen,
                    plan_version=self.version)

    def commit(self, start, actions, binding, tick):
        actions = self._actions(actions)
        if len(actions) != self.timing.horizon:
            raise RuntimeError('Candidate changed its prediction horizon')
        if binding['plan_version'] != self.version:
            raise RuntimeError('Plan changed during serial late-bound action decode')
        frozen = binding['frozen']
        # A right anchor may already have contributed to an issued interpolation.
        protected_end = ((tick-1+self.timing.action_stride-1)//self.timing.action_stride+1
                         if tick else 0)
        if start+frozen < protected_end:
            raise RuntimeError('Candidate would change an already-used interpolation anchor')
        if not np.array_equal(actions[:frozen], self.plan[start:start+frozen]):
            raise RuntimeError('Candidate did not preserve the committed prefix exactly')
        begin, end = start+frozen, start+len(actions)
        old = self.plan
        if end > len(old):
            self.plan = np.concatenate([old, np.zeros((end-len(old), 14), np.float32)])
        self.plan[begin:end] = actions[frozen:]
        self.version += 1
        self.commits.append(dict(start=start, first_mutable_target=begin,
                                 end=end, controller_tick=tick, version=self.version))


def load_recording(args, timing):
    import cv2
    cv2.setNumThreads(1)
    task = args.root/'eval/piperpika'
    episode = next(ep for ep in map(json.loads, (task/'meta/episodes.jsonl').read_text().splitlines())
                   if ep['episode_index'] == args.episode)
    count = args.ticks + timing.observation_stride
    if args.start + count > episode['length']:
        raise ValueError(f'Replay needs {count} frames, episode has {episode["length"]-args.start}')
    # Snapshot windows can start between the original every-fourth-frame grid.
    # VAE inputs within each window still use the trained 4-tick spacing.
    capture_stride = 1 if getattr(args, 'history_mode', 'incremental') == 'snapshot' else timing.observation_stride
    indices = range(0, count, capture_stride)
    images = {tick: {} for tick in indices}
    for camera in CAMERAS:
        cap = cv2.VideoCapture(str(task/'videos'/f'chunk-{args.episode//1000:03d}'/
                                   camera/f'episode_{args.episode:06d}.mp4'))
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)
            for offset in range(count):
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {camera} at {args.start+offset}')
                if offset % capture_stride == 0:
                    images[offset][camera] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (288, 224))
        finally:
            cap.release()
    human = args.human_latent or args.root/'human_latents'/Path(episode['icl']['human_video_path']).with_suffix('.pth')
    if not human.is_file():
        raise FileNotFoundError(human)
    return images, dict(prompt=episode['tasks'][0], icl_latent_path=str(human),
                        seed=args.seed+args.episode*10000+args.start)


def share_recording(images, path):
    """Read-only RGB store; IPC then sends timestamp indices and action arrays.

    The coordinator still selects only captures at or before each observation
    timestamp. Offline warmup may read ahead, and its contexts are discarded.
    """
    shape = images[0][CAMERAS[0]].shape
    array = np.lib.format.open_memmap(path, mode='w+', dtype=np.uint8,
                                     shape=(max(images)+1, len(CAMERAS), *shape))
    for tick, image in images.items():
        for index, camera in enumerate(CAMERAS):
            array[tick, index] = image[camera]
    array.flush()
    del array
    return {tick:tick for tick in images}


def recording_image(image, recording):
    if isinstance(image, (int, np.integer)):
        if recording is None or not 0 <= image < len(recording):
            raise ValueError('RGB capture index is outside the shared recording')
        return {camera:recording[image, index] for index, camera in enumerate(CAMERAS)}
    return image


def make_runtime(options):
    """All torch imports are child-local, after device visibility is selected."""
    import torch
    from einops import rearrange
    from wan_va.wan_va_server import VA_Server
    from wan_va.configs.va_piperpika_train_cfg import va_piperpika_train_cfg
    from wan_va.dataset.piperpika import model_quantiles, verify_checkpoint_contract
    from wan_va.action_sampling import (inference_action_stride, inference_actions_per_latent,
                                        validate_inference_sampling_dataset)
    from wan_va.rtc.server import RTCSession
    from wan_va.rtc.guidance import guided_noise_velocity, prefix_weights
    from wan_va.modules.icl_model import PREDICTION_CACHE_TYPE
    from wan_va.utils import data_seq_to_patch

    torch.set_num_threads(1)
    torch.cuda.set_device(0)
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    print(f'Device visibility={os.environ.get("ASCEND_RT_VISIBLE_DEVICES")}; '
          f'free={free_bytes/2**30:.2f} GiB / {total_bytes/2**30:.2f} GiB', flush=True)
    if free_bytes < MIN_REPLICA_FREE_MIB*2**20:
        raise RuntimeError('Need at least 48 GiB free before loading a complete RTC replica')
    verify_checkpoint_contract(options['model'], options['root'], False)
    validate_inference_sampling_dataset(options['model'], 'piperpika')
    cfg = deepcopy(va_piperpika_train_cfg)
    cfg.model_path = options['model']
    cfg.save_root = options['output']
    cfg.norm_stat = model_quantiles(Path(options['root'])/'train/piperpika/meta/action_stats.json')
    cfg.action_stride, _ = inference_action_stride(cfg.model_path, 3)
    cfg.native_action_per_frame = cfg.action_per_frame
    cfg.action_per_frame, _ = inference_actions_per_latent(cfg.model_path, 32, cfg.native_action_per_frame)
    cfg.actions_per_latent = 32
    cfg.frame_chunk_size = 4
    cfg.num_inference_steps = options['video_steps']
    cfg.action_num_inference_steps = options['action_steps']
    cfg.enable_rtc = True
    cfg.enable_offload = False
    cfg.rank = cfg.local_rank = 0
    cfg.world_size = 1
    # Used only for model initialization and tensor layout. The experimental
    # coordinator below owns its distinct multi-inflight timing contract.
    cfg.rtc_execution_horizon = 32
    cfg.rtc_initial_latency = 0.
    cfg.rtc_attn_window = 4
    cfg.icl_guidance_scale = 1.

    class Runtime(VA_Server):
        @torch.no_grad()
        def reset_probe(self, metadata, image):
            torch.manual_seed(metadata['seed'])
            self._reset_icl(metadata['prompt'], True, '', metadata['icl_latent_path'], -1., 1.)
            names = ('type_ids_cache', 'seq_ids_cache', 'frame_ids_cache', 'cache_type_ids_cache')
            self.task_metadata = {name:getattr(self.transformer, name) for name in names}
            self.task_kv = [dict(block.attn1.attn_caches[self.cache_name])
                            for block in self.transformer.blocks]
            self.reset_robot(image)

        @torch.no_grad()
        def reset_robot(self, image):
            # The human task prefix is immutable. All robot and predicted KV,
            # plus the causal VAE state, belong to the new observed window.
            for name, value in self.task_metadata.items():
                setattr(self.transformer, name, value)
            for block, cache in zip(self.transformer.blocks, self.task_kv):
                block.attn1.attn_caches[self.cache_name] = dict(cache)
            self.init_latent = self.last_predicted_latents = self.last_predicted_actions = None
            self._rtc_guidance = None
            self.chunk_idx = 0
            if self.streaming_vae_half is not None:
                self.streaming_vae_half.clear_cache()
            self.session = RTCSession(self, self.rtc_config)
            self.session._prefill({'obs': image})
            self.history_origin = 0
            self.history_actions = 0
            self._rtc_stream_position = (1, 1)

        @torch.no_grad()
        def prepare_snapshot(self, snapshot):
            self.reset_robot(snapshot['initial_image'])
            self.history_origin = self.history_actions = snapshot['origin']
            self.advance_history(snapshot['history'])
            if self.history_actions != snapshot['start']:
                raise RuntimeError('Snapshot does not end at its observation time')

        @torch.no_grad()
        def advance_history(self, chunks):
            self.transformer.clear_prediction_cache(-1, self.cache_name)
            for chunk in chunks:
                start = chunk['start']
                if start != self.history_actions:
                    raise RuntimeError(f'History gap: expected {self.history_actions}, got {start}')
                count = len(chunk['actions'])
                if count not in (32,64) or len(chunk['images']) != count*3//4:
                    raise RuntimeError('History update requires a complete 32/64-target executed request')
                _, executed = self.session._actions(chunk['actions'], count, 'executed_actions')
                latent = self._encode_obs({'obs': chunk['images']})
                if latent.shape[2] != 6*(count//32):
                    raise RuntimeError(f'Unexpected native feedback latent count: {latent.shape}')
                latent = latent[:, :, 5::6].contiguous()
                # Production s=64 feeds two latent/action blocks in ONE phase.
                # The frame origin advances by two; the request phase by one.
                self._rtc_stream_position = ((start-self.history_origin)//32+1,self.session.phase)
                self.transformer(self._prepare_icl_stream(latent, 0, 'video', 0),
                                 update_cache=1, cache_name=self.cache_name, mode='forward_latent_only')
                self.transformer(self._prepare_icl_stream(executed, 0, 'action', 0),
                                 update_cache=1, cache_name=self.cache_name,
                                 mode='forward_action_only', clean_window_cache=True)
                self.history_actions += count
                self.session.phase += 1
            self._rtc_stream_position = ((self.history_actions-self.history_origin)//32+1,self.session.phase)

        @torch.no_grad()
        def video_proposal(self):
            self.proposal = torch.randn(1, 48, 4, self.latent_height, self.latent_width,
                                        device=self.device, dtype=self.dtype)
            self.pending_actions = torch.randn(1, 30, 4, 32, 1, device=self.device, dtype=self.dtype)
            _ = torch.randn_like(self.pending_actions)[:, ~self.action_mask]
            self.scheduler.set_timesteps(self.job_config.num_inference_steps)
            self.action_scheduler.set_timesteps(self.job_config.action_num_inference_steps)
            steps = list(self.scheduler.timesteps)+[0]
            for index, timestep in enumerate(steps):
                last = index == len(steps)-1
                output = self.transformer(
                    self._prepare_icl_stream(self.proposal.clone(), timestep, 'video', PREDICTION_CACHE_TYPE),
                    update_cache=int(last), cache_name=self.cache_name, mode='forward_latent_only')
                if not last:
                    noise = data_seq_to_patch(self.transformer.patch_size, output, 4,
                                              self.latent_height, self.latent_width)
                    self.proposal = self.scheduler.step(noise, timestep, self.proposal, return_dict=False)

        @torch.no_grad()
        def decode_actions(self, binding, *, guidance=True):
            if guidance:
                _, target = self.session._actions(binding['reference'], 128, 'reference')
                weights = prefix_weights(binding['frozen'], binding['overlap'], 128,
                                         device=self.device).reshape(1, 1, 4, 32, 1)
                weights = weights*self.action_mask.reshape(1, -1, 1, 1, 1)
            actions = self.pending_actions
            for index, timestep in enumerate(self.action_scheduler.timesteps):
                value = actions.clone()
                value[:, ~self.action_mask] = 0

                def velocity_fn(sample):
                    output = self.transformer(
                        self._prepare_icl_stream(sample, timestep, 'action', PREDICTION_CACHE_TYPE),
                        update_cache=0, cache_name=self.cache_name, mode='forward_action_only')
                    return rearrange(output, '1 (f n) c -> 1 c f n 1', f=4)

                velocity = (guided_noise_velocity(value, velocity_fn,
                    self.action_scheduler.sigmas[index].item(), target, weights, 5.)
                    if guidance else velocity_fn(value))
                actions = self.action_scheduler.step(velocity, timestep, actions, return_dict=False)
            actions[:, ~self.action_mask] = 0
            raw = self.postprocess_action(actions).transpose(1, 2, 0).reshape(128, 14).copy()
            # Restore absolute commands after de-normalization, including its
            # floating-point roundoff, as the production RTC runtime does.
            if guidance:
                raw[:binding['frozen']] = binding['reference'][:binding['frozen']]
            if not np.isfinite(raw).all():
                raise RuntimeError('Action decode returned nonfinite commands')
            return raw

    return Runtime(cfg)


def worker_main(device, connection, options):
    os.environ['ASCEND_RT_VISIBLE_DEVICES'] = str(device)
    os.environ['ASCEND_DEVICE_ID'] = '0'
    os.environ['LOCAL_RANK'] = '0'
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    log = open(Path(options['output'])/f'worker_npu{device}.log', 'a', buffering=1)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    try:
        runtime = make_runtime(options)
        import torch
        from wan_va.rtc.cache import request_cache
        recording = (np.load(options['recording_store'], mmap_mode='r')
                     if options.get('recording_store') else None)

        def unpack_history(chunks):
            return [dict(chunk, images=[recording_image(image, recording) for image in chunk['images']])
                    for chunk in chunks]

        def measured(fn):
            torch.cuda.synchronize()
            started = time.monotonic()
            result = fn()
            torch.cuda.synchronize()
            return result, time.monotonic()-started

        connection.send(dict(kind='LOADED', device=device))
        while True:
            command = connection.recv()
            if command['kind'] == 'STOP':
                break
            if command['kind'] == 'CANCEL':
                connection.send(dict(kind='CANCELLED', device=device, token=command.get('token')))
                continue
            if command['kind'] == 'RESET':
                _, elapsed = measured(lambda: runtime.reset_probe(command['metadata'], recording_image(command['image'], recording)))
                connection.send(dict(kind='RESET_READY', seconds=elapsed, device=device))
            elif command['kind'] == 'RESET_ROBOT':
                if 'seed' in command:
                    torch.manual_seed(command['seed'])
                _, elapsed = measured(lambda: runtime.reset_robot(recording_image(command['image'], recording)))
                connection.send(dict(kind='RESET_READY', seconds=elapsed, device=device))
            elif command['kind'] == 'HISTORY':
                with request_cache():
                    _, seconds = measured(lambda: runtime.advance_history(unpack_history(command['history'])))
                connection.send(dict(kind='HISTORY_READY', device=device,
                                     history_actions=runtime.history_actions, history_seconds=seconds))
            elif command['kind'] == 'CHECK':
                # Compare with the existing production sampler using identical
                # seed, observed context, soft mask and absolute-action prefix.
                from wan_va.rtc.guidance import prefix_weights
                metadata, image, reference = command['metadata'], recording_image(command['image'], recording), command['reference']
                # NPU attention/input-VJP reductions are not bitwise stable in
                # the default mode, even for two production sampler repeats.
                # Check structural equivalence with deterministic algorithms,
                # then restore the original mode before measuring throughput.
                previous_determinism = torch.are_deterministic_algorithms_enabled()
                torch.use_deterministic_algorithms(True)
                with torch.no_grad():
                    runtime.reset_probe(metadata, image)
                    _, target = runtime.session._actions(reference, 128, 'reference')
                    weights = prefix_weights(61, 96, 128, device=runtime.device).reshape(1, 1, 4, 32, 1)
                    weights = weights*runtime.action_mask.reshape(1, -1, 1, 1, 1)
                    runtime._rtc_guidance = dict(target=target, weights=weights, delay=61, beta=5.)
                    with request_cache():
                        original = runtime._infer_icl({}).transpose(1, 2, 0).reshape(128, 14).copy()
                    original[:61] = reference[:61]
                    original_video = runtime.last_predicted_latents.detach().clone()
                    runtime.reset_probe(metadata, image)
                    with request_cache():
                        runtime.advance_history([])
                        runtime.video_proposal()
                        split = runtime.decode_actions(dict(reference=reference, frozen=61, overlap=96))
                    video_error = float((runtime.proposal-original_video).abs().max().item())
                    action_error = float(np.max(np.abs(split-original)))
                    if video_error > 1e-6 or action_error > 1e-6:
                        raise RuntimeError(f'Split sampler differs from production: video={video_error}, action={action_error}')
                    runtime.reset_probe(metadata, image)
                torch.use_deterministic_algorithms(previous_determinism)
                connection.send(dict(kind='CHECK_READY', device=device,
                                     deterministic_algorithms=True,
                                     video_max_abs_error=video_error, action_max_abs_error=action_error))
            elif command['kind'] == 'WORLD':
                started = time.monotonic()
                with request_cache():
                    snapshot = command.get('snapshot')
                    if snapshot is not None:
                        snapshot = dict(snapshot,
                            initial_image=recording_image(snapshot['initial_image'], recording),
                            history=unpack_history(snapshot['history']))
                    prepare = (lambda: runtime.prepare_snapshot(snapshot)) if snapshot is not None else (
                        lambda: runtime.advance_history(unpack_history(command['history'])))
                    _, history_seconds = measured(prepare)
                    _, video_seconds = measured(runtime.video_proposal)
                    connection.send(dict(kind='WORLD_READY', job=command['job'], device=device,
                        history_actions=runtime.history_actions, history_seconds=history_seconds,
                        video_seconds=video_seconds, worker_started=started,
                        snapshot_origin=None if snapshot is None else snapshot['origin'],
                        anchor_ticks=None if snapshot is None else snapshot['anchor_ticks'],
                        world_ready=time.monotonic()))
                    binding = connection.recv()
                    if binding['kind'] == 'STOP':
                        break
                    if binding['kind'] == 'CANCEL':
                        connection.send(dict(kind='CANCELLED', device=device, token=binding.get('token')))
                        continue
                    if binding['kind'] != 'ACTION' or binding['job'] != command['job']:
                        raise RuntimeError('Action binding does not match the frozen video proposal')
                    actions, action_seconds = measured(lambda: runtime.decode_actions(binding['binding']))
                    connection.send(dict(kind='ACTION_READY', job=command['job'], device=device,
                        actions=actions, action_seconds=action_seconds,
                        world_total_seconds=time.monotonic()-started,
                        peak_memory_bytes=torch.cuda.max_memory_allocated()))
            else:
                raise RuntimeError(f'Unknown worker command: {command["kind"]}')
    except BaseException:
        error = traceback.format_exc()
        print(error, flush=True)
        try:
            connection.send(dict(kind='ERROR', device=device, error=error))
        except (BrokenPipeError, EOFError):
            pass
        raise
    finally:
        connection.close()
        log.close()


def percentiles(values):
    return ({f'p{p}': float(np.percentile(values, p)) for p in (50, 95, 99)}
            if values else {})


class WorkerPool:
    def __init__(self, args):
        self.events = queue.Queue()
        self.closed = False
        self.workers = {}
        context = mp.get_context('spawn')
        options = {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()}
        for device in args.devices:
            parent, child = context.Pipe()
            process = context.Process(target=worker_main, args=(device, child, options),
                                      name=f'rtc-probe-npu{device}')
            process.start()
            child.close()
            writer = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f'rtc-send-{device}')
            self.workers[device] = dict(process=process, pipe=parent, writer=writer,
                                        writes=[], history_actions=0)
            threading.Thread(target=self._read, args=(device, parent), daemon=True).start()

    def _read(self, device, connection):
        try:
            while not self.closed:
                message = connection.recv()
                message['received'] = time.monotonic()
                self.events.put(message)
        except (EOFError, OSError):
            if not self.closed:
                self.events.put(dict(kind='ERROR', device=device, error='Worker connection closed'))

    def send(self, device, message):
        worker = self.workers[device]
        worker['writes'].append(worker['writer'].submit(worker['pipe'].send, message))

    def receive(self, timeout=0.):
        for device, worker in self.workers.items():
            for future in worker['writes']:
                if future.done():
                    future.result()
            worker['writes'] = [f for f in worker['writes'] if not f.done()]
            if not worker['process'].is_alive():
                raise RuntimeError(f'NPU {device} worker exited: {worker["process"].exitcode}')
        try:
            message = self.events.get(timeout=timeout)
        except queue.Empty:
            return None
        if message['kind'] == 'ERROR':
            raise RuntimeError(f'NPU {message["device"]}: {message["error"]}')
        return message

    def collect(self, kind, count, timeout=600):
        results = []
        deadline = time.monotonic()+timeout
        while len(results) < count:
            if time.monotonic() > deadline:
                raise TimeoutError(f'Waiting for {kind}: {len(results)}/{count}')
            message = self.receive(.2)
            if message is None:
                continue
            if message['kind'] != kind:
                raise RuntimeError(f'Expected {kind}, received {message["kind"]}')
            results.append(message)
        return results

    def close(self):
        self.closed = True
        for device, worker in self.workers.items():
            if worker['process'].is_alive():
                self.send(device, dict(kind='STOP'))
        for worker in self.workers.values():
            worker['process'].join(timeout=10)
            if worker['process'].is_alive():
                worker['process'].terminate()
                worker['process'].join(timeout=5)
            worker['writer'].shutdown(wait=False, cancel_futures=True)
            worker['pipe'].close()


def initialize(pool, args, metadata, images, timing):
    print(f'Loading {len(args.devices)} complete model replicas on NPUs {args.devices}', flush=True)
    pool.collect('LOADED', len(args.devices))
    print('Models loaded; caching the human prompt and initial observation', flush=True)
    for device in args.devices:
        pool.send(device, dict(kind='RESET', metadata=metadata, image=images[0]))
    reset = pool.collect('RESET_READY', len(args.devices))
    stats = json.loads((args.root/'train/piperpika/meta/action_stats.json').read_text())['norm_stats']
    low = stats['action.arm.position']['q01']+stats['action.effector.position']['q01']
    high = stats['action.arm.position']['q99']+stats['action.effector.position']['q99']
    reference = np.repeat(((np.array(low)+np.array(high))*.5)[None], timing.horizon, axis=0).astype(np.float32)
    warmup = []
    for round_id in range(args.warmup_rounds):
        for device in args.devices:
            pool.send(device, dict(kind='WORLD', job=-round_id-1, history=[]))
        worlds = pool.collect('WORLD_READY', len(args.devices))
        for world in worlds:
            pool.send(world['device'], dict(kind='ACTION', job=world['job'], binding=dict(
                reference=reference, frozen=61, overlap=96, plan_version=0)))
        actions = pool.collect('ACTION_READY', len(args.devices))
        for action in actions:
            world = next(w for w in worlds if w['device'] == action['device'])
            warmup.append(dict(device=action['device'], round=round_id,
                history_seconds=world['history_seconds'], video_seconds=world['video_seconds'],
                action_seconds=action['action_seconds'], peak_memory_bytes=action['peak_memory_bytes']))
        print('Warmup '+str(round_id+1)+': '+json.dumps(warmup[-len(args.devices):]), flush=True)
    if not args.profile_only:
        # Compile and exercise the steady-state history/attention shapes before
        # starting the controller clock. These offline warmups are discarded.
        print('Warming native-rate feedback and steady-state cache shapes', flush=True)
        for row in profile_history(pool, args, images, timing, reference):
            warmup.append(dict(row, round=args.warmup_rounds-1, includes_history=True))
    # Warmup outputs and their speculative caches never become robot history.
    for device in args.devices:
        pool.send(device, dict(kind='RESET', metadata=metadata, image=images[0]))
    pool.collect('RESET_READY', len(args.devices))
    equivalence = []
    if args.check_equivalence:
        print('Checking split sampler against production action-only RTC', flush=True)
        for device in args.devices:
            pool.send(device, dict(kind='CHECK', metadata=metadata, image=images[0], reference=reference))
        equivalence = pool.collect('CHECK_READY', len(args.devices))
        print('Equivalence: '+json.dumps(equivalence), flush=True)
    return reset, warmup, equivalence


def profile_history(pool, args, images, timing, reference):
    chunks = []
    for block in range(2):
        start = block*timing.history_ticks
        chunks.append(dict(start=block*32, actions=reference[:32],
            images=[images[tick] for tick in range(start+4, start+timing.history_ticks+1, 4)]))
    for device in args.devices:
        command = dict(kind='WORLD', job=0, history=chunks)
        if getattr(args, 'history_mode', 'incremental') == 'snapshot':
            issued = np.repeat(reference[:1], 2*timing.history_ticks, axis=0)
            command = dict(kind='WORLD', job=0, history=[], snapshot=observed_snapshot(
                64, issued, images, timing, getattr(args, 'snapshot_history_intervals', 2)))
        pool.send(device, command)
    worlds = pool.collect('WORLD_READY', len(args.devices))
    for world in worlds:
        pool.send(world['device'], dict(kind='ACTION', job=0, binding=dict(
            reference=reference, frozen=61, overlap=96, plan_version=0)))
    actions = pool.collect('ACTION_READY', len(args.devices))
    return [dict(device=a['device'],
                 **{k:w[k] for k in ('history_seconds', 'video_seconds')},
                 action_seconds=a['action_seconds'], peak_memory_bytes=a['peak_memory_bytes'])
            for a in actions for w in worlds if a['device'] == w['device']]


def run_pipeline(pool, args, images, timing, warmup):
    """One controller clock, concurrent worlds, ordered late action bindings."""
    bootstrap_device = args.devices[0]
    pool.send(bootstrap_device, dict(kind='WORLD', job=0, history=[]))
    bootstrap_world = pool.collect('WORLD_READY', 1)[0]
    pool.send(bootstrap_device, dict(kind='ACTION', job=0, binding=dict(
        reference=np.zeros((timing.horizon, 14), np.float32), frozen=0,
        overlap=0, plan_version=0)))
    bootstrap = pool.collect('ACTION_READY', 1)[0]
    ledger = PlanLedger(bootstrap['actions'], timing)
    np.save(args.output/'bootstrap_actions.npy', bootstrap['actions'])
    stable_warmup = [r for r in warmup if r['round'] == args.warmup_rounds-1]
    action_latencies = [r['action_seconds'] for r in stable_warmup]
    jobs, measurements, lateness, launches = {}, [], [], []
    busy = {device: False for device in args.devices}
    history_prefetch = []
    available_history = 0
    active_action = None
    next_action_job = 1
    origin = time.monotonic()
    error = None
    history_mode = getattr(args, 'history_mode', 'incremental')
    if history_mode == 'incremental' and timing.receding_actions % timing.actions_per_latent:
        raise ValueError('Incremental history requires whole 32-target intervals; use snapshot history for fractional replans')
    min_coverage = math.inf
    print(f'Controller starts: {timing.control_hz:g} Hz, '
          f'launch interval={timing.period:.3f}s, horizon={timing.horizon/timing.action_hz:.1f}s', flush=True)

    def histories(device, target):
        chunks = []
        start = pool.workers[device]['history_actions']
        for model_start in range(start, target, timing.actions_per_latent):
            native = model_start*timing.action_stride
            end = native+timing.history_ticks
            chunks.append(dict(start=model_start,
                actions=np.stack([ledger.issued[t] for t in range(native, end, timing.action_stride)]),
                images=[images[t] for t in range(native+timing.observation_stride,
                                                 end+1, timing.observation_stride)]))
        return chunks

    def commit_result(job_id, now):
        job = jobs[job_id]
        message = job['result']
        elapsed = now-job['obs_time']
        protected = job['binding']['frozen']
        if elapsed*timing.action_hz > protected-1:
            raise RuntimeError(f'RTC forecast missed for job {job_id}: {elapsed:.3f}s')
        start = job['start']
        future_end = min(len(ledger.plan), start+timing.horizon)
        begin = start+protected
        delta = message['actions'][protected:future_end-start]-ledger.plan[begin:future_end]
        ledger.commit(start, message['actions'], job['binding'], len(ledger.issued))
        world = job['world']
        row = dict(job=job_id, device=message['device'], observation_action=start,
            observation_seconds=job['obs_time']-origin,
            launch_seconds=job['launch_time']-origin, commit_seconds=now-origin,
            action_ready_seconds=job['result_received']-origin,
            world_ready_seconds=job['world']['world_ready']-origin,
            action_bind_seconds=job['bind_time']-origin,
            result_hold_seconds=now-job['result_received'],
            scheduled_commit_seconds=(job['commit_at']-origin) if args.handoff_latency else None,
            observation_to_commit_seconds=elapsed,
            history_seconds=world['history_seconds'], video_seconds=world['video_seconds'],
            action_seconds=message['action_seconds'],
            action_wait_seconds=job['bind_time']-world['world_ready'],
            action_budget_seconds=job['action_budget'],
            binding_plan_version=job['binding']['plan_version'],
            frozen_targets=protected, overlap_targets=job['binding']['overlap'],
            first_mutable_target=start+protected,
            observation_to_first_mutable_seconds=protected/timing.action_hz,
            existing_future_update_max_abs=float(np.max(np.abs(delta))) if delta.size else 0.,
            peak_memory_bytes=message['peak_memory_bytes'])
        row.update(history_mode=history_mode, snapshot_origin=world.get('snapshot_origin'),
                   anchor_ticks=world.get('anchor_ticks'))
        measurements.append(row)
        # File/stdout writes on shared storage may block for several control
        # periods. Persist results after the timed loop; the authoritative
        # committed plan and issued-action log stay in memory during control.

    def drain():
        nonlocal active_action, next_action_job
        while True:
            message = pool.receive()
            if message is None:
                break
            if message['kind'] == 'HISTORY_READY':
                if busy[message['device']] != 'HISTORY':
                    raise RuntimeError('Unrequested history update')
                pool.workers[message['device']]['history_actions'] = message['history_actions']
                busy[message['device']] = False
                history_prefetch.append({k:message[k] for k in (
                    'device', 'history_actions', 'history_seconds')})
                continue
            job = jobs[message['job']]
            if message['device'] != job['device']:
                raise RuntimeError('Worker identity changed for a pending job')
            if message['kind'] == 'WORLD_READY':
                if message['history_actions'] != job['start']:
                    raise RuntimeError('Video proposal was built against the wrong history time')
                pool.workers[message['device']]['history_actions'] = message['history_actions']
                job['world'] = message
            elif message['kind'] == 'ACTION_READY':
                if message['job'] != active_action:
                    raise RuntimeError('Action decodes completed out of the committed dependency order')
                job['result'] = message
                job['result_received'] = time.monotonic()
                action_latencies.append(message['action_seconds'])
                # Gradients and device context are no longer needed by this
                # result. The replica may prefetch real history while the
                # controller waits for the scheduled atomic handoff.
                busy[message['device']] = False
            else:
                raise RuntimeError(f'Unexpected controller event {message["kind"]}')

        if active_action is not None:
            active = jobs[active_action]
            now = time.monotonic()
            if args.handoff_latency and now > active['commit_at']+1/timing.control_hz:
                raise RuntimeError(f'Missed fixed handoff deadline for job {active_action}')
            if 'result' in active and (not args.handoff_latency or now >= active['commit_at']):
                commit_result(active_action, now)
                active_action = None
                next_action_job += 1

        candidate = jobs.get(next_action_job)
        if active_action is None and candidate is not None and 'world' in candidate:
            now = time.monotonic()
            # Same bounded max-history forecast as RTC, applied to the serial
            # action stage; elapsed world time is already known at this point.
            budget = max(action_latencies[-10:])*args.latency_multiplier
            elapsed = now-candidate['obs_time']
            protected_duration = elapsed+budget
            if args.handoff_latency:
                protected_duration = max(protected_duration, args.handoff_latency)
            frozen = timing.protected_steps(protected_duration, 0, args.deadline_margin)
            binding = ledger.bind(candidate['start'], frozen)
            candidate.update(binding=binding, bind_time=now, action_budget=budget)
            pool.send(candidate['device'], dict(kind='ACTION', job=next_action_job, binding=binding))
            active_action = next_action_job

    def maintain_idle_histories():
        # Only already captured frames and already issued commands enter these
        # updates. Busy workers retain the immutable context of their proposal.
        if history_mode == 'snapshot':
            return
        for device in args.devices:
            if not busy[device] and pool.workers[device]['history_actions'] < available_history:
                busy[device] = 'HISTORY'
                pool.send(device, dict(kind='HISTORY', history=histories(device, available_history)))

    try:
        for tick in range(args.ticks):
            deadline = origin+tick/timing.control_hz
            while True:
                drain()
                maintain_idle_histories()
                remaining = deadline-time.monotonic()
                if remaining <= 0:
                    break
                # Poll completions independently of the next 30-Hz control tick.
                time.sleep(min(remaining, .002))
            drain()
            if tick and tick % timing.period_ticks == 0:
                job_id = tick//timing.period_ticks
                preferred = (job_id-1) % len(args.devices)
                ordered = args.devices[preferred:]+args.devices[:preferred]
                device = next((d for d in ordered if not busy[d]), None)
                if device is None:
                    raise RuntimeError(f'No replica available at the {timing.period:.1f}s launch boundary')
                start = tick//timing.action_stride
                available_history = start
                started = time.monotonic()
                jobs[job_id] = dict(device=device, start=start, obs_time=deadline, launch_time=started,
                                   commit_at=deadline+args.handoff_latency)
                busy[device] = True
                if history_mode == 'snapshot':
                    payload = dict(history=[], snapshot=observed_snapshot(start, ledger.issued, images,
                        timing, getattr(args, 'snapshot_history_intervals', 2)))
                else:
                    payload = dict(history=histories(device, start))
                pool.send(device, dict(kind='WORLD', job=job_id, **payload))
                launches.append(started-origin)
            maintain_idle_histories()
            # Measure at command issue, including plan commits and launches.
            lateness.append(max(0., time.monotonic()-deadline))
            if lateness[-1] > 1/timing.control_hz:
                raise RuntimeError(f'30-Hz controller missed a full control period: {lateness[-1]*1000:.1f} ms late')
            ledger.issue(tick)
            # The final target is the last interpolation anchor. Commands
            # after it require a new right anchor, rather than holding joints.
            min_coverage = min(min_coverage, ((len(ledger.plan)-1)*timing.action_stride-tick)/timing.control_hz)
        drain()
        if len(measurements) < 4:
            raise RuntimeError(f'Need at least four real queue commits, observed {len(measurements)}')
    except BaseException:
        error = traceback.format_exc()
    finally:
        stopped_at = time.monotonic()
        np.save(args.output/'issued_actions.npy', np.asarray(ledger.issued))
        np.save(args.output/'controller_lateness_seconds.npy', np.asarray(lateness))
        with (args.output/'measurements.jsonl').open('w') as stream:
            for row in measurements:
                job_id = row['job']
                np.save(args.output/f'job_{job_id:03d}_actions.npy', jobs[job_id]['result']['actions'])
                stream.write(json.dumps(row)+'\n')
                print(f'Commit job={job_id} NPU={row["device"]} '
                      f't={row["commit_seconds"]:.3f}s age={row["observation_to_commit_seconds"]:.3f}s '
                      f'history={row["history_seconds"]:.3f}s video={row["video_seconds"]:.3f}s '
                      f'action={row["action_seconds"]:.3f}s frozen={row["frozen_targets"]}', flush=True)
        commits = [m['commit_seconds'] for m in measurements]
        intervals = np.diff(commits).tolist()
        tolerance = min(args.handoff_tolerance, 1/timing.control_hz) if args.handoff_latency else args.handoff_tolerance
        passed = (error is None and len(intervals) >= 3
                  and max(abs(value-timing.period) for value in intervals) <= tolerance)
        result = dict(success=passed, error=error, controller_ticks=len(ledger.issued),
            history_mode=history_mode,
            bootstrap=dict(video_seconds=bootstrap_world['video_seconds'],
                           action_seconds=bootstrap['action_seconds']),
            launches_seconds=launches, launch_intervals_seconds=np.diff(launches).tolist(),
            measurements=measurements, commits=ledger.commits,
            idle_history_prefetch=history_prefetch,
            handoff_intervals_seconds=intervals,
            handoff_tolerance_seconds=tolerance, handoff_latency_seconds=args.handoff_latency,
            max_controller_lateness_ms=max(lateness, default=0.)*1000,
            min_remaining_coverage_seconds=min_coverage,
            incomplete_jobs=[job for job in jobs if job >= next_action_job],
            pending_job_details=[dict(job=number,device=job['device'],
                observation_seconds=job['obs_time']-origin,
                scheduled_commit_seconds=job['commit_at']-origin,
                world_ready_seconds=(job['world']['world_ready']-origin) if 'world' in job else None,
                action_bind_seconds=(job['bind_time']-origin) if 'bind_time' in job else None,
                result_received_seconds=(job['result_received']-origin) if 'result' in job else None,
                elapsed_at_stop_seconds=stopped_at-job['obs_time'],
                frozen_targets=job['binding']['frozen'] if 'binding' in job else None,
                action_seconds=job['result']['action_seconds'] if 'result' in job else None)
                for number,job in jobs.items() if number >= next_action_job],
            statistics={name:percentiles([m[name] for m in measurements]) for name in (
                'history_seconds', 'video_seconds', 'action_seconds', 'observation_to_commit_seconds')},
            handoff_statistics=percentiles(intervals),
            invariant_checks=dict(serial_late_binding=True, absolute_timestamp_alignment=True,
                exact_frozen_prefix=True, executed_history_only=True,
                controller_within_one_period=max(lateness, default=0.) <= 1/timing.control_hz,
                no_action_queue_underflow=min_coverage > 0))
        if error is None and not passed:
            result['error'] = 'Usable handoff intervals exceeded the configured tolerance'
        return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--root', type=Path, default=Path(DEFAULT_ROOT))
    parser.add_argument('--device-count', '--device_count', type=int, default=None,
                        help='Complete model replicas, default 2; 2-4 for playback, 1 for --profile-only')
    parser.add_argument('--devices', type=int, nargs='+',
                        help='Explicit runtime NPU IDs; when supplied alone, infer device_count from this list')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--episode', type=int, default=15)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--ticks', type=int, default=600)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--video-steps', type=int, default=3)
    parser.add_argument('--action-steps', type=int, default=10)
    parser.add_argument('--receding-actions', type=int, default=32,
                        help='10-Hz targets between launches; prediction horizon stays 128 targets')
    parser.add_argument('--history-mode', choices=('incremental', 'snapshot'), default='incremental',
                        help='snapshot rebuilds a real observed window ending at each launch; changes the history policy')
    parser.add_argument('--snapshot-history-intervals', type=int, choices=(1, 2), default=2,
                        help='Maximum complete 3.2s intervals in snapshot history, followed by four predicted intervals')
    parser.add_argument('--shared-recording', action='store_true',
                        help='Share preloaded RGB via a read-only mmap; pass capture indices over IPC')
    parser.add_argument('--warmup-rounds', type=int, default=2)
    parser.add_argument('--human-latent', type=Path)
    parser.add_argument('--profile-only', action='store_true')
    parser.add_argument('--check-equivalence', action='store_true',
                        help='Compare the split sampler to the production RTC sampler on real NPUs')
    parser.add_argument('--latency-multiplier', type=float, default=1.2)
    parser.add_argument('--deadline-margin', type=float, default=.05)
    parser.add_argument('--handoff-tolerance', type=float, default=.25)
    parser.add_argument('--handoff-latency', type=float, default=0.,
                        help='0: publish immediately; otherwise fixed observation-to-commit delay in seconds')
    args = parser.parse_args(argv)
    if args.devices is not None:
        if len(set(args.devices)) != len(args.devices) or min(args.devices) < 0:
            parser.error('--devices must contain distinct nonnegative NPU IDs')
        if args.device_count is not None and args.device_count != len(args.devices):
            parser.error('--device-count must equal the number of IDs in --devices')
        args.device_count = len(args.devices)
    elif args.device_count is None:
        args.device_count = 2
    if args.device_count not in (1, 2, 3, 4) or (args.device_count == 1 and not args.profile_only):
        parser.error('device_count must be 2-4; 1 is allowed only for --profile-only')
    if min(args.video_steps, args.action_steps, args.warmup_rounds, args.ticks) < 1:
        parser.error('Step counts, warmup rounds and ticks must be positive')
    if (not all(math.isfinite(v) for v in (args.latency_multiplier, args.deadline_margin,
                                         args.handoff_tolerance, args.handoff_latency))
            or args.latency_multiplier < 1 or args.deadline_margin < 0
            or args.handoff_tolerance <= 0 or args.handoff_latency < 0):
        parser.error('Need multiplier>=1, margin>=0, handoff tolerance>0')
    try:
        Timing(receding_actions=args.receding_actions).validate()
    except ValueError as error:
        parser.error(str(error))
    if args.history_mode == 'incremental' and args.receding_actions % 32:
        parser.error('Incremental history requires s to be a multiple of 32; use --history-mode snapshot')
    return args


def main():
    args = parse_args()
    try:
        device_selection = resolve_devices(args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f'RTC device selection failed: {error}') from error
    print(f'RTC device_count={args.device_count}, devices={args.devices}, '
          f'source={device_selection["source"]}', flush=True)
    timing = Timing(receding_actions=args.receding_actions).validate()
    args.output.mkdir(parents=True, exist_ok=True)
    images, metadata = load_recording(args, timing)
    if args.shared_recording:
        args.recording_store = args.output/'recorded_frames.npy'
        images = share_recording(images, args.recording_store)
    pool = WorkerPool(args)
    report = dict(success=False, model=str(args.model.resolve()), device_count=args.device_count,
        devices=args.devices, device_selection=device_selection,
        metadata=metadata, video_steps=args.video_steps, action_steps=args.action_steps,
        timing=timing.__dict__, receding_seconds=timing.period,
        protocol='Recorded RGB only; no robot/simulator execution; full replicas, late-bound action-only RTC')
    try:
        reset, warmup, equivalence = initialize(pool, args, metadata, images, timing)
        report.update(reset=reset, warmup=warmup, sampler_equivalence=equivalence)
        if args.profile_only:
            stats = json.loads((args.root/'train/piperpika/meta/action_stats.json').read_text())['norm_stats']
            low = stats['action.arm.position']['q01']+stats['action.effector.position']['q01']
            high = stats['action.arm.position']['q99']+stats['action.effector.position']['q99']
            reference = np.repeat(((np.array(low)+np.array(high))*.5)[None], timing.horizon, axis=0).astype(np.float32)
            measured = profile_history(pool, args, images, timing, reference)
            report.update(history_profile=measured, success=True)
            print('History profile: '+json.dumps(measured), flush=True)
        else:
            report.update(run_pipeline(pool, args, images, timing, warmup))
            if not report['success']:
                raise RuntimeError(report.get('error') or 'Pipeline verification failed')
    except BaseException:
        report['error'] = traceback.format_exc()
        raise
    finally:
        pool.close()
        (args.output/'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: report[k] for k in ('success', 'receding_seconds', 'device_count', 'devices')}, indent=2), flush=True)


if __name__ == '__main__':
    main()
