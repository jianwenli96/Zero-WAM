"""30 Hz nonblocking action queue and latency buffer for the RTC server.

The controller calls get_action with the current RGB observation at every tick.
All network/model work after start runs in one background thread.
"""
from concurrent.futures import ThreadPoolExecutor
import math
import time

import numpy as np

from wan_va.rtc.config import DelayBuffer, RTCConfig, RTCError, RTCTimingError

CAMERAS = ('observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist')


class WebsocketRTCTransport:
    def __init__(self, host='127.0.0.1', port=29536, timeout=120):
        from websockets.sync.client import connect
        from evaluation.robotwin.msgpack_numpy import Packer, unpackb
        self._unpack = unpackb
        self._packer = Packer()
        self._timeout = timeout
        self._context = connect(f'ws://{host}:{port}', max_size=None, compression=None,
                                ping_interval=None, open_timeout=30, proxy=None)
        self._ws = self._context.__enter__()
        try:
            unpackb(self._ws.recv(timeout=30))
        except Exception:
            self._context.__exit__(None, None, None)
            raise

    def infer(self, message):
        self._ws.send(self._packer.pack(message))
        raw = self._ws.recv(timeout=self._timeout)
        if isinstance(raw, str):
            raise RTCError(f'RTC server rejected the request:\n{raw}')
        return self._unpack(raw)

    def close(self):
        self._context.__exit__(None, None, None)


class RTCPolicy:
    def __init__(self, transport, *, clock=time.monotonic):
        self.transport = transport
        self.clock = clock
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='zero-wam-rtc')
        self._future = None
        self._error = None
        self._queue = None
        self._closed = False
        self._issued = {}
        self._images = {}
        self.measurements = []

    def _snapshot(self, observation):
        import cv2
        cv2.setNumThreads(1)
        frames = {}
        for key in CAMERAS:
            image = np.asarray(observation[key])
            if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
                raise RTCError(f'{key} must be uint8 RGB [height,width,3]')
            h, w = self.image_size
            frames[key] = (image.copy() if image.shape[:2] == (h, w) else
                           cv2.resize(image, (w, h), interpolation=cv2.INTER_LINEAR))
        return frames

    def _reset(self, message):
        metadata = self.transport.infer(dict(message, reset=True, enable_rtc=True))
        if not metadata.get('enable_rtc') or metadata.get('rtc', {}).get('protocol_version') not in (1, 2):
            raise RTCError('Server must be launched with --enable-rtc and a supported protocol')
        remote = metadata['rtc']
        config = RTCConfig(**{k: remote[k] for k in RTCConfig.__dataclass_fields__ if k in remote}).validate()
        if config.action_stride > 1 and remote.get('protocol_version') != 2:
            raise RTCError('Joint time strides require RTC protocol_version=2')
        self.session_id = remote['session_id']
        self.image_size = tuple(metadata.get('image_size', (224, 288)))
        if hasattr(self, 'config') and self.config != config:
            raise RTCError('Server changed RTC configuration during startup')
        self.config = config
        return metadata

    def _request(self, action_start, observation, *, feedback=None, executed=None,
                 previous=None, warmup=False, started=None):
        started = self.clock() if started is None else started
        message = dict(obs=observation, rtc=dict(
            session_id=self.session_id, action_start=action_start,
            delay_steps=self.delays.delay_steps, warmup=warmup))
        if feedback is not None:
            message.update(feedback=feedback, executed_actions=executed, previous_actions=previous)
        response = self.transport.infer(message)
        rtc = response.get('rtc', {})
        if rtc.get('session_id') != self.session_id or rtc.get('action_start') != action_start:
            raise RTCError('Received stale or misaligned RTC response')
        raw = np.asarray(response.get('action'))
        if raw.ndim != 3 or raw.shape[1:] != (self.config.horizon//self.config.action_per_frame,
                                            self.config.action_per_frame):
            raise RTCError(f'Invalid RTC action shape: {raw.shape}')
        actions = raw.transpose(1, 2, 0).reshape(self.config.horizon, -1).copy()
        if not np.isfinite(actions).all():
            raise RTCError('Received nonfinite RTC actions')
        if self.config.action_stride > 1 and actions.shape[1] == 28:
            for start in (3, 10):
                norm = np.linalg.norm(actions[:, start:start+4], axis=1, keepdims=True)
                if (norm < 1e-8).any():
                    raise RTCError('Cannot interpolate a zero quaternion')
                actions[:, start:start+4] /= norm
        return actions, self.clock()-started, rtc

    def start(self, initial_observation, reset_message, *, warmup_rounds=3):
        """Warm up while idle, calibrate, reset fake history, then fill the queue.

        initial_observation may be a callable to acquire a fresh image after
        warmup. No warmup action is returned to or executed by the controller.
        """
        if self._closed or self._queue is not None or self._future is not None:
            raise RTCError('Create a new RTCPolicy before starting another run')
        if warmup_rounds < 3:
            raise RTCError('RTC requires at least three guided warmup rounds')
        try:
            self._reset(reset_message)
            self.delays = DelayBuffer(self.config)
            source = initial_observation if callable(initial_observation) else lambda: initial_observation
            image = self._snapshot(source())
            queue, _, _ = self._request(0, image, warmup=True)
            measured = []
            s = self.config.execution_horizon
            for index in range(1, warmup_rounds+1):
                queue, latency, _ = self._request(index*s, image,
                    feedback=[image]*self.config.feedback_frames,
                    executed=queue[:s].copy(), previous=queue[s:].copy(), warmup=True)
                measured.append(latency)
            # All cold samples remain available for diagnosis. Only the final
            # two warm samples seed the online buffer alongside initial_latency.
            self.warmup_latencies = measured
            for latency in measured[-2:]:
                self.delays.record(latency)
            self._reset(reset_message)
            image = self._snapshot(source())
            self._queue, _, _ = self._request(0, image)
            self._plan_start = self._tick = 0
            self._images = {0: image}
            self._issued = {}
            self._error = None
            return dict(config=self.config.as_dict(), warmup_latencies=measured,
                        latency_buffer=list(self.delays.samples), delay_steps=self.delays.delay_steps)
        except Exception as exc:
            self._error = exc
            raise

    def _accept(self):
        if self._future is None or not self._future.done():
            return
        queue, rpc_latency, rtc = self._future.result()
        elapsed = max(rpc_latency, self.clock()-self._request_started)
        actual_delay = max(self.config.elapsed_delay_steps(elapsed),
            math.ceil((self._tick-self._request_tick)/self.config.action_stride)+int(self.config.action_stride > 1))
        self.delays.record(elapsed)
        if actual_delay > self._forecast_delay:
            raise RTCTimingError(f'RTC response missed forecast: actual d={actual_delay}, forecast d={self._forecast_delay}')
        consumed_native = self._tick-self._request_tick
        consumed = math.ceil(consumed_native/self.config.action_stride)
        if consumed < 0 or consumed >= self.config.horizon:
            raise RTCError('RTC queue swap has invalid controller timestamps')
        # Preserve every command already issued while the request was running.
        for offset in range(consumed):
            queue[offset] = self._issued[self._request_tick+offset*self.config.action_stride]
        self._queue = queue
        self._plan_start = self._request_tick
        self.measurements.append(dict(action_start=self._plan_start//self.config.action_stride, latency_seconds=elapsed,
            actual_delay_steps=actual_delay, forecast_delay_steps=self._forecast_delay,
            latency_buffer=list(self.delays.samples), server=rtc))
        self._issued = {k: v for k, v in self._issued.items() if k >= self._plan_start}
        self._images = {k: v for k, v in self._images.items() if k > self._plan_start}
        self._future = None

    def _schedule(self):
        cfg, tick = self.config, self._tick
        self._request_started = self.clock()
        self._request_tick = tick
        self._forecast_delay = self.delays.delay_steps
        s = cfg.execution_horizon
        native_s = s*cfg.action_stride
        feedback = [self._images[i] for i in range(tick-native_s+cfg.observation_stride, tick+1, cfg.observation_stride)]
        executed = np.stack([self._issued[i] for i in range(tick-native_s, tick, cfg.action_stride)])
        previous = self._queue[s:].copy()
        self._future = self._executor.submit(self._request, tick//cfg.action_stride, self._images[tick],
            feedback=feedback, executed=executed, previous=previous, started=self._request_started)

    def get_action(self, observation, *, executed_action=None):
        """Return one absolute command; call once per control tick at control_hz.

        If a downstream controller changed the preceding command, provide its
        actual executed value so it, rather than the prediction, enters history.
        Errors are latched and propagated; the queue never silently holds/repeats.
        """
        if self._closed or self._error is not None:
            raise RTCError('RTC stopped or failed') from self._error
        if self._queue is None:
            raise RTCError('Call start before consuming RTC actions')
        try:
            tick = self._tick
            if tick and executed_action is not None:
                value = np.asarray(executed_action, dtype=self._queue.dtype)
                if value.shape != self._queue.shape[1:] or not np.isfinite(value).all():
                    raise RTCError('executed_action has invalid shape or values')
                self._issued[tick-1] = value.copy()
            if tick and tick % self.config.observation_stride == 0:
                self._images[tick] = self._snapshot(observation)
            self._accept()
            if self._future is not None:
                delay = max(math.ceil((tick-self._request_tick)/self.config.action_stride)+int(self.config.action_stride > 1),
                            self.config.elapsed_delay_steps(self.clock()-self._request_started))
                if delay > self._forecast_delay:
                    raise RTCTimingError('RTC inference is still pending after its forecast deadline')
            boundary = self._plan_start+self.config.execution_horizon*self.config.action_stride
            if self._future is None and tick >= boundary:
                if tick != boundary:
                    raise RTCTimingError('RTC missed the next observation/replanning boundary')
                self._schedule()
            index = tick-self._plan_start
            if not 0 <= index < len(self._queue)*self.config.action_stride:
                raise RTCTimingError('RTC action queue exhausted')
            action = interpolate_action(self._queue, index, self.config.action_stride)
            self._issued[tick] = action.copy()
            self._tick += 1
            return action
        except Exception as exc:
            self._error = exc
            raise

    def close(self):
        self._closed = True
        self._executor.shutdown(wait=False, cancel_futures=True)
        if hasattr(self.transport, 'close'):
            self.transport.close()


def interpolate_action(queue, native_index, stride):
    """Native-rate command: linear XYZ/joints, quaternion SLERP, held grippers."""
    base, remainder = divmod(native_index, stride)
    left = queue[base]
    if remainder == 0 or base+1 == len(queue):
        return left.copy()
    right = queue[base+1]
    fraction = remainder/stride
    command = ((1-fraction)*left+fraction*right).astype(queue.dtype)
    if command.shape == (28,):
        for start in (3, 10):
            q0, q1 = left[start:start+4], right[start:start+4]
            dot = float(np.dot(q0, q1))
            if dot < 0:
                q1, dot = -q1, -dot
            dot = min(1., max(-1., dot))
            if dot > .9995:
                q = (1-fraction)*q0+fraction*q1
            else:
                angle = math.acos(dot)
                q = (math.sin((1-fraction)*angle)*q0+math.sin(fraction*angle)*q1)/math.sin(angle)
            command[start:start+4] = q/np.linalg.norm(q)
        command[-2:] = left[-2:]
    return command
