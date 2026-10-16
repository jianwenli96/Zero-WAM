"""Shared RTC timing contract; durations are end-to-end seconds."""
from collections import deque
from dataclasses import asdict, dataclass
import math


class RTCError(RuntimeError):
    """An RTC protocol or state invariant was violated; reset before retrying."""


class RTCTimingError(RTCError):
    """The measured/forecast delay cannot meet the action execution deadline."""


@dataclass(frozen=True)
class RTCConfig:
    horizon: int = 256
    execution_horizon: int = 128
    control_hz: float = 30.0
    action_per_frame: int = 16
    observation_stride: int = 4
    action_stride: int = 1
    delay_buffer_size: int = 10
    initial_latency: float = 3.342
    latency_multiplier: float = 1.2
    latency_margin: float = 0.05
    max_guidance_weight: float = 5.0

    @classmethod
    def from_job_config(cls, cfg):
        return cls(
            horizon=int(cfg.frame_chunk_size)*int(cfg.action_per_frame),
            action_per_frame=int(cfg.action_per_frame),
            action_stride=int(getattr(cfg, 'action_stride', 1)),
            **{name: getattr(cfg, 'rtc_'+name, field.default)
               for name, field in cls.__dataclass_fields__.items()
               if name not in ('horizon', 'action_per_frame', 'action_stride')},
        )

    def validate(self):
        for name in ('horizon', 'execution_horizon', 'action_per_frame',
                     'observation_stride', 'action_stride', 'delay_buffer_size'):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise RTCError(f'{name} must be a positive integer, got {value!r}')
        for name in ('control_hz', 'initial_latency', 'latency_multiplier',
                     'latency_margin', 'max_guidance_weight'):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise RTCError(f'{name} must be finite and nonnegative')
        if self.control_hz == 0 or self.latency_multiplier < 1 or self.max_guidance_weight == 0:
            raise RTCError('Need control_hz>0, latency_multiplier>=1, max_guidance_weight>0')
        if self.action_per_frame != 4*self.observation_stride:
            raise RTCError('Causal VAE requires action_per_frame=4*observation_stride')
        if self.horizon % self.action_per_frame or self.execution_horizon % self.action_per_frame:
            raise RTCError('RTC horizons must be whole latent intervals (multiples of action_per_frame)')
        self.check_delay(self.delay_steps(self.initial_latency), context='initial RTC budget')
        return self

    def reserved_seconds(self, latency):
        if not math.isfinite(latency) or latency < 0:
            raise RTCError('Latency must be finite and nonnegative')
        return latency*self.latency_multiplier+self.latency_margin

    def delay_steps(self, latency):
        return self.elapsed_delay_steps(self.reserved_seconds(latency))

    @property
    def action_hz(self):
        return self.control_hz/self.action_stride

    @property
    def feedback_frames(self):
        return self.execution_horizon*self.action_stride//self.observation_stride

    def elapsed_delay_steps(self, latency):
        # Keep the anchor after the deadline frozen, so a queue swap cannot
        # change the interpolation segment already being executed.
        return math.ceil(latency*self.action_hz) + int(self.action_stride > 1)

    def check_delay(self, delay, *, context='RTC'):
        if isinstance(delay, bool) or not isinstance(delay, int) or delay < 0:
            raise RTCError('delay_steps must be a nonnegative integer')
        h, s = self.horizon, self.execution_horizon
        if not 0 < s < h or not delay <= s <= h-delay:
            raw_limit = ((min(s, h-s)-int(self.action_stride > 1))/self.action_hz
                         - self.latency_margin)/self.latency_multiplier
            raise RTCTimingError(
                f'{context}: need d<=s<=H-d, got d={delay}, s={s}, H={h}, '
                f'control_hz={self.control_hz:g}, model_action_hz={self.action_hz:g}; '
                f'maximum raw latency with configured reserve is {max(0., raw_limit):.3f}s. Increase the horizon, '
                'reduce latency, or change the execution horizon.'
            )

    def as_dict(self):
        return asdict(self)


class DelayBuffer:
    """Algorithm 1: forecast from the maximum of a bounded latency history."""

    def __init__(self, config):
        self.config = config.validate()
        self.samples = deque([config.initial_latency], maxlen=config.delay_buffer_size)

    @property
    def delay_steps(self):
        delay = self.config.delay_steps(max(self.samples))
        self.config.check_delay(delay, context='latency buffer')
        return delay

    def record(self, latency):
        self.config.reserved_seconds(latency)  # validate before changing state
        self.samples.append(float(latency))
        return self.delay_steps
