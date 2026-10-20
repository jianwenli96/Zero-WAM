"""RTC cache commits and overlapping prediction for the ICL runtime."""
import math
import time
import uuid

import numpy as np
import torch

from .cache import request_cache
from .config import RTCError, RTCTimingError
from .guidance import prefix_weights


class RTCSession:
    def __init__(self, model, config):
        self.model = model
        self.config = config.validate()
        self.session_id = uuid.uuid4().hex
        self.action_start = None
        self.phase = 0
        self.failed = False

    def metadata(self):
        return dict(self.config.as_dict(), protocol_version=2 if self.config.action_stride > 1 or self.config.action_per_frame != 4*self.config.observation_stride else 1,
                    session_id=self.session_id, full_future_horizon=True, supports_synchronous=True)

    def _actions(self, value, count, name):
        # MessagePack may expose a read-only buffer; keep ownership before the
        # tensor conversion and before restoring committed absolute commands.
        data = np.array(value, dtype=np.float32, copy=True)
        dimensions = len(self.model.job_config.used_action_channel_ids)
        if data.shape != (count, dimensions) or not np.isfinite(data).all():
            raise RTCError(f'{name} must be finite [{count}, {dimensions}], got {data.shape}')
        packed = data.reshape(-1, self.config.action_per_frame, dimensions).transpose(2, 0, 1)
        result = self.model.preprocess_action(packed).to(self.model.device, self.model.dtype)
        result[:, ~self.model.action_mask] = 0
        return data, result

    def _prefill(self, obs):
        """Cache z0/PAD separately, leaving every predicted frame in the future."""
        m = self.model
        m._rtc_stream_position = (0, 0)
        m.init_latent = m._encode_initial_obs(obs)
        padding = torch.zeros(1, m.job_config.action_dim, 1,
                              self.config.action_per_frame, 1,
                              device=m.device, dtype=m.dtype)
        m.transformer(m._prepare_icl_stream(m.init_latent, 0, 'video', 0),
                      update_cache=1, cache_name=m.cache_name, mode='forward_latent_only')
        m.transformer(m._prepare_icl_stream(padding, 0, 'action', 0),
                      update_cache=1, cache_name=m.cache_name, mode='forward_action_only')
        self.phase = 1

    def infer(self, obs):
        if self.failed:
            raise RTCError('RTC session failed; reset before another inference')
        try:
            return self._infer(obs)
        except Exception:
            self.failed = True
            raise
        finally:
            self.model._rtc_guidance = None

    @torch.no_grad()
    def _infer(self, obs):
        started = time.monotonic()
        m, cfg = self.model, self.config
        req = obs.get('rtc')
        if not isinstance(req, dict) or req.get('session_id') != self.session_id:
            raise RTCError('RTC requires a matching session_id from reset metadata')
        action_start = req.get('action_start')
        delay = req.get('delay_steps')
        execution_mode = req.get('execution_mode', 'asynchronous')
        if execution_mode not in ('asynchronous', 'synchronous'):
            raise RTCError('Unknown execution_mode')
        synchronous = execution_mode == 'synchronous'
        if synchronous and delay != 0:
            raise RTCError('Synchronous requests require delay_steps=0: no commands during inference')
        if hasattr(self, 'execution_mode') and self.execution_mode != execution_mode:
            raise RTCError('Cannot switch execution_mode within a session')
        self.execution_mode = execution_mode
        if isinstance(action_start, bool) or not isinstance(action_start, int):
            raise RTCError('RTC action_start must be an integer controller timestamp')
        cfg.check_delay(delay, context='requested RTC budget')
        bootstrap = self.action_start is None
        expected_start = 0 if bootstrap else self.action_start+cfg.execution_horizon
        if action_start != expected_start:
            raise RTCError(f'RTC action_start must be {expected_start}, got {action_start}')
        previous = None
        with request_cache():
            if bootstrap:
                if obs.get('feedback') or 'executed_actions' in obs:
                    raise RTCError('Initial RTC request must contain only the initial observation')
                self._prefill(obs)
            else:
                images = obs.get('feedback')
                required_images = cfg.feedback_frames
                if not isinstance(images, list) or len(images) != required_images:
                    raise RTCError(f'RTC requires {required_images} sampled feedback frames, got {type(images).__name__}')
                _, executed = self._actions(obs.get('executed_actions'), cfg.execution_horizon, 'executed_actions')
                previous, prefix = self._actions(obs.get('previous_actions'), cfg.horizon-cfg.execution_horizon, 'previous_actions')
                # Only the executed prefix becomes history. Unexecuted actions
                # are supplied to guidance, never cached as observations.
                m.transformer.clear_prediction_cache(-1, m.cache_name)
                latent = m._encode_obs({'obs': images})
                expected_dense = cfg.execution_horizon*cfg.action_stride//(4*cfg.observation_stride)
                if latent.shape[2] != expected_dense:
                    raise RTCError('Native-rate feedback VAE output does not match the action timebase')
                latent = latent[:, :, cfg.latent_stride-1::cfg.latent_stride].contiguous()
                if latent.shape[2] != cfg.execution_horizon//cfg.action_per_frame:
                    raise RTCError('Feedback VAE output does not align with executed action intervals')
                m._rtc_stream_position = (self.action_start//cfg.action_per_frame+1, self.phase)
                m.transformer(m._prepare_icl_stream(latent, 0, 'video', 0), update_cache=1,
                              cache_name=m.cache_name, mode='forward_latent_only')
                m.transformer(m._prepare_icl_stream(executed, 0, 'action', 0), update_cache=1,
                              cache_name=m.cache_name, mode='forward_action_only', clean_window_cache=True)
                self.phase += 1
                target = torch.zeros(1, m.job_config.action_dim,
                                     cfg.horizon//cfg.action_per_frame,
                                     cfg.action_per_frame, 1, device=m.device, dtype=m.dtype)
                target[:, :, :prefix.shape[2]] = prefix
                weights = prefix_weights(delay, cfg.horizon-cfg.execution_horizon,
                                         cfg.horizon, device=m.device).reshape(
                                             1, 1, -1, cfg.action_per_frame, 1)
                weights = weights*m.action_mask.reshape(1, -1, 1, 1, 1)
                m._rtc_guidance = dict(target=target, weights=weights,
                                       delay=delay, beta=cfg.max_guidance_weight)
            m._rtc_stream_position = (action_start//cfg.action_per_frame+1, self.phase)
            action = m._infer_icl(obs)
        # Preserve the committed absolute commands exactly, including quantile
        # roundoff. They will have executed while this request was in flight.
        if previous is not None and delay:
            flat = action.transpose(1, 2, 0).reshape(cfg.horizon, -1).copy()
            flat[:delay] = previous[:delay]
            action = flat.reshape(-1, cfg.action_per_frame, flat.shape[-1]).transpose(2, 0, 1)
        elapsed = time.monotonic()-started
        actual_delay = cfg.elapsed_delay_steps(elapsed)
        warmup = bool(req.get('warmup', False))
        if not synchronous and not bootstrap and not warmup and actual_delay > delay:
            raise RTCTimingError(f'RTC inference exceeded its forecast: actual d={actual_delay}, forecast d={delay}; reset and recalibrate latency')
        self.action_start = action_start
        return dict(action=action, rtc=dict(
            session_id=self.session_id, action_start=action_start,
            horizon_actions=cfg.horizon, bootstrap=bootstrap, warmup=warmup,
            execution_mode=execution_mode,
            guidance_applied=previous is not None,
            frozen_prefix_steps=delay if previous is not None else 0,
            server_latency_seconds=elapsed,
        ))
