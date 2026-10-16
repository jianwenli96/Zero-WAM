"""RTC timing, guidance direction, cache alignment and asynchronous failures."""
import threading

import numpy as np
import pytest
import torch

from evaluation.aloha.rtc_client import CAMERAS, RTCPolicy
from evaluation.aloha.server import parse_args
from wan_va.rtc.config import DelayBuffer, RTCConfig, RTCError, RTCTimingError
from wan_va.rtc.guidance import guided_noise_velocity, prefix_weights
from wan_va.rtc.server import RTCSession
from wan_va.utils.scheduler import FlowMatchScheduler
from test_icl_server import _mock_server
from test_icl_model import _tiny_model, _input
from wan_va.rtc.cache import request_cache


def test_flag_defaults_and_candidate_preset():
    legacy = parse_args(['--model', '/tmp/model'])
    assert not legacy.enable_rtc and legacy.steps == 50 and legacy.chunk_frames == 2
    rtc = parse_args(['--model', '/tmp/model', '--enable_rtc'])
    assert rtc.enable_rtc and rtc.steps == 1 and rtc.chunk_frames == 16
    assert rtc.rtc_execution_horizon == 128 and rtc.rtc_attn_window == 4
    assert rtc.rtc_icl_guidance_scale == 1


def test_initial_budget_and_latent_alignment_fail_before_inference():
    cfg = RTCConfig().validate()
    assert cfg.delay_steps(cfg.initial_latency) == 122
    with pytest.raises(RTCTimingError, match='d<=s<=H-d'):
        RTCConfig(horizon=32).validate()
    with pytest.raises(RTCError, match='whole latent'):
        RTCConfig(execution_horizon=127).validate()
    with pytest.raises(RTCError, match='nonnegative'):
        RTCConfig(initial_latency=float('nan')).validate()


def test_rtc_text_encoder_moves_only_for_reset_and_returns_on_error():
    model = _mock_server()
    moves = []
    class Encoder:
        def to(self, device):
            moves.append(str(device))
            return self
    model.text_encoder = Encoder()
    model.encode_prompt = lambda **kwargs: (kwargs['prompt'], 'negative')
    model.enable_rtc = False
    assert model._encode_reset_prompt(prompt='task') == ('task', 'negative')
    assert moves == []
    model.enable_rtc = True
    assert model._encode_reset_prompt(prompt='task') == ('task', 'negative')
    assert moves == ['cpu', 'cpu']
    def fail(**kwargs):
        raise RuntimeError('prompt error')
    model.encode_prompt = fail
    with pytest.raises(RuntimeError, match='prompt error'):
        model._encode_reset_prompt(prompt='task')
    assert moves == ['cpu', 'cpu', 'cpu', 'cpu']


def test_bounded_delay_buffer_uses_max_and_rejects_infeasible_updates():
    cfg = RTCConfig(horizon=16, execution_horizon=8, action_per_frame=4,
                    observation_stride=1, control_hz=10, initial_latency=.1,
                    latency_multiplier=1, latency_margin=0, delay_buffer_size=3)
    history = DelayBuffer(cfg)
    assert history.record(.3) == 3
    assert history.record(.2) == 3
    assert history.record(.1) == 3
    assert history.record(.1) == 2
    assert len(history.samples) == 3
    with pytest.raises(RTCTimingError):
        history.record(1)


def test_exponential_mask_matches_reference_and_stops_at_overlap():
    actual = prefix_weights(2, 6, 10)
    c = torch.tensor([1, 1, .8, .6, .4, .2, 0, 0, 0, 0])
    expected = c*torch.expm1(c)/(torch.e-1)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(prefix_weights(6, 6, 10), torch.tensor([1.]*6+[0.]*4))


def test_guidance_sign_for_decreasing_sigma_and_input_gradient_only():
    # v_sigma=a*x => denoiser=(1-sigma*a)*x. One Euler step must move
    # toward y under the reference tau convention, not away from y.
    a = torch.tensor(.25, requires_grad=False)
    x = torch.tensor([1.])
    y = torch.tensor([.5])
    velocity = guided_noise_velocity(x, lambda value: a*value, 1, y, torch.ones(1), 1)
    expected = a*x-(1-a)*(y-(1-a)*x)
    torch.testing.assert_close(velocity, expected)
    assert abs((x-velocity-y).item()) < abs((x-a*x-y).item())
    assert a.grad is None
    with torch.inference_mode(), pytest.raises(RTCError, match='inference_mode'):
        guided_noise_velocity(x, lambda value: value, 1, y, torch.ones(1), 1)


def test_real_transformer_vjp_with_kv_cache_and_static_reuse():
    model = _tiny_model('cpu').requires_grad_(False)
    with torch.no_grad():
        model(_input(model, 'video', 2), update_cache=1, mode='forward_latent_only')
        model(_input(model, 'video', 1, frame_id=2), update_cache=1, mode='forward_latent_only')
    data = _input(model, 'action', 1, frame_id=3)
    sample = data['action_res_lst']['noisy_latents']
    target = torch.zeros_like(sample)
    weights = torch.tensor([1., .2]).reshape(1, 1, 1, 2, 1)
    def velocity(x):
        stream = dict(data['action_res_lst'], noisy_latents=x)
        output = model(dict(data, action_res_lst=stream), mode='forward_action_only')
        return output.transpose(1, 2).reshape_as(x)
    expected = guided_noise_velocity(sample, velocity, 1, target, weights, 5)
    data['_rtc_static_key'] = ('test_action',)
    with request_cache():
        actual = guided_noise_velocity(sample, velocity, 1, target, weights, 5)
        repeated = guided_noise_velocity(sample, velocity, 1, target, weights, 5)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    torch.testing.assert_close(repeated, expected, atol=0, rtol=0)
    assert torch.isfinite(actual).all() and all(p.grad is None for p in model.parameters())


@pytest.mark.parametrize('keep', [[True]*8, [False]*8,
    [True, True, True, False, False, False, False, False],
    [False, True, False, True, False, False, True, False]])
def test_optimized_pruning_matches_original_for_all_index_patterns(keep):
    original, optimized = _tiny_model('cpu'), _tiny_model('cpu')
    optimized.rtc_indexed_prune = True
    mask = torch.tensor(keep)
    for model in (original, optimized):
        for name in ('type_ids_cache', 'seq_ids_cache', 'frame_ids_cache', 'cache_type_ids_cache'):
            setattr(model, name, torch.arange(8))
        model.blocks[0].attn1.attn_caches['pos'] = {
            name: torch.arange(8*36).reshape(1, 8, 2, 18).clone() for name in ('k', 'v')}
        model._prune_cache(mask)
    for name in ('type_ids_cache', 'seq_ids_cache', 'frame_ids_cache', 'cache_type_ids_cache'):
        torch.testing.assert_close(getattr(original, name), getattr(optimized, name))
    for name in ('k', 'v'):
        torch.testing.assert_close(original.blocks[0].attn1.attn_caches['pos'][name],
                                   optimized.blocks[0].attn1.attn_caches['pos'][name])


def _rtc_model():
    model = _mock_server()
    model.action_per_frame = 4
    model.job_config.frame_chunk_size = 4
    model.job_config.action_per_frame = 4
    model.job_config.used_action_channel_ids = [0, 2]
    model.job_config.inverse_used_action_channel_ids = [0, 2, 1, 2]
    model.actions_q01 = torch.zeros(4, 1, 1)
    model.actions_q99 = torch.ones(4, 1, 1)
    model.action_norm_method = 'quantiles'
    from wan_va.wan_va_server import VA_Server
    model.postprocess_action = VA_Server.postprocess_action.__get__(model)
    model.scheduler = FlowMatchScheduler(shift=1, sigma_min=0, extra_one_step=True)
    model.action_scheduler = FlowMatchScheduler(shift=1, sigma_min=0, extra_one_step=True)
    model._encode_obs = lambda obs: torch.zeros(1, 48, len(obs['obs'])//4, 1, 1)
    model.transformer.clear_prediction_cache = lambda *_args: None
    cfg = RTCConfig(horizon=16, execution_horizon=8, action_per_frame=4,
                    observation_stride=1, initial_latency=.01,
                    latency_margin=0, latency_multiplier=1)
    return model, RTCSession(model, cfg)


def test_prefill_full_first_horizon_and_commit_only_executed_prefix(monkeypatch):
    monkeypatch.setattr('wan_va.rtc.server.time.monotonic', lambda: 0.)
    model, session = _rtc_model()
    request = dict(session_id=session.session_id, action_start=0, delay_steps=1)
    initial = session.infer(dict(obs={}, rtc=request))
    assert initial['action'].shape == (2, 4, 4)  # full 16, no padded block
    assert initial['rtc']['bootstrap'] and not initial['rtc']['guidance_applied']
    video_calls = [c for c in model.transformer.calls if c[0] == 'forward_latent_only']
    assert video_calls[0][2]['latent_grid_id'][0].min() == 0
    assert video_calls[1][2]['latent_grid_id'][0].min() == 1
    flat = initial['action'].transpose(1, 2, 0).reshape(16, 2)
    executed = flat[:8].copy()
    executed[0] = .42  # actual controller override must become history
    previous = flat[8:].copy()
    model.transformer.calls.clear()
    result = session.infer(dict(obs={}, feedback=[{}]*8, executed_actions=executed,
        previous_actions=previous, rtc=dict(request, action_start=8)))
    assert result['rtc']['guidance_applied']
    np.testing.assert_array_equal(result['action'].transpose(1, 2, 0).reshape(16, 2)[:1], previous[:1])
    history = [c for c in model.transformer.calls if c[1] == 1 and
               c[0] == 'forward_action_only'][0][2]
    cached = history['action_res_lst']['noisy_latents']
    assert cached.shape[2] == 2  # eight executed actions, not 16 predicted
    torch.testing.assert_close(cached[0, [0, 2], 0, 0, 0], torch.tensor([-.16, -.16]), atol=2e-6, rtol=0)
    video_calls = [c for c in model.transformer.calls if c[0] == 'forward_latent_only']
    assert video_calls[0][2]['latent_grid_id'][0].min() == 1  # real z1,z2
    assert video_calls[1][2]['latent_grid_id'][0].min() == 3  # next z3..z6


def test_bad_timestamp_latches_server_failure(monkeypatch):
    monkeypatch.setattr('wan_va.rtc.server.time.monotonic', lambda: 0.)
    _, session = _rtc_model()
    with pytest.raises(RTCError, match='action_start'):
        session.infer(dict(obs={}, rtc=dict(session_id=session.session_id, action_start=1, delay_steps=1)))
    with pytest.raises(RTCError, match='session failed'):
        session.infer(dict(obs={}, rtc=dict(session_id=session.session_id, action_start=0, delay_steps=1)))


class _Clock:
    def __init__(self):
        self.now = 0.
    def __call__(self):
        return self.now
    def advance(self, amount=.1):
        self.now += amount


def _image(value=0):
    return {key: np.full((2, 2, 3), value, dtype=np.uint8) for key in CAMERAS}


class _Transport:
    def __init__(self):
        self.cfg = RTCConfig(horizon=16, execution_horizon=8, action_per_frame=4,
            observation_stride=1, control_hz=10, initial_latency=.3,
            latency_multiplier=1, latency_margin=0)
        self.session = 0
        self.requests = []
        self.block = False
        self.release = threading.Event()
        self.failure = None
    def infer(self, message):
        if message.get('reset'):
            self.session += 1
            return dict(enable_rtc=True, image_size=[2, 2], rtc=dict(self.cfg.as_dict(),
                session_id=str(self.session), protocol_version=1, full_future_horizon=True))
        self.requests.append(message)
        req = message['rtc']
        if not req['warmup'] and req['action_start'] and self.block:
            assert self.release.wait(timeout=3)
        if not req['warmup'] and req['action_start'] and self.failure:
            raise self.failure
        flat = np.repeat((np.arange(16)+req['action_start']+100*(req['action_start']>0))[:, None], 2, axis=1).astype(np.float32)
        if 'previous_actions' in message:
            flat[:req['delay_steps']] = message['previous_actions'][:req['delay_steps']]
        return dict(action=flat.reshape(4, 4, 2).transpose(2, 0, 1),
                    rtc=dict(session_id=str(self.session), action_start=req['action_start']))
    def close(self):
        self.release.set()


def test_async_queue_preserves_issued_actions_and_reports_actual_feedback():
    clock, transport = _Clock(), _Transport()
    policy = RTCPolicy(transport, clock=clock)
    try:
        policy.start(_image(), {})
        transport.block = True
        for tick in range(8):
            np.testing.assert_array_equal(policy.get_action(_image(tick)), [tick, tick])
            clock.advance()
        np.testing.assert_array_equal(policy.get_action(_image(8), executed_action=np.array([999, 999])), [8, 8])
        transport.release.set()
        policy._future.result(timeout=3)
        clock.advance()
        np.testing.assert_array_equal(policy.get_action(_image(9)), [9, 9])
        online = transport.requests[-1]
        np.testing.assert_array_equal(online['executed_actions'][-1], [999, 999])
        assert [int(x[CAMERAS[0]][0, 0, 0]) for x in online['feedback']] == list(range(1, 9))
        assert len(policy.measurements) == 1 and policy._plan_start == 8
        clock.advance()
        np.testing.assert_array_equal(policy.get_action(_image(10)), [10, 10])
        clock.advance()
        np.testing.assert_array_equal(policy.get_action(_image(11)), [111, 111])
    finally:
        policy.close()


def test_pending_deadline_failure_is_latched_in_control_thread():
    clock, transport = _Clock(), _Transport()
    policy = RTCPolicy(transport, clock=clock)
    try:
        policy.start(_image(), {})
        transport.block = True
        for tick in range(9):
            policy.get_action(_image(tick))
            clock.advance()
        clock.advance(1)
        with pytest.raises(RTCTimingError, match='pending'):
            policy.get_action(_image(9))
        with pytest.raises(RTCError, match='stopped or failed'):
            policy.get_action(_image(9))
    finally:
        policy.close()


def test_background_rpc_failure_reaches_controller():
    clock, transport = _Clock(), _Transport()
    policy = RTCPolicy(transport, clock=clock)
    try:
        policy.start(_image(), {})
        transport.failure = RTCError('injected network failure')
        for tick in range(9):
            policy.get_action(_image(tick))
            clock.advance()
        with pytest.raises(RTCError, match='injected network failure'):
            policy._future.result(timeout=3)
        with pytest.raises(RTCError, match='injected network failure'):
            policy.get_action(_image(9))
    finally:
        policy.close()
