"""PiperPika uses real joints, not quaternion channels, in the shared RTC client."""
import json
from dataclasses import replace

import numpy as np
import pytest

from evaluation.aloha.rtc_client import RTCPolicy, interpolate_action
from evaluation.piperpika.server import parse_args
from wan_va.action_sampling import (inference_action_stride, inference_actions_per_latent,
                                    validate_inference_sampling_dataset)
from wan_va.rtc.config import RTCConfig, RTCError, RTCTimingError
from test_action_stride import make_trainer
from test_rtc import _Transport, _Clock, _image


def test_native_and_rtc_server_presets():
    native = parse_args(['--model', '/tmp/model'])
    assert native.steps == native.action_steps == 20 and not native.enable_rtc
    rtc = parse_args(['--model', '/tmp/model', '--enable-rtc', '--action-stride', '3',
                      '--actions-per-latent', '32', '--steps', '3', '--action-steps', '10'])
    assert (rtc.steps, rtc.action_steps, rtc.chunk_frames, rtc.rtc_execution_horizon) == (3, 10, 4, 64)


def test_joint_interpolation_does_not_slerp_joint_channels_and_holds_grippers():
    queue = np.zeros((2, 14), np.float32)
    queue[1] = np.arange(1, 15)
    np.testing.assert_allclose(interpolate_action(queue, 1, 3)[:12], queue[1, :12]/3, rtol=1e-6)
    np.testing.assert_allclose(interpolate_action(queue, 2, 3)[:12], queue[1, :12]*2/3, rtol=1e-6)
    np.testing.assert_array_equal(interpolate_action(queue, 2, 3)[-2:], [0, 0])
    np.testing.assert_array_equal(interpolate_action(queue, 3, 3), queue[1])


class JointTransport(_Transport):
    def __init__(self):
        super().__init__()
        self.cfg = RTCConfig(horizon=128, execution_horizon=64, action_per_frame=32,
                             action_stride=3, initial_latency=.05,
                             latency_multiplier=1, latency_margin=0)

    def infer(self, message):
        if message.get('reset'):
            result = super().infer(message)
            result['rtc']['protocol_version'] = 2
            return result
        self.requests.append(message)
        req = message['rtc']
        if not req['warmup'] and req['action_start'] and self.block:
            assert self.release.wait(timeout=3)
        flat = np.repeat((np.arange(128)+req['action_start'])[:, None], 14, axis=1).astype(np.float32)
        if 'previous_actions' in message:
            flat[:req['delay_steps']] = message['previous_actions'][:req['delay_steps']]
        return dict(action=flat.reshape(4, 32, 14).transpose(2, 0, 1),
                    rtc=dict(session_id=str(self.session), action_start=req['action_start']))


def test_joint_policy_async_swap_and_native_rate_feedback():
    clock, transport = _Clock(), JointTransport()
    policy = RTCPolicy(transport, clock=clock)
    try:
        policy.start(_image(), {})
        transport.block = True
        for tick in range(193):
            command = policy.get_action(_image(tick % 255))
            assert command.shape == (14,)
            np.testing.assert_allclose(command[:12], tick/3, rtol=1e-6)
            np.testing.assert_allclose(command[-2:], tick//3)
            clock.advance(1/30)
        transport.release.set()
        policy._future.result(timeout=3)
        command = policy.get_action(_image())
        np.testing.assert_allclose(command[:12], 193/3, rtol=1e-6)
        request = transport.requests[-1]
        assert len(request['feedback']) == 48
        assert request['executed_actions'].shape == (64, 14)
        assert policy._plan_start == 192
    finally:
        policy.close()


def test_startup_refreshes_observation_and_starts_control_after_bootstrap():
    """Warmup time is outside the controller clock and never issues actions."""
    clock = _Clock()

    class StartupTransport(JointTransport):
        def infer(self, message):
            result = super().infer(message)
            if message.get('reset'):
                clock.advance(1.)
            else:
                clock.advance(3. if message['rtc']['action_start'] == 0 else 4.)
            return result

    transport = StartupTransport()
    policy = RTCPolicy(transport, clock=clock)
    captures = []

    def capture():
        captures.append(clock())
        return _image(len(captures))

    try:
        with pytest.raises(RTCError, match='Call start'):
            policy.get_action(_image())
        policy.start(capture, {}, warmup_rounds=3)
        assert captures == [1., 17.]
        assert clock() == 20.  # reset + warmup + reset + final bootstrap
        assert policy._tick == 0 and policy._issued == {}
        final = transport.requests[-1]
        assert final['rtc']['action_start'] == 0 and not final['rtc']['warmup']
        assert 'feedback' not in final and 'executed_actions' not in final
        assert int(final['obs'][next(iter(final['obs']))][0, 0, 0]) == 2
        assert all(r['rtc']['warmup'] for r in transport.requests[:-1])
        assert policy._queue.shape == (128, 14)
        policy.get_action(_image())
        assert policy._tick == 1 and len(policy._issued) == 1
    finally:
        policy.close()


def test_client_rejects_a_server_for_a_different_robot():
    from wan_va.rtc.config import RTCError
    policy = RTCPolicy(JointTransport(), clock=_Clock())
    try:
        with pytest.raises(RTCError, match='action contract'):
            policy.start(_image(), {'expected_action_contract': 'piperpika_absolute_joint12_gripper2_v1'})
    finally:
        policy.close()


def test_joint_checkpoint_keeps_both_action_and_sampling_contracts(tmp_path):
    trainer = make_trainer(3)
    trainer.config.actions_per_latent = 32
    trainer.config.action_per_frame = 16
    trainer.config.action_sampling_datasets = ['piperpika']
    trainer.save_dir = tmp_path
    trainer.step = 2
    trainer.aloha_action_contract = None
    trainer.piperpika_action_contract = {'stats': {'action_contract': 'joint14'}}
    trainer.save_checkpoint()
    checkpoint = tmp_path/'checkpoint_step_2'
    assert json.loads((checkpoint/'piperpika_action_contract.json').read_text()) == trainer.piperpika_action_contract
    assert inference_action_stride(checkpoint) == (3, True)
    assert inference_actions_per_latent(checkpoint) == (32, True)
    validate_inference_sampling_dataset(checkpoint, 'piperpika')
    with pytest.raises(ValueError, match='not aloha'):
        validate_inference_sampling_dataset(checkpoint, 'aloha')


def test_mixture_only_resamples_the_deployment_robot():
    import argparse
    from wan_va import train
    from wan_va.configs import VA_CONFIGS
    from copy import deepcopy
    config = deepcopy(VA_CONFIGS['piperpika_train'])
    config.action_stride = 3
    config.actions_per_latent = 32
    config.action_sampling_datasets = 'piperpika'
    args = argparse.Namespace(datasets='piperpika:5,oxe:4', **{
        key: None for key in train._DATASET_PATH_OVERRIDES})
    sources = train._build_dataset_sources(config, args, 0, 0, 1)
    piper, replay = [source['config'] for source in sources]
    assert (piper.action_stride, piper.actions_per_latent) == (3, 32)
    assert (replay.action_stride, replay.actions_per_latent) == (1, None)
    assert piper.training_dataset_name == 'piperpika' and replay.training_dataset_name == 'oxe'


def test_real_optimizer_accepts_alternating_native_replay_and_32_action_robot():
    import torch
    trainer = make_trainer(3, True)
    trainer.config.actions_per_latent = 32
    trainer.config.action_sampling_datasets = ['piperpika']
    trainer.config.max_train_frames = None
    seen = []
    def check(module, args):
        seen.append(args[0]['action_dict']['latent'].shape[2:])
    hook = trainer.transformer.register_forward_pre_hook(check)
    try:
        for name, frames, actions in [('piperpika', 33, 16), ('oxe', 3, 4)]:
            batch = {key: torch.randn(shape, dtype=torch.bfloat16) for key, shape in dict(
                latents=(1, 4, frames, 2, 2), actions=(1, 3, frames, actions, 1),
                icl_latents=(1, 4, 5, 2, 2), text_emb=(1, 3, 8), icl_text_emb=(1, 3, 8)).items()}
            batch['actions'][:, :, 0] = 0
            batch['actions_mask'] = torch.ones_like(batch['actions'], dtype=torch.bool)
            batch['_dataset_name'] = [name]
            losses = trainer._train_step(batch, 0)
            assert not losses['optimizer_step_skipped']
            assert all(torch.isfinite(losses[key]) for key in ('latent_loss', 'action_loss', 'mcp_loss'))
    finally:
        hook.remove()
    assert seen == [(6, 32, 1), (3, 4, 1)]


def test_bucket_cost_uses_each_sources_timing_instead_of_global_32_action_layout(tmp_path):
    from types import SimpleNamespace
    import torch
    from wan_va.dataset.data_efficiency import training_sample_costs
    paths = {}
    for name, frames in [('piperpika', 13), ('oxe', 3), ('human', 5)]:
        path = tmp_path/f'{name}.pth'
        torch.save(dict(latent_num_frames=frames, latent_height=2, latent_width=2), path)
        paths[name] = path
    def task(name, native, stride, target):
        return SimpleNamespace(root=name, new_metas=[{}], used_video_keys=['camera'],
            config=SimpleNamespace(action_per_frame=native, action_stride=stride, actions_per_latent=target),
            _latent_file=lambda meta, camera: paths[name], _lookup_icl_sample=lambda meta: None,
            _human_latent_candidate=lambda sample: paths['human'])
    dataset = SimpleNamespace(_datasets=[task('piperpika', 16, 3, 32), task('oxe', 4, 1, None)])
    assert training_sample_costs(dataset, workers=1, action_stride=3, actions_per_latent=32) == [203, 35]


def test_native_four_latent_budget_does_not_inherit_ten_hz_budget():
    native = RTCConfig(horizon=64, execution_horizon=32, initial_latency=.1).validate()
    with pytest.raises(RTCTimingError):
        native.check_delay(native.delay_steps(1))
    coarse = replace(native, horizon=128, execution_horizon=64, action_per_frame=32, action_stride=3)
    coarse.check_delay(coarse.delay_steps(5.2))
    with pytest.raises(RTCTimingError):
        coarse.check_delay(coarse.delay_steps(5.3))
