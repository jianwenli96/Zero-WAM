from dataclasses import replace
import numpy as np
import pytest
import torch
from evaluation.aloha.rtc_client import RTCPolicy, interpolate_action, CAMERAS
from evaluation.aloha.server import parse_args
from wan_va.rtc.config import RTCConfig
from wan_va.rtc.server import RTCSession
from test_rtc import _rtc_model, _Transport, _Clock, _image


def test_coarse_preset_and_budget():
    args=parse_args(['--model','/tmp/model','--enable-rtc','--action-stride','3'])
    assert args.chunk_frames==4 and args.rtc_execution_horizon==32 and args.rtc_initial_latency==2
    cfg=RTCConfig(horizon=64,execution_horizon=32,action_stride=3,initial_latency=2).validate()
    assert cfg.action_hz==10 and cfg.feedback_frames==24
    assert cfg.delay_steps(2)==26 and cfg.horizon/cfg.action_hz==6.4
    assert cfg.execution_horizon/cfg.action_hz==3.2


def test_feedback_encodes_native_rate_then_selects_matching_coarse_latents(monkeypatch):
    monkeypatch.setattr('wan_va.rtc.server.time.monotonic',lambda:0.)
    model,old=_rtc_model();cfg=replace(old.config,action_stride=3,initial_latency=.05)
    session=RTCSession(model,cfg)
    request=dict(session_id=session.session_id,action_start=0,delay_steps=2)
    initial=session.infer(dict(obs={},rtc=request))
    assert session.metadata()['protocol_version']==2
    model._encode_obs=lambda obs:torch.arange(6).reshape(1,1,6,1,1).expand(1,48,6,1,1).float()
    flat=initial['action'].transpose(1,2,0).reshape(16,2)
    model.transformer.calls.clear()
    result=session.infer(dict(obs={},feedback=[{}]*24,executed_actions=flat[:8],
        previous_actions=flat[8:],rtc=dict(request,action_start=8)))
    history=[x for x in model.transformer.calls if x[0]=='forward_latent_only' and x[1]==1][0]
    assert history[2]['latent_res_lst']['noisy_latents'][0,0,:,0,0].tolist()==[2,5]
    assert result['rtc']['guidance_applied']


def test_native_rate_interpolation_and_gripper_hold():
    queue=np.zeros((2,28),np.float32);queue[:,6]=1;queue[:,13]=1
    queue[1,0]=3;queue[1,14]=.6;queue[1,-2:]=1
    np.testing.assert_allclose(interpolate_action(queue,1,3)[[0,14,-1]],[1,.2,0])
    np.testing.assert_allclose(interpolate_action(queue,2,3)[[0,14,-1]],[2,.4,0])
    np.testing.assert_array_equal(interpolate_action(queue,3,3),queue[1])
    queue[1,3:7]=[0,0,1,0]
    for index in (1,2):
        assert abs(np.linalg.norm(interpolate_action(queue,index,3)[3:7])-1)<1e-6


class CoarseTransport(_Transport):
    def __init__(self):
        super().__init__();self.cfg=replace(self.cfg,control_hz=30,action_stride=3)
    def infer(self,message):
        result=super().infer(message)
        if message.get('reset'):result['rtc']['protocol_version']=2
        return result


def test_native_control_ticks_and_model_action_ticks_remain_aligned():
    clock,transport=_Clock(),CoarseTransport();policy=RTCPolicy(transport,clock=clock)
    try:
        policy.start(_image(),{});transport.block=True
        for tick in range(25):
            command=policy.get_action(_image(tick))
            np.testing.assert_allclose(command,[tick/3,tick/3],rtol=1e-6)
            clock.advance(1/30)
        transport.release.set();policy._future.result(timeout=3)
        command=policy.get_action(_image(25))
        np.testing.assert_allclose(command,[25/3,25/3],rtol=1e-6)
        request=transport.requests[-1]
        assert request['rtc']['action_start']==8 and len(request['feedback'])==24
        np.testing.assert_allclose(request['executed_actions'][:,0],np.arange(8))
        assert policy._plan_start==24 and policy.measurements[-1]['action_start']==8
    finally:policy.close()
