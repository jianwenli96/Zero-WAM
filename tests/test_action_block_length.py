import json
import sys
from dataclasses import replace
import numpy as np
import pytest
import torch
import wan_va.train as training
from wan_va.action_sampling import (ACTION_SAMPLING_CONFIG_KEY, action_sampling_metadata,
    latent_stride_for, resample_training_batch, inference_action_stride, inference_actions_per_latent)
from wan_va.rtc.config import RTCConfig, RTCTimingError
from wan_va.rtc.server import RTCSession
from evaluation.aloha.server import parse_args
from test_action_stride import native_batch, make_trainer
from test_mcp_mask_sharing import enable_checkpointing
from test_rtc import _rtc_model


def test_32_actions_at_10hz_align_with_native_z6():
    batch=native_batch(frames=33)
    sampled=resample_training_batch(batch,3,32)
    assert sampled['latents'].flatten().tolist()==[0,6,12,18,24,30]
    assert sampled['actions'].shape==(1,1,6,32,1)
    assert torch.count_nonzero(sampled['actions'][:,:,0])==0
    torch.testing.assert_close(sampled['actions'][:,:,1:].flatten()-16,torch.arange(0,480,3).float())
    assert (sampled['actions'][0,0,1,:,0]-16).tolist()==list(range(0,96,3))
    assert sampled['icl_latents'] is batch['icl_latents']
    assert resample_training_batch(sampled,3,32) is sampled
    with pytest.raises(ValueError,match='different'):resample_training_batch(sampled,3,16)
    assert 32/(30/3)==3.2 and 4*32/(30/3)==12.8
    with pytest.raises(ValueError,match='at least 7'):resample_training_batch(native_batch(frames=6),3,32)


@pytest.mark.parametrize('target,stride,native,expected',[(32,3,16,6),(32,1,16,2),(16,3,16,3),(40,3,20,6),(8,2,16,1)])
def test_layout_ratios(target,stride,native,expected):
    assert latent_stride_for(stride,target,native)==expected


@pytest.mark.parametrize('target',[0,-1,True,1.5])
def test_invalid_action_count(target):
    with pytest.raises(ValueError,match='positive integer'):latent_stride_for(3,target)


def test_fractional_native_latent_interval_is_rejected():
    with pytest.raises(ValueError,match='integer number'):latent_stride_for(3,32,20)


def test_native_dataset_block_mismatch_cannot_save_wrong_contract():
    trainer=make_trainer(5);trainer.config.actions_per_latent=32
    with pytest.raises(ValueError,match='native action blocks matching'):
        trainer._train_step(native_batch(frames=33,a=20),0)


@pytest.mark.parametrize('checkpointed',[False,True])
def test_real_icl_mcp_32_action_optimizer_step(checkpointed):
    trainer=make_trainer(3,True);trainer.config.actions_per_latent=32
    if checkpointed:enable_checkpointing(trainer.transformer)
    batch={k:torch.randn(shape,dtype=torch.bfloat16) for k,shape in dict(
        latents=(1,4,33,2,2),actions=(1,3,33,16,1),icl_latents=(1,4,5,2,2),
        text_emb=(1,3,8),icl_text_emb=(1,3,8)).items()}
    batch['actions'][:,:,0]=0;batch['actions_mask']=torch.ones_like(batch['actions'],dtype=torch.bool)
    seen=[]
    def check(module,args):
        a=args[0]['action_dict']['latent'];v=args[0]['latent_dict']['latent']
        assert a.shape[2:]==(4,32,1) and v.shape[2]==4;seen.append(True)
    handle=trainer.transformer.register_forward_pre_hook(check)
    before=trainer.transformer.action_proj_out.weight.detach().clone()
    losses=trainer._train_step(batch,0);handle.remove()
    assert seen and not losses['optimizer_step_skipped']
    assert all(torch.isfinite(losses[k]) for k in ('latent_loss','action_loss','mcp_loss'))
    assert not torch.equal(before,trainer.transformer.action_proj_out.weight)


def test_checkpoint_records_and_auto_loads_action_block(tmp_path):
    trainer=make_trainer(3);trainer.config.actions_per_latent=32;trainer.config.action_per_frame=16
    trainer.save_dir=tmp_path;trainer.step=7;trainer.aloha_action_contract=None;trainer.save_checkpoint()
    checkpoint=tmp_path/'checkpoint_step_7'
    saved=json.loads((checkpoint/'transformer/config.json').read_text())[ACTION_SAMPLING_CONFIG_KEY]
    assert saved==action_sampling_metadata(3,32) and saved['version']==3 and saved['latent_stride']==6
    assert inference_action_stride(checkpoint)==(3,True)
    assert inference_actions_per_latent(checkpoint)==(32,True)
    with pytest.raises(ValueError,match='conflicts'):inference_actions_per_latent(checkpoint,16)
    loaded=type(trainer.transformer).from_pretrained(checkpoint/'transformer',torch_dtype=torch.bfloat16)
    torch.testing.assert_close(loaded.action_proj_out.weight,trainer.transformer.action_proj_out.weight,rtol=0,atol=0)
    saved['latent_stride']=3
    (checkpoint/'transformer/config.json').write_text(json.dumps({ACTION_SAMPLING_CONFIG_KEY:saved}))
    with pytest.raises(ValueError,match='inconsistent'):inference_action_stride(checkpoint)


def test_legacy_explicit_probe_and_bound_v2_rejection(tmp_path):
    path=tmp_path/'transformer';path.mkdir();(path/'config.json').write_text('{}')
    assert inference_actions_per_latent(tmp_path,32)==(32,False)
    (path/'config.json').write_text(json.dumps({ACTION_SAMPLING_CONFIG_KEY:action_sampling_metadata(3)}))
    assert inference_actions_per_latent(tmp_path)==(16,True)
    with pytest.raises(ValueError,match='conflicts'):inference_actions_per_latent(tmp_path,32)


def test_32_action_rtc_budget_and_feedback():
    cfg=RTCConfig(horizon=128,execution_horizon=64,action_per_frame=32,action_stride=3,initial_latency=2).validate()
    assert cfg.action_hz==10 and cfg.latent_stride==6 and cfg.feedback_frames==48
    assert cfg.delay_steps(5.2)==64
    with pytest.raises(RTCTimingError):cfg.check_delay(cfg.delay_steps(5.3))


def test_rtc_32_action_history_uses_native_z6_z12(monkeypatch):
    monkeypatch.setattr('wan_va.rtc.server.time.monotonic',lambda:0.)
    model,old=_rtc_model();model.action_per_frame=model.job_config.action_per_frame=32
    cfg=replace(old.config,horizon=128,execution_horizon=64,action_per_frame=32,
        observation_stride=4,action_stride=3,control_hz=30,initial_latency=.05)
    session=RTCSession(model,cfg)
    req=dict(session_id=session.session_id,action_start=0,delay_steps=3)
    initial=session.infer(dict(obs={},rtc=req));assert initial['action'].shape==(2,4,32)
    flat=initial['action'].transpose(1,2,0).reshape(128,2)
    model._encode_obs=lambda obs:torch.arange(12).reshape(1,1,12,1,1).expand(1,48,12,1,1).float()
    model.transformer.calls.clear()
    result=session.infer(dict(obs={},feedback=[{}]*48,executed_actions=flat[:64],previous_actions=flat[64:],
        rtc=dict(req,action_start=64)))
    history=[c for c in model.transformer.calls if c[0]=='forward_latent_only' and c[1]==1][0]
    assert history[2]['latent_res_lst']['noisy_latents'][0,0,:,0,0].tolist()==[5,11]
    assert result['rtc']['guidance_applied']
    np.testing.assert_array_equal(result['action'].transpose(1,2,0).reshape(128,2)[:3],flat[64:67])


@pytest.mark.parametrize('flag',['--actions-per-latent','--actions_per_latent'])
def test_train_and_inference_cli(monkeypatch,flag):
    args=parse_args(['--model','/tmp/model','--enable-rtc','--action-stride','3',flag,'32'])
    assert args.rtc_execution_horizon==64 and args.chunk_frames==4
    monkeypatch.setattr(sys,'argv',['train','--config-name','aloha_train','--action-stride','3',flag,'32'])
    class Captured(Exception):pass
    def capture(cfg,*args,**kwargs):
        assert cfg.actions_per_latent==32 and cfg.action_stride==3;raise Captured
    monkeypatch.setattr(training,'_build_dataset_sources',capture)
    with pytest.raises(Captured):training.main()
