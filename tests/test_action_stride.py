import json
import sys
from copy import deepcopy
from types import SimpleNamespace
import pytest
import torch
import wan_va.train as training
from wan_va.action_sampling import (ACTION_SAMPLING_CONFIG_KEY, action_sampling_metadata,
    inference_action_stride, resample_training_batch, validate_action_stride)
from wan_va.dataset.data_efficiency import crop_training_batch
from wan_va.utils import FlowMatchScheduler
from test_mcp_mask_sharing import enable_checkpointing, make_model


def native_batch(frames=13,a=16):
    data=dict(latents=torch.arange(frames).reshape(1,1,frames,1,1).float(),
              actions=torch.arange(frames*a).reshape(1,1,frames,a,1).float(),
              icl_latents=torch.randn(1,1,3,1,1), text_emb=torch.randn(1,2,8))
    data['actions'][:,:,0]=0
    data['actions_mask']=torch.ones_like(data['actions'],dtype=torch.bool)
    return data


def make_trainer(stride=1,action_reweight=False):
    model=make_model(); model.set_requires_gradient_sync=lambda enabled:None
    trainer=training.Trainer.__new__(training.Trainer)
    trainer.transformer=model; trainer.device=torch.device('cpu');trainer.dtype=torch.bfloat16
    trainer.patch_size=model.patch_size;trainer.enable_mcp=True;trainer.gradient_accumulation_steps=1
    trainer.config=SimpleNamespace(action_stride=stride,max_train_frames=4,rank=0,
        frame_chunk_size=1,max_frame_chunk_size=4,noisy_img_prob=0.,
        noisy_cond_min_timestep_bd=0.,noisy_cond_max_timestep_bd=1.,drop_icl=0.,
        icl_rope_h=4,attn_window=4,max_attn_window=4,num_mcp_modules=4,future_chunk_stride=1,
        video_loss_reweight=True,action_loss_reweight=action_reweight,mcp_loss_weights=[.5,.25,.15,.1],
        max_norm=1.,skip_step_grad_norm_multiplier=0.)
    for name,shift in [('latent',5),('action',1),('mcp',10)]:
        scheduler=FlowMatchScheduler(shift=shift,sigma_min=0.,extra_one_step=True)
        scheduler.set_timesteps(1000,training=True);setattr(trainer,'train_scheduler_'+name,scheduler)
    trainer.optimizer=torch.optim.SGD(model.parameters(),lr=.1,foreach=False)
    trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda _:1.)
    return trainer


def test_stride_one_is_noop_and_default():
    batch=native_batch();assert resample_training_batch(batch,1) is batch
    assert training.VA_CONFIGS['aloha_train'].action_stride==1


@pytest.mark.parametrize('value',[0,-1,True,1.5,'3',None])
def test_invalid_stride_rejected(value):
    with pytest.raises(ValueError,match='positive integer'):validate_action_stride(value)


@pytest.mark.parametrize('stride',[2,3,4])
@pytest.mark.parametrize('a',[4,8,12,16])
def test_joint_time_resampling_keeps_a_actions_per_latent(stride,a):
    batch=native_batch(a=a);result=resample_training_batch(batch,stride)
    video_indices=list(range(0,13,stride));f=len(video_indices)
    assert result['latents'].flatten().tolist()==video_indices
    assert result['actions'].shape==(1,1,f,a,1)
    assert torch.count_nonzero(result['actions'][:,:,0])==0
    # Remove native history A and check real source command indices globally.
    actual=result['actions'][:,:,1:].flatten()-a
    torch.testing.assert_close(actual,torch.arange(0,(f-1)*a*stride,stride).float())
    for k in range(1,f):
        last=result['actions'][0,0,k,-1,0].item()-a
        video_end=video_indices[k]*a
        assert video_end-last==stride
    assert result['icl_latents'] is batch['icl_latents']
    assert result['text_emb'] is batch['text_emb']
    assert resample_training_batch(result,stride) is result


def test_f4_stride3_has_64_targets_spanning_6_4_seconds():
    batch=native_batch(frames=13)
    result=resample_training_batch(batch,3)
    assert result['latents'].flatten().tolist()==[0,3,6,9,12]
    assert result['actions'][:,:,1:].numel()==64
    assert 12*16/30==6.4
    assert (result['actions'][0,0,1,:,0]-16).tolist()==list(range(0,48,3))
    cropped=crop_training_batch(result,4,generator=torch.Generator().manual_seed(3))
    assert cropped['latents'].shape[2]==cropped['actions'].shape[2]==4
    assert cropped['actions'].shape[3]==16


def test_incomplete_future_interval_is_rejected():
    with pytest.raises(ValueError,match='at least 4'):resample_training_batch(native_batch(frames=3),3)


@pytest.mark.parametrize('checkpointed',[False,True])
@pytest.mark.parametrize('reweight',[False,True])
def test_real_icl_mcp_optimizer_step(checkpointed,reweight):
    torch.manual_seed(19);trainer=make_trainer(3,reweight)
    if checkpointed:enable_checkpointing(trainer.transformer)
    data={k:torch.randn(shape,dtype=torch.bfloat16) for k,shape in dict(
        latents=(1,4,13,2,2),actions=(1,3,13,16,1),icl_latents=(1,4,5,2,2),
        text_emb=(1,3,8),icl_text_emb=(1,3,8)).items()}
    data['actions_mask']=torch.ones_like(data['actions'],dtype=torch.bool)
    seen=[]
    def check(module,args):
        inputs=args[0];v=inputs['latent_dict'];a=inputs['action_dict']
        assert v['latent'].shape[2]==a['latent'].shape[2]==4
        assert a['latent'].shape[3]==16 and inputs['icl_latent_dict']['latent'].shape[2]==5
        seen.append(True)
    handle=trainer.transformer.register_forward_pre_hook(check)
    before=trainer.transformer.action_proj_out.weight.detach().clone()
    losses=trainer._train_step(data,0);handle.remove()
    assert seen and not losses['optimizer_step_skipped']
    assert all(torch.isfinite(losses[k]) for k in ('latent_loss','action_loss','mcp_loss'))
    assert not torch.equal(before,trainer.transformer.action_proj_out.weight)
    assert data['latents'].shape[2]==13


@pytest.mark.parametrize('flag',['--action-stride','--action_stride'])
def test_cli_override(monkeypatch,flag):
    monkeypatch.setattr(sys,'argv',['train','--config-name','aloha_train',flag,'3'])
    class Captured(Exception):pass
    def capture(config,*args,**kwargs):
        assert config.action_stride==3;raise Captured
    monkeypatch.setattr(training,'_build_dataset_sources',capture)
    with pytest.raises(Captured):training.main()


def test_file_config_survives_default_cli(monkeypatch):
    cfg=deepcopy(training.VA_CONFIGS['aloha_train']);cfg.action_stride=3
    monkeypatch.setitem(training.VA_CONFIGS,'aloha_train',cfg)
    monkeypatch.setattr(sys,'argv',['train','--config-name','aloha_train'])
    class Captured(Exception):pass
    def capture(config,*args,**kwargs):
        assert config.action_stride==3;raise Captured
    monkeypatch.setattr(training,'_build_dataset_sources',capture)
    with pytest.raises(Captured):training.main()


def test_checkpoint_contract_and_weight_roundtrip(tmp_path):
    trainer=make_trainer(3);trainer.save_dir=tmp_path;trainer.step=7;trainer.aloha_action_contract=None
    trainer.save_checkpoint();checkpoint=tmp_path/'checkpoint_step_7'
    cfg=json.loads((checkpoint/'transformer/config.json').read_text())
    assert cfg[ACTION_SAMPLING_CONFIG_KEY]==action_sampling_metadata(3)
    assert inference_action_stride(checkpoint)==(3,True)
    with pytest.raises(ValueError,match='conflicts'):inference_action_stride(checkpoint,1)
    model=type(trainer.transformer).from_pretrained(checkpoint/'transformer',torch_dtype=torch.bfloat16)
    torch.testing.assert_close(model.action_proj_out.weight,trainer.transformer.action_proj_out.weight,rtol=0,atol=0)


def test_obsolete_token_thinning_is_not_silently_reinterpreted(tmp_path):
    p=tmp_path/'transformer';p.mkdir()
    (p/'config.json').write_text(json.dumps({ACTION_SAMPLING_CONFIG_KEY:{'version':1,'action_stride':3}}))
    with pytest.raises(ValueError,match='obsolete'):inference_action_stride(tmp_path)
    (p/'config.json').write_text('{}')
    assert inference_action_stride(tmp_path)==(1,False)
    assert inference_action_stride(tmp_path,3)==(3,False)
