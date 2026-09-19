from types import SimpleNamespace
import torch
from wan_va.train import Trainer
from wan_va.utils import FlowMatchScheduler
from test_mcp_mask_sharing import make_model


def test_fixed_window_crop_runs_real_training_step_with_accumulation():
    torch.manual_seed(19)
    model=make_model()
    sync=[]
    model.set_requires_gradient_sync=lambda enabled:sync.append(enabled)
    trainer=Trainer.__new__(Trainer)
    trainer.transformer=model
    trainer.device=torch.device('cpu')
    trainer.dtype=torch.bfloat16
    trainer.patch_size=model.patch_size
    trainer.enable_mcp=True
    trainer.gradient_accumulation_steps=2
    trainer.config=SimpleNamespace(max_train_frames=2,rank=0,frame_chunk_size=1,max_frame_chunk_size=2,
        noisy_img_prob=0.,noisy_cond_min_timestep_bd=0.,noisy_cond_max_timestep_bd=1.,drop_icl=0.,
        icl_rope_h=4,attn_window=4,max_attn_window=4,num_mcp_modules=4,future_chunk_stride=1,
        video_loss_reweight=True,action_loss_reweight=False,mcp_loss_weights=[.5,.25,.15,.1],
        max_norm=1.,skip_step_grad_norm_multiplier=0.)
    for name,shift in [('latent',5),('action',1),('mcp',10)]:
        scheduler=FlowMatchScheduler(shift=shift,sigma_min=0.,extra_one_step=True)
        scheduler.set_timesteps(1000,training=True)
        setattr(trainer,'train_scheduler_'+name,scheduler)
    trainer.optimizer=torch.optim.SGD(model.parameters(),lr=.01)
    trainer.lr_scheduler=torch.optim.lr_scheduler.LambdaLR(trainer.optimizer,lambda _:1.)
    data={key:torch.randn(shape,dtype=torch.bfloat16) for key,shape in dict(
        latents=(1,4,4,2,2),actions=(1,3,4,2,1),icl_latents=(1,4,5,2,2),
        text_emb=(1,3,8),icl_text_emb=(1,3,8)).items()}
    data['actions_mask']=torch.ones_like(data['actions'],dtype=torch.bool)
    observed=[]
    def verify_input(module,args):
        inputs=args[0]
        observed.append(inputs['icl_latent_dict']['latent'].shape[2])
        assert inputs['latent_dict']['noisy_latents'].shape[2]==2
        assert inputs['action_dict']['noisy_latents'].shape[2]==2
    handle=model.register_forward_pre_hook(verify_input)
    before=model.action_proj_out.weight.detach().clone()
    first=trainer._train_step(data,0)
    assert not first['should_log']
    second=trainer._train_step(data,1)
    handle.remove()
    assert sync==[False,True] and observed==[5,5]
    assert second['should_log'] and not second['optimizer_step_skipped']
    assert second['window_crop']['retained_frames']==2
    assert torch.isfinite(second['latent_loss']) and torch.isfinite(second['mcp_loss'])
    assert not torch.equal(before,model.action_proj_out.weight)
    assert data['latents'].shape[2]==4
