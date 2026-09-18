from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wan_va.dataset.dataset_mixture import DistributedDatasetMixtureSampler
from wan_va.dataset.sample_costs import _latent_shape, training_sample_costs
from wan_va.modules import icl_model
from wan_va.modules.icl_model import ICLAttentionBackend as Backend


def test_tiled_training_and_streaming_masks_match_dense(monkeypatch):
    torch.manual_seed(9)
    seq = torch.tensor([0, 0, 0, 1, 1, -1, 0, 1, 0, -1])
    frame = torch.randint(-1, 8, (10,))
    noise = torch.randint(0, 2, (10,))
    types = torch.randint(0, 2, (10,))
    icl = torch.randint(0, 2, (10,))
    for window in [-1, 0, 4]:
        monkeypatch.setattr(icl_model, '_DENSE_MASK_DIRECT_ELEMENTS', 10000)
        expected = Backend.build_training_self_mask(seq, frame, noise, types, icl,
                                                    window, 'cpu', False)
        streaming = Backend.build_self_mask(types[:7], types, seq[:7], seq,
                                              frame[:7], frame, window, 'cpu', False)
        monkeypatch.setattr(icl_model, '_DENSE_MASK_DIRECT_ELEMENTS', 0)
        monkeypatch.setattr(icl_model, '_DENSE_MASK_TILE_ELEMENTS', 23)
        actual = Backend.build_training_self_mask(seq, frame, noise, types, icl,
                                                  window, 'cpu', False)
        assert torch.equal(actual, expected)
        assert torch.equal(Backend.build_self_mask(types[:7], types, seq[:7], seq,
                            frame[:7], frame, window, 'cpu', False), streaming)


def test_bucket_sampler_preserves_each_window_and_balances_ranks():
    kwargs = dict(dataset_lengths=[100, 100], dataset_weights=[4, 1],
                  seed=19, epoch_size=211, num_replicas=7)
    costs = list(range(1, 201))
    def global_draws(bucket):
        ranks = [list(DistributedDatasetMixtureSampler(**kwargs, rank=rank,
                 sample_costs=costs, bucket_steps=bucket)) for rank in range(7)]
        return [index for step in zip(*ranks) for index in step]
    original = global_draws(0)
    grouped = global_draws(10)
    assert grouped == global_draws(10)
    for start in range(0, len(original), 70):
        assert Counter(original[start:start+70]) == Counter(grouped[start:start+70])
    def work(draws):
        return sum(max(costs[i] for i in draws[j:j+7]) for j in range(0,len(draws),7))
    assert work(grouped) < work(original)
    sampler = DistributedDatasetMixtureSampler(**kwargs, rank=0,
                     sample_costs=costs, bucket_steps=10)
    old = list(sampler)
    sampler.set_epoch(1)
    assert list(sampler) != old


@pytest.mark.parametrize('costs,steps', [(None,2), ([1],2), ([0]*4,2), ([1]*4,-1)])
def test_bucket_sampler_validates_costs(costs,steps):
    with pytest.raises(ValueError):
        DistributedDatasetMixtureSampler([4],[1],sample_costs=costs,bucket_steps=steps)


def test_cost_reader_uses_latent_shapes_and_full_human_prompt(tmp_path):
    paths=[]
    for name,shape in [('robot',(100,14,18)),('human',(46,18,32))]:
        path=tmp_path/f'{name}.pth'
        torch.save(dict(latent=torch.randn(2,3), latent_num_frames=shape[0],
                        latent_height=shape[1],latent_width=shape[2]),path)
        paths.append(path)
    task=SimpleNamespace(new_metas=[{}],used_video_keys=['a','b'],root=tmp_path,
         _latent_file=lambda meta,camera:paths[0],_lookup_icl_sample=lambda meta:{},
         _human_latent_candidate=lambda sample:paths[1])
    assert _latent_shape(str(paths[0])) == (100,14,18)
    assert training_sample_costs(SimpleNamespace(_datasets=[task]),64) == [
        2*(64*7*9*2+64*16)+46*9*16]


def test_shared_mcp_masks_preserve_outputs_and_gradients_across_depths():
    from wan_va.utils import get_mesh_id
    from wan_va.modules.icl_model import WanICLTransformer3DModel
    torch.manual_seed(17)
    model = WanICLTransformer3DModel(
        patch_size=(1,1,1),num_attention_heads=2,attention_head_dim=18,
        in_channels=4,out_channels=4,action_dim=3,text_dim=8,freq_dim=4,
        ffn_dim=16,num_layers=1,rope_max_seq_len=32,action_inner_dim=36,
        action_ffn_dim=16,attn_window=4,enable_mcp=True,num_mcp_modules=2,
        mcp_hidden_collect_layers=(0,),
    ).to(dtype=torch.bfloat16).train()
    reference=deepcopy(model)
    set_masks=reference._set_masks
    # Independent storage per depth is the pre-optimization lifetime behavior.
    reference._set_masks=lambda blocks,self_mask,cross_mask: set_masks(
        blocks,self_mask.clone(),cross_mask.clone())
    def stream(channels,action=False,shift=0):
        data=torch.randn(1,channels,2,2,1,dtype=torch.bfloat16)
        grid=get_mesh_id(2,2,1,int(action),f_shift=shift,action=False)[None]
        return dict(noisy_latents=data,latent=data.clone(),timesteps=torch.zeros(1,2),
                    cond_timesteps=torch.zeros(1,2),grid_id=grid)
    robot=stream(4);action=stream(3,True)
    inputs=dict(latent_dict=robot,action_dict=action,
                icl_latent_dict=dict(latent=robot['latent'],timesteps=torch.zeros(1,2),
                                     grid_id=robot['grid_id']),
                mcp_latent_dicts=[stream(4,shift=2),stream(4,shift=4)],
                text_emb=torch.randn(1,4,8,dtype=torch.bfloat16),
                encoder_seq_ids=torch.tensor([0,0,1,1]),chunk_size=2,
                max_frame_chunk_size=4,window_size=4)
    out=model(deepcopy(inputs),train_mode=True)
    expected=reference(deepcopy(inputs),train_mode=True)
    actual_flat=[out[0],out[1],*out[2]]
    expected_flat=[expected[0],expected[1],*expected[2]]
    for a,b in zip(actual_flat,expected_flat):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    sum(x.float().square().mean() for x in actual_flat).backward()
    sum(x.float().square().mean() for x in expected_flat).backward()
    for a,b in zip(model.parameters(),reference.parameters()):
        if a.grad is not None:
            assert torch.isfinite(a.grad).all()
            torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)
    first=model.mcp_blocks[0][0].attn1.self_block_mask
    assert first is model.mcp_blocks[1][0].attn1.self_block_mask
    model(deepcopy(inputs),train_mode=True)
    assert first is not model.mcp_blocks[0][0].attn1.self_block_mask
