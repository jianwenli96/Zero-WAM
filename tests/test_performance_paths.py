from collections import Counter
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from wan_va.dataset.dataset_mixture import DistributedDatasetMixtureSampler
from wan_va.dataset.sample_costs import _latent_shape, training_sample_costs
from wan_va.modules import icl_model
from wan_va.modules.icl_model import ICLAttentionBackend as Backend


@pytest.mark.parametrize('affine', [False, True])
def test_rms_norm_cpu_fallback_preserves_state_and_gradients(affine):
    torch.manual_seed(431)
    norm = icl_model._NpuRMSNorm(16, eps=1e-6, elementwise_affine=affine)
    reference = torch.nn.RMSNorm(16, eps=1e-6, elementwise_affine=affine)
    reference.load_state_dict(norm.state_dict(), strict=True)
    x = torch.randn(2, 7, 16, requires_grad=True)
    ref_x = x.detach().clone().requires_grad_()
    a, b = norm(x), reference(ref_x)
    upstream = torch.randn_like(a)
    a.backward(upstream)
    b.backward(upstream)
    torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(x.grad, ref_x.grad, rtol=0, atol=0)
    if affine:
        torch.testing.assert_close(norm.weight.grad, reference.weight.grad,
                                   rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('shape', [(2, 127, 3072), (2, 17, 36)])
@pytest.mark.parametrize('scale', [1.0, 0.001])
def test_npu_fused_rms_outputs_and_gradients(dtype, shape, scale):
    pytest.importorskip('torch_npu')
    if not torch.npu.is_available():
        pytest.skip('NPU unavailable')
    torch.manual_seed(431)
    norm = icl_model._NpuRMSNorm(shape[-1], eps=1e-6, device='npu', dtype=dtype)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    reference = torch.nn.RMSNorm(shape[-1], eps=1e-6, device='npu', dtype=dtype)
    reference.load_state_dict(norm.state_dict(), strict=True)
    x = (torch.randn(shape, device='npu', dtype=dtype) * scale).requires_grad_()
    ref_x = x.detach().clone().requires_grad_()
    a, b = norm(x), reference(ref_x)
    upstream = torch.randn_like(a)
    a.backward(upstream)
    b.backward(upstream)
    # Include small inputs where epsilon matters and token-varying gradients.
    # Relative L2 avoids unstable per-element ratios at cancellation zeros.
    for actual, expected in ((a, b), (x.grad, ref_x.grad),
                             (norm.weight.grad, reference.weight.grad)):
        assert torch.isfinite(actual).all()
        relative_error = ((actual.float() - expected.float()).norm()
                          / expected.float().norm().clamp_min(1e-12))
        assert relative_error < (1e-3 if dtype == torch.bfloat16 else 1e-5)


def test_compact_modulation_matches_dense_with_different_spatial_sizes():
    from wan_va.modules.icl_model import (
        FrameTimestepProjection, WanICLTransformerBlock,
        _concat_timestep_projections,
    )

    torch.manual_seed(23)
    values = [torch.randn(1, frames, 6, 5, requires_grad=True)
              for frames in (2, 3)]
    reference_values = [x.detach().clone().requires_grad_() for x in values]
    indices = [torch.arange(frames).repeat_interleave(spatial)
               for frames, spatial in ((2, 3), (3, 2))]
    compact = _concat_timestep_projections([
        FrameTimestepProjection(x, ids) for x, ids in zip(values, indices)])
    dense = torch.cat([x.index_select(1, ids)
                       for x, ids in zip(reference_values, indices)], dim=1)
    table = torch.randn(1, 6, 5, requires_grad=True)
    reference_table = table.detach().clone().requires_grad_()
    actual = WanICLTransformerBlock._modulation(table, compact)
    expected = WanICLTransformerBlock._modulation(reference_table, dense)
    weights = torch.randn(6, 1, 12, 5)
    actual_loss = expected_loss = 0
    for i in range(6):
        torch.testing.assert_close(actual[i], expected[i], rtol=0, atol=0)
        actual_loss = actual_loss + (actual[i] * weights[i]).square().sum()
        expected_loss = expected_loss + (expected[i] * weights[i]).square().sum()
    actual_loss.backward()
    expected_loss.backward()
    for a, b in zip([table, *values], [reference_table, *reference_values]):
        torch.testing.assert_close(a.grad, b.grad)


@pytest.mark.parametrize('mode', ['video', 'action'])
def test_frame_timestep_embedding_matches_token_outputs_and_gradients(mode):
    from test_icl_model import _tiny_model

    torch.manual_seed(41)
    model = _tiny_model('cpu').float().train()
    embedder = (model.condition_embedder if mode == 'video'
                else model.condition_embedder_action)
    reference = deepcopy(embedder)
    channels = 4 if mode == 'video' else 3
    # Different times per frame and different upstream gradients per token:
    # this checks that spatial broadcasting accumulates parameter gradients.
    times = torch.tensor([[0., 100., 900.]])
    stream = dict(noisy_latents=torch.randn(1, channels, 3, 2, 2),
                  timesteps=times)
    rows = []
    hook = embedder.register_forward_pre_hook(
        lambda module, args: rows.append(args[0].numel()))
    _, temb, proj = model._embed_stream(stream, mode)
    proj = proj.materialize()
    hook.remove()
    expected_temb, expected_proj = reference(times.repeat_interleave(4, dim=1),
                                             dtype=torch.float32)
    expected_proj = expected_proj.unflatten(2, (6, -1))
    assert rows == [3]
    torch.testing.assert_close(temb, expected_temb)
    torch.testing.assert_close(proj, expected_proj)
    weights = [torch.randn_like(temb), torch.randn_like(proj)]
    ((temb * weights[0]).sum() + (proj * weights[1]).sum()).backward()
    ((expected_temb * weights[0]).sum()
     + (expected_proj * weights[1]).sum()).backward()
    for (name, param), (_, expected) in zip(embedder.named_parameters(),
                                           reference.named_parameters()):
        if expected.grad is None:
            assert param.grad is None, name
        else:
            torch.testing.assert_close(param.grad, expected.grad,
                                       atol=2e-5, rtol=2e-4, msg=name)


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


@pytest.mark.parametrize('dense_projection', [False, True])
def test_shared_mcp_masks_preserve_outputs_and_gradients_across_depths(dense_projection):
    from wan_va.utils import get_mesh_id
    from wan_va.modules.icl_model import WanICLTransformer3DModel
    torch.manual_seed(17)
    dtype = torch.bfloat16
    model = WanICLTransformer3DModel(
        patch_size=(1,1,1),num_attention_heads=2,attention_head_dim=18,
        in_channels=4,out_channels=4,action_dim=3,text_dim=8,freq_dim=4,
        ffn_dim=16,num_layers=1,rope_max_seq_len=32,action_inner_dim=36,
        action_ffn_dim=16,attn_window=4,enable_mcp=True,num_mcp_modules=2,
        mcp_hidden_collect_layers=(0,),
    ).to(dtype=dtype).train()
    reference=deepcopy(model)
    if dense_projection:
        from wan_va.distributed.fsdp import apply_ac
        embed = reference._embed_stream
        def dense_embed(*args, **kwargs):
            hidden, temb, projection = embed(*args, **kwargs)
            return hidden, temb, projection.materialize()
        reference._embed_stream = dense_embed
        # Exercise the compact pytree through the production checkpoint wrapper.
        apply_ac(model)
        apply_ac(reference)
    set_masks=reference._set_masks
    # Independent storage per depth is the pre-optimization lifetime behavior.
    reference._set_masks=lambda blocks,self_mask,cross_mask: set_masks(
        blocks,self_mask.clone(),cross_mask.clone())
    def stream(channels,action=False,shift=0):
        data=torch.randn(1,channels,2,2,1,dtype=dtype)
        grid=get_mesh_id(2,2,1,int(action),f_shift=shift,action=False)[None]
        return dict(noisy_latents=data,latent=data.clone(),timesteps=torch.tensor([[10.,700.]]),
                    cond_timesteps=torch.zeros(1,2),grid_id=grid)
    robot=stream(4);action=stream(3,True)
    inputs=dict(latent_dict=robot,action_dict=action,
                icl_latent_dict=dict(latent=robot['latent'],timesteps=torch.zeros(1,2),
                                     grid_id=robot['grid_id']),
                mcp_latent_dicts=[stream(4,shift=2),stream(4,shift=4)],
                text_emb=torch.randn(1,4,8,dtype=dtype),
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
            torch.testing.assert_close(a.grad,b.grad,
                                       rtol=0.02 if dense_projection else 0,
                                       atol=2e-3 if dense_projection else 0)
    first=model.mcp_blocks[0][0].attn1.self_block_mask
    assert first is model.mcp_blocks[1][0].attn1.self_block_mask
    model(deepcopy(inputs),train_mode=True)
    assert first is not model.mcp_blocks[0][0].attn1.self_block_mask
