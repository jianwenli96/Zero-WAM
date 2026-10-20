"""Static RTC reuse must preserve dynamic samples and their input gradients."""
import pytest
import torch

from test_icl_model import _tiny_model, _input
from test_icl_server import _mock_server
from wan_va.rtc.cache import request_cache
from wan_va.rtc.guidance import guided_noise_velocity


@pytest.mark.parametrize('cfg', ['none', 'icl', 'text'])
@pytest.mark.parametrize('mode', ['video', 'action'])
def test_mesh_reuse_preserves_cfg_metadata_and_is_request_scoped(monkeypatch, cfg, mode):
    import wan_va.wan_va_server as module
    server = _mock_server()
    server._rtc_stream_position = (4, 3)
    server.use_icl_cfg = cfg == 'icl'
    server.target_text_cfg_active = cfg == 'text'
    server.negative_prompt_embeds = torch.ones_like(server.target_prompt_embeds)
    data = torch.randn(1, 4, 2, 1, 1)
    baseline = server._prepare_icl_stream(data, 123., mode, 1)
    original = module.get_mesh_id
    calls = []
    def tracked(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'get_mesh_id', tracked)
    server.rtc_stream_layout = True
    with request_cache():
        actual = server._prepare_icl_stream(data, 123., mode, 1)
        server._prepare_icl_stream(data+1, 42., mode, 1)
        server._prepare_icl_stream(data+2, 12., mode, 1)
        assert len(calls) == 1
    assert actual['_rtc_padding_cache_type'] == 1
    for key in ('latent_grid_id' if mode == 'video' else 'action_grid_id',
                'current_seq_ids', 'current_frame_ids', 'encoder_seq_ids', 'text_emb'):
        torch.testing.assert_close(actual[key], baseline[key], atol=0, rtol=0)
    stream = 'latent_res_lst' if mode == 'video' else 'action_res_lst'
    for key in ('noisy_latents', 'timesteps', 'cache_type_ids'):
        torch.testing.assert_close(actual[stream][key], baseline[stream][key], atol=0, rtol=0)
    with request_cache():
        server._prepare_icl_stream(data, 123., mode, 1)
    assert len(calls) == 2


def test_cached_text_kv_preserves_repeated_real_action_vjps():
    model = _tiny_model('cpu').requires_grad_(False)
    with torch.no_grad():
        model(_input(model, 'video', 0), update_cache=1, mode='forward_latent_only')
    data = _input(model, 'action', 1, frame_id=1)
    data['_rtc_static_key'] = ('action', 1)
    sample = data['action_res_lst']['noisy_latents']
    target, weights = torch.zeros_like(sample), torch.ones_like(sample)
    count = []
    hook = model.blocks[0].attn2.to_k.register_forward_hook(lambda *_: count.append(1))
    def velocity(x):
        stream = dict(data['action_res_lst'], noisy_latents=x)
        output = model(dict(data, action_res_lst=stream), mode='forward_action_only')
        return output.transpose(1, 2).reshape_as(x)
    try:
        with request_cache():
            expected = [guided_noise_velocity(sample, velocity, s, target, weights, 5)
                        for s in [1., .7, .3]]
        assert len(count) == 3
        count.clear()
        model.blocks[0].attn2.rtc_cache_text_kv = True
        with request_cache():
            actual = [guided_noise_velocity(sample, velocity, s, target, weights, 5)
                      for s in [1., .7, .3]]
        assert len(count) == 1
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert all(p.grad is None for p in model.parameters())
    finally:
        hook.remove()


def test_text_kv_reuse_invalidates_edits_and_preserves_text_gradients():
    model = _tiny_model('cpu').requires_grad_(False)
    attention = model.blocks[0].attn2
    attention.rtc_cache_text_kv = True
    text = torch.randn(1, 2, 36, dtype=torch.bfloat16)
    query = torch.randn(1, 2, 36, dtype=torch.bfloat16)
    padding = torch.zeros(1, 1, 36, dtype=torch.bfloat16)
    calls = []
    hook = attention.to_k.register_forward_hook(lambda *_: calls.append(1))
    try:
        with request_cache():
            first = attention._project(None, query, padding, text)[1]
            repeated = attention._project(None, query, padding, text)[1]
            assert len(calls) == 1
            torch.testing.assert_close(first, repeated, atol=0, rtol=0)
            text.add_(1)
            edited = attention._project(None, query, padding, text)[1]
            assert len(calls) == 2
            assert not torch.equal(first, edited)
        text = text.detach().requires_grad_(True)
        with request_cache():
            for _ in range(2):
                _, key, value, _, _ = attention._project(None, query, padding, text)
                gradient, = torch.autograd.grad(key.float().sum()+value.float().sum(), text)
                assert torch.isfinite(gradient).all()
        assert len(calls) == 4  # differentiable text is recomputed each time
    finally:
        hook.remove()
