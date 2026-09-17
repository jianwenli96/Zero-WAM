import json
import pytest
import torch
from safetensors.torch import save_file
from wan_va.modules.icl_model import WanICLTransformer3DModel
from wan_va.wan_init import build_plan, convert

CONFIG = dict(num_attention_heads=2, attention_head_dim=18, in_channels=4,
              out_channels=4, action_dim=3, text_dim=8, freq_dim=4,
              ffn_dim=16, num_layers=1, action_inner_dim=36, action_ffn_dim=16,
              num_mcp_modules=4, mcp_blocks_per_group=1,
              mcp_hidden_collect_layers=(0,), rope_max_seq_len=32)


def source_checkpoint(path, config=CONFIG):
    torch.manual_seed(9)
    model = WanICLTransformer3DModel(**config)
    source = {}
    replacements = [
        ('condition_embedder.time_embedder.linear_1', 'time_embedding.0'),
        ('condition_embedder.time_embedder.linear_2', 'time_embedding.2'),
        ('condition_embedder.text_embedder.linear_1', 'text_embedding.0'),
        ('condition_embedder.text_embedder.linear_2', 'text_embedding.2'),
        ('condition_embedder.time_proj', 'time_projection.1'),
        ('attn1', 'self_attn'), ('attn2', 'cross_attn'),
        ('.to_out.0.', '.o.'), ('.to_q.', '.q.'), ('.to_k.', '.k.'), ('.to_v.', '.v.'),
        ('ffn.net.0.proj', 'ffn.0'), ('ffn.net.2', 'ffn.2'),
        ('norm2', 'norm3'), ('scale_shift_table', 'modulation')]
    for name, tensor in model.state_dict().items():
        if 'action' in name or name.startswith(('mcp_', 'patch_embedding_mlp')):
            continue
        key = name
        if key.startswith('proj_out.'):
            key = key.replace('proj_out.', 'head.head.')
        elif key == 'scale_shift_table':
            key = 'head.modulation'
        else:
            for a, b in replacements:
                key = key.replace(a, b)
        source[key] = tensor.clone()
    path.mkdir()
    save_file(source, path / 'weights.safetensors')
    (path / 'config.json').write_text(json.dumps({'model_type': 'ti2v'}))
    (path / 'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps({'weight_map': {k: 'weights.safetensors' for k in source}}))
    return source


def test_conversion_reload_copies_and_patch_equivalence(tmp_path):
    source = source_checkpoint(tmp_path / 'source')
    report = convert(tmp_path / 'source', tmp_path / 'out', CONFIG,
                     dtype=torch.float32, shard_bytes=40_000)
    model, info = WanICLTransformer3DModel.from_pretrained(
        tmp_path / 'out' / 'transformer', output_loading_info=True, torch_dtype=torch.float32)
    assert not info['missing_keys'] and not info['unexpected_keys']
    state = model.state_dict()
    for name, entry in report['parameters'].items():
        if entry['source']:
            torch.testing.assert_close(state[name], source[entry['source']].reshape(entry['shape']), rtol=0, atol=0)
    assert model.blocks[0].attn2.action_to_k is model.blocks[0].attn2.to_k
    assert model.blocks[0].attn1.action_to_q.weight is not model.blocks[0].attn1.to_q.weight
    data = torch.randn(1, 4, 2, 4, 6)
    patches = data.reshape(1, 4, 2, 1, 2, 2, 3, 2).permute(0, 2, 4, 6, 1, 3, 5, 7).reshape(1, 12, 16)
    expected = model.patch_embedding(data).flatten(2).transpose(1, 2)
    actual = model.patch_embedding_mlp(patches)
    torch.testing.assert_close(actual, expected)
    loss = actual.square().mean() + model.action_proj_out(model.action_embedder(torch.randn(2, 3))).square().mean()
    loss.backward()
    assert torch.isfinite(model.action_embedder.weight.grad).all()
    assert torch.isfinite(model.patch_embedding_mlp.weight.grad).all()
    with pytest.raises(FileExistsError):
        convert(tmp_path / 'source', tmp_path / 'out', CONFIG)


def test_determinism_and_strict_source_validation(tmp_path):
    source = source_checkpoint(tmp_path / 'source')
    shapes = {k: tuple(v.shape) for k, v in source.items()}
    with pytest.raises(ValueError, match='Unused'):
        build_plan({**shapes, 'extra': (1,)}, CONFIG)
    with pytest.raises(ValueError, match='Shape mismatch'):
        build_plan({**shapes, 'head.head.weight': (1, 1)}, CONFIG)
    for name in ('a', 'b'):
        convert(tmp_path / 'source', tmp_path / name, CONFIG, seed=3)
    a = WanICLTransformer3DModel.from_pretrained(tmp_path / 'a' / 'transformer', torch_dtype=torch.bfloat16)
    b = WanICLTransformer3DModel.from_pretrained(tmp_path / 'b' / 'transformer', torch_dtype=torch.bfloat16)
    for key, tensor in a.state_dict().items():
        assert torch.equal(tensor, b.state_dict()[key])


def test_converted_model_training_with_ifp(tmp_path, monkeypatch):
    import test_icl_model as training_test
    config = dict(CONFIG, patch_size=(1, 1, 1), num_mcp_modules=1)
    source_checkpoint(tmp_path / 'source', config)
    convert(tmp_path / 'source', tmp_path / 'out', config)
    model = WanICLTransformer3DModel.from_pretrained(
        tmp_path / 'out' / 'transformer', torch_dtype=torch.bfloat16)
    monkeypatch.setattr(training_test, '_tiny_model', lambda *a, **kw: model)
    training_test.test_icl_training_forward_and_backward("cpu", True)
