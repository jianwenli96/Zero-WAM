"""Validate original-Wan conversion through serialized ICL checkpoints."""
import json

import pytest
import torch
from safetensors.torch import save_file

from wan_va.modules.icl_model import WanICLTransformer3DModel
from wan_va.wan_weight_init import build_plan, convert


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_wan_conversion_preserves_video_and_initializes_independent_experts(tmp_path, monkeypatch, dtype):
    config = dict(num_attention_heads=2, attention_head_dim=18, num_layers=2,
                  in_channels=4, out_channels=4, action_dim=3, text_dim=8,
                  freq_dim=4, ffn_dim=48, action_inner_dim=36, action_ffn_dim=48,
                  num_mcp_modules=2, mcp_hidden_collect_layers=(0, 1))
    template = WanICLTransformer3DModel(**config)
    rename = {
        'condition_embedder.time_embedder.linear_1': 'time_embedding.0',
        'condition_embedder.time_embedder.linear_2': 'time_embedding.2',
        'condition_embedder.time_proj': 'time_projection.1',
        'condition_embedder.text_embedder.linear_1': 'text_embedding.0',
        'condition_embedder.text_embedder.linear_2': 'text_embedding.2',
        'attn1': 'self_attn', 'attn2': 'cross_attn',
        '.to_q.': '.q.', '.to_k.': '.k.', '.to_v.': '.v.', '.to_out.0.': '.o.',
        'ffn.net.0.proj': 'ffn.0', 'ffn.net.2': 'ffn.2',
        'norm2': 'norm3', 'scale_shift_table': 'modulation',
    }
    original = {}
    for name, value in template.state_dict().items():
        if 'action' in name or name.startswith(('mcp_', 'patch_embedding_mlp')):
            continue
        if name == 'scale_shift_table':
            key = 'head.modulation'
        elif name.startswith('proj_out.'):
            key = name.replace('proj_out.', 'head.head.')
        else:
            key = name
            for before, after in rename.items():
                key = key.replace(before, after)
        original[key] = value.clone()
    source = tmp_path / 'wan'
    source.mkdir()
    (source / 'config.json').write_text('{"model_type": "ti2v"}')
    save_file(original, source / 'source.safetensors')
    (source / 'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps(dict(weight_map={k: 'source.safetensors' for k in original})))
    result = tmp_path / 'converted'
    report = convert(source, result, config=config, seed=91, dtype=dtype, shard_bytes=30_000)
    model, info = WanICLTransformer3DModel.from_pretrained(
        result / 'transformer', torch_dtype=torch.float32, output_loading_info=True)
    assert info['missing_keys'] == [] and info['unexpected_keys'] == []
    for name, entry in report['parameters'].items():
        if entry['source'] is not None:
            torch.testing.assert_close(model.state_dict()[name],
                                       original[entry['source']].reshape(entry['shape']).to(dtype).float(), rtol=0, atol=0)
    assert json.loads((result / 'initialization.json').read_text())['seed'] == 91
    repeated = tmp_path / 'repeated'
    convert(source, repeated, config=config, seed=91, dtype=dtype)
    repeated_model = WanICLTransformer3DModel.from_pretrained(
        repeated / 'transformer', torch_dtype=torch.float32)
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, repeated_model.state_dict()[name], rtol=0, atol=0)
    assert model.blocks[0].attn1.to_q.weight.data_ptr() != model.blocks[0].attn1.action_to_q.weight.data_ptr()
    assert model.blocks[0].attn2.to_k is model.blocks[0].attn2.action_to_k
    patches = torch.randn(1, 4, 2, 4, 4)
    expected = model.patch_embedding(patches).flatten(2).transpose(1, 2)
    embedded, _, _ = model._embed_stream(dict(noisy_latents=patches, timesteps=torch.zeros(1, 2)), 'video')
    torch.testing.assert_close(embedded, expected)
    with pytest.raises(FileExistsError):
        convert(source, result, config=config)
    shapes = {key: tuple(tensor.shape) for key, tensor in original.items()}
    with pytest.raises(ValueError, match='Unused'):
        build_plan(dict(shapes, unexpected=(1,)), config)
    with pytest.raises(ValueError, match='Shape mismatch'):
        build_plan(dict(shapes, **{'head.head.weight': (1, 1)}), config)
    with pytest.raises(ValueError, match='Index mismatch'):
        (source / 'diffusion_pytorch_model.safetensors.index.json').write_text(
            json.dumps(dict(weight_map={k: 'source.safetensors' for k in original if k != 'head.head.weight'})))
        convert(source, tmp_path / 'bad-index', config=config)
    (source / 'diffusion_pytorch_model.safetensors.index.json').write_text(
        json.dumps(dict(weight_map={k: 'source.safetensors' for k in original})))
    def fail_write(*args, **kwargs):
        raise OSError('simulated disk failure')
    monkeypatch.setattr('wan_va.wan_weight_init.save_file', fail_write)
    with pytest.raises(OSError, match='simulated disk failure'):
        convert(source, tmp_path / 'failed', config=config)
    assert not (tmp_path / 'failed').exists()
    assert not list(tmp_path.glob('failed.partial-*'))
    assert not list(tmp_path.glob('converted.partial-*'))
