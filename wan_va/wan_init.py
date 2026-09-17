"""Initialize the ICL MoT from original Wan weights without loading a full model."""
import argparse
from collections import Counter
from contextlib import ExitStack
import json
import math
from pathlib import Path
import re
import tempfile

import torch
from diffusers.loaders.single_file_utils import convert_wan_transformer_to_diffusers
from safetensors import safe_open
from safetensors.torch import save_file

from .modules.icl_model import WanICLTransformer3DModel


def build_plan(source_shapes, config):
    # The upstream converter only renames keys for TI2V; values can be metadata.
    renamed = convert_wan_transformer_to_diffusers(
        {key: key for key in source_shapes}
    )
    with torch.device('meta'):
        model = WanICLTransformer3DModel(**config)
    shapes = {k: tuple(v.shape) for k, v in model.state_dict().items()}
    if config.get('mcp_blocks_per_group', 1) != 1:
        raise ValueError('This initializer supports one block per IFP module')
    plan = {}
    used = set()
    for name, shape in shapes.items():
        key = name
        kind = 'video'
        if key.startswith(('action_embedder.', 'action_proj_out.')):
            plan[name] = dict(shape=shape, kind='new_action', source=None)
            continue
        if key.startswith(('mcp_mlp_hidden.', 'mcp_projections.')):
            plan[name] = dict(shape=shape, kind='new_ifp', source=None)
            continue
        if key.startswith('mcp_blocks.'):
            key = re.sub(r'^mcp_blocks\.\d+\.0\.',
                         f'blocks.{config["num_layers"] - 1}.', key)
            kind = 'ifp_copy'
        if 'action_' in key or '_action' in key:
            key = key.replace('condition_embedder_action', 'condition_embedder')
            key = key.replace('scale_shift_table_action', 'scale_shift_table')
            key = key.replace('action_', '')
            if kind == 'video':
                kind = 'action_copy'
        if key.startswith('patch_embedding_mlp.'):
            key = key.replace('patch_embedding_mlp.', 'patch_embedding.')
            kind = 'patch_reshape'
        source = renamed.get(key)
        if source is None:
            raise ValueError(f'No source for {name} ({key})')
        source_shape = tuple(source_shapes[source])
        if kind == 'patch_reshape' and name.endswith('weight'):
            source_shape = (source_shape[0], math.prod(source_shape[1:]))
        if source_shape != shape:
            raise ValueError(f'Shape mismatch: {name}: {shape} != {source_shape}')
        used.add(source)
        plan[name] = dict(shape=shape, kind=kind, source=source)
    if used != set(source_shapes):
        raise ValueError(f'Unused Wan parameters: {sorted(set(source_shapes) - used)}')
    return model, plan


def new_tensor(name, entry, plan, generator):
    tensor = torch.empty(entry['shape'], dtype=torch.float32)
    if entry['kind'] == 'new_ifp':
        if name.endswith('.bias'):
            tensor.zero_()
        else:
            tensor.normal_(std=0.02, generator=generator)
    else:
        weight_shape = plan[name.rsplit('.', 1)[0] + '.weight']['shape']
        bound = 1 / math.sqrt(weight_shape[1])
        tensor.uniform_(-bound, bound, generator=generator)
    return tensor


def convert(source_dir, output_dir, config=None, seed=42, dtype=torch.bfloat16,
            shard_bytes=3_000_000_000):
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite {output_dir}')
    if shard_bytes <= 0 or dtype not in (torch.float32, torch.bfloat16):
        raise ValueError('Use a positive shard size and float32 or bfloat16')
    raw = json.loads((source_dir / 'config.json').read_text())
    if raw.get('model_type') != 'ti2v':
        raise ValueError('Expected original Wan TI2V checkpoint')
    if config is None:
        config = dict(num_attention_heads=raw['num_heads'],
                      attention_head_dim=raw['dim'] // raw['num_heads'],
                      in_channels=raw['in_dim'], out_channels=raw['out_dim'],
                      ffn_dim=raw['ffn_dim'], freq_dim=raw['freq_dim'],
                      num_layers=raw['num_layers'], eps=raw['eps'],
                      action_inner_dim=raw['dim'], action_ffn_dim=raw['ffn_dim'])
    index = json.loads((source_dir / 'diffusion_pytorch_model.safetensors.index.json').read_text())
    weight_map = index['weight_map']
    with ExitStack() as stack:
        readers = {f: stack.enter_context(safe_open(source_dir / f, framework='pt', device='cpu'))
                   for f in sorted(set(weight_map.values()))}
        for filename, reader in readers.items():
            expected = {k for k, f in weight_map.items() if f == filename}
            if set(reader.keys()) != expected:
                raise ValueError(f'Index mismatch in {filename}')
        shapes = {k: readers[f].get_slice(k).get_shape() for k, f in weight_map.items()}
        model, plan = build_plan(shapes, config)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        # Publish only after every shard and its metadata have been written.
        with tempfile.TemporaryDirectory(prefix=output_dir.name + '.partial-',
                                         dir=output_dir.parent) as temp:
            root = Path(temp)
            target = root / 'transformer'
            target.mkdir()
            model.save_config(target)
            generator = torch.Generator(device='cpu').manual_seed(seed)
            shards, shard, size, total = [], {}, 0, 0

            def flush():
                nonlocal shard, size
                if not shard:
                    return
                filename = f'part-{len(shards) + 1:05d}.safetensors'
                save_file(shard, target / filename, metadata={'format': 'pt'})
                # Verify the serialized schema before publishing the checkpoint.
                with safe_open(target / filename, framework='pt') as saved:
                    assert set(saved.keys()) == set(shard)
                    for key, tensor in shard.items():
                        assert saved.get_slice(key).get_shape() == list(tensor.shape)
                shards.append((filename, list(shard)))
                print(f'Wrote shard {len(shards)} ({size / 1e9:.2f} GB)', flush=True)
                shard, size = {}, 0

            for name, entry in plan.items():
                nbytes = math.prod(entry['shape']) * torch.empty((), dtype=dtype).element_size()
                if size and size + nbytes > shard_bytes:
                    flush()
                if entry['source'] is None:
                    tensor = new_tensor(name, entry, plan, generator)
                else:
                    key = entry['source']
                    tensor = readers[weight_map[key]].get_tensor(key).reshape(entry['shape'])
                # Own the storage: independent copies and shared aliases serialize safely.
                shard[name] = tensor.to(dtype=dtype, copy=True).contiguous()
                size += nbytes
                total += nbytes
            flush()
            output_map = {}
            for i, (filename, names) in enumerate(shards, 1):
                final = f'diffusion_pytorch_model-{i:05d}-of-{len(shards):05d}.safetensors'
                (target / filename).rename(target / final)
                output_map.update({name: final for name in names})
            (target / 'diffusion_pytorch_model.safetensors.index.json').write_text(
                json.dumps({'metadata': {'total_size': total}, 'weight_map': output_map}, indent=2) + '\n')
            report = dict(source=str(source_dir), seed=seed, dtype=str(dtype),
                          counts=dict(Counter(e['kind'] for e in plan.values())),
                          total_size=total, parameters=plan,
                          initialization={'new_action': 'Linear default uniform +/- 1/sqrt(fan_in), including bias',
                                          'new_ifp': 'normal std=0.02, zero bias; implementation choice, not a confirmed paper recipe'},
                          scope='Transformer only; VAE/text encoder/tokenizer are not converted')
            (root / 'initialization.json').write_text(json.dumps(report, indent=2) + '\n')
            root.rename(output_dir)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', choices=['bfloat16', 'float32'], default='bfloat16')
    parser.add_argument('--shard-size-gb', type=float, default=3)
    args = parser.parse_args()
    report = convert(args.source, args.output, seed=args.seed,
                     dtype=getattr(torch, args.dtype), shard_bytes=int(args.shard_size_gb * 1e9))
    print(json.dumps({k: v for k, v in report.items() if k != 'parameters'}, indent=2))


if __name__ == '__main__':
    main()
