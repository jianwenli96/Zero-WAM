from copy import deepcopy
from unittest.mock import patch

import pytest
import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

from wan_va.modules.icl_model import WanICLTransformer3DModel
from wan_va.utils import get_mesh_id


def make_model(cls=WanICLTransformer3DModel):
    return cls(patch_size=(1, 1, 1), num_attention_heads=2, attention_head_dim=18,
               in_channels=4, out_channels=4, action_dim=3, text_dim=8, freq_dim=4,
               ffn_dim=16, num_layers=1, action_inner_dim=36, action_ffn_dim=16,
               num_mcp_modules=4, mcp_hidden_collect_layers=(0,)).to(dtype=torch.bfloat16).train()


def make_inputs(frames=2, window=4):
    def stream(channels, height, width, shift=0, action=False):
        data = torch.randn(1, channels, frames, height, width, dtype=torch.bfloat16)
        grid = get_mesh_id(frames, height, width, 1 if action else 0, action=False)[None]
        grid[:, 0] += shift
        return dict(noisy_latents=data, latent=data.clone(),
                    timesteps=torch.full((1, frames), float(shift)),
                    cond_timesteps=torch.zeros(1, frames), grid_id=grid)
    return dict(latent_dict=stream(4, 1, 2), action_dict=stream(3, 2, 1, action=True),
                icl_latent_dict=stream(4, 1, 2),
                mcp_latent_dicts=[stream(4, 1, 2, i+1) for i in range(4)],
                text_emb=torch.randn(1, 4, 8, dtype=torch.bfloat16), encoder_seq_ids=torch.tensor([0, 0, 1, 1]),
                chunk_size=1, max_frame_chunk_size=4, window_size=window)


def masks(model):
    return [(group[0].attn1.self_block_mask, group[0].attn2.cross_block_mask)
            for group in model.mcp_blocks]


def flatten_output(output):
    return [output[0], output[1], *output[2]]


def enable_checkpointing(model):
    for i, block in enumerate(model.blocks):
        model.blocks[i] = checkpoint_wrapper(block, preserve_rng_state=False)
    for group in model.mcp_blocks:
        for i, block in enumerate(group):
            group[i] = checkpoint_wrapper(block, preserve_rng_state=False)


@pytest.mark.parametrize('checkpointed', [False, True])
def test_mcp_masks_share_storage_preserve_gradients_and_refresh(checkpointed):
    torch.manual_seed(7)
    model = make_model()
    independent = deepcopy(model)
    # Reference uses identical masks with separate storage, as before sharing.
    original_set = independent._set_masks
    def separate(blocks, self_mask, cross_mask):
        original_set(blocks, self_mask.clone(), cross_mask.clone())
    independent._set_masks = separate
    if checkpointed:
        enable_checkpointing(model)
        enable_checkpointing(independent)
    previous = None
    for frames, window in [(2, 4), (3, 0)]:
        inputs = make_inputs(frames, window)
        with patch.object(model, '_build_training_masks', wraps=model._build_training_masks) as build:
            actual = flatten_output(model(deepcopy(inputs), train_mode=True))
            assert build.call_count == 2  # backbone + one shared MCP pair
        expected = flatten_output(independent(deepcopy(inputs), train_mode=True))
        for a, b in zip(actual, expected):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        current = masks(model)
        assert all(a is current[0][0] and b is current[0][1] for a, b in current)
        assert model.blocks[0].attn1.self_block_mask is not current[0][0]
        if previous is not None:
            assert current[0][0] is not previous[0]
            assert current[0][1] is not previous[1]
        unique_bytes = sum(t.numel() * t.element_size() for t in current[0])
        old_bytes = sum(t.numel() * t.element_size() for pair in masks(independent) for t in pair)
        assert old_bytes == 4 * unique_bytes
        sum(t.float().square().mean() for t in actual).backward()
        sum(t.float().square().mean() for t in expected).backward()
        for (name, a), (_, b) in zip(model.named_parameters(), independent.named_parameters()):
            if a.grad is None:
                assert b.grad is None, name
            else:
                assert torch.isfinite(a.grad).all(), name
                torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
        previous = current[0]
        model.zero_grad(set_to_none=True)
        independent.zero_grad(set_to_none=True)
