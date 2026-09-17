import pytest
import torch

from wan_va.modules.icl_model import (
    ICLAttentionBackend,
    ICL_CACHE_TYPE,
    OBSERVATION_CACHE_TYPE,
    PREDICTION_CACHE_TYPE,
    WanICLTransformer3DModel,
)
from wan_va.utils import get_mesh_id


@pytest.fixture(params=["cpu", "cuda", "npu"])
def attention_device(request):
    device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "npu":
        pytest.importorskip("torch_npu")
        if not torch.npu.is_available():
            pytest.skip("NPU unavailable")
    return device


def _mask_values(mask, query_length, key_length):
    if isinstance(mask, torch.Tensor):
        return mask[0, 0].cpu()
    index = torch.tensor(0)
    return torch.tensor([
        [bool(mask.mask_mod(index, index, q, k)) for k in range(key_length)]
        for q in range(query_length)
    ])


def _tiny_model(device, enable_mcp=False):
    model = WanICLTransformer3DModel(
        patch_size=(1, 1, 1),
        num_attention_heads=2,
        attention_head_dim=18,
        in_channels=4,
        out_channels=4,
        action_dim=3,
        text_dim=8,
        freq_dim=4,
        ffn_dim=16,
        num_layers=1,
        rope_max_seq_len=32,
        action_inner_dim=36,
        action_ffn_dim=16,
        attn_window=4,
        enable_mcp=enable_mcp,
        num_mcp_modules=1,
        mcp_hidden_collect_layers=(0,),
    )
    return model.to(device=device, dtype=torch.bfloat16).eval()


def _input(model, mode, cache_type, frame_id=0):
    device = next(model.parameters()).device
    if mode == "video":
        data = torch.randn(1, 4, 1, 1, 2, device=device, dtype=torch.bfloat16)
        grid = get_mesh_id(1, 1, 2, 0).to(device)
        stream_key, grid_key = "latent_res_lst", "latent_grid_id"
    else:
        data = torch.randn(1, 3, 1, 2, 1, device=device, dtype=torch.bfloat16)
        grid = get_mesh_id(1, 2, 1, 1, action=False).to(device)
        stream_key, grid_key = "action_res_lst", "action_grid_id"
    token_count = grid.shape[1]
    return {
        stream_key: {
            "noisy_latents": data,
            "timesteps": torch.zeros(1, device=device),
            "cache_type_ids": torch.full(
                [token_count], cache_type, device=device, dtype=torch.int
            ),
        },
        grid_key: grid,
        "current_seq_ids": torch.zeros(token_count, device=device, dtype=torch.int),
        "current_frame_ids": torch.full(
            [token_count], frame_id, device=device, dtype=torch.int
        ),
        "encoder_seq_ids": torch.zeros(2, device=device, dtype=torch.int),
        "text_emb": torch.randn(
            1, 2, 8, device=device, dtype=torch.bfloat16
        ),
    }


def _training_mask_values(device):
    seq_ids = torch.zeros(9, dtype=torch.int)
    frame_ids = torch.tensor([0, 2, 0, 2, 0, 1, 3, 1, 3])
    noise_ids = torch.tensor([0, 0, 1, 1, 1, 0, 0, 1, 1])
    type_ids = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1])
    icl_ids = torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0])
    mask = ICLAttentionBackend.build_training_self_mask(
        seq_ids=seq_ids.to(device),
        frame_ids=frame_ids.to(device),
        noise_ids=noise_ids.to(device),
        type_ids=type_ids.to(device),
        icl_ids=icl_ids.to(device),
        window_size=4,
        device=torch.device(device),
        compile_mask=False,
    )
    actual = _mask_values(mask, 9, 9)

    expected = torch.zeros(9, 9, dtype=torch.bool)
    for q_idx in range(9):
        for kv_idx in range(9):
            if icl_ids[q_idx] == 0 and icl_ids[kv_idx] == 1:
                expected[q_idx, kv_idx] = type_ids[q_idx] == 0
                continue
            if icl_ids[q_idx] == 1 and icl_ids[kv_idx] == 1:
                expected[q_idx, kv_idx] = True
                continue
            if icl_ids[q_idx] != 0 or icl_ids[kv_idx] != 0:
                continue
            if abs(frame_ids[q_idx] - frame_ids[kv_idx]) > 4:
                continue
            clean_to_clean = (
                noise_ids[q_idx] == 1
                and noise_ids[kv_idx] == 1
                and frame_ids[kv_idx] <= frame_ids[q_idx]
            )
            noisy_to_clean = (
                noise_ids[q_idx] == 0
                and noise_ids[kv_idx] == 1
                and frame_ids[kv_idx] < frame_ids[q_idx]
            )
            noisy_to_noisy = (
                noise_ids[q_idx] == 0
                and noise_ids[kv_idx] == 0
                and frame_ids[kv_idx] == frame_ids[q_idx]
            )
            expected[q_idx, kv_idx] = bool(
                clean_to_clean or noisy_to_clean or noisy_to_noisy
            )
    return actual, expected


def test_training_mask_extends_next_forcing_with_icl_rules(attention_device):
    actual, expected = _training_mask_values(attention_device)
    assert torch.equal(actual, expected)


def test_icl_cache_survives_prediction_cleanup(attention_device):
    model = _tiny_model(attention_device)
    with torch.inference_mode():
        model(
            _input(model, "video", ICL_CACHE_TYPE),
            update_cache=1,
            mode="forward_latent_only",
        )
        assert model.cache_counts()[ICL_CACHE_TYPE] == 2
        assert model.cache_type_ids_cache.shape[0] == 128
        assert (model.cache_type_ids_cache == ICL_CACHE_TYPE).all()

        prediction = model(
            _input(model, "video", PREDICTION_CACHE_TYPE, frame_id=2),
            update_cache=1,
            mode="forward_latent_only",
        )
        assert torch.isfinite(prediction).all()
        assert model.cache_counts()[PREDICTION_CACHE_TYPE] == 2

        action = model(
            _input(model, "action", PREDICTION_CACHE_TYPE, frame_id=3),
            mode="forward_action_only",
        )
        assert torch.isfinite(action).all()

        model.clear_prediction_cache()
        assert model.cache_counts()[ICL_CACHE_TYPE] == 2
        assert model.cache_counts()[PREDICTION_CACHE_TYPE] == 0
        assert model.cache_type_ids_cache.shape[0] == 128

        model(
            _input(model, "video", OBSERVATION_CACHE_TYPE, frame_id=0),
            update_cache=1,
            mode="forward_latent_only",
        )
        model.clear_prediction_cache()
        assert model.cache_counts()[ICL_CACHE_TYPE] == 2
        assert model.cache_counts()[OBSERVATION_CACHE_TYPE] == 2
        assert model.cache_type_ids_cache.shape[0] == 256


def test_released_checkpoint_schema():
    with torch.device("meta"):
        model = WanICLTransformer3DModel()
    state = model.state_dict()
    assert len(state) == 1880
    assert state["action_embedder.weight"].shape == (3072, 30)
    assert state["mcp_mlp_hidden.0.weight"].shape == (3072, 12288)
    assert len(model.mcp_blocks) == 4
    assert all(len(group) == 1 for group in model.mcp_blocks)


def test_rope_frequency_buffers_follow_model_dtype():
    model = _tiny_model("cpu")
    assert model.rope.f_freqs_base.dtype == torch.bfloat16
    assert model.rope.h_freqs_base.dtype == torch.bfloat16
    assert model.rope.w_freqs_base.dtype == torch.bfloat16


def test_observation_window_uses_next_frame_boundary():
    model = _tiny_model("cpu")
    model.type_ids_cache = torch.tensor([2, 0, 0, 0])
    model.seq_ids_cache = torch.zeros(4, dtype=torch.int)
    model.frame_ids_cache = torch.tensor([0, 0, 5, 6])
    model.cache_type_ids_cache = torch.tensor(
        [ICL_CACHE_TYPE, OBSERVATION_CACHE_TYPE, OBSERVATION_CACHE_TYPE,
         PREDICTION_CACHE_TYPE]
    )
    for block in model.blocks:
        block.attn1.attn_caches["pos"] = {
            "k": torch.zeros(1, 4, model.config.num_attention_heads, 18),
            "v": torch.zeros(1, 4, model.config.num_attention_heads, 18),
        }

    model.clear_prediction_cache(attn_window=4)

    assert model.cache_type_ids_cache.tolist() == [
        ICL_CACHE_TYPE,
        OBSERVATION_CACHE_TYPE,
    ]
    assert model.frame_ids_cache.tolist() == [0, 5]


@pytest.mark.parametrize("enable_mcp", [False, True])
def test_icl_training_forward_and_backward(attention_device, enable_mcp):
    model = _tiny_model(attention_device, enable_mcp=enable_mcp).train()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    latent_data = torch.randn(1, 4, 2, 1, 2, device=device, dtype=dtype)
    action_data = torch.randn(1, 3, 2, 2, 1, device=device, dtype=dtype)
    icl_data = torch.randn(1, 4, 2, 1, 2, device=device, dtype=dtype)
    latent_grid = get_mesh_id(2, 1, 2, 0).to(device)[None]
    action_grid = get_mesh_id(2, 2, 1, 1, action=False).to(device)[None]
    icl_grid = get_mesh_id(2, 1, 2, 0, h_shift=4).to(device)[None]

    def stream(data, grid):
        return {
            "noisy_latents": data,
            "latent": data.clone(),
            "timesteps": torch.zeros(1, 2, device=device),
            "cond_timesteps": torch.zeros(1, 2, device=device),
            "grid_id": grid,
        }

    output = model(
        {
            "mcp_latent_dicts": [stream(latent_data, latent_grid)] if enable_mcp else [],
            "latent_dict": stream(latent_data, latent_grid),
            "action_dict": stream(action_data, action_grid),
            "icl_latent_dict": {
                "latent": icl_data,
                "timesteps": torch.zeros(1, 2, device=device),
                "grid_id": icl_grid,
            },
            "text_emb": torch.randn(1, 4, 8, device=device, dtype=dtype),
            "encoder_seq_ids": torch.tensor([0, 0, 1, 1], device=device),
            "chunk_size": 2,
            "max_frame_chunk_size": 4,
            "window_size": 4,
        },
        train_mode=True,
    )
    latent_output, action_output = output[:2]
    assert latent_output.shape == (1, 4, 4)
    assert action_output.shape == (1, 4, 3)
    loss = latent_output.float().square().mean() + action_output.float().square().mean()
    if enable_mcp:
        assert len(output[2]) == 1
        assert output[2][0].shape == latent_output.shape
        loss = loss + output[2][0].float().square().mean()
    loss.backward()
    assert torch.isfinite(model.patch_embedding_mlp.weight.grad).all()
    assert torch.isfinite(model.action_embedder.weight.grad).all()
    if enable_mcp:
        assert torch.isfinite(model.mcp_projections[0].weight.grad).all()


@pytest.mark.parametrize("window_size", [-1, 0, 4])
def test_streaming_masks_preserve_icl_window_and_sequence_rules(
    attention_device, window_size
):
    device = attention_device
    # Video/action queries, two sequences, padding, persistent ICL keys,
    # and a rectangular cache (more keys than queries).
    query_types = [0, 1, 0, -1]
    key_types = [2, 0, 1, 2, 0, -1]
    query_seqs = [0, 0, 1, -1]
    key_seqs = [0, 0, 0, 1, 1, -1]
    query_frames = [8, 8, 0, -1]
    key_frames = [0, 8, 7, 20, 0, -1]
    metadata = [query_types, key_types, query_seqs, key_seqs,
                query_frames, key_frames]
    mask = ICLAttentionBackend.build_self_mask(
        *(torch.tensor(values, device=device) for values in metadata),
        window_size, device, compile_mask=False,
    )
    expected = torch.zeros(4, 6, dtype=torch.bool)
    for q in range(4):
        for k in range(6):
            if query_seqs[q] != key_seqs[k] or key_types[k] == -1:
                continue
            if key_types[k] == ICL_CACHE_TYPE:
                expected[q, k] = query_types[q] == 0
            else:
                expected[q, k] = (
                    window_size == -1
                    or abs(query_frames[q] - key_frames[k]) <= window_size
                )
    assert torch.equal(_mask_values(mask, 4, 6), expected)

    cross = ICLAttentionBackend.build_cross_mask(
        torch.tensor(query_seqs, device=device),
        torch.tensor([0, 0, 1], device=device), device,
        compile_mask=False,
    )
    assert torch.equal(_mask_values(cross, 4, 3), torch.tensor([
        [True, True, False], [True, True, False],
        [False, False, True], [False, False, False],
    ]))


def test_attention_matches_reference_with_fully_masked_rows(attention_device):
    torch.manual_seed(42)
    # Use a non-square cross mask and B > 1 to catch layout/broadcast errors.
    q_cpu = torch.randn(2, 4, 2, 32, dtype=torch.bfloat16)
    k_cpu = torch.randn(2, 6, 2, 32, dtype=torch.bfloat16)
    v_cpu = torch.randn_like(k_cpu)
    q, k, v = [x.to(attention_device).detach().requires_grad_() for x in (q_cpu, k_cpu, v_cpu)]
    mask = ICLAttentionBackend.build_cross_mask(
        torch.tensor([0, 0, 1, -1], device=attention_device),
        torch.tensor([0, 0, 0, 1, 1, 1], device=attention_device),
        attention_device, compile_mask=False,
    )
    actual = ICLAttentionBackend.apply(q, k, v, cross_attention=True, block_mask=mask)
    q_ref, k_ref, v_ref = [x.float().detach().requires_grad_() for x in (q_cpu, k_cpu, v_cpu)]
    allowed = torch.tensor([
        [True, True, True, False, False, False],
        [True, True, True, False, False, False],
        [False, False, False, True, True, True],
        [False, False, False, False, False, False],
    ])
    scores = (q_ref.transpose(1, 2) @ k_ref.transpose(1, 2).transpose(-1, -2)) / 32**0.5
    # Avoid undefined softmax on the all-masked padding row.
    scores = scores.masked_fill(~allowed, float('-inf'))
    scores = torch.where(allowed.any(-1, keepdim=True), scores, 0.0)
    probabilities = scores.softmax(-1).masked_fill(~allowed, 0.0)
    expected = (probabilities @ v_ref.transpose(1, 2)).transpose(1, 2)
    torch.testing.assert_close(actual.float().cpu(), expected, atol=0.02, rtol=0.02)
    assert torch.count_nonzero(actual[:, -1]).item() == 0
    actual.float().square().sum().backward()
    expected.square().sum().backward()
    for tensor, reference in zip((q, k, v), (q_ref, k_ref, v_ref)):
        assert torch.isfinite(tensor.grad).all()
        torch.testing.assert_close(tensor.grad.float().cpu(), reference.grad, atol=0.05, rtol=0.05)
