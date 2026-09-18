import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from wan_va.dataset.sample_costs import training_sample_costs
from wan_va.dataset.sequence_crop import SequenceCapacity, crop_training_batch


PROFILE = Path(__file__).resolve().parents[1] / 'wan_va/configs/sequence_capacity_8npu.json'


@pytest.fixture
def capacity():
    return SequenceCapacity.load(PROFILE)


def batch(frames=177, height=14, width=54, actions=8, human=(46, 20, 28)):
    ids = torch.arange(frames).reshape(1, 1, frames, 1, 1)
    return dict(latents=ids.expand(1, 1, frames, height, width).clone(),
                actions=(ids + 1000).expand(1, 2, frames, actions, 1).clone(),
                actions_mask=(ids % 2 == 0).expand(1, 2, frames, actions, 1).clone(),
                icl_latents=torch.zeros(1, 1, *human),
                text_emb=torch.randn(1, 3, 4), icl_text_emb=torch.randn(1, 5, 4))


def test_random_windows_align_modalities_and_keep_full_conditions(capacity):
    original = batch()
    generator = torch.Generator().manual_seed(17)
    starts = set()
    for _ in range(20):
        result = crop_training_batch(original, capacity=capacity, generator=generator)
        window = result['_window_crop']
        starts.add(window['start'])
        assert window['retained_frames'] == 92
        assert window['end'] <= 177
        ids = result['latents'][0, 0, :, 0, 0]
        torch.testing.assert_close(ids, torch.arange(window['start'], window['end']))
        assert torch.equal(result['actions'][0, 0, :, 0, 0], ids + 1000)
        assert torch.equal(result['actions_mask'][0, 0, :, 0, 0], ids % 2 == 0)
        for key in ('icl_latents', 'text_emb', 'icl_text_emb'):
            assert result[key] is original[key]
    assert len(starts) > 1
    assert original['latents'].shape[2] == 177


def test_short_and_long_single_camera_samples_remain_whole(capacity):
    for sample in [batch(frames=52, human=(22, 20, 28)),
                   batch(frames=175, width=18, actions=12)]:
        assert crop_training_batch(sample, capacity=capacity) is sample


def test_spatial_action_and_full_human_cost_control_window(capacity):
    assert capacity.batch_frame_limit(batch()) == 92
    assert capacity.batch_frame_limit(batch(height=12, width=66, human=(46, 18, 32))) == 88
    assert capacity.batch_frame_limit(batch(human=(16, 20, 28))) == 95
    assert capacity.batch_frame_limit(batch(actions=24)) < 92
    result = crop_training_batch(batch(), 64, capacity=capacity)
    assert result['latents'].shape[2] == 64


def test_seed_reproducibility_and_last_window_is_reachable(capacity, monkeypatch):
    a = crop_training_batch(batch(), capacity=capacity,
                            generator=torch.Generator().manual_seed(33))
    b = crop_training_batch(batch(), capacity=capacity,
                            generator=torch.Generator().manual_seed(33))
    assert a['_window_crop'] == b['_window_crop']
    monkeypatch.setattr(torch, 'randint', lambda high, *a, **kw: torch.tensor([high - 1]))
    result = crop_training_batch(batch(), capacity=capacity)
    assert result['_window_crop']['end'] == 177


def test_uncovered_human_condition_and_invalid_shapes_fail_before_transfer(capacity):
    with pytest.raises(ValueError, match='Full human condition'):
        crop_training_batch(batch(human=(100, 20, 28)), capacity=capacity)
    with pytest.raises(ValueError, match='align with patch_size'):
        crop_training_batch(batch(height=13), capacity=capacity)
    sample = batch()
    sample['actions_mask'] = sample['actions_mask'][:, :, :-1]
    with pytest.raises(ValueError, match='align'):
        crop_training_batch(sample, capacity=capacity)


@pytest.mark.parametrize('field,value', [('world_size', 7), ('num_mcp_modules', 3),
    ('gradient_accumulation_steps', 2), ('fsdp_granularity', 'block'),
    ('param_dtype', torch.float32)])
def test_profile_rejects_uncalibrated_training_configs(capacity, field, value):
    config = SimpleNamespace(**capacity.training, patch_size=capacity.patch_size)
    capacity.validate_training(config)
    setattr(config, field, value)
    with pytest.raises(ValueError, match=field):
        capacity.validate_training(config)


def test_profile_rejects_changed_model_and_invalid_margin(capacity, tmp_path):
    capacity.validate_model(capacity.model)
    with pytest.raises(ValueError, match='num_layers'):
        capacity.validate_model(dict(capacity.model, num_layers=40))
    payload = json.loads(PROFILE.read_text())
    payload['robot_token_margin'] = 1
    path = tmp_path / 'invalid.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match='robot_token_margin'):
        SequenceCapacity.load(path)


def test_bucket_cost_matches_cropped_batch_with_actual_action_stride(capacity, tmp_path):
    robot, human = tmp_path / 'robot.pth', tmp_path / 'human.pth'
    torch.save(dict(latent_num_frames=150, latent_height=14, latent_width=18,
                    frame_ids=[0, 2], latent=torch.ones(1)), robot)
    torch.save(dict(latent_num_frames=46, latent_height=18, latent_width=32,
                    latent=torch.ones(1)), human)
    task = SimpleNamespace(new_metas=[{}], used_video_keys=['left', 'right'], root=tmp_path,
        _latent_file=lambda meta, camera: robot, _lookup_icl_sample=lambda meta: {},
        _human_latent_candidate=lambda sample: human)
    sample = batch(frames=150, width=36, actions=8, human=(46, 18, 32))
    cropped = crop_training_batch(sample, capacity=capacity)
    f = cropped['latents'].shape[2]
    assert f == 135
    expected = 2 * f * (126 + 8) + 6624
    assert training_sample_costs(SimpleNamespace(_datasets=[task]), capacity=capacity) == [expected]


def test_training_applies_window_before_device_transfer(capacity):
    from wan_va.train import Trainer
    trainer = Trainer.__new__(Trainer)
    trainer.config = SimpleNamespace(max_train_frames=None, rank=0)
    trainer.capacity_profile = capacity
    original = batch()

    class ReachedTransfer(Exception):
        pass

    def transfer(cropped):
        assert cropped['latents'].shape[2] == 92
        assert cropped['actions'].shape[2] == 92
        assert cropped['icl_latents'] is original['icl_latents']
        raise ReachedTransfer

    trainer.convert_input_format = transfer
    with pytest.raises(ReachedTransfer):
        trainer._train_step(original, 0)
