import pytest
import torch

from wan_va.train import crop_training_batch


def batch(frames=20):
    values = torch.arange(frames).reshape(1, 1, frames, 1, 1)
    return dict(latents=values.clone(), actions=values.clone() + 100,
                actions_mask=values.clone() + 200,
                icl_latents=torch.randn(1, 48, 25, 2, 2))


def test_crop_preserves_frame_alignment_and_full_prompt(monkeypatch):
    original = batch()
    monkeypatch.setattr(torch, 'randint', lambda *a, **kw: torch.tensor([3]))
    result = crop_training_batch(original, 8)
    torch.testing.assert_close(result['latents'].flatten(), torch.arange(3, 11))
    torch.testing.assert_close(result['actions'], result['latents'] + 100)
    torch.testing.assert_close(result['actions_mask'], result['latents'] + 200)
    assert result['icl_latents'] is original['icl_latents']
    assert original['latents'].shape[2] == 20


def test_disabled_or_short_crop_leaves_batch_intact():
    original = batch(5)
    assert crop_training_batch(original, None) is original
    assert crop_training_batch(original, 16) is original


def test_crop_rejects_bad_limits_and_misaligned_actions():
    with pytest.raises(ValueError, match='positive'):
        crop_training_batch(batch(), 0)
    original = batch()
    original['actions'] = original['actions'][:, :, :-1]
    with pytest.raises(ValueError, match='align'):
        crop_training_batch(original, 8)
