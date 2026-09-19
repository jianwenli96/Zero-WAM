from types import SimpleNamespace
import pytest
import torch
from wan_va.dataset.data_efficiency import crop_training_batch


def batch(frames=12):
    time=torch.arange(frames).view(1,1,frames,1,1)
    return dict(latents=time.expand(1,4,frames,2,2), actions=(time+100).expand(1,3,frames,2,1),
                actions_mask=(time%2==0).expand(1,3,frames,2,1),
                icl_latents=torch.randn(1,4,20,2,2),text_emb=torch.randn(1,3,8))


def test_random_windows_stay_aligned_and_keep_full_human():
    original=batch()
    starts=set()
    for seed in range(12):
        result=crop_training_batch(original,4,generator=torch.Generator().manual_seed(seed))
        start=result['_window_crop']['start']; starts.add(start)
        for key in ['latents','actions','actions_mask']:
            torch.testing.assert_close(result[key],original[key][:,:,start:start+4])
            assert result[key].is_contiguous()
        assert result['icl_latents'] is original['icl_latents']
        assert result['text_emb'] is original['text_emb']
    assert len(starts)>1 and original['latents'].shape[2]==12
    assert crop_training_batch(original) is original
    assert crop_training_batch(original,20) is original
    with pytest.raises(ValueError):crop_training_batch(original,0)
    bad=dict(original,actions=original['actions'][:,:,:2])
    with pytest.raises(ValueError,match='align'):crop_training_batch(bad,4)


def test_train_step_crops_before_device_transfer():
    from wan_va.train import Trainer
    trainer=Trainer.__new__(Trainer)
    trainer.config=SimpleNamespace(max_train_frames=4,rank=0)
    class ReachedDeviceTransfer(Exception):pass
    def transfer(data):
        assert data['latents'].shape[2]==data['actions'].shape[2]==4
        assert data['icl_latents'].shape[2]==20
        assert data['latents'].device.type=='cpu'
        raise ReachedDeviceTransfer
    trainer.convert_input_format=transfer
    with pytest.raises(ReachedDeviceTransfer):trainer._train_step(batch(),0)
