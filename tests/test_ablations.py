from types import SimpleNamespace

import torch
from easydict import EasyDict

from wan_va.ablations import apply_ablation_flags
from wan_va.dataset.icl_lerobot_latent_dataset import ICLLeRobotLatentDataset
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset


def test_ablation_flags_override_dropout_and_mixture():
    config = EasyDict(enable_text=True, enable_human_video=True, enable_mcp=True,
                      droptext_target=0.4, length_bucket_steps=8)
    args = SimpleNamespace(disable_text=True, disable_mcp=True,
                           robotwin_only=True, datasets='oxe:1,robotwin:1')
    apply_ablation_flags(config, args)
    assert config.enable_human_video
    assert not config.enable_text
    assert not config.enable_mcp
    assert config.cfg_prob == 1.0
    assert args.datasets == 'robotwin:1.0'
    assert config.length_bucket_steps == 0


def test_disabled_text_removes_both_cached_instructions(monkeypatch):
    dataset = ICLLeRobotLatentDataset.__new__(ICLLeRobotLatentDataset)
    dataset.new_metas = [{}]
    dataset.config = SimpleNamespace(enable_text=False)
    dataset.empty_emb = torch.full((2, 4), -3.0)
    monkeypatch.setattr(dataset, '_lookup_icl_sample', lambda meta: {})
    monkeypatch.setattr(dataset, '_human_latent_file', lambda sample: 'unused')
    human = torch.randn(48, 1, 2, 2)
    monkeypatch.setattr(dataset, '_load_human_latent',
                        lambda path: (human, torch.ones(2, 4)))
    monkeypatch.setattr(LatentLeRobotDataset, '__getitem__',
                        lambda self, idx: {'text_emb': torch.ones(2, 4) * 7})
    item = dataset[0]
    torch.testing.assert_close(item['text_emb'], dataset.empty_emb)
    torch.testing.assert_close(item['icl_text_emb'], dataset.empty_emb)
    assert item['icl_latents'] is human
