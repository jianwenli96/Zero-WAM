import pytest
import torch

import wan_va.modules.icl_model as icl


@pytest.mark.parametrize('kind', ['training', 'streaming', 'cross'])
@pytest.mark.parametrize('window', [-1, 0, 3])
def test_real_mask_predicates_match_when_tiled(monkeypatch, kind, window):
    backend = icl.ICLAttentionBackend
    seq = torch.tensor([0, 0, 0, 0, 1, 1, 1, -1, -1])
    frames = torch.tensor([0, 2, 4, 6, 0, 2, 4, -1, -1])
    types = torch.tensor([0, 1, 0, 1, 0, 1, 0, -1, -1])
    def build():
        if kind == 'training':
            return backend.build_training_self_mask(
                seq, frames, torch.tensor([0, 1, 1, 0, 0, 1, 1, -1, -1]),
                types, torch.tensor([0, 0, 1, 0, 0, 0, 1, -1, -1]),
                window, 'cpu', False)
        if kind == 'cross':
            return backend.build_cross_mask(seq, torch.tensor([0, 0, 1, 1, -1]), 'cpu', False)
        return backend.build_self_mask(types, types, seq, seq, frames, frames,
                                       window, 'cpu', False)
    direct = build()
    monkeypatch.setattr(icl, '_DENSE_MASK_DIRECT_ELEMENTS', 0)
    monkeypatch.setattr(icl, '_DENSE_MASK_TILE_ELEMENTS', 18)
    tiled = build()
    assert tiled.dtype == torch.bool
    assert torch.equal(direct, tiled)


def test_tiling_bounds_temporary_query_key_grids(monkeypatch):
    monkeypatch.setattr(icl, '_DENSE_MASK_DIRECT_ELEMENTS', 0)
    monkeypatch.setattr(icl, '_DENSE_MASK_TILE_ELEMENTS', 34)
    sizes = []
    def predicate(b, h, q, k):
        sizes.append(q.numel() * k.numel())
        return (q >= k) & (q - k < 3)
    mask = icl.ICLAttentionBackend._build_mask(predicate, 13, 17, 'cpu', False)
    assert len(sizes) == 7 and max(sizes) <= 34
    q, k = torch.arange(13)[:, None], torch.arange(17)[None, :]
    assert torch.equal(mask[0, 0], (q >= k) & (q - k < 3))
