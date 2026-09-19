from collections import Counter
from types import SimpleNamespace

import pytest
import torch
from wan_va.dataset.dataset_mixture import DistributedDatasetMixtureSampler
from wan_va.dataset.data_efficiency import training_sample_costs


def test_bucketing_preserves_draws_and_balances_rank_costs():
    costs = list(range(1, 65))
    def draw(bucket, epoch=0):
        samplers = [DistributedDatasetMixtureSampler([32, 32], [4, 1], 4, rank,
                    epoch_size=129, sample_costs=costs, bucket_steps=bucket) for rank in range(4)]
        for sampler in samplers: sampler.set_epoch(epoch)
        ranks = [list(s) for s in samplers]
        assert all(len(row) == 33 for row in ranks)
        return list(zip(*ranks))
    plain, grouped = draw(0), draw(8)
    assert Counter(i for row in plain for i in row) == Counter(i for row in grouped for i in row)
    spread = lambda rows: sum(max(costs[i] for i in row)-min(costs[i] for i in row) for row in rows)
    assert spread(grouped) < spread(plain)
    assert grouped == draw(8) and grouped != draw(8, 1)


@pytest.mark.parametrize('costs', [[1], [1, float('nan')], [1, -2]])
def test_invalid_costs_rejected(costs):
    with pytest.raises(ValueError):
        DistributedDatasetMixtureSampler([2], [1], sample_costs=costs, bucket_steps=2)


def test_costs_read_metadata_without_tensor_loading(tmp_path, monkeypatch):
    robot, human = tmp_path/'robot.pth', tmp_path/'human.pth'
    torch.save(dict(latent=torch.zeros(48,8,2,2), latent_num_frames=8,
                    latent_height=2, latent_width=2, frame_ids=list(range(0,32,4))), robot)
    torch.save(dict(latent=torch.zeros(48,5,2,2), latent_num_frames=5,
                    latent_height=2, latent_width=2), human)
    task=SimpleNamespace(root=tmp_path, used_video_keys=['camera'], new_metas=[{}],
                         _latent_file=lambda *a: robot, _lookup_icl_sample=lambda *a: {},
                         _human_latent_candidate=lambda *a: human)
    monkeypatch.setattr(torch, 'load', lambda *a, **k: pytest.fail('must not load tensor storage'))
    assert training_sample_costs(SimpleNamespace(_datasets=[task]),max_frames=3,workers=1)==[107]
