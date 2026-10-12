#!/usr/bin/env python3
"""Fail fast before torchrun if any expected ALOHA latent is absent or invalid."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def check(root, deep=False, load_samples=False):
    root = root.resolve()
    if deep or load_samples:
        import torch
        torch.set_num_threads(4)
    from wan_va.dataset.aloha_integrity import checked_cache, file_identity
    provenance = json.loads((root / 'encoding_contract.json').read_text())
    for source in provenance['sources'] + provenance['prepared']:
        if file_identity(source['path']) != source:
            raise ValueError(f'Source changed after preparation: {source["path"]}')
    for component in provenance['model'].values():
        for source in component:
            if file_identity(source['path']) != source:
                raise ValueError(f'Encoder checkpoint changed: {source["path"]}')
    jobs = json.loads((root / 'encoding_jobs.json').read_text())
    missing = [j['output'] for j in jobs if not Path(j['output']).is_file()]
    if missing:
        raise RuntimeError(f'{len(missing)}/{len(jobs)} latents missing; run script/prepare_aloha.py encode. First: {missing[0]}')
    for job in jobs:
        if not checked_cache(job, provenance['model']):
            raise ValueError(f'Uncommitted/incomplete latent: {job["output"]}; rerun encode')
    train = [json.loads(x) for x in (root / 'train/breakfast/meta/episodes.jsonl').read_text().splitlines()]
    val = [json.loads(x) for x in (root / 'eval/breakfast/meta/episodes.jsonl').read_text().splitlines()]
    assert not {e['episode_index'] for e in train} & {e['episode_index'] for e in val}
    stats = json.loads((root / 'train/breakfast/meta/action_stats.json').read_text())
    assert set(stats['train_episode_ids']) == {e['episode_index'] for e in train}
    assert all(e['icl']['human_video_path'].startswith('run_train/') for e in train)
    assert all(e['icl']['human_video_path'].startswith('run_eval/') for e in val)
    if deep:
        import torch
        for job in jobs:
            item = torch.load(job['output'], map_location='cpu', weights_only=False)
            f, h, w = [item[k] for k in ['latent_num_frames', 'latent_height', 'latent_width']]
            assert item['latent'].shape == (f*h*w, 48), job['output']
            assert torch.isfinite(item['latent']).all(), job['output']
            assert item['text_emb'].shape == (512, 4096), job['output']
            assert torch.isfinite(item['text_emb']).all(), job['output']
            assert f == (len(item['frame_ids'])-1)//4+1
            if not job['human']: assert item['frame_ids'] == job['frame_ids']
    if load_samples:
        from copy import deepcopy
        import torch
        from wan_va.configs.va_aloha_train_cfg import va_aloha_train_cfg
        from wan_va.dataset.icl_lerobot_latent_dataset import MultiICLLeRobotLatentDataset
        for split in ('train', 'eval'):
            cfg = deepcopy(va_aloha_train_cfg)
            cfg.dataset_path = str(root / split)
            cfg.icl_manifest_path = str(root / 'icl_manifest.json')
            cfg.human_latent_path = str(root / 'human_latents')
            cfg.init_worker = 1
            cfg.rebuild_dataset_index_cache = True
            cfg.cfg_prob = 0
            dataset = MultiICLLeRobotLatentDataset(cfg)
            expected = sum(len(e['action_config']) for e in (train if split == 'train' else val))
            assert len(dataset) == expected, (split, len(dataset), expected)
            for index in range(len(dataset)):
                sample = dataset[index]
                assert sample['latents'].shape[0] == 48
                assert sample['actions'].shape[0] == 30
                assert sample['actions'].shape[2:] == (16, 1)
                assert sample['actions'].shape[1] == sample['latents'].shape[1]
                for key in ('latents', 'actions', 'text_emb', 'icl_latents', 'icl_text_emb'):
                    assert torch.isfinite(sample[key]).all(), (split,index,key)
            print(f'{split}: all {len(dataset)} actual training-loader samples passed')
    print(f'ALOHA ready: {len(train)} train / {len(val)} held-out episodes; {len(jobs)} latents verified' + (' (deep)' if deep else ' (existence)'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3'))
    parser.add_argument('--deep', action='store_true')
    parser.add_argument('--load-samples', action='store_true')
    args = parser.parse_args()
    check(args.root, args.deep, args.load_samples)
