#!/usr/bin/env python3
"""Check real external ICL samples for each robot/action-schema group on CPU."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wan_va.configs import TRAIN_DATASET_CONFIGS
from wan_va.dataset.humangen_paths import configure_humangen_source
from wan_va.dataset.icl_lerobot_latent_dataset import ICLLeRobotLatentDataset
from prepare_humangen_training import SOURCES


def check(root, all_repos=False, samples_per_repo=1):
    root = Path(root).resolve()
    prepared = json.loads((root / 'external-preparation.json').read_text())
    reports = {}
    for name in SOURCES:
        source = prepared['sources'][name]
        config = configure_humangen_source(deepcopy(TRAIN_DATASET_CONFIGS[name]), name, root)
        config.cfg_prob = 0.0
        config.init_worker = 1
        config.rank = 0
        repos = list(source['per_repo_segments']) if all_repos else source['representative_repos']
        samples, indexed = [], 0
        for relative in repos:
            repo = root / f'{name}_data' / relative
            dataset = ICLLeRobotLatentDataset(repo_id=repo, latent_root=repo / 'latents', config=config)
            expected = source['per_repo_segments'][relative]
            if len(dataset) != expected:
                raise ValueError(f'Loader filtered samples in {repo}: {len(dataset)} != {expected}')
            indexed += len(dataset)
            for i in range(min(samples_per_repo, len(dataset))):
                item = dataset[i]
                for key in ('latents', 'actions', 'text_emb', 'icl_latents', 'icl_text_emb'):
                    if not torch.isfinite(item[key]).all():
                        raise ValueError(f'Non-finite {repo}/{i}/{key}')
                v, a, h = item['latents'], item['actions'], item['icl_latents']
                if v.shape[0] != 48 or h.shape[0] != 48 or a.shape[0] != 30:
                    raise ValueError(f'Channel mismatch: {repo}')
                if v.shape[1] != a.shape[1] or a.shape != item['actions_mask'].shape or not item['actions_mask'].any():
                    raise ValueError(f'Action/latent alignment mismatch: {repo}')
                if any(d % 2 for d in (*v.shape[-2:], *h.shape[-2:])):
                    raise ValueError(f'Invalid patch dimensions: {repo}')
                tokens = 2 * v.shape[1] * v.shape[2] * v.shape[3] // 4 + 2 * a.shape[1] * a.shape[2] + h.shape[1] * h.shape[2] * h.shape[3] // 4
                samples.append(dict(repo=relative, index=i, video_shape=list(v.shape),
                                    action_shape=list(a.shape), human_shape=list(h.shape),
                                    valid_action_channels=item['actions_mask'].flatten(1).any(dim=1).nonzero().flatten().tolist(),
                                    backbone_tokens_before_padding=tokens))
            print(f'{name}/{relative}: indexed {len(dataset)}, decoded {min(samples_per_repo, len(dataset))}', flush=True)
        reports[name] = dict(total_repos=source['repos'], total_segments=source['segments'],
                             indexed_repos=len(repos), indexed_segments=indexed,
                             decoded_samples=len(samples), samples=samples)
    report = dict(scope='all repos' if all_repos else 'one repo per action-schema group',
                  source=prepared['source'], output=str(root), sources=reports)
    (root / 'external-validation.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({n: {k: v for k, v in r.items() if k != 'samples'} for n, r in reports.items()}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/HumanGen')
    parser.add_argument('--all-repos', action='store_true')
    parser.add_argument('--samples-per-repo', type=int, default=1)
    args = parser.parse_args()
    if args.samples_per_repo < 1:
        parser.error('--samples-per-repo must be positive')
    check(args.root, args.all_repos, args.samples_per_repo)


if __name__ == '__main__':
    main()
