#!/usr/bin/env python3
"""Build training indexes and exercise real paired samples without loading a policy."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from wan_va.configs.va_robotwin_train_cfg import va_robotwin_train_cfg
from wan_va.configs.va_robotwin_cfg import load_robotwin_norm_stat
from wan_va.dataset.icl_lerobot_latent_dataset import MultiICLLeRobotLatentDataset


def check(root, samples_per_task=1):
    root = Path(root).resolve()
    prepared = json.loads((root / 'preparation.json').read_text())
    config = deepcopy(va_robotwin_train_cfg)
    config.dataset_path = str(root / 'robotwin_data')
    config.icl_manifest_path = str(root / 'icl_configs/ICL_config_robotwin_train.json')
    config.human_latent_path = str(root / 'human_latents/robotwin')
    config.norm_stat = load_robotwin_norm_stat(root / 'robotwin_data/meta/action_stats.json')
    config.cfg_prob = 0.0
    config.rank = 0
    config.init_worker = 1
    dataset = MultiICLLeRobotLatentDataset(config, num_init_worker=1)
    if len(dataset) != prepared['counts']['training_segments']:
        raise ValueError(f'Loader silently filtered samples: {len(dataset)} vs preparation report')
    records = []
    for task in dataset._datasets:
        count = len(task) if samples_per_task == 0 else min(samples_per_task, len(task))
        for index in range(count):
            item = task[index]
            for key in ('latents', 'actions', 'text_emb', 'icl_latents', 'icl_text_emb'):
                if not torch.isfinite(item[key]).all():
                    raise ValueError(f'Non-finite {task.root.name}/{index}/{key}')
            video, action, human = item['latents'], item['actions'], item['icl_latents']
            if video.shape[0] != 48 or action.shape[0] != 30 or human.shape[0] != 48:
                raise ValueError(f'Invalid channel counts in {task.root}')
            if video.shape[1] != action.shape[1] or action.shape != item['actions_mask'].shape:
                raise ValueError(f'Video/action alignment mismatch in {task.root}')
            if action.shape[2:] != (16, 1) or not item['actions_mask'].any():
                raise ValueError(f'Invalid action packing/mask in {task.root}')
            if any(d % 2 for d in (*video.shape[-2:], *human.shape[-2:])):
                raise ValueError(f'Latents not divisible by spatial patch size in {task.root}')
            tokens = (2 * video.shape[1] * video.shape[2] * video.shape[3] // 4
                      + 2 * action.shape[1] * action.shape[2]
                      + human.shape[1] * human.shape[2] * human.shape[3] // 4)
            records.append(dict(task=task.root.name, index=index,
                                shapes={k: list(item[k].shape) for k in ('latents', 'actions', 'icl_latents', 'text_emb', 'icl_text_emb')},
                                backbone_tokens_before_padding=tokens))
        print(f'Checked {task.root.name}: {len(task)} indexed, {count} decoded', flush=True)
    result = dict(training_tasks=len(dataset._datasets), training_samples=len(dataset),
                  excluded_tasks=[p.name for p in dataset.excluded_dataset_roots],
                  decoded_samples=len(records), index_cache_hits=dataset.index_cache_hits,
                  token_range=[min(r['backbone_tokens_before_padding'] for r in records),
                               max(r['backbone_tokens_before_padding'] for r in records)], samples=records)
    (root / 'validation.json').write_text(json.dumps(result, indent=2) + '\n')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='data/HumanGen')
    parser.add_argument('--samples-per-task', type=int, default=1, help='0 decodes every training sample')
    args = parser.parse_args()
    if args.samples_per_task < 0:
        parser.error('--samples-per-task must be nonnegative')
    report = check(args.root, args.samples_per_task)
    print(json.dumps({k: v for k, v in report.items() if k != 'samples'}, indent=2))


if __name__ == '__main__':
    main()
