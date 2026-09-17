#!/usr/bin/env python3
"""Create a local, writable training view; keep public data read-only."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

HELD_OUT = {'place_object_scale', 'stamp_seal', 'open_microwave',
            'move_stapler_pad', 'place_bread_basket', 'place_empty_cup',
            'stack_blocks_three'}
CAMERAS = ['observation.images.cam_high', 'observation.images.cam_left_wrist',
           'observation.images.cam_right_wrist']


def link(destination, source):
    source = source.resolve(strict=True)
    if os.path.lexists(destination):
        if not destination.is_symlink() or destination.resolve() != source:
            raise FileExistsError(f'Refusing to replace {destination}')
    else:
        destination.symlink_to(source, target_is_directory=source.is_dir())


def prepare(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Output must be separate from the source collection')
    robot = source / 'robotwin_data'
    tasks = sorted(p.parent.parent for p in robot.glob('*/meta/info.json'))
    train = [p for p in tasks if p.name.split('-')[0] not in HELD_OUT]
    held = [p for p in tasks if p.name.split('-')[0] in HELD_OUT]
    if len(train) != 43 or {p.name.split('-')[0] for p in held} != HELD_OUT or len(held) != 7:
        raise ValueError(f'Expected 43/7 task split, found {len(train)}/{len(held)}')
    manifest = json.loads((source / 'icl_configs/ICL_config_robotwin.json').read_text())
    pairs = {}
    for sample in manifest['samples']:
        video = sample.get('status_info', {}).get('video_rel_path') or sample['robot_video_path']
        parts = Path(video).parts
        episode = int(re.fullmatch(r'episode_(\d+)\.mp4', parts[-1]).group(1))
        key = (parts[0], episode)
        if key in pairs:
            raise ValueError(f'Duplicate human pairing: {key}')
        human_parts = Path(sample['human_video_path']).parts
        start = next(i for i, part in enumerate(human_parts) if part.startswith('run_'))
        human = source / 'human_latents/robotwin' / Path(*human_parts[start:]).with_suffix('.pth')
        if not human.is_file():
            raise FileNotFoundError(human)
        pairs[key] = sample
    counts = Counter()
    per_task, active_keys = {}, set()
    for task in tasks:
        info = json.loads((task / 'meta/info.json').read_text())
        selected = 0
        for line in (task / 'meta/episodes.jsonl').read_text().splitlines():
            row = json.loads(line)
            if not row.get('action_config'):
                continue
            episode = row['episode_index']
            key = (task.name, episode)
            if key not in pairs:
                raise ValueError(f'Missing human pair for {key}')
            active_keys.add(key)
            chunk = episode // info['chunks_size']
            parquet = task / info['data_path'].format(episode_chunk=chunk, episode_index=episode)
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
            for segment in row['action_config']:
                start, end = segment['start_frame'], segment['end_frame']
                if not 0 <= start < end <= row['length']:
                    raise ValueError(f'Invalid segment: {key}: {segment}')
                for camera in CAMERAS:
                    latent = task / 'latents' / f'chunk-{chunk:03d}' / camera / f'episode_{episode:06d}_{start}_{end}.pth'
                    if not latent.is_file():
                        raise FileNotFoundError(latent)
                    counts['robot_latent_files'] += 1
                selected += 1
        per_task[task.name] = selected
        counts['held_out_segments' if task in held else 'training_segments'] += selected
    if active_keys != set(pairs):
        raise ValueError(f'Manifest/episode mismatch: {len(set(pairs) - active_keys)} unmatched pairs')
    # Directory roots are local so loader index caches cannot modify public data.
    output.mkdir(parents=True, exist_ok=True)
    local_robot = output / 'robotwin_data'
    local_robot.mkdir(exist_ok=True)
    (local_robot / 'meta').mkdir(exist_ok=True)
    for file in (robot / 'meta').iterdir():
        if file.is_file():
            link(local_robot / 'meta' / file.name, file)
    for task in tasks:
        local = local_robot / task.name
        local.mkdir(exist_ok=True)
        (local / 'meta').mkdir(exist_ok=True)
        for file in (task / 'meta').iterdir():
            if file.is_file():
                link(local / 'meta' / file.name, file)
        for component in ('data', 'latents', 'videos'):
            if (task / component).exists():
                link(local / component, task / component)
    (output / 'human_latents').mkdir(exist_ok=True)
    link(output / 'human_latents/robotwin', source / 'human_latents/robotwin')
    (output / 'icl_configs').mkdir(exist_ok=True)
    train_names = {p.name for p in train}
    training_samples = [sample for (task, _), sample in pairs.items() if task in train_names]
    training_manifest = output / 'icl_configs/ICL_config_robotwin_train.json'
    training_manifest.write_text(json.dumps({'samples': training_samples}, ensure_ascii=False) + '\n')
    report = dict(source=str(source), output=str(output), training_tasks=[p.name for p in train],
                  held_out_tasks=[p.name for p in held], manifest_pairs=len(pairs),
                  training_pairs=len(training_samples), counts=dict(counts), per_task_segments=per_task,
                  checks='All manifest human latents, selected parquet files, and three camera latent paths exist; tensor contents checked separately')
    (output / 'preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/mnt/sfs_turbo/public/datasets/HumanGen')
    parser.add_argument('--output', default='data/HumanGen')
    args = parser.parse_args()
    report = prepare(args.source, args.output)
    print(json.dumps({k: v for k, v in report.items() if k not in ('per_task_segments', 'training_tasks', 'held_out_tasks')}, indent=2))


if __name__ == '__main__':
    main()
