#!/usr/bin/env python3
"""Audit released external HumanGen assets and create local cache-safe views."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import re

import yaml
from prepare_robotwin_training import link

SOURCES = ('agibot', 'robocoin', 'robomind', 'interna1', 'oxe')


def discover_repos(root):
    repos = []
    for parent, dirs, _ in os.walk(root):
        parent = Path(parent)
        if 'meta' in dirs and (parent / 'meta/info.json').is_file():
            repos.append(parent)
            dirs[:] = []
        else:
            dirs[:] = [d for d in dirs if d not in ('data', 'latents', 'videos', '.cache')]
    return sorted(repos)


def video_key(path, strip_interval=False):
    parts = str(path).replace('\\', '/').split('/')
    if 'videos' in parts:
        i = parts.index('videos')
        if len(parts) > i + 3 and parts[i + 1].startswith('chunk-') and parts[i + 3].startswith('episode_'):
            del parts[i + 2]
    if strip_interval:
        parts[-1] = re.sub(r'^(episode_\d{6})_\d+_\d+(\.[^.]+)$', r'\1\2', parts[-1])
    return '/'.join(parts)


def human_path(root, sample):
    parts = Path(sample['human_video_path']).parts
    start = next(i for i, part in enumerate(parts) if part.startswith('run_'))
    return root / Path(*parts[start:]).with_suffix('.pth')


def action_metadata(repo):
    transform = next((p for p in (repo / 'meta/action_transform.yaml', repo.parent / 'meta/action_transform.yaml') if p.is_file()), None)
    if transform is None:
        raise FileNotFoundError(f'No action transform for {repo}')
    payload = yaml.safe_load(transform.read_text())
    ref = payload.get('norm_stats', 'action_stats.json')
    stats = Path(ref) if Path(ref).is_absolute() else transform.parent / ref
    if not stats.is_file():
        raise FileNotFoundError(stats)
    if json.loads(stats.read_text()).get('method') != 'abs':
        raise ValueError(f'Unsupported normalization: {stats}')
    cameras = []
    for item in payload['images']:
        spec = next(iter(item.values()))['origin_keys']
        cameras.extend([spec] if isinstance(spec, str) else [next(iter(v)) for v in spec])
    if not cameras:
        raise ValueError(f'No cameras: {transform}')
    return transform, stats, cameras


def mirror_meta(source, target):
    target.mkdir(parents=True, exist_ok=True)
    for file in source.iterdir():
        if file.is_file():
            link(target / file.name, file)
        elif file.is_dir():
            mirror_meta(file, target / file.name)


def prepare_source(source, output, name):
    collection = source / f'{name}_data'
    repos = discover_repos(collection)
    if not repos:
        raise FileNotFoundError(f'No extracted LeRobot repos: {collection}')
    manifest_file = source / 'icl_configs' / f'ICL_config_{name}.json'
    samples = json.loads(manifest_file.read_text())['samples']
    exact, episodes = {}, {}
    for sample in samples:
        video = sample.get('status_info', {}).get('video_rel_path') or sample.get('robot_video_path', '')
        exact.setdefault(video_key(video), sample)
        episodes.setdefault(video_key(video, True), sample)
        if not human_path(source / 'human_latents' / name, sample).is_file():
            raise FileNotFoundError(human_path(source / 'human_latents' / name, sample))
    manifest_humans = {s['human_video_path'] for s in samples}
    counts, per_repo, groups, used_humans = Counter(), {}, {}, set()
    schemas = {}
    for repo in repos:
        relative = str(repo.relative_to(collection))
        info = json.loads((repo / 'meta/info.json').read_text())
        schema_key = repo if (repo / 'meta/action_transform.yaml').is_file() else repo.parent
        if schema_key not in schemas:
            schemas[schema_key] = action_metadata(repo)
        transform, stats, cameras = schemas[schema_key]
        groups.setdefault(str(transform.relative_to(collection)), relative)
        n = 0
        for line in (repo / 'meta/episodes.jsonl').read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if not row.get('action_config'):
                continue
            episode = int(row['episode_index'])
            chunk = episode // int(info['chunks_size'])
            parquet = repo / info['data_path'].format(episode_index=episode, episode_chunk=chunk)
            if not parquet.is_file():
                raise FileNotFoundError(parquet)
            counts['parquet_files'] += 1
            for segment in row['action_config']:
                sample = row.get('icl')
                source_video = segment.get('source_video_rel_path')
                if not sample or not sample.get('human_video_path'):
                    if not source_video:
                        raise ValueError(f'No ICL correspondence: {repo}/{episode}')
                    sample = exact.get(video_key(source_video)) or episodes.get(video_key(source_video, True))
                if not sample or sample.get('human_video_path') not in manifest_humans:
                    raise ValueError(f'Pair is absent from manifest: {repo}/{episode}')
                used_humans.add(sample['human_video_path'])
                start, end = int(segment['start_frame']), int(segment['end_frame'])
                if not 0 <= start < end <= row['length']:
                    raise ValueError(f'Invalid segment: {repo}/{episode}')
                for camera in cameras:
                    candidates = [repo / 'latents' / f'chunk-{chunk:03d}' / camera / f'episode_{episode:06d}_{start}_{end}.pth']
                    if all(k in segment for k in ('source_episode_index', 'source_frame_start', 'source_frame_end')):
                        se = int(segment['source_episode_index'])
                        candidates.append(repo / 'latents' / f'chunk-{se // 1000:03d}' / camera / f"episode_{se:06d}_{segment['source_frame_start']}_{segment['source_frame_end']}.pth")
                    if not any(p.is_file() for p in candidates):
                        raise FileNotFoundError(candidates[-1])
                    counts['robot_latent_files'] += 1
                n += 1
        if n == 0:
            raise ValueError(f'Empty extracted repo: {repo}')
        per_repo[relative] = n
    if used_humans != manifest_humans:
        raise ValueError(f'{name}: {len(manifest_humans - used_humans)} manifest human videos have no robot segment')
    # Retain original nesting so collection-level action transforms remain correct.
    target_collection = output / f'{name}_data'
    for repo in repos:
        target = target_collection / repo.relative_to(collection)
        target.mkdir(parents=True, exist_ok=True)
        mirror_meta(repo / 'meta', target / 'meta')
        ancestor = repo.parent
        while ancestor == collection or collection in ancestor.parents:
            if (ancestor / 'meta').is_dir():
                mirror_meta(ancestor / 'meta', target_collection / ancestor.relative_to(collection) / 'meta')
            if ancestor == collection:
                break
            ancestor = ancestor.parent
        for component in ('data', 'latents', 'videos'):
            if (repo / component).is_dir():
                link(target / component, repo / component)
    (output / 'human_latents').mkdir(parents=True, exist_ok=True)
    link(output / 'human_latents' / name, source / 'human_latents' / name)
    (output / 'icl_configs').mkdir(exist_ok=True)
    link(output / 'icl_configs' / manifest_file.name, manifest_file)
    return dict(repos=len(repos), manifest_pairs=len(samples), segments=sum(per_repo.values()),
                unique_human_videos=len(used_humans), counts=dict(counts), per_repo_segments=per_repo,
                representative_repos=list(groups.values()), action_schema_groups=len(groups))


def prepare(source, output, names=SOURCES):
    source, output = Path(source).resolve(), Path(output).resolve()
    if output == source or source in output.parents:
        raise ValueError('Keep output separate from the public source')
    report = dict(source=str(source), output=str(output), sources={})
    for name in names:
        if name not in SOURCES:
            raise ValueError(f'Unknown external source: {name}')
        result = prepare_source(source, output, name)
        report['sources'][name] = result
        print(name, {k: v for k, v in result.items() if k not in ('per_repo_segments', 'representative_repos')}, flush=True)
    (output / 'external-preparation.json').write_text(json.dumps(report, indent=2) + '\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='/mnt/sfs_turbo/public/datasets/HumanGen')
    parser.add_argument('--output', default='data/HumanGen')
    args = parser.parse_args()
    prepare(args.source, args.output)


if __name__ == '__main__':
    main()
