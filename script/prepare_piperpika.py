#!/usr/bin/env python3
"""Prepare manifest-aligned joint-only data; torchrun mode prepares then encodes all shards."""
import argparse
from datetime import timedelta
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
import yaml
from prepare_aloha import encode, write_json, jsonl
from wan_va.dataset.aloha import ACTION_NAMES, CAMERAS, segments
from wan_va.dataset.piperpika import joint_metadata, CHANNELS
from wan_va.dataset.aloha_integrity import file_identity, model_identity, checked_cache

ROBOT_VERSION = 'lerobot28_v21_h264_crf18_20261017_160250'


def split_groups(groups, seed=42, eval_groups=6):
    ids = [g['alignment_task_number'] for g in groups]
    episodes = [e['episode_index'] for g in groups for e in g['robot_episodes']]
    humans = [g['human_video_path'] for g in groups]
    if len(set(ids)) != len(ids) or len(set(episodes)) != len(episodes) or len(set(humans)) != len(humans):
        raise ValueError('Duplicate group, episode or human video; resolve grouping before splitting')
    if not 0 < eval_groups < len(ids):
        raise ValueError('Require nonempty train and validation groups')
    return sorted(random.Random(seed).sample(sorted(ids), eval_groups))


def audit_video(path, length=None, expected_fps=None):
    import cv2
    cap = cv2.VideoCapture(str(path))
    frames, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
    ok, _ = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
    end_ok, _ = cap.read()
    shape = [int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))]
    cap.release()
    if not ok or not end_ok or frames < 5 or fps <= 0 or (length is not None and frames != length) or (expected_fps is not None and abs(fps - expected_fps) > .1):
        raise ValueError(f'Invalid video: {path}, frames={frames}, fps={fps}, expected={length}')
    return {'frames': frames, 'fps': fps, 'shape': shape, 'seconds': frames/fps}


def prepare(args):
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or source in out.parents or out in source.parents:
        raise ValueError('Prepared output must be separate from source')
    alignment_path = source/'human_robot_alignment.json'
    alignment = json.loads(alignment_path.read_text())
    groups = alignment['groups']
    heldout = split_groups(groups, args.seed, args.eval_groups)
    robot = source / ROBOT_VERSION
    info = json.loads((robot/'meta/info.json').read_text())
    if info['fps'] != 30 or info['features']['action']['names'] != ACTION_NAMES:
        raise ValueError('Unsupported source action ordering or frame rate')
    episodes = {e['episode_index']: e for e in jsonl(robot/'meta/episodes.jsonl')}
    if set(episodes) != {e['episode_index'] for g in groups for e in g['robot_episodes']}:
        raise ValueError('Alignment must cover every source episode exactly once')
    contract = {'source': str(source), 'robot_version': ROBOT_VERSION, 'seed': args.seed,
                'heldout_groups': heldout, 'sampled_frames': 129, 'stride': 4,
                'human_max_frames': 129, 'version': 1, 'action_channels': CHANNELS,
                'alignment_identity': file_identity(alignment_path)}
    marker = out/'preparation.json'
    if marker.exists() and json.loads(marker.read_text()) != contract:
        raise ValueError('Preparation contract changed: use a new output directory')
    if marker.exists() and (out/'encoding_contract.json').exists():
        # Do not rewrite metadata on resume: provenance intentionally includes mtimes.
        provenance = json.loads((out/'encoding_contract.json').read_text())
        validate_provenance(provenance)
        if model_identity(args.model) != provenance['model']:
            raise ValueError('Encoding model changed')
        print('Prepared metadata verified; resuming latent jobs', flush=True)
        return
    jobs, pairs, all_train, audit = [], [], [], []
    records = {'train': [], 'eval': []}
    stats_records = {'train': [], 'eval': []}
    sources = {alignment_path, *robot.glob('meta/*.json*')}
    for group in groups:
        gid = group['alignment_task_number']
        split = 'eval' if gid in heldout else 'train'
        human = source/group['human_video_path']
        human_stats = audit_video(human)
        sources.add(human)
        # Unique group names avoid the repeated basename video.mp4.
        human_key = f'run_{split}/group_{gid:03d}.mp4'
        prompts = {episodes[e['episode_index']]['tasks'][0] for e in group['robot_episodes']}
        if len(prompts) != 1:
            raise ValueError(f'Group {gid} has incompatible task descriptions')
        prompt = next(iter(prompts))
        jobs.append({'video': str(human), 'output': str(out/'human_latents'/f'run_{split}'/f'group_{gid:03d}.pth'),
                     'human': True, 'prompt': prompt, 'height': 320, 'width': 480})
        for entry in group['robot_episodes']:
            eid = entry['episode_index']; ep = episodes[eid]
            relative = info['data_path'].format(episode_chunk=eid//1000, episode_index=eid)
            parquet = robot/relative
            if (source/entry['data_path']).resolve() != parquet.resolve():
                raise ValueError(f'Episode {eid} does not point to the required NEW robot dataset')
            sources.add(parquet)
            table = pq.read_table(parquet)
            action = np.asarray(table['action'].to_pylist(), dtype=np.float32)
            if len(action) != ep['length'] or not np.isfinite(action).all():
                raise ValueError(f'Invalid action data at episode {eid}')
            if not np.all(np.asarray(table['episode_index']) == eid):
                raise ValueError(f'Parquet episode ID mismatch: {eid}')
            timestamps = np.asarray(table['timestamp'])
            if len(timestamps)>1 and not np.allclose(np.diff(timestamps), 1/30, atol=1e-3):
                raise ValueError(f'Nonuniform timestamps: {eid}')
            if split == 'train': all_train.append(action)
            task = out/split/'piperpika'
            dest = task/relative; dest.parent.mkdir(parents=True, exist_ok=True)
            if not dest.exists(): dest.symlink_to(parquet)
            pair = {'sample_id': f'piperpika_{eid:06d}', 'robot_video_path': f'piperpika/videos/chunk-{eid//1000:03d}/episode_{eid:06d}.mp4',
                    'human_video_path': human_key, 'source_human_video_path': str(human)}
            pairs.append(pair)
            windows = list(segments(len(action)))
            record = dict(ep, icl=pair, alignment_group=gid,
                          action_config=[{'start_frame': s, 'end_frame': e} for s,e,_ in windows])
            records[split].append(record)
            stats_records[split].append({'episode_index': eid, 'stats': {'action': {
                'min': action.min(0).tolist(), 'max': action.max(0).tolist(), 'mean': action.mean(0).tolist(),
                'std': action.std(0).tolist(), 'count': [len(action)]}}})
            video_stats = {}
            for cam in CAMERAS:
                video = robot/info['video_path'].format(episode_chunk=eid//1000, episode_index=eid, video_key=cam)
                if (source/entry['videos'][cam]).resolve() != video.resolve():
                    raise ValueError('Alignment video points outside the new dataset')
                sources.add(video)
                video_stats[cam] = audit_video(video, len(action), 30)
                for start,end,ids in windows:
                    jobs.append({'video': str(video), 'output': str(task/'latents'/f'chunk-{eid//1000:03d}'/cam/f'episode_{eid:06d}_{start}_{end}.pth'),
                                 'frame_ids': ids, 'human': False, 'prompt': prompt, 'height': 224, 'width': 288})
            audit.append({'episode': eid, 'group': gid, 'split': split, 'frames': len(action),
                          'windows': len(windows), 'human': human_stats, 'robot': video_stats})
    transform, stats = joint_metadata(np.concatenate(all_train))
    stats['train_episode_ids'] = sorted(e['episode_index'] for e in records['train'])
    for split, eps in records.items():
        meta = out/split/'piperpika/meta'; meta.mkdir(parents=True, exist_ok=True)
        videos = meta.parent/'videos'
        if not videos.exists(): videos.symlink_to(robot/'videos', target_is_directory=True)
        write_json(meta/'info.json', dict(info, total_episodes=len(eps), total_frames=sum(e['length'] for e in eps), total_videos=3*len(eps), chunks_size=1000, total_chunks=max(e['episode_index']//1000 for e in eps)+1, splits={'train': f'0:{len(eps)}'}))
        for name, values in [('episodes',eps), ('episodes_stats',stats_records[split])]:
            (meta/f'{name}.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in values))
        (meta/'tasks.jsonl').write_text((robot/'meta/tasks.jsonl').read_text())
        (meta/'action_transform.yaml').write_text(yaml.safe_dump(transform, sort_keys=False))
        write_json(meta/'action_stats.json', stats)
    write_json(out/'icl_manifest.json', {'samples':pairs})
    write_json(out/'encoding_jobs.json', jobs)
    report = {'robot_version': ROBOT_VERSION, 'groups': len(groups), 'heldout_groups': heldout,
              'splits': {s: {'episodes':len(eps), 'windows':sum(len(e['action_config']) for e in eps)} for s,eps in records.items()},
              'encoding_jobs':len(jobs), 'episodes':audit}
    write_json(out/'audit.json', report)
    # Commit only once all metadata has been constructed.
    write_json(marker, contract)
    prepared = [p for p in out.glob('*/piperpika/meta/*') if p.is_file()]
    prepared += [out/f for f in ('preparation.json','icl_manifest.json','encoding_jobs.json','audit.json')]
    write_json(out/'encoding_contract.json', {'model':model_identity(args.model),
        'sources':[file_identity(p) for p in sorted(sources)], 'prepared':[file_identity(p) for p in sorted(prepared)]})
    print(json.dumps({k:v for k,v in report.items() if k != 'episodes'}, indent=2), flush=True)


def validate_provenance(provenance):
    for identity in provenance['sources'] + provenance['prepared'] + [i for v in provenance['model'].values() for i in v]:
        if file_identity(identity['path']) != identity:
            raise ValueError(f'Source/prepared/encoder file changed: {identity["path"]}')


def check(root, load_samples=False):
    provenance = json.loads((root/'encoding_contract.json').read_text())
    validate_provenance(provenance)
    jobs = json.loads((root/'encoding_jobs.json').read_text())
    if not all(checked_cache(j, provenance['model']) for j in jobs):
        raise ValueError('Latents incomplete; run distributed preparation')
    if load_samples:
        from copy import deepcopy
        import torch
        torch.set_num_threads(2)
        from wan_va.configs.va_piperpika_train_cfg import va_piperpika_train_cfg
        from wan_va.dataset.icl_lerobot_latent_dataset import MultiICLLeRobotLatentDataset
        for split in ('train','eval'):
            cfg = deepcopy(va_piperpika_train_cfg)
            cfg.dataset_path = str(root/split); cfg.icl_manifest_path = str(root/'icl_manifest.json')
            cfg.human_latent_path = str(root/'human_latents'); cfg.init_worker = 1; cfg.cfg_prob = 0
            ds = MultiICLLeRobotLatentDataset(cfg)
            expected = json.loads((root/'audit.json').read_text())['splits'][split]['windows']
            if len(ds) != expected: raise ValueError('Dataset silently dropped samples')
            for i in range(len(ds)):
                x = ds[i]
                if x['actions'].shape[0] != 30 or x['actions'].shape[1] != x['latents'].shape[1]:
                    raise ValueError('Action/video alignment mismatch')
                if x['actions'][:14].any() or x['actions_mask'][:14].any():
                    raise ValueError('EEF leaked into joint-only data')
                for key in ('actions','latents','icl_latents','text_emb','icl_text_emb'):
                    if not torch.isfinite(x[key]).all(): raise ValueError(f'Nonfinite {key}')
            print(f'{split}: verified {len(ds)} real loader samples', flush=True)
    print(f'PiperPika: {len(jobs)} latent artifacts ready', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare','distributed','check'])
    p.add_argument('--source', type=Path, default=Path(os.environ.get('PIPERPIKA_SOURCE_ROOT','/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16')))
    p.add_argument('--output', type=Path, default=Path(os.environ.get('PIPERPIKA_PREPARED_ROOT','/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2')))
    p.add_argument('--model', type=Path, default=Path(os.environ.get('MODEL_PATH','/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain')))
    p.add_argument('--seed', type=int, default=42); p.add_argument('--eval-groups', type=int, default=6)
    p.add_argument('--load-samples', action='store_true')
    args=p.parse_args()
    if args.stage == 'prepare': prepare(args)
    elif args.stage == 'check': check(args.output,args.load_samples)
    else:
        import torch.distributed as dist
        dist.init_process_group('gloo', timeout=timedelta(hours=4))
        try:
            if dist.get_rank()==0: prepare(args)
            dist.barrier()
            args.num_shards=dist.get_world_size();args.shard_index=dist.get_rank()
            args.device=f"{os.environ.get('DEVICE_TYPE','npu')}:{os.environ['LOCAL_RANK']}"
            encode(args)
            dist.barrier()
            if dist.get_rank()==0: check(args.output, True)
            dist.barrier()
        finally: dist.destroy_process_group()


if __name__=='__main__': main()
