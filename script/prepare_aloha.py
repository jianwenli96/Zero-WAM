#!/usr/bin/env python3
"""Prepare immutable-source LeRobot views and resumable Wan/T5 latent jobs."""
import argparse
import json
import os
from pathlib import Path
import random
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from wan_va.dataset.aloha import CAMERAS, ACTION_NAMES, action_metadata, segments, validate_actions, canonical_actions
from wan_va.dataset.aloha_integrity import file_identity, model_identity, job_identity, checked_cache


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(payload, indent=2) + '\n')
    tmp.replace(path)


def jsonl(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def prepare(args):
    source, out = args.source.resolve(), args.output.resolve()
    if source == out or source in out.parents:
        raise ValueError('Output must be separate from the original data')
    robot = source / 'robot_data'
    info = json.loads((robot / 'meta/info.json').read_text())
    episodes = jsonl(robot / 'meta/episodes.jsonl')
    if info['features']['action']['names'] != ACTION_NAMES:
        raise ValueError('Action names/order differ from the supported ALOHA 28D contract')
    if info['total_tasks'] != 1 or len({tuple(ep['tasks']) for ep in episodes}) != 1:
        raise ValueError('Random pairing requires a single task; group multi-task data before preparation')
    if info['fps'] != 30:
        raise ValueError('This preset expects 30 Hz robot recordings')
    humans = sorted((source / 'human_videos/train').glob('*.mp4'))
    eval_humans = sorted((source / 'human_videos/eval').glob('*.mp4'))
    if not humans or not eval_humans:
        raise ValueError('Both train and eval human videos are required')
    # Audit camera headers and endpoints before allocating accelerator memory.
    import cv2
    rejected = []
    for ep in episodes:
        eid = ep['episode_index']
        for camera in CAMERAS:
            video = robot / info['video_path'].format(episode_chunk=eid // 1000, episode_index=eid, video_key=camera)
            cap = cv2.VideoCapture(str(video))
            count, fps = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), cap.get(cv2.CAP_PROP_FPS)
            first, _ = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, ep['length'] - 1)
            last, _ = cap.read()
            cap.release()
            if not first or not last or count < ep['length'] or abs(fps - info['fps']) > .1:
                rejected.append({'episode_index': eid, 'camera': camera, 'video': str(video),
                                 'frames': count, 'expected_frames': ep['length'],
                                 'reason': 'unreadable video, missing endpoint, frame count or fps mismatch'})
    rejected_ids = {e['episode_index'] for e in rejected}
    episodes = [e for e in episodes if e['episode_index'] not in rejected_ids]
    ids = sorted(e['episode_index'] for e in episodes)
    rng = random.Random(args.seed)
    marker = out / 'preparation.json'
    old = json.loads(marker.read_text()) if marker.exists() else None
    heldout = set(rng.sample(ids, args.eval_episodes))
    if old and old['seed'] == args.seed and len(old['heldout']) == args.eval_episodes:
        heldout = set(old['heldout'])
        if not heldout <= set(ids):
            raise ValueError('A held-out episode is now unavailable; use a new output directory to change evaluation split')
    if not 0 < len(heldout) < len(ids):
        raise ValueError('Require nonempty train and held-out robot splits')
    contract = {'source': str(source), 'seed': args.seed, 'heldout': sorted(heldout),
                'sampled_frames': args.sampled_frames, 'stride': 4,
                'human_max_frames': 129, 'excluded_episodes': sorted(rejected_ids), 'version': 3, 'quaternions': 'continuous_unit_xyzw'}
    marker = out / 'preparation.json'
    if marker.exists() and json.loads(marker.read_text()) != contract and not args.rebuild_metadata:
        raise ValueError('Existing output has a different contract; use a new output directory')
    write_json(marker, contract)
    write_json(out / 'rejected_videos.json', rejected)
    jobs, train_actions, records = [], [], {'train': [], 'eval': []}
    episode_stats = {}
    prompt = episodes[0]['tasks'][0]
    for split, paths in [('train', humans), ('eval', eval_humans)]:
        for video in paths:
            jobs.append({'video': str(video), 'output': str(out / 'human_latents' / f'run_{split}' / (video.stem + '.pth')),
                         'human': True, 'prompt': prompt, 'height': 320, 'width': 480})
    pairs = []
    for ep in episodes:
        eid = ep['episode_index']
        split = 'eval' if eid in heldout else 'train'
        task = out / split / 'breakfast'
        (task / 'meta').mkdir(parents=True, exist_ok=True)
        relative = info['data_path'].format(episode_chunk=eid // 1000, episode_index=eid)
        table = pq.read_table(robot / relative)
        actions = canonical_actions(table['action'].to_pylist())
        if len(actions) != ep['length']:
            raise ValueError(f'Episode {eid}: metadata length mismatch')
        episode_stats[eid] = {'episode_index': eid, 'stats': {'action': {
            'min': actions.min(0).tolist(), 'max': actions.max(0).tolist(),
            'mean': actions.mean(0).tolist(), 'std': actions.std(0).tolist(),
            'count': [len(actions)]}}}
        if split == 'train':
            train_actions.append(actions)
        link = task / relative
        link.parent.mkdir(parents=True, exist_ok=True)
        if link.is_symlink():
            raise ValueError(f'Use a new v3 output directory; refusing to overwrite a source symlink: {link}')
        column = pa.array(actions.tolist(), type=table.schema.field('action').type)
        table = table.set_column(table.schema.get_field_index('action'), 'action', column)
        temporary = link.with_suffix('.tmp.parquet')
        pq.write_table(table, temporary)
        temporary.replace(link)
        videos = task / 'videos'
        if not videos.exists():
            videos.symlink_to(robot / 'videos', target_is_directory=True)
        human = rng.choice(humans if split == 'train' else eval_humans)
        pair = {'sample_id': f'aloha_{eid:06d}',
                'robot_video_path': f'breakfast/videos/chunk-{eid // 1000:03d}/episode_{eid:06d}.mp4',
                'human_video_path': f'run_{split}/{human.name}', 'source_human_video_path': str(human)}
        pairs.append(pair)
        ep = dict(ep, icl=pair, action_config=[])
        for start, end, frame_ids in segments(len(actions), args.sampled_frames):
            ep['action_config'].append({'start_frame': start, 'end_frame': end})
            for camera in CAMERAS:
                video = robot / info['video_path'].format(episode_chunk=eid // 1000, episode_index=eid, video_key=camera)
                if not video.is_file():
                    raise FileNotFoundError(video)
                jobs.append({'video': str(video), 'output': str(task / 'latents' / f'chunk-{eid // 1000:03d}' / camera / f'episode_{eid:06d}_{start}_{end}.pth'),
                             'frame_ids': frame_ids, 'human': False, 'prompt': ep['tasks'][0], 'height': 224, 'width': 288})
        records[split].append(ep)
    transform, stats = action_metadata(np.concatenate(train_actions))
    stats['train_episode_ids'] = sorted(set(ids) - heldout)
    for split, eps in records.items():
        meta = out / split / 'breakfast/meta'
        split_info = dict(info, total_episodes=len(eps), total_frames=sum(e['length'] for e in eps),
                          chunks_size=1000, total_chunks=1, total_videos=3 * len(eps),
                          splits={'train': f'0:{len(eps)}'})
        write_json(meta / 'info.json', split_info)
        (meta / 'episodes.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in eps))
        (meta / 'tasks.jsonl').write_text((robot / 'meta/tasks.jsonl').read_text())
        selected = {e['episode_index'] for e in eps}
        (meta / 'episodes_stats.jsonl').write_text(''.join(json.dumps(episode_stats[eid]) + '\n' for eid in sorted(selected)))
        (meta / 'action_transform.yaml').write_text(yaml.safe_dump(transform, sort_keys=False))
        write_json(meta / 'action_stats.json', stats)
    write_json(out / 'icl_manifest.json', {'samples': pairs})
    write_json(out / 'encoding_jobs.json', jobs)
    provenance = {'model': model_identity(args.model),
                  'sources': [file_identity(p) for p in sorted(set(
                      [Path(j['video']) for j in jobs] + list(robot.glob('meta/*.json*'))
                      + list(robot.glob('data/*/*.parquet'))))]}
    provenance['prepared'] = [file_identity(p) for p in sorted(
        list(out.glob('*/breakfast/meta/*')) + list(out.glob('*/breakfast/data/*/*.parquet'))
        + [out/'preparation.json', out/'icl_manifest.json', out/'encoding_jobs.json']) if p.is_file()]
    write_json(out / 'encoding_contract.json', provenance)
    print(json.dumps({'train_episodes': len(records['train']), 'eval_episodes': len(records['eval']), 'encoding_jobs': len(jobs), 'output': str(out)}, indent=2))


def encode(args):
    import torch
    if args.device.startswith('npu'):
        import torch_npu  # noqa: F401
    from wan_va.modules.utils import load_vae, load_text_encoder, load_tokenizer, WanVAEStreamingWrapper
    from diffusers.pipelines.wan.pipeline_wan import prompt_clean
    import cv2
    torch.set_num_threads(4)
    jobs = json.loads((args.output / 'encoding_jobs.json').read_text())
    jobs = jobs[args.shard_index::args.num_shards]
    provenance = json.loads((args.output / 'encoding_contract.json').read_text())
    if model_identity(args.model) != provenance['model']:
        raise ValueError('Encoder checkpoint differs from the preparation contract')
    pending = [j for j in jobs if not checked_cache(j, provenance['model'])]
    if not pending:
        print('All latent jobs complete'); return
    dtype = torch.bfloat16
    tokenizer = load_tokenizer(str(args.model / 'tokenizer'))
    encoder = load_text_encoder(str(args.model / 'text_encoder'), dtype, args.device).eval()
    embeddings = {}
    with torch.inference_mode():
        for prompt in sorted({j['prompt'] for j in pending}):
            inputs = tokenizer([prompt_clean(prompt)], padding='max_length', max_length=512,
                               truncation=True, return_tensors='pt')
            result = encoder(inputs.input_ids.to(args.device), inputs.attention_mask.to(args.device)).last_hidden_state[0]
            result[int(inputs.attention_mask.sum()):] = 0
            embeddings[prompt] = result.cpu()
    del encoder
    if args.device.startswith('npu'): torch.npu.empty_cache()
    elif args.device.startswith('cuda'): torch.cuda.empty_cache()
    vae = load_vae(str(args.model / 'vae'), dtype, args.device).eval()
    wrapper = WanVAEStreamingWrapper(vae)
    mean = torch.tensor(vae.config.latents_mean, device=args.device).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device=args.device).view(1, -1, 1, 1, 1)
    with torch.inference_mode():
        for ji, job in enumerate(pending):
            cap = cv2.VideoCapture(job['video'])
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            ids = job.get('frame_ids')
            if ids is None:
                fps = cap.get(cv2.CAP_PROP_FPS)
                n = min(129, max(5, int(count / fps * 12)))
                n = (n - 1) // 4 * 4 + 1
                if count < n: raise ValueError('Human video too short')
                ids = np.linspace(0, count - 1, n).astype(int).tolist()
            wrapper.clear_cache()
            encoded, batch = [], []
            cursor = 0
            for index in ids:
                while cursor <= index:
                    ok, bgr = cap.read(); cursor += 1
                    if not ok: raise ValueError(f'Cannot decode frame {index}: {job["video"]}')
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                frame = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float()
                frame = torch.nn.functional.interpolate(frame[None], size=(job['height'], job['width']), mode='bilinear', align_corners=False)[0]
                batch.append(frame)
                if len(batch) == (1 if not encoded else 4):
                    video = (torch.stack(batch, dim=1)[None] / 255 * 2 - 1).to(device=args.device, dtype=dtype)
                    mu = wrapper.encode_chunk(video).chunk(2, dim=1)[0]
                    encoded.append(((mu - mean) / std).to(dtype).cpu())
                    batch = []
            cap.release()
            latent = torch.cat(encoded, dim=2)[0]
            c, f, h, w = latent.shape
            if f != (len(ids)-1)//4+1 or c != 48 or not torch.isfinite(latent).all():
                raise ValueError(f'Invalid latent shape/value: {latent.shape}')
            payload = {'latent': latent.permute(1,2,3,0).reshape(-1,c).contiguous(),
                       'latent_num_frames': f, 'latent_height': h, 'latent_width': w,
                       'frame_ids': ids, 'text_emb': embeddings[job['prompt']],
                       'source_video': job['video'], 'prompt': job['prompt']}
            dest = Path(job['output']); dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(f'.tmp-{os.getpid()}'); torch.save(payload, tmp); tmp.replace(dest)
            write_json(dest.with_suffix('.provenance.json'), {'identity': job_identity(job, provenance['model']), 'size': dest.stat().st_size})
            print(f'[{ji+1}/{len(pending)}] {dest}', flush=True)
    missing = [j['output'] for j in jobs if not Path(j['output']).is_file()]
    if missing: raise RuntimeError(f'{len(missing)} missing latents')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage', choices=['prepare', 'encode'])
    p.add_argument('--source', type=Path, default=Path('/mnt/sfs_turbo/lijianwen/Datasets/aloha_icl_data'))
    p.add_argument('--output', type=Path, default=Path('/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3'))
    p.add_argument('--model', type=Path, default=Path(os.environ.get('MODEL_PATH', '/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain')))
    p.add_argument('--rebuild-metadata', action='store_true', help='Rebuild split/stats/job metadata after source repair; existing latent files are reused')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--eval-episodes', type=int, default=5)
    p.add_argument('--sampled-frames', type=int, default=129)
    p.add_argument('--device', default='npu:0')
    p.add_argument('--num-shards', type=int, default=1)
    p.add_argument('--shard-index', type=int, default=0)
    args = p.parse_args()
    if not 0 <= args.shard_index < args.num_shards: p.error('Invalid shard')
    (prepare if args.stage == 'prepare' else encode)(args)


if __name__ == '__main__': main()
