#!/usr/bin/env python3
"""Independent q01/q99, action-slot, window alignment and encoded-frame audit."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pyarrow.parquet as pq
from wan_va.dataset.piperpika import CHANNELS, model_quantiles
from wan_va.dataset.lerobot_action import LeRobotActionProcessor


def audit(root):
    episodes = {}
    arrays = {}
    groups = {}
    for split in ('train', 'eval'):
        task = root/split/'piperpika'
        episodes[split] = [json.loads(l) for l in (task/'meta/episodes.jsonl').read_text().splitlines()]
        groups[split] = {e['alignment_group'] for e in episodes[split]}
        arrays[split] = {e['episode_index']: np.asarray(pq.read_table(
            task/'data'/f'chunk-{e["episode_index"]//1000:03d}'/f'episode_{e["episode_index"]:06d}.parquet',
            columns=['action'])['action'].to_pylist(), dtype=np.float32) for e in episodes[split]}
    assert not groups['train'] & groups['eval']
    assert not arrays['train'].keys() & arrays['eval'].keys()
    path = root/'train/piperpika/meta/action_stats.json'
    stats = json.loads(path.read_text())
    assert json.loads((root/'eval/piperpika/meta/action_stats.json').read_text()) == stats
    assert set(stats['train_episode_ids']) == set(arrays['train'])
    training = np.concatenate(list(arrays['train'].values()))
    max_stat_error = 0.
    for name, section in [('arm.position', slice(14,26)), ('effector.position', slice(26,28))]:
        values = training[:,section]
        for key, expected in [('q01',np.quantile(values,.01,axis=0)), ('q99',np.quantile(values,.99,axis=0)),
                              ('mean',values.mean(0)), ('std',values.std(0))]:
            actual = np.asarray(stats['norm_stats']['action.'+name][key])
            np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)
            max_stat_error = max(max_stat_error,float(np.abs(actual-expected).max()))
    quantiles = model_quantiles(path)
    lo, hi = [np.asarray(quantiles[k])[CHANNELS] for k in ('q01','q99')]
    assert np.all(hi > lo), 'Constant/reversed quantile range in an active action channel'
    report = {'max_stat_error': max_stat_error, 'active_q99_minus_q01': (hi-lo).tolist(),
              'splits': {}, 'existing_latents_checked': 0}
    robot_jobs = {j['output']:j for j in json.loads((root/'encoding_jobs.json').read_text()) if not j['human']}
    for split in ('train','eval'):
        task = root/split/'piperpika'; processor = LeRobotActionProcessor(task)
        count = clipped = elements = dropped = 0; frame_counts = []; max_roundtrip = 0.
        for ep in episodes[split]:
            eid = ep['episode_index']; a = arrays[split][eid]; previous_end = 0
            for window in ep['action_config']:
                start,end = window['start_frame'],window['end_frame']
                assert start == previous_end and (end-start-1)%16 == 0
                previous_end = end-1
                ids = list(range(start,end,4)); f = (len(ids)-1)//4+1
                # Mimic the Dataset's sliced parquet range, crucial for nonzero starts.
                x, mask, stride = processor.process({'action':a[start:end]},start,ids)
                expected = (a[start:end-1,14:]-lo)/(hi-lo+1e-6)*2-1
                np.testing.assert_allclose(x[16:,CHANNELS],np.clip(expected,-2,2),atol=1e-6)
                assert x.shape == (f*16,30) and stride==16
                assert not x[:16].any() and mask[:,CHANNELS].all()
                inactive = [i for i in range(30) if i not in CHANNELS]
                assert not x[:,inactive].any() and not mask[:,inactive].any()
                unclipped = np.abs(expected)<=2
                restored = (x[16:,CHANNELS]+1)/2*(hi-lo+1e-6)+lo
                max_roundtrip = max(max_roundtrip,float(np.abs(restored-a[start:end-1,14:])[unclipped].max()))
                clipped += int((~unclipped).sum()); elements += expected.size
                frame_counts.append(f);count += 1
                jobs = [robot_jobs[str(task/'latents'/f'chunk-{eid//1000:03d}'/cam/f'episode_{eid:06d}_{start}_{end}.pth')]
                        for cam in processor.video_keys]
                assert len(jobs)==3 and all(j['frame_ids']==ids for j in jobs)
                for job in jobs:
                    output=Path(job['output'])
                    if output.exists():
                        import torch
                        item=torch.load(output,map_location='cpu',weights_only=False)
                        assert item['frame_ids']==ids and item['latent_num_frames']==f
                        assert item['latent_height']==14 and item['latent_width']==18
                        assert tuple(item['latent'].shape)==(f*14*18,48)
                        assert torch.isfinite(item['latent']).all()
                        report['existing_latents_checked']+=1
            dropped += len(a)-1-previous_end
        report['splits'][split] = {'groups':len(groups[split]), 'episodes':len(episodes[split]),
            'windows':count,'latent_frame_min_max':[min(frame_counts),max(frame_counts)],
            'latent_frame_histogram':dict(sorted(Counter(frame_counts).items())),
            'windows_at_most_4_latent_frames':sum(f<=4 for f in frame_counts),
            'clipped_fraction_excluding_padding':clipped/elements,
            'unclipped_roundtrip_max_error':max_roundtrip,'unused_tail_action_rows':dropped}
    report['padding_note']='First 16 rows are zero padding with active-channel mask true, following upstream convention.'
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(os.environ.get('PIPERPIKA_PREPARED_ROOT','/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2')))
    parser.add_argument('--output',type=Path)
    args=parser.parse_args(); report=audit(args.root)
    text=json.dumps(report,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n')
    print(text)
