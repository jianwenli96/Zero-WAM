"""Independently replay saved RTC quality actions and compare profiles to GT."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from .rtc_quality import PROFILES, scores


def replay(data, report, arm):
    """Reconstruct absolute plans without using the evaluation PlanLedger."""
    profile=report['profile']
    plan=data['bootstrap'].copy()
    version=0
    rows=report['proposals']
    binds={row['bind_tick']:row for row in rows}
    commits={row['commit_tick']:row for row in rows}
    assert len(binds)==len(commits)==len(rows)
    pending={}
    issued=[]
    for tick in range(len(data['valid'])):
        if tick in commits:
            row=commits[tick]
            reference,bound_version,frozen=pending[row['job']]
            assert version==bound_version
            start=row['observation_tick']//3
            assert start+frozen>=(tick+1)//3+1
            key=f"job{row['job']}_"+('rtc' if arm=='rtc' else 'ordinary_independent')
            candidate=data[key].copy()
            if arm=='rtc':
                np.testing.assert_array_equal(candidate[:frozen],reference[:frozen])
            else:
                candidate[:frozen]=reference[:frozen]
            end=start+128
            if end>len(plan):
                plan=np.concatenate([plan,np.zeros((end-len(plan),14),np.float32)])
            plan[start+frozen:end]=candidate[frozen:]
            version+=1
        if tick in binds:
            row=binds[tick]
            start=row['observation_tick']//3
            frozen=row['frozen'] if arm=='rtc' else profile['ordinary_prefix']
            overlap=min(128,len(plan)-start)
            reference=np.zeros((128,14),np.float32)
            reference[:overlap]=plan[start:start+overlap]
            assert 0<=frozen<=overlap and row['plan_version']==version
            if arm=='rtc':
                np.testing.assert_array_equal(reference,data[f"job{row['job']}_reference"])
                assert overlap==row['overlap']
            pending[row['job']]=(reference,version,frozen)
        left,remainder=divmod(tick,3)
        value=plan[left].copy()
        if remainder:
            w=remainder/3
            value[:12]=value[:12]*(1-w)+plan[left+1,:12]*w
        issued.append(value)
    np.testing.assert_array_equal(np.stack(issued),data[arm+'_issued'])
    np.testing.assert_array_equal(data['rtc_issued'][:profile['first_handoff_tick']],
                                  data['ordinary_issued'][:profile['first_handoff_tick']])
    actual=scores(data[arm+'_issued'][data['valid']],data['gt'][data['valid']],data['span'])
    assert abs(actual['normalized_mae']-report[arm]['normalized_mae'])<1e-12


def pool(items):
    items=[item for item in items if len(item['target'])]
    if not items:
        return None
    span=items[0]['span']
    target=np.concatenate([item['target'] for item in items])
    result={arm:scores(np.concatenate([item[arm] for item in items]),target,span)
            for arm in ('ordinary','rtc')}
    a,b=[result[arm]['normalized_mae'] for arm in ('ordinary','rtc')]
    result.update(delta_percentage_points=100*(b-a),relative_change_percent=100*(b/a-1),
        episodes=len(items),episode_wins_for_rtc=sum(
            scores(item['rtc'],item['target'],span)['normalized_mae']<
            scores(item['ordinary'],item['target'],span)['normalized_mae'] for item in items))
    groups={}
    for item in items:
        sums=groups.setdefault(item['group'],np.zeros(2,np.float64))
        sums[0]+=np.sum((np.abs(item['rtc'].astype(np.float64)-item['target'])-
                        np.abs(item['ordinary'].astype(np.float64)-item['target']))/span)
        sums[1]+=item['target'].size
    values=np.array(list(groups.values()))
    draws=np.random.default_rng(20261020).integers(0,len(values),size=(10000,len(values)))
    combined=values[draws].sum(1)
    result['paired_human_group_bootstrap_delta_pp_95pct']=np.percentile(
        100*combined[:,0]/combined[:,1],[2.5,97.5]).tolist()
    result['bootstrap_human_groups']=len(values)
    return result


def summarize(root, gt_path, metadata_path, profiles):
    gt=np.load(gt_path)
    metadata=json.loads(metadata_path.read_text())
    groups={ep['episode_index']:ep['icl']['human_video_path'] for ep in metadata['episodes']}
    common_tick=max(PROFILES[name].first_editable_tick for name in profiles)
    output={}
    episode_rows=[]
    bootstraps={}
    for name in profiles:
        paths=sorted(root.glob(f'npu*/{name}/episode_*/report.json'))
        if len(paths)!=len(groups):
            raise RuntimeError(f'{name}: expected {len(groups)} episodes, found {len(paths)}')
        collections={key:[] for key in ('whole_episode','after_first_handoff','from_first_rtc_target',
            'common_after_first_editable','paired_mutable_suffix','paired_first16_native')}
        changes={arm:[] for arm in ('ordinary','rtc','gt')}
        qa=[]
        seen=set()
        for path in paths:
            report=json.loads(path.read_text())
            eid=report['episode']
            assert eid not in seen
            seen.add(eid)
            data=np.load(path.parent/'predictions.npz')
            truth,valid,span=data['gt'],data['valid'],data['span']
            np.testing.assert_array_equal(truth,gt[f'action_{eid}'][:len(truth)])
            np.testing.assert_array_equal(valid,gt[f'valid_{eid}'][:len(truth)])
            np.testing.assert_array_equal(span,gt['span'])
            if eid in bootstraps:
                np.testing.assert_array_equal(data['bootstrap'],bootstraps[eid])
            else:
                bootstraps[eid]=data['bootstrap'].copy()
            for arm in ('ordinary','rtc'):
                replay(data,report,arm)
            profile=report['profile']
            tick=np.arange(len(truth))
            masks=dict(whole_episode=valid,
                after_first_handoff=valid & (tick>=profile['first_handoff_tick']),
                from_first_rtc_target=valid & (tick>=profile['first_editable_tick']),
                common_after_first_editable=valid & (tick>=common_tick))
            base=dict(episode=eid,group=groups[eid],span=span)
            for scope,mask in masks.items():
                collections[scope].append(dict(base,target=truth[mask],
                    ordinary=data['ordinary_issued'][mask],rtc=data['rtc_issued'][mask]))
            paired_rtc,paired_plain,paired_gt=[],[],[]
            for row in report['proposals']:
                indices=row['observation_tick']+3*np.arange(128)
                within=np.flatnonzero(indices<len(valid))
                mask=np.zeros(128,bool)
                mask[within]=valid[indices[within]]
                mask[:row['frozen']]=False
                paired_rtc.append(data[f"job{row['job']}_rtc"][mask])
                paired_plain.append(data[f"job{row['job']}_ordinary_same_context"][mask])
                paired_gt.append(truth[indices[mask]])
            for values,key in ((paired_rtc,'guided'),(paired_plain,'ordinary'),(paired_gt,'target')):
                expected=np.concatenate(values) if values else np.empty((0,14))
                np.testing.assert_array_equal(expected,data[key])
            collections['paired_mutable_suffix'].append(dict(base,
                ordinary=data['ordinary'],rtc=data['guided'],target=data['target']))
            collections['paired_first16_native'].append(dict(base,
                ordinary=data['short_ordinary'],rtc=data['short_guided'],target=data['short_target']))
            for arm in changes:
                values=truth if arm=='gt' else data[arm+'_issued']
                change=np.mean(np.abs(np.diff(values.astype(np.float64)[:,:12],axis=0))/span[:12],axis=1)
                selection=masks['after_first_handoff'][1:] & valid[:-1]
                changes[arm].append(change[selection])
            qa.append(dict(episode=eid,exact_issued_command_replays_per_arm=len(truth),
                commits=report['commits'],zero_guidance_max_abs=report['zero_guidance_max_abs'],
                frozen_prefix_exact=True,gt_matches_safe_archive=True))
            episode_rows.append(dict(profile=name,episode=eid,valid_ticks=int(valid.sum()),
                commits=report['commits'],ordinary_percent=100*report['ordinary']['normalized_mae'],
                rtc_percent=100*report['rtc']['normalized_mae'],
                delta_pp=100*(report['rtc']['normalized_mae']-report['ordinary']['normalized_mae'])))
        assert seen==set(groups)
        change_summary={}
        for arm,arrays in changes.items():
            values=np.concatenate(arrays)*100
            change_summary[arm]=dict(samples=len(values),mean_percent_range_per_native_tick=float(values.mean()),
                p99_percent_range_per_native_tick=float(np.percentile(values,99)),
                maximum_percent_range_per_native_tick=float(values.max()))
        output[name]=dict(profile=PROFILES[name].metadata(),summaries={k:pool(v) for k,v in collections.items()},
            joint_step_change_after_first_handoff=change_summary,qa=qa,
            total_rtc_commits=sum(q['commits'] for q in qa))
    source_paths=[Path(__file__),Path(__file__).with_name('rtc_quality.py'),
                  Path(__file__).with_name('rtc_pipeline_probe.py'),gt_path,metadata_path]
    configuration_path=root/'run_configuration.json'
    run_configuration=json.loads(configuration_path.read_text()) if configuration_path.exists() else None
    parity_path=root/'production_parity.json'
    production_parity=json.loads(parity_path.read_text()) if parity_path.exists() else None
    for path in (configuration_path,parity_path):
        if path.exists():
            source_paths.append(path)
    if production_parity is not None:
        assert production_parity['success'] and production_parity['production_vs_quality_action_max_abs']==0
    result=dict(success=True,checkpoint_step=5000,video_steps=3,action_steps=10,frames=4,
        actions_per_latent=32,horizon_targets=128,target_hz=10,control_hz=30,seed=42,
        deterministic_algorithms=True,normalization=metadata['normalization'],
        gt_reader=metadata['pyarrow'],profiles=output,episodes=episode_rows,
        run_configuration=run_configuration,production_path_parity=production_parity,
        common_scoring_start_tick=common_tick,common_scoring_start_seconds=common_tick/30,
        audit=dict(all_episode_bootstraps_equal_across_profiles=True,
            replayed_controller_commands=sum(q['exact_issued_command_replays_per_arm']*2
                for profile in output.values() for q in profile['qa'])),
        limitations=[
            'Offline fixed-clock replay of measured reservation traces, not new all-episode real-time validation.',
            's32 uses incremental 32-target updates with dual-replica scheduling; s64 uses production one-phase 64-target updates.',
            'The 6.4s profile rounds prior measured 4.895/4.995s latency to a common 5.0s handoff; reservations 58/61 are held fixed.',
            'Ordinary sampling uses the same clock as RTC; its faster unguided inference is not exploited.',
            'Recorded RGB never responds to generated actions. GT measures imitation error, not closed-loop success.',
            'Profiles change history grouping, delay, reservation and update period together; cross-profile changes are not a pure ablation of s.',
            'One paired seed and 12 episodes from six human demonstration groups; bootstrap intervals are exploratory.',
            'The virtual clock starts after the common bootstrap; startup model/first prediction latency is excluded.',
            'Per-proposal mutable-suffix scores may count overlapping future targets more than once; issued-command metrics do not.'
        ],sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths})
    (root/'comparison_report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    with (root/'per_episode.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episode_rows[0]))
        writer.writeheader();writer.writerows(episode_rows)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True)
    parser.add_argument('--metadata',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=PROFILES,default=['s32_dual','s64_single'])
    args=parser.parse_args()
    result=summarize(args.root,args.gt,args.metadata,args.profiles)
    print(json.dumps({name:{scope:dict(
        ordinary_percent=100*pair['ordinary']['normalized_mae'],rtc_percent=100*pair['rtc']['normalized_mae'],
        delta_pp=pair['delta_percentage_points'],targets=pair['rtc']['targets'])
        for scope,pair in profile['summaries'].items()} for name,profile in result['profiles'].items()},indent=2))


if __name__=='__main__':
    main()
