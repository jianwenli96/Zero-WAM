"""Fixed-clock recorded-RGB quality ablation of ordinary sampling and RTC.

This does not measure latency or closed-loop robot success. Both policies use
the same measured deployment clock, model/noise settings and recorded RGB.
GT is passed only to the scorer, never to the virtual controller or sampler.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path

import numpy as np

from .rtc_pipeline_probe import CAMERAS, PlanLedger, Timing, observed_snapshot


@dataclass(frozen=True)
class QualityProfile:
    name: str
    receding_actions: int
    handoff_ticks: int
    history_mode: str
    frozen_targets: tuple[int, ...]
    binding_ticks: tuple[int, ...]
    timing_source: str

    @property
    def ordinary_prefix(self):
        # At an integral target boundary, the right anchor for the preceding
        # native command has already been used. Protect that anchor as well.
        return (self.handoff_ticks+1)//3+1

    @property
    def first_handoff_tick(self):
        return self.receding_actions*3+self.handoff_ticks

    @property
    def first_editable_tick(self):
        return (self.receding_actions+self.frozen_targets[0])*3

    def reservation(self, number):
        return self.frozen_targets[min(number-1,len(self.frozen_targets)-1)]

    def binding_offset(self, number):
        return self.binding_ticks[min(number-1,len(self.binding_ticks)-1)]

    def metadata(self):
        return dict(asdict(self), ordinary_prefix=self.ordinary_prefix,
                    first_handoff_tick=self.first_handoff_tick,
                    first_editable_tick=self.first_editable_tick,
                    receding_seconds=self.receding_actions/10,
                    handoff_seconds=self.handoff_ticks/30)


PROFILES = {
    's28_snapshot': QualityProfile('s28_snapshot',28,162,'snapshot',(56,61),(24,78),
        'zerowam_rtc_min_s_20261020/final: 5.4s snapshot replay'),
    's32_dual': QualityProfile('s32_dual',32,132,'incremental',(46,46,49,48),(41,41,48,45),
        'zerowam_rtc_min_s_20261020/final/trial00_incremental_s32_handoff4.4/report.json; '
        'binding offsets rounded up from action_ready-action_seconds-observation to native ticks'),
    's64_single': QualityProfile('s64_single',64,150,'incremental',(58,61),(0,),
        'zerowam_posttrain_eval_20261019/rtc_step5000/rtc_replay_report.json; '
        'measured 4.895/4.995s rounded to a common fixed 5.0s handoff, measured reservations 58/61'),
}


def incremental_observation(start, issued, images, timing):
    """Capture one request's executed interval, including its endpoint RGB.

    A 64-target production request stays one clean cache update containing
    two video latents and two action blocks, rather than two 32-target updates.
    """
    previous = start-timing.receding_actions
    native, end = previous*3, start*3
    if previous < 0 or start % 32 or len(issued) != end:
        raise ValueError('Incremental history requires complete executed latent intervals')
    chunk = dict(start=previous,
        actions=np.stack([issued[t] for t in range(native,end,3)]),
        images=[images[t] for t in range(native+4,end+1,4)])
    return dict(origin=0,start=start,history=[chunk],
                anchor_ticks=list(range(native,end+1,96)))


def scores(prediction, target, span):
    prediction, target, span = [np.asarray(x, dtype=np.float64) for x in (prediction, target, span)]
    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 14:
        raise ValueError('Need matching [T,14] predictions and GT')
    if not len(prediction):
        return None
    if span.shape != (14,) or np.any(span <= 0) or not all(np.isfinite(x).all() for x in (prediction,target,span)):
        raise ValueError('Need finite actions and positive training spans')
    error = np.abs(prediction-target)
    return dict(normalized_mae=float((error/span).mean()), joint_mae=float(error[:,:12].mean()),
                gripper_mae=float(error[:,12:].mean()), mae_per_channel=error.mean(0).tolist(),
                targets=len(prediction))


def native_sample(actions, tick):
    left, remainder = divmod(tick, 3)
    value = actions[left].copy()
    if remainder:
        weight = remainder/3
        value[:12] = value[:12]*(1-weight)+actions[left+1,:12]*weight
    return value


def rollout(images, ticks, seed, bootstrap, infer, *, rtc, profile=PROFILES['s28_snapshot']):
    """Virtual control clock; infer sees only captured RGB and issued actions.

    The selected profile replays a measured deployment delay/reservation trace.
    Ordinary sampling uses no guidance/reference in the network; its commit
    preserves only already-used targets and the right interpolation anchor.
    """
    timing = Timing(receding_actions=profile.receding_actions).validate()
    ledger = PlanLedger(bootstrap, timing)
    jobs, proposals = {}, []
    launches = {timing.period_ticks*j:j for j in range(1,ticks//timing.period_ticks+1)
                if timing.period_ticks*j+profile.handoff_ticks < ticks}
    bindings = {tick+profile.binding_offset(job):job for tick,job in launches.items()}
    commits = {tick+profile.handoff_ticks:job for tick,job in launches.items()}
    for tick in range(ticks):
        if tick in commits:
            job = jobs[commits[tick]]
            binding = job['binding']
            candidate = job['output']['actions'].copy()
            candidate[:binding['frozen']] = binding['reference'][:binding['frozen']]
            ledger.commit(job['start'], candidate, binding, tick)
            proposals.append(dict(job=commits[tick], observation_tick=job['start']*3,
                bind_tick=job['bind_tick'], commit_tick=tick, start=job['start'],
                binding=binding, **job['output']))
        if tick in launches:
            job = launches[tick]
            capture = (observed_snapshot(tick//3,ledger.issued,images,timing,2)
                       if profile.history_mode == 'snapshot' else
                       incremental_observation(tick//3,ledger.issued,images,timing))
            jobs[job] = dict(start=tick//3,snapshot=capture)
        if tick in bindings:
            number = bindings[tick]
            job = jobs[number]
            frozen = profile.reservation(number) if rtc else profile.ordinary_prefix
            binding = ledger.bind(job['start'], frozen)
            job['binding'], job['bind_tick'] = binding, tick
            job['output'] = infer(job['snapshot'], binding, seed+number*100003, rtc=rtc)
        ledger.issue(tick)
    return dict(issued=np.asarray(ledger.issued), proposals=proposals, commits=ledger.commits)


def load_rgb(root, episode, ticks):
    import cv2
    cv2.setNumThreads(1)
    images = {tick:{} for tick in range(ticks)}
    for camera in CAMERAS:
        path = root/'eval/piperpika/videos'/f'chunk-{episode//1000:03d}'/camera/f'episode_{episode:06d}.mp4'
        capture = cv2.VideoCapture(str(path))
        try:
            for tick in range(ticks):
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f'Cannot decode {episode}/{camera}/{tick}')
                images[tick][camera] = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (288,224))
        finally:
            capture.release()
    return images


def evaluate_episode(runtime, ep, root, gt, valid, span, output, *, check_zero,
                     profile=PROFILES['s28_snapshot'], images=None):
    import torch
    from wan_va.rtc.cache import request_cache

    episode = ep['episode_index']
    ticks = int(np.flatnonzero(valid)[-1])+1
    if images is None:
        images = load_rgb(root, episode, ticks)
    seed = 42+episode*10000
    metadata = dict(prompt=ep['tasks'][0], seed=seed,
        icl_latent_path=str(root/'human_latents'/Path(ep['icl']['human_video_path']).with_suffix('.pth')))
    task_key = (metadata['prompt'], metadata['icl_latent_path'])
    if getattr(runtime, 'quality_task_key', None) != task_key:
        runtime.reset_probe(metadata, images[0])
        runtime.quality_task_key = task_key
    else:
        runtime.reset_robot(images[0])
    torch.manual_seed(seed)
    empty = dict(reference=np.zeros((128,14),np.float32), frozen=0, overlap=0, plan_version=0)
    with request_cache():
        runtime.video_proposal()
        initial_noise = runtime.pending_actions.clone()
        bootstrap = runtime.decode_actions(empty, guidance=False)
        zero_error = None
        if check_zero:
            runtime.pending_actions = initial_noise
            guided_zero = runtime.decode_actions(empty)
            zero_error = float(np.max(np.abs(guided_zero-bootstrap)))
            if zero_error > 1e-6:
                raise RuntimeError(f'Zero-weight RTC differs from ordinary sampler: {zero_error}')

    def infer(snapshot, binding, job_seed, *, rtc):
        torch.manual_seed(job_seed)
        with request_cache():
            if profile.history_mode == 'snapshot':
                runtime.prepare_snapshot(snapshot)
            else:
                runtime.advance_history(snapshot['history'])
                if runtime.history_actions != snapshot['start']:
                    raise RuntimeError('Incremental context differs from its observation timestamp')
            runtime.video_proposal()
            noise = runtime.pending_actions.clone()
            if rtc:
                shadow = runtime.decode_actions(binding, guidance=False)
                runtime.pending_actions = noise
                actions = runtime.decode_actions(binding)
                return dict(actions=actions, ordinary_same_context=shadow)
            return dict(actions=runtime.decode_actions(binding, guidance=False))

    runtime.reset_robot(images[0])
    guided = rollout(images,ticks,seed,bootstrap,infer,rtc=True,profile=profile)
    runtime.reset_robot(images[0])
    ordinary = rollout(images,ticks,seed,bootstrap,infer,rtc=False,profile=profile)
    np.testing.assert_array_equal(guided['issued'][:profile.first_handoff_tick],
                                  ordinary['issued'][:profile.first_handoff_tick])
    paired = {name:[] for name in ('guided','ordinary','target','short_guided','short_ordinary','short_target')}
    rows = []
    arrays = dict(bootstrap=bootstrap, rtc_issued=guided['issued'], ordinary_issued=ordinary['issued'],
                  gt=gt[:ticks], valid=valid[:ticks], span=span)
    for proposal in guided['proposals']:
        start, frozen = proposal['start'], proposal['binding']['frozen']
        target_ticks = (start+np.arange(128))*3
        mask = target_ticks < len(valid)
        mask[mask] &= valid[target_ticks[mask]]
        mask[:frozen] = False
        target = gt[target_ticks[mask]]
        a, b = proposal['actions'][mask], proposal['ordinary_same_context'][mask]
        paired['guided'].append(a); paired['ordinary'].append(b); paired['target'].append(target)
        row = dict(job=proposal['job'], observation_tick=proposal['observation_tick'],
            bind_tick=proposal['bind_tick'],commit_tick=proposal['commit_tick'],frozen=frozen,
            overlap=proposal['binding']['overlap'],plan_version=proposal['binding']['plan_version'],
            guided=scores(a,target,span),ordinary_same_context=scores(b,target,span))
        short_ticks = proposal['observation_tick']+3*frozen+np.arange(16)
        short_ticks = short_ticks[short_ticks < len(valid)]
        short_ticks = short_ticks[valid[short_ticks]]
        for name, key in [('short_guided','actions'),('short_ordinary','ordinary_same_context')]:
            values = np.stack([native_sample(proposal[key], t-proposal['observation_tick'])
                               for t in short_ticks]) if len(short_ticks) else np.empty((0,14))
            paired[name].append(values)
        paired['short_target'].append(gt[short_ticks])
        arrays[f"job{proposal['job']}_rtc"] = proposal['actions']
        arrays[f"job{proposal['job']}_ordinary_same_context"] = proposal['ordinary_same_context']
        arrays[f"job{proposal['job']}_reference"] = proposal['binding']['reference']
        rows.append(row)
    for proposal in ordinary['proposals']:
        arrays[f"job{proposal['job']}_ordinary_independent"] = proposal['actions']
    for name, values in paired.items():
        arrays[name] = np.concatenate(values) if values else np.empty((0,14))
    valid = valid[:ticks]
    after = valid & (np.arange(ticks) >= profile.first_handoff_tick)
    new = valid & (np.arange(ticks) >= profile.first_editable_tick)
    report = dict(episode=episode,ticks=ticks,valid_ticks=int(valid.sum()),human_latent=metadata['icl_latent_path'],
        profile=profile.metadata(),seed=seed,
        zero_guidance_max_abs=zero_error,commits=len(guided['commits']),
        ordinary=scores(ordinary['issued'][valid],gt[:ticks][valid],span),
        rtc=scores(guided['issued'][valid],gt[:ticks][valid],span),
        after_first_handoff=dict(ordinary=scores(ordinary['issued'][after],gt[:ticks][after],span),
                                 rtc=scores(guided['issued'][after],gt[:ticks][after],span)),
        from_first_rtc_editable=dict(ordinary=scores(ordinary['issued'][new],gt[:ticks][new],span),
                                    rtc=scores(guided['issued'][new],gt[:ticks][new],span)),
        paired_mutable_suffix=dict(ordinary=scores(arrays['ordinary'],arrays['target'],span),
                                  rtc=scores(arrays['guided'],arrays['target'],span)),
        paired_first_16_editable_native_commands=dict(
            ordinary=scores(arrays['short_ordinary'],arrays['short_target'],span),
            rtc=scores(arrays['short_guided'],arrays['short_target'],span)),proposals=rows)
    output.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output/'predictions.npz',**arrays)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',type=int,required=True)
    parser.add_argument('--episodes',type=int,nargs='+',required=True)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--gt',type=Path,required=True,help='Safe exported GT npz; only the scorer reads it')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--profiles',nargs='+',choices=PROFILES,default=['s28_snapshot'])
    args=parser.parse_args()
    os.environ['ASCEND_RT_VISIBLE_DEVICES']=str(args.device)
    os.environ['ASCEND_DEVICE_ID']=os.environ['LOCAL_RANK']=os.environ['RANK']='0'
    os.environ['WORLD_SIZE']='1'
    args.output.mkdir(parents=True,exist_ok=True)
    from .rtc_pipeline_probe import make_runtime
    import torch
    torch.use_deterministic_algorithms(True)
    runtime=make_runtime(dict(model=str(args.model),root=str(args.root),output=str(args.output),video_steps=3,action_steps=10))
    gt=np.load(args.gt)
    episodes=list(map(json.loads,(args.root/'eval/piperpika/meta/episodes.jsonl').read_text().splitlines()))
    rows=[]
    for ep in episodes:
        if ep['episode_index'] not in args.episodes:
            continue
        eid=ep['episode_index']
        ticks=int(np.flatnonzero(gt[f'valid_{eid}'])[-1])+1
        images=load_rgb(args.root,eid,ticks)
        for name in args.profiles:
            print(f'EVAL_START profile={name} episode={eid}',flush=True)
            destination=args.output if args.profiles==['s28_snapshot'] else args.output/name
            report=evaluate_episode(runtime,ep,args.root,gt[f'action_{eid}'],gt[f'valid_{eid}'],gt['span'],
                destination/f'episode_{eid:06d}',check_zero=not rows,profile=PROFILES[name],images=images)
            rows.append(report)
            (args.output/'partial.json').write_text(json.dumps(rows,indent=2)+'\n')
            print('EVAL_RESULT '+json.dumps(dict(profile=name,
                **{k:report[k] for k in ('episode','ticks','commits','ordinary','rtc')})),flush=True)
    (args.output/'worker_report.json').write_text(json.dumps(dict(success=True,episodes=args.episodes,results=rows),indent=2)+'\n')


if __name__=='__main__':
    main()
