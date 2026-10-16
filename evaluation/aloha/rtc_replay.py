"""Exercise the real asynchronous RTC client at 30 Hz using recorded RGB.

No robot is connected. Observations are recording playback, independent of the
issued model actions; this validates software/timing, not task success.
"""
import argparse
import json
from pathlib import Path
import time

import cv2
import numpy as np

from evaluation.aloha.rtc_client import CAMERAS, RTCPolicy, WebsocketRTCTransport


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3'))
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, default=29536)
    p.add_argument('--episode', type=int, default=1)
    p.add_argument('--start', type=int, default=0)
    p.add_argument('--ticks', type=int, default=512)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--warmup-rounds', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.ticks < 1 or args.start < 0:
        p.error('Need positive ticks and nonnegative start')
    cv2.setNumThreads(1)
    task = args.root/'eval/breakfast'
    ep = next(e for e in map(json.loads, (task/'meta/episodes.jsonl').read_text().splitlines())
              if e['episode_index'] == args.episode)
    if args.start+args.ticks > ep['length']:
        p.error('Replay exceeds the recorded episode')
    indices = list(range(args.start, args.start+args.ticks, 4))
    frames = {i: {} for i in indices}
    for camera in CAMERAS:
        cap = cv2.VideoCapture(str(task/'videos'/f'chunk-{args.episode//1000:03d}'/camera/f'episode_{args.episode:06d}.mp4'))
        for index in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f'Cannot decode {camera}/{index}')
            frames[index][camera] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        cap.release()
    args.output.mkdir(parents=True, exist_ok=True)
    policy = RTCPolicy(WebsocketRTCTransport(args.host, args.port))
    actions, lateness, startup, error = [], [], None, None
    try:
        startup = policy.start(frames[args.start], dict(
            seed=args.seed+args.episode*10000+args.start, prompt=ep['tasks'][0],
            use_icl=True, video_guidance_scale=-1,
            icl_latent_path=str(args.root/'human_latents'/Path(ep['icl']['human_video_path']).with_suffix('.pth')),
        ), warmup_rounds=args.warmup_rounds)
        period = 1/policy.config.control_hz
        deadline = time.monotonic()
        for tick in range(args.ticks):
            remaining = deadline-time.monotonic()
            if remaining > 0:
                time.sleep(remaining)
            lateness.append(max(0., time.monotonic()-deadline))
            actions.append(policy.get_action(frames[args.start+(tick//4)*4]))
            deadline += period
        if len(policy.measurements) < 2:
            raise RuntimeError('Replay too short to validate two asynchronous queue swaps; increase --ticks')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report = dict(success=error is None, error=error,
            protocol='30 Hz asynchronous RTC with real soft-mask VJP guidance; recorded RGB, no robot execution',
            episode=args.episode, start=args.start, num_actions=len(actions),
            startup=startup, measurements=policy.measurements,
            max_controller_lateness_ms=max(lateness, default=0)*1000)
        if actions:
            np.save(args.output/'issued_actions.npy', np.asarray(actions))
        (args.output/'rtc_replay_report.json').write_text(json.dumps(report, indent=2))
        policy.close()
        print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
