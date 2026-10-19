"""Independent-window open-loop evaluation through the production ICL server.

No robot is connected. Every window resets caches; only its initial RGB is sent.
The first latent action block is padding and is discarded, exactly as in training.
"""
import argparse
import os
import time
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from wan_va.dataset.piperpika import CAMERAS, CHANNELS, model_quantiles


def flatten_first_chunk(action):
    action = np.asarray(action)
    if action.ndim != 3 or action.shape[0] != 14 or action.shape[1] < 2 or action.shape[2] != 16:
        raise ValueError(f'Expected [14, F>=2, 16] absolute PiperPika actions, got {action.shape}')
    prediction = action[:, 1:, :].transpose(1, 2, 0).reshape(-1, 14)
    if not np.isfinite(prediction).all(): raise ValueError('Nonfinite prediction')
    return prediction


def metrics(pred, target, q01, q99):
    error = np.asarray(pred) - np.asarray(target)
    if error.ndim != 2 or error.shape[1] != 14:
        raise ValueError('Expected [T,14] joint/gripper predictions')
    return {'mae_per_channel': np.abs(error).mean(0).tolist(),
            'rmse_per_channel': np.sqrt((error**2).mean(0)).tolist(),
            'normalized_mae': float(np.abs(error / np.maximum(q99-q01, 1e-6)).mean()),
            'joint_mae': float(np.abs(error[:,:12]).mean()),
            'gripper_mae': float(np.abs(error[:,12:]).mean())}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('PIPERPIKA_PREPARED_ROOT', '/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2')))
    p.add_argument('--split', choices=['train','eval'], default='eval')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=29536)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-windows', type=int, default=20)
    p.add_argument('--human-latent', type=Path, help='Override with an encoded self-collected human demonstration')
    p.add_argument('--no-icl', action='store_true', help='Run matched no-human ablation')
    p.add_argument('--target-text', choices=['enabled', 'disabled'], default='disabled',
                   help='Enable target prompt conditioning; default preserves the original text-free target protocol')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--window-position', choices=['start', 'middle'], default='start',
                   help='Evaluate from the prepared segment start or its middle (aligned to 16 actions)')
    args = p.parse_args()
    if args.max_windows < 1: p.error('--max-windows must be positive')
    import cv2
    import websockets.sync.client
    from evaluation.robotwin.msgpack_numpy import Packer, unpackb
    # Bounded connection/response waits, unlike the interactive client's retry loop.
    ws = websockets.sync.client.connect(f'ws://{args.host}:{args.port}', open_timeout=30,
                                       max_size=None, compression=None, ping_interval=None)
    unpackb(ws.recv(timeout=30)); packer = Packer()
    def infer(message):
        ws.send(packer.pack(message)); response = ws.recv(timeout=1800)
        if isinstance(response, str): raise RuntimeError(response)
        return unpackb(response)
    task = args.root / args.split / 'piperpika'
    episodes = [json.loads(x) for x in (task / 'meta/episodes.jsonl').read_text().splitlines()]
    action_contract = json.loads((task / 'meta/action_stats.json').read_text()).get('action_contract', 'piperpika_absolute_joint12_gripper2_v1')
    quantiles = model_quantiles(task / 'meta/action_stats.json')
    q01, q99 = [np.asarray(quantiles[q])[CHANNELS] for q in ('q01','q99')]
    windows = [(ep, segment) for ep in episodes for segment in ep['action_config']]
    np.random.default_rng(args.seed).shuffle(windows)
    predictions, targets, baselines, rows, skipped = [], [], [], [], []
    args.output.mkdir(parents=True, exist_ok=True)
    try:
        for ep, segment in windows:
            if len(rows) >= args.max_windows:
                break
            eid, start = ep['episode_index'], segment['start_frame']
            if args.window_position == 'middle':
                start += ((segment['end_frame'] - 1 - start) // 32) * 16
            human = args.human_latent or args.root / 'human_latents' / Path(ep['icl']['human_video_path']).with_suffix('.pth')
            reset_result = infer({'reset': True, 'seed': args.seed + eid * 10000 + start, 'prompt': ep['tasks'][0], 'use_icl': not args.no_icl,
                   'icl_latent_path': str(human.resolve()),
                   'video_guidance_scale': 1 if args.target_text == 'enabled' else -1, 'icl_guidance_scale': 5})
            if reset_result.get('target_text_enabled') != (args.target_text == 'enabled'):
                raise ValueError('Server target text condition differs from requested evaluation protocol')
            if reset_result.get('action_contract') != action_contract:
                raise ValueError('Connect to evaluation.piperpika.server, not a Robotwin server')
            if reset_result.get('norm_stat') != quantiles:
                raise ValueError('Client and server action statistics differ')
            horizon = (reset_result['chunk_frames'] - 1) * 16
            if horizon > segment['end_frame'] - 1 - start:
                skipped.append({'episode': eid, 'start': start, 'reason': 'shorter than inference horizon'})
                continue
            images = {}
            for camera in CAMERAS:
                cap = cv2.VideoCapture(str(task / 'videos' / f'chunk-{eid//1000:03d}' / camera / f'episode_{eid:06d}.mp4'))
                cap.set(cv2.CAP_PROP_POS_FRAMES, start)
                ok, frame = cap.read(); cap.release()
                if not ok: raise ValueError(f'Cannot decode {eid}/{camera}/{start}')
                images[camera] = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            started = time.perf_counter()
            pred = flatten_first_chunk(infer({'obs': images})['action'])
            latency = time.perf_counter() - started
            actions = np.asarray(pq.read_table(task / 'data' / f'chunk-{eid//1000:03d}' / f'episode_{eid:06d}.parquet', columns=['action'])['action'].to_pylist())
            if start + len(pred) > segment['end_frame'] - 1:
                raise ValueError('Inference horizon exceeds prepared window')
            target = actions[start:start+len(pred),14:]
            baseline = np.repeat(actions[start:start+1,14:], len(pred), axis=0)
            predictions.append(pred); targets.append(target); baselines.append(baseline)
            rows.append({'episode': eid, 'start': start, 'human_latent': str(human), 'prompt': ep['tasks'][0],
                         'num_actions': len(pred), 'inference_seconds': latency, 'metrics': metrics(pred, target, q01, q99)})
            print(f'Episode {eid} frame {start}: normalized MAE={rows[-1]["metrics"]["normalized_mae"]:.4f}', flush=True)
    finally:
        ws.close()
    if not rows:
        raise ValueError('No windows long enough for the server inference horizon')
    pred, target, baseline = map(np.concatenate, [predictions, targets, baselines])
    np.savez_compressed(args.output / 'predictions.npz', prediction=pred, target=target, hold_baseline=baseline)
    action_names = json.loads((task / 'meta/info.json').read_text())['features']['action']['names'][14:]
    report = {'action_names': action_names, 'split': args.split, 'use_icl': not args.no_icl, 'seed': args.seed,
              'window_position': args.window_position,
              'target_text': args.target_text, 'video_guidance_scale': reset_result['video_guidance_scale'],
              'icl_guidance_scale': reset_result['icl_guidance_scale'],
              'action_contract': action_contract, 'checkpoint_contract_bound': reset_result['checkpoint_contract_bound'],
              'model_path': reset_result['model_path'], 'inference_steps': reset_result['inference_steps'],
              'protocol': 'independent windows; initial RGB only; absolute joints/grippers; first padded block discarded',
              'mean_inference_seconds': float(np.mean([r['inference_seconds'] for r in rows])),
              'metrics': metrics(pred,target,q01,q99), 'hold_baseline': metrics(baseline,target,q01,q99), 'skipped_windows': skipped, 'windows': rows}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__': main()
