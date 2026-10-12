"""Independent-window open-loop evaluation through the production ICL server.

No robot is connected. Every window resets caches; only its initial RGB is sent.
The first latent action block is padding and is discarded, exactly as in training.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
from wan_va.dataset.aloha import CAMERAS, CHANNELS, model_quantiles


def flatten_first_chunk(action):
    action = np.asarray(action)
    if action.ndim != 3 or action.shape[0] != 28 or action.shape[1] < 2 or action.shape[2] != 16:
        raise ValueError(f'Expected [28, F>=2, 16] absolute ALOHA actions, got {action.shape}')
    prediction = action[:, 1:, :].transpose(1, 2, 0).reshape(-1, 28)
    if not np.isfinite(prediction).all(): raise ValueError('Nonfinite prediction')
    return prediction


def metrics(pred, target, q01, q99):
    raw_error = pred - target
    aligned = pred.copy()
    for start in (3, 10):
        signs = np.where((pred[:, start:start+4] * target[:, start:start+4]).sum(-1) < 0, -1, 1)
        aligned[:, start:start+4] *= signs[:, None]
    error = aligned - target
    result = {'raw_normalized_mae': float(np.abs(raw_error / np.maximum(q99-q01, 1e-6)).mean()),
              'mae_per_channel': np.abs(error).mean(0).tolist(),
              'rmse_per_channel': np.sqrt((error**2).mean(0)).tolist(),
              'normalized_mae': float(np.abs(error / np.maximum(q99-q01, 1e-6)).mean())}
    for name, indices in [('position', [0,1,2,7,8,9]), ('joint', list(range(14,26))), ('gripper', [26,27])]:
        result[name + '_mae'] = float(np.abs(error[:, indices]).mean())
    for side, start in [('left', 3), ('right', 10)]:
        p, t = pred[:, start:start+4], target[:, start:start+4]
        norm = np.linalg.norm(p, axis=1)
        result[side + '_quaternion_norm_error'] = float(np.abs(norm-1).mean())
        result[side + '_invalid_quaternion_count'] = int((norm < 1e-8).sum())
        dots = np.abs(np.sum(p*t, axis=1) / np.maximum(norm*np.linalg.norm(t,axis=1), 1e-8))
        result[side + '_rotation_deg'] = float(np.degrees(2*np.arccos(np.clip(dots,0,1))).mean())
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3'))
    p.add_argument('--split', choices=['train','eval'], default='eval')
    p.add_argument('--host', default='127.0.0.1'); p.add_argument('--port', type=int, default=29536)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--max-windows', type=int, default=20)
    p.add_argument('--human-latent', type=Path, help='Override with an encoded self-collected human demonstration')
    p.add_argument('--no-icl', action='store_true', help='Run matched no-human ablation')
    p.add_argument('--seed', type=int, default=42)
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
    task = args.root / args.split / 'breakfast'
    episodes = [json.loads(x) for x in (task / 'meta/episodes.jsonl').read_text().splitlines()]
    action_contract = json.loads((task / 'meta/action_stats.json').read_text()).get('action_contract', 'aloha_absolute_28_v1')
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
            human = args.human_latent or args.root / 'human_latents' / Path(ep['icl']['human_video_path']).with_suffix('.pth')
            reset_result = infer({'reset': True, 'seed': args.seed + eid * 10000 + start, 'prompt': ep['tasks'][0], 'use_icl': not args.no_icl,
                   'icl_latent_path': str(human.resolve()), 'video_guidance_scale': -1, 'icl_guidance_scale': 5})
            if reset_result.get('action_contract') != action_contract:
                raise ValueError('Connect to evaluation.aloha.server, not a Robotwin server')
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
            pred = flatten_first_chunk(infer({'obs': images})['action'])
            actions = np.asarray(pq.read_table(task / 'data' / f'chunk-{eid//1000:03d}' / f'episode_{eid:06d}.parquet', columns=['action'])['action'].to_pylist())
            if start + len(pred) > segment['end_frame'] - 1:
                raise ValueError('Inference horizon exceeds prepared window')
            target = actions[start:start+len(pred)]
            baseline = np.repeat(actions[start:start+1], len(pred), axis=0)
            predictions.append(pred); targets.append(target); baselines.append(baseline)
            rows.append({'episode': eid, 'start': start, 'human_latent': str(human),
                         'num_actions': len(pred), 'metrics': metrics(pred, target, q01, q99)})
            print(f'Episode {eid} frame {start}: normalized MAE={rows[-1]["metrics"]["normalized_mae"]:.4f}', flush=True)
    finally:
        ws.close()
    if not rows:
        raise ValueError('No windows long enough for the server inference horizon')
    pred, target, baseline = map(np.concatenate, [predictions, targets, baselines])
    np.savez_compressed(args.output / 'predictions.npz', prediction=pred, target=target, hold_baseline=baseline)
    action_names = json.loads((task / 'meta/info.json').read_text())['features']['action']['names']
    report = {'action_names': action_names, 'split': args.split, 'use_icl': not args.no_icl, 'seed': args.seed,
              'action_contract': action_contract, 'checkpoint_contract_bound': reset_result['checkpoint_contract_bound'],
              'model_path': reset_result['model_path'], 'inference_steps': reset_result['inference_steps'],
              'protocol': 'independent windows; initial RGB only; absolute actions; first padded block discarded',
              'metrics': metrics(pred,target,q01,q99), 'hold_baseline': metrics(baseline,target,q01,q99), 'skipped_windows': skipped, 'windows': rows}
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['metrics'], indent=2))


if __name__ == '__main__': main()
