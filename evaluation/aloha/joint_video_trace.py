"""Optional ICL prediction recording; leaves model inference and caches unchanged.

Import install() before launching evaluation.aloha.server. Decode saved tensors
in a separate process with this file's --trace argument.
"""
import argparse
import json
import os
from pathlib import Path
import time
import uuid


def install():
    import numpy as np
    import torch
    from wan_va.wan_va_server import VA_Server
    original = VA_Server.infer
    root = Path(os.environ['ZERO_WAM_PREDICTION_ROOT'])
    def infer(self, obs):
        if obs.get('reset'):
            d = root/(time.strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:8])
            d.mkdir(parents=True, exist_ok=False)
            self._joint_video_dir = d
            self._joint_video_index = 0
            (d/'reset.json').write_text(json.dumps(obs, indent=2, default=str))
            (d/'config.json').write_text(json.dumps(dict(self.job_config), indent=2, default=str))
        result = original(self, obs)
        if obs.get('reset'):
            result['prediction_trace_dir'] = str(self._joint_video_dir)
        elif 'action' in result and hasattr(self, '_joint_video_dir'):
            d = self._joint_video_dir/f'chunk_{self._joint_video_index:03d}'
            d.mkdir(exist_ok=False)
            images = obs['obs']
            if isinstance(images, list):
                images = images[0]
            np.savez_compressed(d/'observation.npz', **images)
            np.save(d/'raw_action.npy', result['action'])
            torch.save({'predicted': self.last_predicted_latents.detach().cpu().clone(),
                        'initial': self.init_latent.detach().cpu().clone(),
                        'cached_actions': self.last_predicted_actions.detach().cpu().clone()}, d/'final_latents.pt')
            (d/'saved.json').write_text(json.dumps({'complete': True, 'chunk': self._joint_video_index,
                'time': time.time(), 'camera_order': list(self.job_config.obs_cam_keys)}))
            self._joint_video_index += 1
        return result
    VA_Server.infer = infer


def decode(trace, device):
    import types
    import numpy as np
    import torch
    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    from wan_va import wan_va_server as server
    if device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(device)
    cfg = json.loads((trace.parent/'config.json').read_text())
    saved = json.loads((trace/'saved.json').read_text())
    if not saved['complete']:
        raise ValueError('Incomplete trace')
    vae = server.load_vae(str(Path(cfg['model_path'])/'vae'), torch_dtype=torch.bfloat16, torch_device=device)
    obj = types.SimpleNamespace(vae=vae, video_processor=server.VideoProcessor(vae_scale_factor=1))
    data = torch.load(trace/'final_latents.pt', map_location='cpu', weights_only=False)
    def rgb(latent):
        if latent.shape[-1] % 3:
            raise ValueError('Expected three views concatenated along latent width')
        views = []
        for part in latent.chunk(3, dim=-1):
            vae.clear_cache()
            with torch.no_grad():
                result = server.VA_Server.decode_one_video(obj, part.to(device), 'np')[0]
            views.append(np.rint(np.clip(result, 0, 1)*255).astype(np.uint8))
        return views
    views, initial = rgb(data['predicted']), rgb(data['initial'])
    keys = saved['camera_order']
    joined = np.concatenate(views, axis=2)
    # Prepared dataset: 30 Hz action/capture, stride 4 robot video sampling.
    fps = 7.5
    for name, frames, rate in [('predicted_three_views.mp4', joined, fps),
                               ('predicted_three_views_slow.mp4', joined, 2)]:
        iio.mimwrite(str(trace/name), frames, fps=rate, codec='libx264', macro_block_size=1)
    for key, frames in zip(keys, views):
        iio.mimwrite(str(trace/(key.split('.')[-1]+'.mp4')), frames, fps=fps, codec='libx264', macro_block_size=1)
    with np.load(trace/'observation.npz') as obs:
        h, w = joined.shape[1], views[0].shape[2]
        canvas = Image.new('RGB', (3*w, 3*(h+24)), 'white')
        draw = ImageDraw.Draw(canvas)
        for row, (label, images) in enumerate([
                ('Actual input', [obs[k] for k in keys]),
                ('VAE reconstruction of input', [v[0] for v in initial]),
                ('Predicted last frame (rerun)', [v[-1] for v in views])]):
            draw.text((4, row*(h+24)+4), label, fill='black')
            for col, im in enumerate(images):
                canvas.paste(Image.fromarray(im).resize((w,h)), (col*w,row*(h+24)+24))
        canvas.save(trace/'observation_vs_prediction.jpg', quality=95)
    (trace/'decode_report.json').write_text(json.dumps({'frames': len(joined), 'fps': fps,
        'camera_order': keys, 'video_timebase': 'dataset 30 fps / stride 4',
        'first_frame': 'observation-conditioned', 'slow_fps': 2}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--device', default='npu:0')
    args = p.parse_args()
    decode(args.trace, args.device)
