"""Launch the existing ICL runtime with ALOHA's 28-channel absolute action contract."""
import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('ALOHA_PREPARED_ROOT', '/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3')))
    p.add_argument('--model', type=Path, required=True, help='Complete model root: transformer, vae, tokenizer, text_encoder')
    p.add_argument('--allow-unbound-model', action='store_true', help='Explicitly allow a base/legacy model with no saved ALOHA statistics')
    p.add_argument('--port', type=int, default=29536)
    p.add_argument('--steps', type=int, help='Default: 50, or 1 with RTC')
    p.add_argument('--action-steps', '--action_steps', type=int,
                   help='Override action denoising steps; --steps sets video steps and the action default')
    p.add_argument('--chunk-frames', type=int, help='Default: 2; RTC: 16 native-rate or 4 for time-strided checkpoints')
    p.add_argument('--action-stride', '--action_stride', type=int, default=None,
                   help='Use checkpoint joint time stride; explicit override allowed only for legacy timing probes')
    p.add_argument('--actions-per-latent', '--actions_per_latent', type=int, default=None,
                   help='Action targets per video latent, e.g. 32; defaults to the checkpoint layout')
    p.add_argument('--enable-rtc', '--enable_rtc', action='store_true', default=False,
                   help='Enable training-free RTC; requires the RTC client protocol')
    p.add_argument('--rtc-control-hz', type=float, default=30)
    p.add_argument('--rtc-execution-horizon', type=int, default=None)
    p.add_argument('--rtc-delay-buffer-size', type=int, default=10)
    p.add_argument('--rtc-initial-latency', type=float, default=None,
                   help='Seed end-to-end latency in seconds; client must warm up and measure it')
    p.add_argument('--rtc-latency-multiplier', type=float, default=1.2)
    p.add_argument('--rtc-latency-margin', type=float, default=0.05)
    p.add_argument('--rtc-max-guidance-weight', type=float, default=5)
    p.add_argument('--rtc-attn-window', type=int, default=4)
    p.add_argument('--rtc-icl-guidance-scale', type=float, default=1)
    args = p.parse_args(argv)
    args.chunk_frames_explicit = args.chunk_frames is not None
    args.execution_horizon_explicit = args.rtc_execution_horizon is not None
    args.initial_latency_explicit = args.rtc_initial_latency is not None
    coarse = (args.action_stride is not None and args.action_stride > 1) or args.actions_per_latent not in (None, 16)
    args.steps = args.steps if args.steps is not None else (1 if args.enable_rtc else 50)
    args.action_steps = args.action_steps if args.action_steps is not None else args.steps
    args.chunk_frames = args.chunk_frames if args.chunk_frames is not None else ((4 if coarse else 16) if args.enable_rtc else 2)
    args.rtc_execution_horizon = args.rtc_execution_horizon if args.rtc_execution_horizon is not None else (2*(args.actions_per_latent or 16) if coarse else 128)
    args.rtc_initial_latency = args.rtc_initial_latency if args.rtc_initial_latency is not None else (2.0 if coarse else 3.342)
    if args.chunk_frames < 2 or min(args.steps, args.action_steps) < 1:
        p.error('Need chunk-frames >= 2 and positive video/action steps')
    if args.actions_per_latent is not None and args.actions_per_latent < 1:
        p.error('Need positive actions-per-latent')
    if args.enable_rtc and not 4 <= args.rtc_attn_window <= 64:
        p.error('RTC attention window must be between 4 and 64')
    return args


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    from wan_va.wan_va_server import VA_Server, init_distributed, run_async_server_mode
    if os.environ.get('ZERO_WAM_PREDICTION_ROOT'):
        from evaluation.aloha.joint_video_trace import install
        install()
    from wan_va.configs.va_aloha_train_cfg import va_aloha_train_cfg
    from wan_va.dataset.aloha import model_quantiles, verify_checkpoint_contract
    contract, bound = verify_checkpoint_contract(args.model, args.root, args.allow_unbound_model)
    cfg = deepcopy(va_aloha_train_cfg)
    cfg.norm_stat = model_quantiles(args.root / 'train/breakfast/meta/action_stats.json')
    cfg.model_path = str(args.model)
    from wan_va.action_sampling import inference_action_stride, inference_actions_per_latent
    cfg.action_stride, cfg.sampling_contract_bound = inference_action_stride(args.model, args.action_stride)
    cfg.native_action_per_frame = cfg.action_per_frame
    cfg.action_per_frame, _ = inference_actions_per_latent(args.model, args.actions_per_latent,
                                                         cfg.native_action_per_frame)
    cfg.actions_per_latent = cfg.action_per_frame
    if cfg.action_stride > 1 or cfg.action_per_frame != cfg.native_action_per_frame:
        if not args.enable_rtc:
            raise ValueError('Time-strided checkpoints require --enable-rtc for clock and feedback alignment')
        if not args.chunk_frames_explicit:
            args.chunk_frames = 4
        if not args.execution_horizon_explicit:
            args.rtc_execution_horizon = 2*cfg.action_per_frame
        if not args.initial_latency_explicit:
            args.rtc_initial_latency = 2.0
        if not cfg.sampling_contract_bound:
            logging.warning('Legacy checkpoint has no joint-time-stride training contract; this is a timing probe, not quality validation')
    cfg.frame_chunk_size = args.chunk_frames
    cfg.num_inference_steps = args.steps
    cfg.action_num_inference_steps = args.action_steps
    cfg.enable_rtc = args.enable_rtc
    if args.enable_rtc:
        from wan_va.rtc.config import RTCConfig, RTCError
        for name in ('control_hz', 'execution_horizon', 'delay_buffer_size',
                     'initial_latency', 'latency_multiplier', 'latency_margin',
                     'max_guidance_weight', 'attn_window'):
            setattr(cfg, 'rtc_'+name, getattr(args, 'rtc_'+name))
        cfg.icl_guidance_scale = args.rtc_icl_guidance_scale
        RTCConfig.from_job_config(cfg).validate()
        if int(os.environ.get('WORLD_SIZE', '1')) != 1:
            raise RTCError('RTC requires NGPU=1; distributed FSDP inference is unsupported')
    cfg.rank = int(os.environ.get('RANK', 0)); cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    cfg.world_size = int(os.environ.get('WORLD_SIZE', 1))
    init_distributed(cfg.world_size, cfg.local_rank, cfg.rank)
    class AlohaServer(VA_Server):
        def infer(self, obs):
            if obs.get('reset'):
                if obs.get('enable_rtc') is not None and bool(obs['enable_rtc']) != self.enable_rtc:
                    raise ValueError('Client/server enable_rtc settings do not match')
                import torch
                seed = int(obs.get('seed', 42))
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
            result = super().infer(obs)
            if obs.get('reset'):
                result['action_contract'] = contract['stats'].get('action_contract', 'aloha_absolute_28_v1')
                result['checkpoint_contract_bound'] = bound
                result['chunk_frames'] = self.job_config.frame_chunk_size
                result['norm_stat'] = self.job_config.norm_stat
                result['model_path'] = self.job_config.model_path
                result['inference_steps'] = self.job_config.num_inference_steps
                result['action_inference_steps'] = self.job_config.action_num_inference_steps
                result['action_stride'] = self.job_config.action_stride
                result['actions_per_latent'] = self.job_config.action_per_frame
                result['sampling_contract_bound'] = self.job_config.sampling_contract_bound
                result['enable_rtc'] = self.enable_rtc
                result['image_size'] = [self.height, self.width]
            return result
    model = AlohaServer(cfg)
    run_async_server_mode(model, cfg.local_rank, '127.0.0.1', args.port)


if __name__ == '__main__': main()
