"""Launch the existing ICL runtime with ALOHA's 28-channel absolute action contract."""
import argparse
from copy import deepcopy
import logging
import os
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('ALOHA_PREPARED_ROOT', '/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3')))
    p.add_argument('--model', type=Path, required=True, help='Complete model root: transformer, vae, tokenizer, text_encoder')
    p.add_argument('--allow-unbound-model', action='store_true', help='Explicitly allow a base/legacy model with no saved ALOHA statistics')
    p.add_argument('--port', type=int, default=29536)
    p.add_argument('--steps', type=int, default=50)
    p.add_argument('--chunk-frames', type=int, default=2)
    args = p.parse_args()
    if args.chunk_frames < 2 or args.steps < 1: p.error('Need chunk-frames >= 2 and positive steps')
    logging.basicConfig(level=logging.INFO)
    from wan_va.wan_va_server import VA_Server, init_distributed, run_async_server_mode
    from wan_va.configs.va_aloha_train_cfg import va_aloha_train_cfg
    from wan_va.dataset.aloha import model_quantiles, verify_checkpoint_contract
    contract, bound = verify_checkpoint_contract(args.model, args.root, args.allow_unbound_model)
    cfg = deepcopy(va_aloha_train_cfg)
    cfg.norm_stat = model_quantiles(args.root / 'train/breakfast/meta/action_stats.json')
    cfg.model_path = str(args.model)
    cfg.frame_chunk_size = args.chunk_frames
    cfg.num_inference_steps = cfg.action_num_inference_steps = args.steps
    cfg.rank = int(os.environ.get('RANK', 0)); cfg.local_rank = int(os.environ.get('LOCAL_RANK', 0))
    cfg.world_size = int(os.environ.get('WORLD_SIZE', 1))
    init_distributed(cfg.world_size, cfg.local_rank, cfg.rank)
    class AlohaServer(VA_Server):
        def infer(self, obs):
            if obs.get('reset'):
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
            return result
    model = AlohaServer(cfg)
    run_async_server_mode(model, cfg.local_rank, '127.0.0.1', args.port)


if __name__ == '__main__': main()
