"""Conservative 64-device post-training with real-data replay."""
from copy import deepcopy
import os
from pathlib import Path
from .va_aloha_train_cfg import va_aloha_train_cfg
from ..dataset.piperpika import CHANNELS

va_piperpika_train_cfg = deepcopy(va_aloha_train_cfg)
cfg = va_piperpika_train_cfg
root = Path(os.environ.get('PIPERPIKA_PREPARED_ROOT', '/mnt/sfs_turbo/lijianwen/Datasets/piperpika_10_16_zerowam_joint_v2'))
cfg.__name__ = 'Config: PiperPika joints with real HumanGen replay'
cfg.dataset_path = str(root / 'train')
cfg.icl_manifest_path = str(root / 'icl_manifest.json')
cfg.human_latent_path = str(root / 'human_latents')
cfg.action_stats_path = str(root / 'train/piperpika/meta/action_stats.json')
cfg.used_action_channel_ids = CHANNELS
cfg.inverse_used_action_channel_ids = [CHANNELS.index(i) if i in CHANNELS else len(CHANNELS) for i in range(30)]
cfg.learning_rate = 2e-6
cfg.warmup_steps = 100
cfg.lr_schedule = 'cosine'
cfg.min_lr_ratio = 0.1
cfg.num_steps = 5000
cfg.save_interval = 500
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 1
cfg.max_train_frames = 60
# Random causal attention chunks, in latent frames (not the video crop length).
cfg.frame_chunk_size = 0
cfg.max_frame_chunk_size = 4
# Preserve the pretrained joint video/action/ICL/MCP objective and noise schedule.
cfg.drop_icl = 0.1
cfg.droptext_target = 0.4
# Inference defaults only; training continues to cover the 1000-step noise grid.
cfg.num_inference_steps = 20
cfg.action_num_inference_steps = 20
