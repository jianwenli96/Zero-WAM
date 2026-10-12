"""ALOHA + HumanGen finetuning, with dataset-local absolute action statistics."""
from copy import deepcopy
import os
from pathlib import Path
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from ..dataset.aloha import CHANNELS

va_aloha_train_cfg = deepcopy(va_robotwin_train_cfg)
cfg = va_aloha_train_cfg
root = Path(os.environ.get('ALOHA_PREPARED_ROOT', '/mnt/sfs_turbo/lijianwen/Datasets/aloha_zerowam_v3'))
cfg.model_path = os.environ.get('MODEL_PATH', '/mnt/sfs_turbo/public/ckpts/Zero-WAM/zero-wam-pretrain')
cfg.__name__ = 'Config: ALOHA HumanGen train'
cfg.dataset_path = str(root / 'train')
cfg.icl_manifest_path = str(root / 'icl_manifest.json')
cfg.human_latent_path = str(root / 'human_latents')
cfg.robot_latent_path = ''
cfg.load_robotwin_stats_from_dataset = False
cfg.action_stats_path = str(root / 'train/breakfast/meta/action_stats.json')
cfg.excluded_task_names = []
cfg.expected_num_train_tasks = 1
cfg.used_action_channel_ids = CHANNELS
cfg.inverse_used_action_channel_ids = [CHANNELS.index(i) if i in CHANNELS else 28 for i in range(30)]
cfg.learning_rate = 1e-5
cfg.batch_size = 1
cfg.gradient_accumulation_steps = 1
cfg.num_steps = 10000
