"""Resolve every source in a mixture against one prepared HumanGen root."""
from pathlib import Path


def configure_humangen_source(config, name, root):
    root = Path(root).expanduser().resolve()
    config.dataset_path = str(root / f'{name}_data')
    config.human_latent_path = str(root / 'human_latents' / name)
    config.robot_latent_path = ''
    manifest = root / 'icl_configs' / f'ICL_config_{name}.json'
    if name == 'robotwin':
        training_manifest = root / 'icl_configs/ICL_config_robotwin_train.json'
        if training_manifest.is_file():
            manifest = training_manifest
    config.icl_manifest_path = str(manifest)
    return config
