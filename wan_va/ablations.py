"""Independent ablation controls shared by training entry points."""


def apply_ablation_flags(config, args):
    for name in ('human_video', 'text', 'mcp'):
        if getattr(args, f'disable_{name}', False):
            config[f'enable_{name}'] = False
    if not config.get('enable_text', True):
        # Use the pretrained null embedding, never an all-zero replacement.
        config.droptext_target = 1.0
    if getattr(args, 'robotwin_only', False):
        args.datasets = 'robotwin:1.0'
        # The length-bucket sampler currently supports mixtures only.
        config.length_bucket_steps = 0
    config.cfg_prob = config.droptext_target
