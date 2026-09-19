from copy import deepcopy
from types import SimpleNamespace
import sys

import pytest
import wan_va.train as training
from wan_va.configs.zerowam_train_config import zerowam_train_cfg


@pytest.mark.parametrize('override', [False, True])
def test_file_configuration_survives_default_cli_and_explicit_cli_wins(monkeypatch, override):
    cfg = deepcopy(training.VA_CONFIGS['robotwin_train'])
    values = dict(length_bucket_steps=8, max_train_frames=32)
    cfg.update(values)
    monkeypatch.setitem(training.VA_CONFIGS, 'robotwin_train', cfg)
    args = ['train', '--config-name', 'robotwin_train']
    if override:
        values = dict(length_bucket_steps=0, max_train_frames=16)
        for key, value in values.items():
            args.extend(['--'+key.replace('_','-'), str(value)])
    monkeypatch.setattr(sys, 'argv', args)
    class CapturedConfiguration(Exception): pass
    def capture(config, *args, **kwargs):
        for key, value in values.items():
            assert config[key] == value
        raise CapturedConfiguration
    monkeypatch.setattr(training, '_build_dataset_sources', capture)
    with pytest.raises(CapturedConfiguration):
        training.main()


def test_shared_defaults_and_effective_bucket_validation():
    expected = {key: zerowam_train_cfg[key] for key in
                ('length_bucket_steps', 'max_train_frames')}
    for key, value in expected.items():
        assert zerowam_train_cfg[key] == value
        assert training.VA_CONFIGS['robotwin_train'][key] == value
    # Validation must read the effective config, even when CLI has no override.
    with pytest.raises(ValueError, match='mixture'):
        training._build_dataset_sources(SimpleNamespace(length_bucket_steps=8),
            SimpleNamespace(datasets='robotwin:1', length_bucket_steps=None), 0, 0, 1)
