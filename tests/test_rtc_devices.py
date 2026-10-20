"""Replica-count and read-only NPU selection; no accelerator runtime needed."""
import pytest

from evaluation.piperpika import rtc_devices as devices
from evaluation.piperpika import rtc_pipeline_probe as probe


def arguments(*extra):
    return probe.parse_args(['--model', '/model', '--output', '/output', *extra])


def inventory():
    return [devices.NPUDevice(device, device, 0, 'OK', used, 65536, busy)
            for device, used, busy in [(0, 3425, False), (1, 35862, True),
                                      (3, 3414, False), (4, 3422, False),
                                      (5, 8610, True), (6, 3425, False)]]


def test_default_is_two_and_device_count_spellings_are_equivalent():
    assert arguments().device_count == 2
    assert arguments().devices is None
    for flag in ('--device-count', '--device_count'):
        assert arguments(flag, '4').device_count == 4


def test_explicit_devices_infer_count_and_preserve_existing_commands():
    args = arguments('--devices', '0', '3', '4', '6')
    assert args.device_count == 4
    assert args.devices == [0, 3, 4, 6]
    assert arguments('--device_count', '3', '--devices', '6', '0', '3').device_count == 3


@pytest.mark.parametrize('extra', [
    ['--device-count', '0'], ['--device-count', '-2'], ['--device-count', '5'],
    ['--device-count', '1'], ['--devices', '0'], ['--devices', '-1', '3'],
    ['--devices', '0', '0'], ['--device-count', '4', '--devices', '0', '3'],
])
def test_invalid_counts_and_explicit_list_conflicts_fail_before_device_discovery(extra):
    with pytest.raises(SystemExit):
        arguments(*extra)


def test_one_device_remains_available_for_profile_only():
    assert arguments('--device-count', '1', '--profile-only').device_count == 1
    assert arguments('--devices', '3', '--profile-only').device_count == 1


def test_automatic_selection_skips_busy_cards_and_prefers_free_memory():
    args = arguments()
    selection = devices.resolve_devices(args, inventory=inventory(), environ={})
    assert args.devices == [3, 4]
    assert selection['source'] == 'npu-smi_idle'
    assert [item['device'] for item in selection['selected']] == [3, 4]
    assert all(item['has_processes'] is False for item in selection['selected'])
    args = arguments('--device_count', '4')
    devices.resolve_devices(args, inventory=inventory(), environ={})
    assert args.devices == [3, 4, 0, 6]
    assert args.receding_actions == 32
    assert (args.video_steps, args.action_steps, args.handoff_latency) == (3, 10, 0.)


def test_visible_noncontiguous_runtime_ids_and_mask_order_are_preserved():
    args = arguments()
    selection = devices.resolve_devices(args, inventory=inventory(),
                                        environ={'ASCEND_RT_VISIBLE_DEVICES': '6, 0,3'})
    assert args.devices == [6, 0]
    assert selection['visible_devices'] == [6, 0, 3]
    assert selection['source'] == 'visible_mask'


def test_explicit_list_keeps_order_and_can_share_a_sufficiently_free_card():
    args = arguments('--devices', '5', '0')
    selection = devices.resolve_devices(args, inventory=inventory(), environ={})
    assert args.devices == [5, 0]
    assert selection['source'] == 'explicit_devices'


@pytest.mark.parametrize('extra,env,reason', [
    (['--devices', '0', '4'], {'ASCEND_RT_VISIBLE_DEVICES': '0,3'}, 'exceeds'),
    (['--device-count', '3'], {'ASCEND_RT_VISIBLE_DEVICES': '0,3'}, '2 visible'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': ''}, '0 visible'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': '-1'}, '0 visible'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': '0,0'}, 'duplicate'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': '0,,3'}, 'comma-separated'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': '0,unknown'}, 'comma-separated'),
    ([], {'ASCEND_RT_VISIBLE_DEVICES': '0,99'}, 'not found'),
    (['--devices', '0', '99'], {}, 'not found'),
    (['--devices', '0', '1'], {}, 'at least 48 GiB'),
])
def test_visibility_capacity_and_unknown_id_errors(extra, env, reason):
    with pytest.raises(ValueError, match=reason):
        devices.resolve_devices(arguments(*extra), inventory=inventory(), environ=env)


@pytest.mark.parametrize('bad', [
    devices.NPUDevice(3, 3, 0, 'Warning', 3414, 65536, False),
    devices.NPUDevice(3, 3, 0, 'OK', 3414, 65536, None),
    devices.NPUDevice(3, 3, 0, 'OK', 18000, 65536, False),
])
def test_auto_selection_never_shrinks_count_or_guesses_unknown_process_state(bad):
    args = arguments()
    with pytest.raises(ValueError, match='only 1 visible'):
        devices.resolve_devices(args, inventory=[inventory()[0], bad], environ={})
    assert args.device_count == 2
    assert args.devices is None


def test_npu_smi_parser_maps_runtime_ids_and_ignores_mcu_rows():
    mapping = '''NPU ID  Chip ID  Chip Logic ID  Chip Name
    4  0  2  Ascend 910B3
    4  1  -  Mcu
    8  0  7  Ascend 910B3
    8  1  -  Mcu
    '''
    status = '''
| NPU   Name | Health | Power(W) Temp(C) Hugepages-Usage(page) |
| Chip | Bus-Id | AICore(%) Memory-Usage(MB) HBM-Usage(MB) |
| 4 910B3 | OK | 89.2 36 0 / 0 |
| 0 | 0000:C1:00.0 | 0 0 / 0 3425 / 65536 |
| 8 910B3 | OK | 91.0 36 0 / 0 |
| 0 | 0000:02:00.0 | 0 0 / 0 8610/ 65536 |
| NPU Chip | Process id | Process name | Process memory(MB) |
| No running processes found in NPU 4 |
| 8 0 | 1254597 | python | 5098 |
'''
    result = devices.parse_npu_inventory(mapping, status)
    assert result == [devices.NPUDevice(2, 4, 0, 'OK', 3425, 65536, False),
                      devices.NPUDevice(7, 8, 0, 'OK', 8610, 65536, True)]
    assert result[0].free_mib == 62111
    with pytest.raises(ValueError, match='Incomplete'):
        devices.parse_npu_inventory(mapping, status.replace('3425 / 65536', 'unknown'))
    with pytest.raises(ValueError, match='Invalid HBM'):
        devices.parse_npu_inventory(mapping, status.replace('3425 / 65536', '70000 / 65536'))
    with pytest.raises(ValueError, match='No runtime'):
        devices.parse_npu_inventory('unrecognized topology', status)


def test_npu_smi_unknown_process_status_is_not_considered_idle():
    result = devices.parse_npu_inventory('0 0 0 Ascend 910B3',
                                        '| 0 910B3 | OK | 90 36 0 / 0 |\n'
                                        '| 0 | 0000:C1:00.0 | 0 0 / 0 3425 / 65536 |')
    assert result[0].has_processes is None


def test_missing_npu_smi_has_actionable_error(monkeypatch):
    def missing(*args, **kwargs):
        raise FileNotFoundError('npu-smi')
    monkeypatch.setattr(devices.subprocess, 'run', missing)
    with pytest.raises(RuntimeError, match='Ascend driver, device access'):
        devices.discover_npus()


def test_main_rejects_insufficient_devices_before_loading_recordings_or_models(monkeypatch):
    args = arguments()
    monkeypatch.setattr(probe, 'parse_args', lambda: args)
    monkeypatch.setattr(devices, 'discover_npus', lambda: inventory()[:1])
    monkeypatch.delenv('ASCEND_RT_VISIBLE_DEVICES', raising=False)
    def forbidden(*args, **kwargs):
        pytest.fail('Must select enough devices before any dataset I/O or model loading')
    monkeypatch.setattr(probe, 'load_recording', forbidden)
    monkeypatch.setattr(probe, 'WorkerPool', forbidden)
    with pytest.raises(SystemExit, match='only 1 visible'):
        probe.main()
