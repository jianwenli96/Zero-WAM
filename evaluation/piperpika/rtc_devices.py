"""Read-only NPU discovery for independent RTC replicas; no torch initialization."""
from dataclasses import asdict, dataclass
import os
import re
import subprocess


MIN_REPLICA_FREE_MIB = 48 * 1024


@dataclass(frozen=True)
class NPUDevice:
    device: int
    card: int
    chip: int
    health: str
    used_mib: int
    total_mib: int
    has_processes: bool | None

    @property
    def free_mib(self):
        return self.total_mib - self.used_mib


def parse_npu_inventory(mapping, status):
    """Join npu-smi's card/chip rows to runtime device IDs (not board IDs)."""
    runtime_ids = {}
    for line in mapping.splitlines():
        match = re.fullmatch(r'\s*(\d+)\s+(\d+)\s+(\d+)\s+.+', line)
        if match:
            card, chip, device = map(int, match.groups())
            if (card, chip) in runtime_ids or device in runtime_ids.values():
                raise ValueError('Ambiguous runtime device IDs in npu-smi info -m')
            runtime_ids[card, chip] = device
    if not runtime_ids:
        raise ValueError('No runtime device IDs found in npu-smi info -m')

    memory, busy, process_cards = {}, set(), set()
    card = health = None
    processes = False
    for line in status.splitlines():
        if 'Process id' in line:
            processes = True
            continue
        cells = [cell.strip() for cell in line.split('|')[1:-1]]
        if not cells:
            continue
        if processes:
            empty = re.search(r'No running processes found in NPU\s+(\d+)', line)
            if empty:
                process_cards.add(int(empty[1]))
            identity = cells[0].split()
            if len(cells) >= 2 and len(identity) == 2 and all(v.isdigit() for v in identity) and cells[1].isdigit():
                pair = tuple(map(int, identity))
                busy.add(pair)
                process_cards.add(pair[0])
            continue
        identity = cells[0].split()
        if len(identity) == 2 and identity[0].isdigit() and len(cells) == 3:
            card, health = int(identity[0]), cells[1]
        elif card is not None and cells[0].isdigit() and len(cells) == 3:
            hbm = re.search(r'(\d+)\s*/\s*(\d+)\s*$', cells[-1])
            if hbm:
                used, total = map(int, hbm.groups())
                if not 0 <= used <= total or total == 0:
                    raise ValueError('Invalid HBM capacity in npu-smi info')
                memory[card, int(cells[0])] = health, used, total

    result = []
    for (card, chip), device in runtime_ids.items():
        if (card, chip) in memory:
            health, used, total = memory[card, chip]
            occupied = (card, chip) in busy if card in process_cards else None
            result.append(NPUDevice(device, card, chip, health, used, total, occupied))
    if len(result) != len(runtime_ids):
        raise ValueError('Incomplete NPU memory status; cannot safely select RTC replicas')
    return sorted(result, key=lambda item: item.device)


def discover_npus():
    def query(*arguments):
        try:
            return subprocess.run(['npu-smi', 'info', *arguments], capture_output=True,
                                  text=True, check=True, timeout=15,
                                  env={**os.environ, 'LC_ALL': 'C'}).stdout
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError('Cannot query npu-smi; check the Ascend driver, device access '
                               'and PATH before starting RTC replicas') from error

    return parse_npu_inventory(query('-m'), query())


def visible_device_ids(environ):
    """None means unrestricted; an empty mask means no devices are visible."""
    if 'ASCEND_RT_VISIBLE_DEVICES' not in environ:
        return None
    value = environ['ASCEND_RT_VISIBLE_DEVICES'].strip()
    if value in ('', '-1'):
        return []
    fields = [field.strip() for field in value.split(',')]
    if not all(field.isdecimal() for field in fields):
        raise ValueError('ASCEND_RT_VISIBLE_DEVICES must contain comma-separated nonnegative device IDs')
    devices = list(map(int, fields))
    if len(set(devices)) != len(devices):
        raise ValueError('ASCEND_RT_VISIBLE_DEVICES contains duplicate device IDs')
    return devices


def resolve_devices(args, *, inventory=None, environ=None):
    """Select replicas before recording I/O or model loading; never shrink count.

    Automatic selection skips occupied NPUs. Explicit --devices preserves order
    and permits sharing only when the existing 48-GiB free-memory check passes.
    Each worker rechecks free memory because this discovery is not a reservation.
    """
    mask = visible_device_ids(os.environ if environ is None else environ)
    if mask is not None and len(mask) < args.device_count:
        raise ValueError(f'device_count={args.device_count} requires more than the '
                         f'{len(mask)} visible NPUs in ASCEND_RT_VISIBLE_DEVICES={mask}')
    if args.devices is not None and mask is not None and not set(args.devices) <= set(mask):
        raise ValueError(f'--devices {args.devices} exceeds ASCEND_RT_VISIBLE_DEVICES={mask}')
    inventory = discover_npus() if inventory is None else inventory
    known = {item.device: item for item in inventory}
    if mask is not None and not set(mask) <= set(known):
        raise ValueError(f'Visible device IDs {sorted(set(mask)-set(known))} were not found by npu-smi')
    visible = list(known) if mask is None else mask
    if args.devices is None:
        eligible = [known[device] for device in visible if known[device].health == 'OK'
                    and known[device].has_processes is False
                    and known[device].free_mib >= MIN_REPLICA_FREE_MIB]
        if mask is None:
            eligible.sort(key=lambda item: (-item.free_mib, item.device))
        if len(eligible) < args.device_count:
            raise ValueError(f'device_count={args.device_count}, but only {len(eligible)} visible '
                             f'healthy, idle NPUs have at least 48 GiB free; visible IDs={visible}. '
                             'No replicas were started; free devices or select an eligible --devices list')
        devices = [item.device for item in eligible[:args.device_count]]
        source = 'visible_mask' if mask is not None else 'npu-smi_idle'
    else:
        devices = args.devices
        source = 'explicit_devices'
        for device in devices:
            if device not in known:
                raise ValueError(f'NPU {device} was not found by npu-smi')
            item = known[device]
            if item.health != 'OK' or item.free_mib < MIN_REPLICA_FREE_MIB:
                raise ValueError(f'NPU {device}: health={item.health}, free={item.free_mib/1024:.2f} GiB; '
                                 'each RTC replica requires a healthy NPU with at least 48 GiB free')
    args.devices = devices
    return dict(source=source, visible_devices=visible,
                selected=[dict(asdict(known[device]), free_mib=known[device].free_mib) for device in devices])
