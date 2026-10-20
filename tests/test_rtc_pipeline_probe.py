"""Controller invariants for the experimental multi-inflight RTC protocol."""
import numpy as np
import pytest
import heapq
from types import SimpleNamespace

from evaluation.piperpika import rtc_pipeline_probe as probe
from evaluation.piperpika.rtc_pipeline_probe import PlanLedger, Timing, observed_snapshot


def initial_plan():
    return np.repeat(np.arange(128, dtype=np.float32)[:, None], 14, axis=1)


def test_late_binding_can_cover_delay_longer_than_receding_interval():
    timing = Timing().validate()
    assert timing.period == 3.2
    assert timing.horizon/timing.action_hz == 12.8
    ledger = PlanLedger(initial_plan(), timing)
    binding = ledger.bind(32, 55)
    assert binding['overlap'] == 96
    assert binding['frozen'] > timing.receding_actions
    np.testing.assert_array_equal(binding['reference'][:96], initial_plan()[32:])
    with pytest.raises(RuntimeError, match='exceeds overlap'):
        ledger.bind(32, 97)


def test_absolute_time_commit_preserves_all_issued_interpolation_anchors():
    timing = Timing()
    ledger = PlanLedger(initial_plan(), timing)
    issued = np.stack([ledger.issue(tick) for tick in range(250)])
    binding = ledger.bind(32, 55)
    candidate = np.full((128, 14), -10, np.float32)
    candidate[:55] = binding['reference'][:55]
    ledger.commit(32, candidate, binding, tick=250)
    np.testing.assert_array_equal(ledger.plan[:87], initial_plan()[:87])
    np.testing.assert_array_equal(ledger.plan[87:160], candidate[55:])
    np.testing.assert_array_equal(np.asarray(ledger.issued), issued)
    # Replaying the final plan still reproduces every command already issued.
    replay = PlanLedger(ledger.plan[:128], timing)
    np.testing.assert_array_equal(np.stack([replay.issue(t) for t in range(250)]), issued)


def test_protection_includes_right_anchor_used_between_model_ticks():
    ledger = PlanLedger(initial_plan(), Timing())
    for tick in range(5):
        ledger.issue(tick)
    binding = ledger.bind(0, 2)
    candidate = binding['reference'].copy()
    candidate[2:] = -1
    with pytest.raises(RuntimeError, match='interpolation anchor'):
        ledger.commit(0, candidate, binding, tick=5)


def test_rejects_changed_committed_prefix_and_stale_plan_binding():
    ledger = PlanLedger(initial_plan(), Timing())
    binding = ledger.bind(32, 55)
    changed = binding['reference'].copy()
    changed[0, 0] += 1
    with pytest.raises(RuntimeError, match='preserve the committed prefix'):
        ledger.commit(32, changed, binding, tick=0)
    ledger.commit(32, binding['reference'], binding, tick=0)
    with pytest.raises(RuntimeError, match='Plan changed'):
        ledger.commit(32, binding['reference'], binding, tick=0)


def test_grippers_are_held_and_joint_targets_are_interpolated():
    ledger = PlanLedger(initial_plan(), Timing())
    first, second = ledger.issue(0), ledger.issue(1)
    np.testing.assert_array_equal(first, np.zeros(14))
    np.testing.assert_allclose(second[:12], 1/3)
    np.testing.assert_array_equal(second[12:], np.zeros(2))


def test_deadline_budget_preserves_extra_interpolation_anchor():
    timing = Timing()
    frozen = timing.protected_steps(elapsed=2.5, decode_budget=2.4, margin=.05)
    assert frozen == 51
    assert (frozen-1)/timing.action_hz >= 2.5+2.4+.05


def test_fractional_replanning_preserves_trained_history_timebase():
    timing = Timing(receding_actions=27).validate()
    assert timing.period == 2.7
    assert timing.period_ticks == 81
    assert timing.history_ticks == 96
    issued = np.repeat(np.arange(243, dtype=np.float32)[:, None], 14, axis=1)
    snapshot = observed_snapshot(81, issued, dict(enumerate(range(244))), timing)
    assert snapshot['origin'] == 17
    assert snapshot['anchor_ticks'] == [51, 147, 243]
    assert snapshot['initial_image'] == 51
    for index, chunk in enumerate(snapshot['history']):
        first = 51+96*index
        np.testing.assert_array_equal(chunk['actions'], issued[first:first+96:3])
        assert chunk['images'] == list(range(first+4, first+97, 4))
        assert len(chunk['actions']) == 32
        assert len(chunk['images']) == 24
    assert snapshot['history'][-1]['images'][-1] == 243


def test_early_snapshot_uses_current_observation_and_no_invented_past():
    timing = Timing(receding_actions=27)
    issued = np.zeros((81, 14), np.float32)
    snapshot = observed_snapshot(27, issued, {81: 'current RGB'}, timing)
    assert snapshot['history'] == []
    assert snapshot['origin'] == snapshot['start'] == 27
    assert snapshot['initial_image'] == 'current RGB'
    with pytest.raises(ValueError, match='already-issued'):
        observed_snapshot(27, issued[:-1], {81: 'RGB'}, timing)


def test_fractional_replanning_requires_explicit_snapshot_mode(tmp_path):
    argv = ['--model', '/model', '--devices', '0', '3', '--output', str(tmp_path),
            '--receding-actions', '27']
    with pytest.raises(SystemExit):
        probe.parse_args(argv)
    assert probe.parse_args(argv+['--history-mode', 'snapshot']).receding_actions == 27


def test_shared_recording_preserves_rgb_and_is_read_only(tmp_path):
    images = {tick:{camera:np.full((2, 3, 3), tick+index, np.uint8)
                    for index, camera in enumerate(probe.CAMERAS)} for tick in range(5)}
    path = tmp_path/'rgb.npy'
    indices = probe.share_recording(images, path)
    assert indices == dict(enumerate(range(5)))
    recording = np.load(path, mmap_mode='r')
    for tick in images:
        actual = probe.recording_image(indices[tick], recording)
        for camera in probe.CAMERAS:
            np.testing.assert_array_equal(actual[camera], images[tick][camera])
            assert not actual[camera].flags.writeable
    with pytest.raises(ValueError, match='outside'):
        probe.recording_image(5, recording)
    with pytest.raises(ValueError, match='outside'):
        probe.recording_image(0, None)


class VirtualClock:
    now = 0.

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class SimulatedPool:
    """Timed worker protocol, including history freshness; no model inference."""

    def __init__(self, clock, devices, action_seconds):
        self.clock = clock
        self.workers = {device: {'history_actions': 0} for device in devices}
        self.action_seconds = action_seconds
        self.events = []
        self.serial = 0
        self.origin = None
        self.world_job = {}

    def event(self, after, **message):
        self.serial += 1
        ready = self.clock.now+after
        heapq.heappush(self.events, (ready, self.serial, dict(message, received=ready)))

    def send(self, device, message):
        kind = message['kind']
        if kind in ('WORLD', 'HISTORY'):
            chunks = message['history']
            snapshot = message.get('snapshot')
            if snapshot is not None:
                chunks = snapshot['history']
                assert snapshot['anchor_ticks'][-1] == snapshot['start']*3
                assert max(snapshot['anchor_ticks']) <= (self.clock.now-self.origin)*30+1e-5
            for chunk in chunks:
                assert self.origin is not None
                # A preloaded recording must not leak future feedback to a worker.
                assert max(chunk['images']) <= (self.clock.now-self.origin)*30+1e-5
            history = (snapshot['start'] if snapshot is not None else
                       self.workers[device]['history_actions']+32*len(chunks))
            seconds = .9*len(chunks)+(.2 if snapshot is not None else 0.)
            if kind == 'HISTORY':
                self.event(seconds, kind='HISTORY_READY', device=device,
                           history_actions=history, history_seconds=seconds)
            else:
                self.world_job[device] = message['job']
                self.event(seconds+.5, kind='WORLD_READY', device=device, job=message['job'],
                           history_actions=history, history_seconds=seconds, video_seconds=.5,
                           world_ready=self.clock.now+seconds+.5)
        elif kind == 'ACTION':
            assert self.world_job[device] == message['job']
            binding = message['binding']
            action = binding['reference'].copy()
            action[binding['frozen']:] = message['job']+.25
            self.event(self.action_seconds, kind='ACTION_READY', job=message['job'],
                       device=device, action_seconds=self.action_seconds,
                       actions=action, peak_memory_bytes=0)
        else:
            raise AssertionError(kind)

    def receive(self, timeout=0):
        if self.events and self.events[0][0] <= self.clock.now+1e-10:
            return heapq.heappop(self.events)[2]

    def collect(self, kind, count):
        results = []
        for _ in range(count):
            ready, _, result = heapq.heappop(self.events)
            self.clock.now = ready
            assert result['kind'] == kind
            results.append(result)
        if kind == 'ACTION_READY':
            self.origin = self.clock.now
        return results


@pytest.mark.parametrize('devices', [[0, 3], [0, 3, 4], [0, 3, 4, 6]])
def test_fixed_handoff_protocol_with_two_to_four_replicas(monkeypatch, tmp_path, devices):
    clock = VirtualClock()
    monkeypatch.setattr(probe, 'time', clock)
    args = SimpleNamespace(devices=devices, output=tmp_path, ticks=600, warmup_rounds=2,
        latency_multiplier=1.2, deadline_margin=.05, handoff_tolerance=.25, handoff_latency=5.)
    pool = SimulatedPool(clock, devices, action_seconds=2.7)
    report = probe.run_pipeline(pool, args, {tick: tick for tick in range(0, 604, 4)},
                                Timing(), [dict(round=1, action_seconds=2.7)])
    assert report['success'], report['error']
    assert report['controller_ticks'] == 600
    assert len(report['measurements']) == 4
    np.testing.assert_allclose(report['handoff_intervals_seconds'], 3.2, atol=.003)
    assert all(row['result_hold_seconds'] > 0 for row in report['measurements'])
    assert report['min_remaining_coverage_seconds'] > 0


@pytest.mark.parametrize('devices', [[0, 3, 4], [0, 3, 4, 6]])
def test_more_video_replicas_do_not_hide_missed_serial_action_deadline(monkeypatch, tmp_path, devices):
    clock = VirtualClock()
    monkeypatch.setattr(probe, 'time', clock)
    args = SimpleNamespace(devices=devices, output=tmp_path, ticks=600, warmup_rounds=2,
        latency_multiplier=1.2, deadline_margin=.05, handoff_tolerance=.25, handoff_latency=5.)
    pool = SimulatedPool(clock, args.devices, action_seconds=3.5)
    report = probe.run_pipeline(pool, args, {tick: tick for tick in range(0, 604, 4)},
                                Timing(), [dict(round=1, action_seconds=2.7)])
    assert not report['success']
    assert 'Missed fixed handoff deadline' in report['error']


def test_four_distinct_devices_are_accepted_but_duplicates_rejected(tmp_path):
    argv=['--model','/model','--output',str(tmp_path),'--devices','0','3','4','6']
    assert probe.parse_args(argv).devices==[0,3,4,6]
    with pytest.raises(SystemExit):
        probe.parse_args(argv[:-1]+['4'])


@pytest.mark.parametrize('devices,passed',[([0,3],False),([0,3,4,6],True)])
def test_four_replicas_allow_longer_world_latency_without_changing_action_dependencies(
        monkeypatch,tmp_path,devices,passed):
    clock=VirtualClock()
    monkeypatch.setattr(probe,'time',clock)
    args=SimpleNamespace(devices=devices,output=tmp_path,ticks=600,warmup_rounds=2,
        history_mode='snapshot',snapshot_history_intervals=2,
        latency_multiplier=1.2,deadline_margin=.05,handoff_tolerance=1/30,handoff_latency=6.)
    pool=SimulatedPool(clock,devices,action_seconds=2.55)
    report=probe.run_pipeline(pool,args,dict(enumerate(range(604))),Timing(receding_actions=26),
                              [dict(round=1,action_seconds=2.55)])
    assert report['success']==passed,report['error']
    if passed:
        assert report['controller_ticks']==600
        assert all(report['invariant_checks'].values())
        np.testing.assert_allclose(report['handoff_intervals_seconds'],2.6,atol=.003)
    else:
        assert 'No replica available' in report['error']


@pytest.mark.parametrize('receding,passed', [(28, True), (27, True), (26, False)])
def test_snapshot_scheduler_admits_only_feasible_serial_action_periods(monkeypatch, tmp_path, receding, passed):
    clock = VirtualClock()
    monkeypatch.setattr(probe, 'time', clock)
    args = SimpleNamespace(devices=[0, 3], output=tmp_path, ticks=600, warmup_rounds=2,
        history_mode='snapshot', snapshot_history_intervals=2,
        latency_multiplier=1.2, deadline_margin=.05, handoff_tolerance=.25, handoff_latency=5.3)
    pool = SimulatedPool(clock, args.devices, action_seconds=2.65)
    report = probe.run_pipeline(pool, args, dict(enumerate(range(604))),
        Timing(receding_actions=receding), [dict(round=1, action_seconds=2.65)])
    assert report['success'] == passed, report['error']
    if passed:
        np.testing.assert_allclose(report['handoff_intervals_seconds'], receding/10, atol=.003)
        assert report['controller_ticks'] == 600
        assert report['invariant_checks']['no_action_queue_underflow']
    else:
        assert ('Missed fixed handoff deadline' in report['error']
                or 'No replica available' in report['error'])


def test_controller_missed_period_fails_even_with_action_coverage(monkeypatch, tmp_path):
    clock = VirtualClock()
    monkeypatch.setattr(probe, 'time', clock)
    class StalledPool(SimulatedPool):
        stalled = False
        def receive(self, timeout=0):
            if self.origin is not None and clock.now-self.origin >= 1 and not self.stalled:
                self.stalled = True
                clock.now += .07
            return super().receive(timeout)
    args = SimpleNamespace(devices=[0,3], output=tmp_path, ticks=600, warmup_rounds=2,
        latency_multiplier=1.2, deadline_margin=.05, handoff_tolerance=.25, handoff_latency=5.)
    pool = StalledPool(clock, args.devices, action_seconds=2.6)
    report = probe.run_pipeline(pool, args, dict(enumerate(range(604))),
        Timing(), [dict(round=1, action_seconds=2.6)])
    assert not report['success']
    assert 'missed a full control period' in report['error']
    assert report['min_remaining_coverage_seconds'] > 0
    assert not report['invariant_checks']['controller_within_one_period']


def test_slow_artifact_writes_are_outside_controller_loop(monkeypatch, tmp_path):
    clock = VirtualClock()
    monkeypatch.setattr(probe, 'time', clock)
    original_save = np.save
    def slow_save(*args, **kwargs):
        clock.now += .1  # A shared-filesystem pause longer than three ticks.
        return original_save(*args, **kwargs)
    monkeypatch.setattr(probe.np, 'save', slow_save)
    args = SimpleNamespace(devices=[0,3], output=tmp_path, ticks=600, warmup_rounds=2,
        latency_multiplier=1.2, deadline_margin=.05, handoff_tolerance=.25, handoff_latency=5.)
    pool = SimulatedPool(clock, args.devices, action_seconds=2.6)
    report = probe.run_pipeline(pool, args, dict(enumerate(range(604))),
        Timing(), [dict(round=1, action_seconds=2.6)])
    assert report['success'], report['error']
    assert report['controller_ticks'] == 600
    assert report['invariant_checks']['controller_within_one_period']
    assert (tmp_path/'bootstrap_actions.npy').is_file()
    assert len(list(tmp_path.glob('job_*_actions.npy'))) == len(report['commits'])
