import numpy as np
import pytest

from evaluation.piperpika.rtc_quality import (
    PROFILES, incremental_observation, native_sample, rollout, scores,
)
from evaluation.piperpika.rtc_pipeline_probe import Timing


def test_normalized_mae_uses_training_span_and_original_channel_units():
    target = np.zeros((2,14))
    prediction = np.full((2,14),2.)
    result = scores(prediction,target,np.full(14,4.))
    assert result['normalized_mae'] == .5
    assert result['joint_mae'] == result['gripper_mae'] == 2.
    assert scores(np.empty((0,14)),np.empty((0,14)),np.ones(14)) is None
    with pytest.raises(ValueError):
        scores(prediction,target,np.zeros(14))


def test_native_interpolation_keeps_grippers_at_left_target():
    actions = np.zeros((2,14),np.float32)
    actions[1,:12] = 3
    actions[1,12:] = 9
    np.testing.assert_array_equal(native_sample(actions,1),np.r_[np.ones(12),[0,0]])
    np.testing.assert_array_equal(native_sample(actions,3),actions[1])


def test_virtual_rollout_uses_only_issued_history_but_latest_binding():
    initial = np.repeat(np.arange(128,dtype=np.float32)[:,None],14,axis=1)
    seen = []
    def infer(snapshot,binding,seed,*,rtc):
        assert snapshot['anchor_ticks'][-1] == snapshot['start']*3
        if snapshot['start'] == 56:
            # This observation predates the first commit, so its executed
            # history is bootstrap; the later binding sees that commit.
            np.testing.assert_array_equal(snapshot['history'][0]['actions'],initial[24:56])
            assert binding['plan_version'] == 1
            assert binding['reference'][28,0] == 1
        number = len(seen)+1
        seen.append(snapshot['start'])
        value = np.full((128,14),number,np.float32)
        return dict(actions=value,ordinary_same_context=value.copy())
    result = rollout(dict(enumerate(range(600))),600,42,initial,infer,rtc=True)
    assert seen == [28,56,84,112,140]
    assert [c['controller_tick'] for c in result['commits']] == [246,330,414,498,582]
    np.testing.assert_array_equal(result['issued'][:246],np.stack([
        native_sample(initial,t) for t in range(246)]))
    assert result['proposals'][0]['binding']['frozen'] == 56
    assert result['proposals'][1]['binding']['frozen'] == 61


def test_ordinary_rollout_preserves_only_used_interpolation_anchors():
    initial = np.zeros((128,14),np.float32)
    bindings = []
    def infer(snapshot,binding,seed,*,rtc):
        assert not rtc
        bindings.append(binding['frozen'])
        return dict(actions=np.ones((128,14),np.float32))
    result = rollout(dict(enumerate(range(300))),300,42,initial,infer,rtc=False)
    assert bindings == [55]
    np.testing.assert_array_equal(result['issued'][246],np.zeros(14))
    np.testing.assert_allclose(result['issued'][247,:12],1/3)
    np.testing.assert_array_equal(result['issued'][247,12:],np.zeros(2))


def test_episode_ending_before_first_commit_has_no_rtc_correction():
    def forbidden(*args,**kwargs):
        raise AssertionError('No candidate can affect this short recording')
    result = rollout(dict(enumerate(range(240))),240,42,np.zeros((128,14)),forbidden,rtc=True)
    assert result['commits'] == result['proposals'] == []


@pytest.mark.parametrize('name,starts,commits,reservations',[
    ('s32_dual',[32,64,96,128],[228,324,420,516],[46,46,49,48]),
    ('s64_single',[64,128],[342,534],[58,61]),
])
def test_incremental_profiles_capture_executed_requests_and_latest_plan(name,starts,commits,reservations):
    initial=np.zeros((128,14),np.float32)
    seen=[]
    profile=PROFILES[name]
    def infer(capture,binding,seed,*,rtc):
        number=len(seen)+1
        chunk,=capture['history']
        start=capture['start']
        assert chunk['start']==start-profile.receding_actions
        assert chunk['actions'].shape==(profile.receding_actions,14)
        assert len(chunk['images'])==profile.receding_actions*3//4
        assert max(chunk['images'])==start*3
        assert binding['plan_version']==number-1
        assert binding['frozen']==profile.reservation(number)
        # No generated future is cached as history: the second launch is
        # before the first handoff for s32, and after it for s64.
        if number==2:
            if name=='s32_dual':
                np.testing.assert_array_equal(chunk['actions'],0)
            else:
                assert chunk['actions'][-1,0]==1
        seen.append(start)
        value=np.full((128,14),number,np.float32)
        return dict(actions=value,ordinary_same_context=value.copy())
    result=rollout(dict(enumerate(range(600))),600,42,initial,infer,rtc=True,profile=profile)
    assert seen==starts
    assert [c['controller_tick'] for c in result['commits']]==commits
    assert [r['binding']['frozen'] for r in result['proposals']]==reservations
    np.testing.assert_array_equal(result['issued'][:profile.first_editable_tick-2],0)


@pytest.mark.parametrize('name',['s32_dual','s64_single'])
def test_ordinary_profile_retains_used_right_anchor_at_commit(name):
    profile=PROFILES[name]
    def infer(capture,binding,seed,*,rtc):
        assert not rtc and binding['frozen']==profile.ordinary_prefix
        return dict(actions=np.ones((128,14),np.float32))
    result=rollout(dict(enumerate(range(600))),600,42,np.zeros((128,14)),infer,
                   rtc=False,profile=profile)
    handoff=profile.first_handoff_tick
    np.testing.assert_array_equal(result['issued'][handoff],0)
    np.testing.assert_allclose(result['issued'][handoff+1,:12],1/3)
    np.testing.assert_array_equal(result['issued'][handoff+1,12:],0)


def test_incremental_history_rejects_partial_or_future_execution():
    images=dict(enumerate(range(300)))
    with pytest.raises(ValueError):
        incremental_observation(28,[np.zeros(14)]*84,images,Timing(receding_actions=28))
    with pytest.raises(ValueError):
        incremental_observation(32,[np.zeros(14)]*97,images,Timing(receding_actions=32))


@pytest.mark.parametrize('name',['s32_dual','s64_single'])
def test_independent_audit_reconstructs_both_policies_and_rejects_changed_prefix(name):
    from evaluation.piperpika.rtc_quality_summary import replay
    profile=PROFILES[name]
    bootstrap=np.zeros((128,14),np.float32)
    def infer(capture,binding,seed,*,rtc):
        actions=np.full((128,14),capture['start']/32,np.float32)
        if rtc:
            actions[:binding['frozen']]=binding['reference'][:binding['frozen']]
        return dict(actions=actions,ordinary_same_context=actions.copy())
    guided,ordinary=[rollout(dict(enumerate(range(600))),600,42,bootstrap,infer,
                            rtc=rtc,profile=profile) for rtc in (True,False)]
    data=dict(bootstrap=bootstrap,rtc_issued=guided['issued'],ordinary_issued=ordinary['issued'],
              gt=np.zeros((600,14)),valid=np.ones(600,bool),span=np.ones(14))
    rows=[]
    for row in guided['proposals']:
        binding=row['binding']
        rows.append(dict(job=row['job'],observation_tick=row['observation_tick'],bind_tick=row['bind_tick'],
            commit_tick=row['commit_tick'],frozen=binding['frozen'],overlap=binding['overlap'],
            plan_version=binding['plan_version']))
        data[f"job{row['job']}_rtc"]=row['actions']
        data[f"job{row['job']}_reference"]=binding['reference']
    for row in ordinary['proposals']:
        data[f"job{row['job']}_ordinary_independent"]=row['actions']
    report=dict(profile=profile.metadata(),proposals=rows,
                **{arm:scores(data[arm+'_issued'],data['gt'],data['span']) for arm in ('ordinary','rtc')})
    replay(data,report,'ordinary')
    replay(data,report,'rtc')
    data['job1_rtc'][0,0]=10
    with pytest.raises(AssertionError):
        replay(data,report,'rtc')
