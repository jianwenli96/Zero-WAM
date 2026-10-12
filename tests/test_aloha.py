import json
import numpy as np
import pytest
import yaml
from wan_va.dataset.aloha import CHANNELS, action_metadata, segments, model_quantiles
from wan_va.dataset.lerobot_action import LeRobotActionProcessor
from evaluation.aloha.open_loop import flatten_first_chunk, metrics


def actions(n=100):
    a = np.random.default_rng(4).normal(size=(n,28)).astype(np.float32)
    for i in (3,10): a[:,i:i+4] /= np.linalg.norm(a[:,i:i+4],axis=1,keepdims=True)
    return a


def test_absolute_action_round_trip_and_padding(tmp_path):
    a = actions()
    transform, stats = action_metadata(a)
    meta = tmp_path / 'meta'; meta.mkdir()
    (meta/'action_transform.yaml').write_text(yaml.safe_dump(transform))
    (meta/'action_stats.json').write_text(json.dumps(stats))
    processor = LeRobotActionProcessor(tmp_path)
    normalized, mask, stride = processor.process({'action': a}, 0, list(range(0,65,4)))
    assert stride == 16
    assert mask[:,CHANNELS].all()
    assert not mask[:,26:28].any()
    np.testing.assert_array_equal(normalized[:16],0)
    qs = model_quantiles(meta/'action_stats.json')
    lo, hi = [np.asarray(qs[k])[CHANNELS] for k in ('q01','q99')]
    restored = (normalized[16:,CHANNELS]+1)/2*(hi-lo+1e-6)+lo
    np.testing.assert_allclose(restored, a[:64], atol=1e-6)
    # Changing the first action must not rebase later absolute actions.
    b = a.copy(); b[0,0] += .01
    changed, _, _ = processor.process({'action': b}, 0, list(range(0,65,4)))
    np.testing.assert_array_equal(changed[17:], normalized[17:])


def test_segment_alignment_and_tail():
    for length in (17, 33, 513, 514, 2775):
        last = 0
        for start,end,ids in segments(length):
            assert start == last
            assert end <= length and ids[-1] == end-1
            assert (len(ids)-1)%4 == 0
            assert np.diff(ids).tolist() == [4]*(len(ids)-1)
            assert ((len(ids)-1)//4)*16 == end-1-start
            last = end-1
        assert length-last <= 17
    with pytest.raises(ValueError): list(segments(100, sampled_frames=128))


def test_first_block_is_never_scored_and_axes_are_chronological():
    a = np.arange(28*3*16).reshape(28,3,16)
    flat = flatten_first_chunk(a)
    np.testing.assert_array_equal(flat[0], a[:,1,0])
    np.testing.assert_array_equal(flat[16], a[:,2,0])
    with pytest.raises(ValueError): flatten_first_chunk(np.zeros((16,2,16)))


def test_rotation_metrics_ignore_quaternion_sign():
    a = actions(20); b = a.copy()
    b[:,3:7] *= -1; b[:,10:14] *= -1
    m = metrics(b,a,np.zeros(28),np.ones(28))
    assert m['left_rotation_deg'] < .1
    assert m['right_rotation_deg'] < .1
    assert m['joint_mae'] == 0


def test_quaternion_sign_canonicalization_preserves_rotations_and_other_channels():
    from wan_va.dataset.aloha import canonical_actions
    # Smooth trajectory crossing qw=0: forcing qw>=0 would introduce a jump.
    angle = np.linspace(2.8, 3.5, 60)
    a = actions(60)
    for k in (3,10):
        a[:,k:k+4] = np.stack([np.sin(angle/2), angle*0, angle*0, np.cos(angle/2)],axis=-1)
    flipped = a.copy()
    flipped[::2,3:7] *= -1; flipped[1::3,10:14] *= -1
    out = canonical_actions(flipped)
    np.testing.assert_allclose(out, canonical_actions(a), atol=2e-7)
    for k in (3,10):
        np.testing.assert_allclose(np.abs((out[:,k:k+4]*a[:,k:k+4]).sum(-1)),1,atol=2e-7)
        assert ((out[1:,k:k+4]*out[:-1,k:k+4]).sum(-1) >= 0).all()
    other = [i for i in range(28) if i not in list(range(3,7))+list(range(10,14))]
    np.testing.assert_array_equal(out[:,other], a[:,other])


def test_antipodal_prediction_not_penalized_by_summary_metric():
    a = actions(20); b = a.copy(); b[:,3:7] *= -1; b[:,10:14] *= -1
    result = metrics(b,a,np.zeros(28),np.ones(28))
    assert result['normalized_mae'] == 0
    assert result['raw_normalized_mae'] > 0


def test_stale_encoding_cache_is_rejected(tmp_path):
    from wan_va.dataset.aloha_integrity import checked_cache, job_identity
    video = tmp_path/'source.mp4'; video.write_bytes(b'video')
    output = tmp_path/'latent.pth'
    job = {'video':str(video), 'output':str(output), 'prompt':'task'}
    assert not checked_cache(job, {})
    output.write_bytes(b'latent')
    assert not checked_cache(job,{})  # resume after tensor save but before commit record
    sidecar = output.with_suffix('.provenance.json')
    sidecar.write_text(json.dumps({'identity':job_identity(job,{}),'size':output.stat().st_size}))
    assert checked_cache(job,{})
    with pytest.raises(ValueError,match='Stale'): checked_cache(dict(job,prompt='changed'),{})
    with pytest.raises(ValueError,match='Stale'): checked_cache(job,{'vae':'other'})
    video.write_bytes(b'changed video')
    with pytest.raises(ValueError,match='Stale'): checked_cache(job,{})


def test_checkpoint_statistics_binding(tmp_path):
    from wan_va.dataset.aloha import capture_action_contract, verify_checkpoint_contract
    root=tmp_path/'data'; meta=root/'train/breakfast/meta'; meta.mkdir(parents=True)
    stats=meta/'action_stats.json'; stats.write_text(json.dumps({'q01':[1]}))
    (root/'preparation.json').write_text(json.dumps({'seed':42}))
    model=tmp_path/'model'; model.mkdir()
    with pytest.raises(ValueError,match='no ALOHA'): verify_checkpoint_contract(model,root)
    assert verify_checkpoint_contract(model,root,True)[1] is False
    snapshot=capture_action_contract(root)
    (model/'aloha_action_contract.json').write_text(json.dumps(snapshot))
    assert verify_checkpoint_contract(model,root)[1] is True
    stats.write_text(json.dumps({'q01':[2]}))
    with pytest.raises(ValueError,match='differ'): verify_checkpoint_contract(model,root)
