import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pytest
import yaml
from wan_va.dataset.piperpika import CHANNELS, joint_metadata, model_quantiles, verify_checkpoint_contract
from wan_va.dataset.lerobot_action import LeRobotActionProcessor
from wan_va.utils.finetune import cosine_warmup_ratio


def test_joint_only_roundtrip_and_eef_independence(tmp_path):
    a = np.random.default_rng(3).uniform(-.5,.5,(100,28)).astype(np.float32)
    transform, stats = joint_metadata(a)
    meta = tmp_path/'meta'; meta.mkdir()
    (meta/'action_transform.yaml').write_text(yaml.safe_dump(transform))
    (meta/'action_stats.json').write_text(json.dumps(stats))
    processor = LeRobotActionProcessor(tmp_path)
    x, mask, stride = processor.process({'action':a}, 0, list(range(0,65,4)))
    assert stride == 16
    assert mask[:,CHANNELS].all()
    assert not mask[:,:14].any() and not mask[:,26:28].any()
    assert not x[:,:14].any() and not x[:,26:28].any()
    q = model_quantiles(meta/'action_stats.json')
    lo,hi = [np.array(q[k])[CHANNELS] for k in ('q01','q99')]
    restored = (x[16:,CHANNELS]+1)/2*(hi-lo+1e-6)+lo
    np.testing.assert_allclose(restored,a[:64,14:],atol=1e-6)
    b=a.copy();b[:,:14]=10000
    y,_,_=processor.process({'action':b},0,list(range(0,65,4)))
    np.testing.assert_array_equal(x,y)
    assert 'action.hand.position' not in stats['norm_stats']


def test_cosine_learning_rate():
    assert cosine_warmup_ratio(0,100,1000)==0
    assert cosine_warmup_ratio(100,100,1000)==1
    assert cosine_warmup_ratio(1000,100,1000)==pytest.approx(.1)
    assert cosine_warmup_ratio(2000,100,1000)==pytest.approx(.1)
    assert cosine_warmup_ratio(500,100,1000)>cosine_warmup_ratio(900,100,1000)
    with pytest.raises(ValueError): cosine_warmup_ratio(0,100,50)


def preparation_module():
    scripts=Path(__file__).resolve().parents[1]/'script'
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('prepare_piperpika',scripts/'prepare_piperpika.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_group_split_never_splits_human_pair():
    prep=preparation_module()
    groups=[{'alignment_task_number':i,'human_video_path':f'human/{i}.mp4',
             'robot_episodes':[{'episode_index':2*i},{'episode_index':2*i+1}]} for i in range(60)]
    heldout=prep.split_groups(groups)
    assert len(heldout)==6 and heldout==prep.split_groups(groups)
    train={e['episode_index'] for g in groups if g['alignment_task_number'] not in heldout for e in g['robot_episodes']}
    val={e['episode_index'] for g in groups if g['alignment_task_number'] in heldout for e in g['robot_episodes']}
    assert len(train)==108 and len(val)==12 and not train&val
    groups[1]['human_video_path']=groups[0]['human_video_path']
    with pytest.raises(ValueError): prep.split_groups(groups)


def test_config_keeps_pretrained_dimensions_and_replay_unchanged():
    from wan_va.configs import VA_CONFIGS, TRAIN_DATASET_CONFIGS
    cfg=VA_CONFIGS['piperpika_train']
    assert cfg.action_dim==30 and cfg.used_action_channel_ids==CHANNELS
    assert cfg.learning_rate==2e-6 and cfg.enable_mcp
    assert TRAIN_DATASET_CONFIGS['aloha'].used_action_channel_ids != CHANNELS
    assert len(cfg.inverse_used_action_channel_ids)==30
    assert cfg.inverse_used_action_channel_ids[0]==14


def test_contract_requires_matching_stats(tmp_path):
    root=tmp_path/'data';meta=root/'train/piperpika/meta';meta.mkdir(parents=True)
    (meta/'action_stats.json').write_text('{}');(root/'preparation.json').write_text('{}')
    model=tmp_path/'model';model.mkdir()
    with pytest.raises(ValueError):verify_checkpoint_contract(model,root)
    contract,bound=verify_checkpoint_contract(model,root,True)
    assert not bound
    (model/'piperpika_action_contract.json').write_text(json.dumps(contract))
    assert verify_checkpoint_contract(model,root)[1]
    (meta/'action_stats.json').write_text('{"changed":true}')
    with pytest.raises(ValueError):verify_checkpoint_contract(model,root)


def test_launcher_prepares_before_training_and_honors_cluster(tmp_path):
    import os
    import subprocess
    log=tmp_path/'calls'
    fake=tmp_path/'python'
    fake.write_text('#!/bin/bash\nprintf "%s\\n" "$*" >> "$CALL_LOG"\n')
    fake.chmod(0o755)
    env=dict(os.environ, PYTHON_BIN=str(fake), CALL_LOG=str(log),
             NNODES='8', NGPU='8', NODE_RANK='3', MASTER_ADDR='10.0.0.1',
             MASTER_PORT='29501', CONFIG_NAME='piperpika_train', PREPARE_DATA='1')
    env.pop('DATASETS',None)
    subprocess.run(['bash','script/train_dist.sh','--num-steps','400'],env=env,check=True)
    calls=log.read_text().splitlines()
    assert len(calls)==2
    assert '--nnodes=8 --node_rank=3' in calls[0]
    assert '--master_port=29502' in calls[0]
    assert 'prepare_piperpika.py distributed' in calls[0]
    assert '--num-steps 400' in calls[1] and 'piperpika:5,agibot:4' in calls[1]
    assert 'robotwin:' not in calls[1]


def test_preparation_new_version_metadata_and_resume(tmp_path, monkeypatch):
    import pyarrow as pa
    import pyarrow.parquet as pq
    from types import SimpleNamespace
    from wan_va.dataset.aloha import ACTION_NAMES, CAMERAS
    prep=preparation_module()
    source=tmp_path/'raw';robot=source/prep.ROBOT_VERSION;meta=robot/'meta';meta.mkdir(parents=True)
    info={'fps':30,'features':{'action':{'names':ACTION_NAMES}},'total_tasks':1,
          'data_path':'data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet',
          'video_path':'videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4'}
    (meta/'info.json').write_text(json.dumps(info));(meta/'tasks.jsonl').write_text('{"task_index":0,"task":"pick"}\n')
    episodes=[];groups=[]
    for gid in range(2):
        human=source/f'human/{gid}.mp4';human.parent.mkdir(exist_ok=True);human.touch()
        entries=[]
        for eid in (2*gid,2*gid+1):
            ep={'episode_index':eid,'length':33,'tasks':['pick']};episodes.append(ep)
            data=robot/info['data_path'].format(episode_chunk=0,episode_index=eid)
            data.parent.mkdir(parents=True,exist_ok=True)
            pq.write_table(pa.table({'action':[[float(eid)]*28]*33,'episode_index':[eid]*33,'timestamp':np.arange(33)/30}),data)
            videos={}
            for cam in CAMERAS:
                v=robot/info['video_path'].format(episode_chunk=0,episode_index=eid,video_key=cam)
                v.parent.mkdir(parents=True,exist_ok=True);v.touch();videos[cam]=str(v.relative_to(source))
            entries.append({'episode_index':eid,'data_path':str(data.relative_to(source)),'videos':videos})
        groups.append({'alignment_task_number':gid,'human_video_path':str(human.relative_to(source)),'robot_episodes':entries})
    (meta/'episodes.jsonl').write_text(''.join(json.dumps(e)+'\n' for e in episodes))
    (source/'human_robot_alignment.json').write_text(json.dumps({'groups':groups}))
    monkeypatch.setattr(prep,'audit_video',lambda *a,**kw: {'frames':33,'fps':30})
    monkeypatch.setattr(prep,'model_identity',lambda p: {})
    args=SimpleNamespace(source=source,output=tmp_path/'prepared',seed=42,eval_groups=1,model=tmp_path/'model')
    prep.prepare(args)
    for split in ('train','eval'):
        info=json.loads((args.output/split/'piperpika/meta/info.json').read_text())
        assert info['chunks_size']==1000 and info['total_chunks']==1
        assert info['total_episodes']==2
    marker=args.output/'encoding_contract.json';before=marker.read_bytes();mtime=marker.stat().st_mtime_ns
    prep.prepare(args)
    assert marker.read_bytes()==before and marker.stat().st_mtime_ns==mtime
    # Resume rejects drift rather than silently changing the training split/data.
    (meta/'tasks.jsonl').write_text('changed')
    with pytest.raises(ValueError,match='changed'):prep.prepare(args)


def test_joint_evaluation_order_and_padding():
    from evaluation.piperpika.open_loop import flatten_first_chunk, metrics
    action=np.arange(14*3*16).reshape(14,3,16)
    pred=flatten_first_chunk(action)
    np.testing.assert_array_equal(pred[0],action[:,1,0])
    np.testing.assert_array_equal(pred[16],action[:,2,0])
    result=metrics(pred,pred,np.zeros(14),np.ones(14))
    assert result['joint_mae']==result['gripper_mae']==0
    with pytest.raises(ValueError):flatten_first_chunk(np.zeros((28,2,16)))
