from collections import Counter
import copy
import pytest
from verl_distill.data.qwen_image21 import RankCursor
from verl_distill.engine.qwen_checkpoint import infrastructure_contract_compatible


def epoch(cursors):
    count=(cursors[0].size+cursors[0].world_size-1)//cursors[0].world_size
    return [[c.next_index() for c in cursors] for _ in range(count)]


def test_bucket_coverage_cost_balance_and_resume():
    size, world = 257, 8
    costs=[1+(i%6)*10 for i in range(size)]
    plain=[RankCursor(size,r,world,42) for r in range(world)]
    bucket=[RankCursor(size,r,world,42,costs=costs,bucket_batches=64) for r in range(world)]
    a,b=epoch(plain),epoch(bucket)
    assert Counter(sum(a,[])) == Counter(sum(b,[]))
    assert set(sum(b,[])) == set(range(size))
    assert sum(max(costs[i] for i in row) for row in b) < sum(max(costs[i] for i in row) for row in a)*.75
    clone=[RankCursor(size,r,world,42,costs=costs,bucket_batches=64) for r in range(world)]
    for c,d in zip(bucket,clone):d.load_state_dict(c.state_dict())
    assert epoch(bucket)==epoch(clone)


def test_legacy_resume_rebuckets_only_pending_suffix_and_replays():
    size,world,consumed=81,8,3
    costs=[i%5+1 for i in range(size)]
    old=[RankCursor(size,r,world,7) for r in range(world)]
    for _ in range(consumed):
        for c in old:c.next_index()
    migrated=[RankCursor(size,r,world,7,costs=costs,allow_ordering_migration=True) for r in range(world)]
    for c,d in zip(old,migrated):d.load_state_dict(c.state_dict())
    remaining=(size+world-1)//world-consumed
    expected=[[c.next_index() for c in old] for _ in range(remaining)]
    first=[c.next_index() for c in migrated]
    replay=[RankCursor(size,r,world,7,costs=costs) for r in range(world)]
    for c,d in zip(migrated,replay):d.load_state_dict(c.state_dict())
    actual=[[c.next_index() for c in migrated] for _ in range(remaining-1)]
    repeated=[[c.next_index() for c in replay] for _ in range(remaining-1)]
    assert actual==repeated
    assert Counter(sum([first]+actual,[]))==Counter(sum(expected,[]))
    assert all(c.bucket_start==consumed for c in migrated)
    assert epoch(migrated)==epoch(replay)
    denied=RankCursor(size,0,world,7,costs=costs)
    with pytest.raises(ValueError,match='explicit'):denied.load_state_dict(RankCursor(size,0,world,7).state_dict())


def test_infra_migration_does_not_allow_training_changes():
    old={'runtime':{'seed':42,'gradient_accumulation_steps':4},'optimizer':{'lr':1e-4},'manifest_hashes':{'train':'fixed'}}
    new=copy.deepcopy(old);new['runtime'].update(data_ordering='bucketed_v1',bucket_batches=64,attention_backend='flash2_segmented')
    assert infrastructure_contract_compatible(old,new)
    for key in ['seed','gradient_accumulation_steps']:
        bad=copy.deepcopy(new);bad['runtime'][key]+=1
        assert not infrastructure_contract_compatible(old,bad)
    bad=copy.deepcopy(new);bad['optimizer']['lr']=5e-4
    assert not infrastructure_contract_compatible(old,bad)
    bad=copy.deepcopy(new);bad['manifest_hashes']['train']='changed'
    assert not infrastructure_contract_compatible(old,bad)


def test_explicit_refinement_contract_is_narrow():
    from verl_distill.engine.qwen_checkpoint import refinement_contract_compatible
    old={'updates':{'reflow_updates':500,'fake_updates':0,'generator_updates':0,'dmd_initialized':False},
         'contract':{'params':{'reflow_updates':1000,'fake_updates':3000},
                     'runtime':{'reflow_gradient_accumulation_steps':1,'gradient_accumulation_steps':4,'reflow_checkpoint_steps':[1,500,1000]},
                     'optimizer':{'reflow':{'lr':1e-4},'generator':{'lr':5e-7}},
                     'manifest_hashes':{'train':'old','eval':'same'},'condition_index':'old'}}
    new=copy.deepcopy(old['contract'])
    new['params']['reflow_updates']=510
    new['runtime'].update(reflow_gradient_accumulation_steps=16,reflow_checkpoint_steps=[510],data_ordering='bucketed_v1',bucket_batches=64,attention_backend='flash2_segmented')
    new['optimizer']['reflow']['lr']=2e-5
    new['manifest_hashes']['train']='new';new['condition_index']='new'
    plan={'source_step':500,'updates':10,'ga':16,'lr_divisor':5,'replace_training_data':True}
    assert refinement_contract_compatible(old,new,plan,allow_infra_change=True)
    for mutate in [lambda c:c['optimizer']['generator'].update(lr=1e-3),
                   lambda c:c['runtime'].update(gradient_accumulation_steps=16),
                   lambda c:c['params'].update(reflow_updates=511),
                   lambda c:c['manifest_hashes'].update(eval='changed')]:
        bad=copy.deepcopy(new);mutate(bad)
        assert not refinement_contract_compatible(old,bad,plan,allow_infra_change=True)
    assert not refinement_contract_compatible(old,new,None,allow_infra_change=True)
    assert not refinement_contract_compatible(old,new,{**plan,'source_step':499},allow_infra_change=True)


def test_reused_cache_provenance_and_payload(tmp_path):
    import json, torch
    from verl_distill.models.qwen_image21.modeling import save_condition,ConditionStore
    from verl_distill.data.qwen_image21 import sha256
    identity={'test':'identity'};record={'id':'sample'}
    base=tmp_path/'base';new=tmp_path/'new';new.mkdir()
    key,entry=save_condition(base,record,identity,{'tensor':torch.tensor([1.])})
    base_index={'schema':1,'model_identity':identity,'manifest_hashes':{'train':'old'},'entries':{key:entry}}
    (base/'index.json').write_text(json.dumps(base_index))
    new_index={'schema':1,'model_identity':identity,'manifest_hashes':{'train':'new'},'entries':{key:{**entry,'source':'base'}},'reused_cache':{'root':str(base),'index_sha256':sha256(base/'index.json')}}
    (new/'index.json').write_text(json.dumps(new_index))
    store=ConditionStore(new,identity,{'train':'new'})
    assert store.get(record,'cpu')['tensor'].item()==1
    new_index['entries'][key]['sha256']='corrupt'
    (new/'index.json').write_text(json.dumps(new_index))
    with pytest.raises(ValueError,match='differs'):
        ConditionStore(new,identity,{'train':'new'}).get(record,'cpu')
    (base/'index.json').write_text('{}')
    with pytest.raises(ValueError,match='digest changed'):
        ConditionStore(new,identity,{'train':'new'})
