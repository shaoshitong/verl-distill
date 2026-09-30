import copy
import pytest
import torch
from verl_distill.algorithms.dmd.qwen_rollout import initial_rollout_noise
from verl_distill.engine.qwen_checkpoint import initial_noise_contract_compatible

@pytest.mark.parametrize('phase',['generator','fake_score'])
def test_fresh_noise_is_randn_like_and_independent(phase):
    clean=torch.full((1,8,64),99.,dtype=torch.float32)
    stored=torch.zeros_like(clean)
    torch.manual_seed(32);expected=torch.randn_like(clean)
    torch.manual_seed(32)
    actual=initial_rollout_noise(clean,stored,phase=phase,rollout_input=True,source='gaussian')
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    other=initial_rollout_noise(clean,stored,phase=phase,rollout_input=True,source='gaussian')
    assert not torch.equal(actual,other)
    assert actual.dtype==clean.dtype and actual.shape==clean.shape

@pytest.mark.parametrize('phase,source',[('reflow','gaussian'),('reflow','dataset'),('generator','dataset'),('fake_score','dataset')])
def test_paired_mode_and_reflow_preserve_rng(phase,source):
    clean=torch.zeros(1,4,64);stored=torch.ones_like(clean)
    before=torch.get_rng_state()
    result=initial_rollout_noise(clean,stored,phase=phase,rollout_input=True,source=source)
    assert torch.equal(result,stored)
    assert torch.equal(before,torch.get_rng_state())


def test_resume_contract_only_authorized_changes():
    saved={'updates':{'dmd_initialized':True},'models':['generator','fake'],
      'contract':{'params':{'generator_input':'rollout_dataset_noise','fake_loss':'velocity_mse'},
      'runtime':{'save_every_fake_updates':200,'gradient_accumulation_steps':1},
      'optimizer':{'lr':1e-6}}}
    current=copy.deepcopy(saved['contract'])
    current['params']['rollout_initial_noise']='gaussian'
    current['runtime']['save_every_fake_updates']=100
    plan={'source_state_sha256':'a'*64}
    assert initial_noise_contract_compatible(saved,current,plan)
    assert not initial_noise_contract_compatible(saved,current,None)
    for section,key,value in [('optimizer','lr',2e-6),('params','dmd_rollout_loss_mode','all_exits'),('runtime','gradient_accumulation_steps',2)]:
        changed=copy.deepcopy(current);changed[section][key]=value
        assert not initial_noise_contract_compatible(saved,changed,plan)
