import unittest
from unittest.mock import patch
import torch
from verl_distill.models.qwen_image21.guidance import predict_fake_cfg
class SingleCFG(unittest.TestCase):
 def test_forward_gradient_and_cache_decision(self):
  for scale in (1,):
   weight=torch.tensor(2.,requires_grad=True);x=torch.tensor(3.)
   def predict(model,noisy,sigma,condition):return weight*noisy*(1 if condition=='positive' else .25)
   with patch('verl_distill.models.qwen_image21.modeling.predict_velocity',side_effect=predict) as call:
    y=predict_fake_cfg(None,x,None,'positive',None,scale)
    self.assertEqual(call.call_count,2 if scale>1 else 1)
    self.assertAlmostEqual(y.item(),6. if scale<=1 else 10.5)
    y.backward();self.assertGreater(weight.grad.item(),0)
   with torch.no_grad(),patch('verl_distill.models.qwen_image21.modeling.predict_velocity',side_effect=predict) as call:
    y=predict_fake_cfg(None,x,None,'positive',None,scale)
    self.assertFalse(y.requires_grad);self.assertEqual(call.call_count,2 if scale>1 else 1)
 def test_invalid_scale_no_forward(self):
  for scale in (0,.5,2,-1,float('nan'),float('inf')):
   with patch('verl_distill.models.qwen_image21.modeling.predict_velocity') as call:
    with self.assertRaises(ValueError):predict_fake_cfg(None,None,None,None,None,scale)
    call.assert_not_called()

class ConfigContract(unittest.TestCase):
 def test_actual_configs_and_cache_contract(self):
  import copy,yaml
  from pathlib import Path
  from verl_distill.models.qwen_image21.configuration import validate_qwen_config
  for name in ('bench8','bench32'):
   c=yaml.safe_load((Path(__file__).resolve().parents[1]/'configs/recipes/qwen_image21/resume200_singlefake_fa3.yaml').read_text())
   c['method']['params']['fake_cfg_scale']=1
   validate_qwen_config(c)
   c['method']['params']['teacher_cfg_scale']=1
   c['data'].pop('negative_condition_cache',None)
   validate_qwen_config(c)
   for scale in (0,.5,2):
    bad=copy.deepcopy(c);bad['method']['params']['fake_cfg_scale']=scale
    with self.assertRaises(ValueError):validate_qwen_config(bad)
   c['method']['params']['teacher_cfg_scale']=2
   with self.assertRaises(ValueError):validate_qwen_config(c)
 def test_no_fake_cache_lookup_in_trainer(self):
  from pathlib import Path
  p=Path(__file__).resolve().parents[1]/'src/verl_distill/trainers/qwen_image21.py'
  source=p.read_text()
  self.assertNotIn('load Fake negative condition',source)
  self.assertNotIn('fake_negative_condition',source)

if __name__ == '__main__': unittest.main()
