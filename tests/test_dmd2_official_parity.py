import unittest,torch
from verl_distill.algorithms.dmd.qwen_image21 import dmd_surrogate
def candidate(*args):
 loss,aux=dmd_surrogate(*args,loss_weight=1)
 return loss,{**aux,"direction":aux["normalized_direction"]}

def reference(g,z,f,r,s):
 with torch.no_grad():
  s=s.double().reshape(-1,1,1)
  fake=z.double()-s*f.double();real=z.double()-s*r.double()
  pr=g-real;pf=g-fake
  grad=torch.nan_to_num((pr-pf)/pr.abs().mean((1,2),keepdim=True))
 return .5*torch.nn.functional.mse_loss(g.float(),(g-grad).detach().float())
class Parity(unittest.TestCase):
 def check(self,g,z,f,r,s):
  a=g.clone().requires_grad_();b=g.clone().requires_grad_();la,_=candidate(a,z,f,r,s);lb=reference(b,z,f,r,s)
  ga=torch.autograd.grad(la,a)[0];gb=torch.autograd.grad(lb,b)[0]
  torch.testing.assert_close(la,lb,rtol=0,atol=0,equal_nan=True);torch.testing.assert_close(ga,gb,rtol=0,atol=0,equal_nan=True)
 def test_finite(self):
  torch.manual_seed(42);self.check(*[torch.randn(2,8,4) for _ in range(4)],torch.tensor([.2,.8]))
 def test_near_zero(self):
  self.check(torch.full((1,2,2),1e-9),torch.zeros(1,2,2),torch.ones(1,2,2),torch.zeros(1,2,2),torch.tensor([.5]))
 def test_nan_score(self):
  g=torch.ones(1,2,2);self.check(g,g,torch.full_like(g,float('nan')),torch.zeros_like(g),torch.tensor([.5]))
  loss,a=candidate(g,g,torch.full_like(g,float('nan')),torch.zeros_like(g),torch.tensor([.5]));self.assertEqual(loss.item(),0);self.assertEqual(a['direction'].abs().sum().item(),0)
 def test_zero_over_zero(self):
  z=torch.zeros(1,2,2);self.check(z,z,z,z,torch.tensor([.5]))
 def test_inf_is_not_always_safe(self):
  z=torch.zeros(1,2,2);loss,_=candidate(z,z,torch.ones_like(z),z,torch.tensor([.5]));self.assertFalse(torch.isfinite(loss))
if __name__=='__main__':unittest.main()
