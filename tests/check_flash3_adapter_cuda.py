"""Explicit GPU check; run only after baseline finishes and resources are released.
Tests the real processor segmentation against FP32 explicit masks, including all QKV gradients.
"""
import json
import types
from unittest.mock import patch
import torch
from verl_distill.models.qwen_image21 import attention as a

def run_case(length, segments, q_length=None):
    torch.manual_seed(739)
    q_length=q_length or length
    inputs=[torch.randn(1,n,4,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            for n in (q_length,length,length)]
    refs=[x.detach().float().requires_grad_() for x in inputs]
    q,k,v=inputs
    attn=types.SimpleNamespace(to_out=[torch.nn.Identity(),torch.nn.Identity()])
    with patch.object(a, '_qwenimage21_prepare_qkv', return_value=(q,k,v,q_length)):
        out=a.QwenImage21Flash3Processor()(attn, q.flatten(2), segments=segments)
    mask=torch.ones(q_length,length,dtype=torch.bool,device='cuda')
    if segments is not None:
        mask.zero_()
        for start,end,is_text in segments:
            mask[start:end,:end]=True
            if is_text:
                rows=torch.arange(start,end,device='cuda')[:,None]
                keys=torch.arange(end,device='cuda')[None,:]
                mask[start:end,:end]=keys<=rows
        prefix=segments[-1][1] if segments else 0
        mask[prefix:,:]=True
    scores=torch.einsum('bqhd,bkhd->bhqk',refs[0],refs[1]) / (128**.5)
    expected=torch.einsum('bhqk,bkhd->bqhd',scores.masked_fill(~mask,float('-inf')).softmax(-1),refs[2]).flatten(2)
    upstream=torch.randn_like(out)
    (out*upstream).sum().backward()
    (expected*upstream.float()).sum().backward()
    torch.testing.assert_close(out.float(),expected,atol=.025,rtol=.025)
    errors={'out_max_abs':(out.float()-expected).abs().max().item()}
    for name,x,r in zip(('dq','dk','dv'),inputs,refs):
        torch.testing.assert_close(x.grad.float(),r.grad,atol=.06,rtol=.04)
        errors[name+'_max_abs']=(x.grad.float()-r.grad).abs().max().item()
    return errors

def run_long_fa2_comparison(length, segments, q_length=None):
    """Model-sized 32 heads, 128 dim; no dense NxN reference tensor."""
    torch.manual_seed(941)
    q_length = q_length or length
    inputs = [torch.randn(1,n,32,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
              for n in (q_length,length,length)]
    refs = [x.detach().clone().requires_grad_() for x in inputs]
    attn = types.SimpleNamespace(to_out=[torch.nn.Identity(),torch.nn.Identity()])
    def predict(processor, xs):
        q,k,v = xs
        with patch.object(a, '_qwenimage21_prepare_qkv', return_value=(q,k,v,q_length)):
            return processor(attn, q.flatten(2), segments=segments)
    out = predict(a.QwenImage21Flash3Processor(), inputs)
    expected = predict(a.QwenImage21Flash2Processor(), refs)
    upstream = torch.randn_like(out)
    # Divide by sqrt(numel) so long-sequence scalar losses remain well-scaled.
    loss = (out.float()*upstream).sum() / out.numel()**.5
    reference_loss = (expected.float()*upstream).sum() / expected.numel()**.5
    assert torch.isfinite(loss) and torch.isfinite(reference_loss)
    loss.backward()
    reference_loss.backward()
    results = {'length':length, 'q_length':q_length, 'heads':32, 'head_dim':128,
               'loss_fa3':loss.item(), 'loss_fa2':reference_loss.item()}
    pairs = [('out',out,expected)] + [(name,x.grad,r.grad) for name,x,r in zip(('dq','dk','dv'),inputs,refs)]
    for name,x,r in pairs:
        assert x is not None and r is not None and torch.isfinite(x).all() and torch.isfinite(r).all(), name
        delta = x.float()-r.float()
        rms = r.float().square().mean().sqrt().clamp_min(1e-12)
        relative_rms = (delta.square().mean().sqrt()/rms).item()
        # Aggregate BF16 numerical tolerance; max abs is reported separately.
        # Near-zero entries make an elementwise relative tolerance misleading.
        assert relative_rms < .03, (name, relative_rms)
        results[name] = {'max_abs':delta.abs().max().item(), 'relative_rms':relative_rms,
                         'reference_rms':rms.item(), 'tolerance_relative_rms':.03}
    return results

if __name__=='__main__':
    a.require_flash3()  # No silent skip: absent build/wrong device must fail.
    results={
      'text_image_text_target':run_case(31,[(0,3,True),(3,13,False),(13,18,True)]),
      'image_target':run_case(23,[(0,11,False)]),
      'target_only':run_case(19,[]),
      'cached_q_less_than_k':run_case(29,None,q_length=7),
    }
    results['long_32heads_segmented_fa2_comparison'] = run_long_fa2_comparison(16384, [(0,17,True),(17,4113,False),(4113,4142,True)])
    results['long_32heads_cached_fa2_comparison'] = run_long_fa2_comparison(16384, None, q_length=1024)
    torch.cuda.synchronize()
    print(json.dumps({'device':torch.cuda.get_device_name(),'capability':torch.cuda.get_device_capability(), 'cases':results},indent=2))
