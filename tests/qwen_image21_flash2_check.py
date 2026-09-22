"""Small CUDA correctness check; no production model loaded or training interrupted."""
import copy
import json
import torch
from diffusers import QwenImage21Transformer2DModel
from verl_distill.models.qwen_image21 import attention
from verl_distill.models.qwen_image21.modeling import predict_velocity


def check():
    torch.manual_seed(711)
    device='cuda:0'
    records=[]
    real_flash=attention.require_flash2()
    for references,padded in [(0,False),(1,False),(2,False),(5,False),(2,True)]:
        native=QwenImage21Transformer2DModel(num_layers=2, attention_head_dim=64,
            num_attention_heads=2,context_in_dim=16,axes_dims_rope=(16,24,24)).to(device,dtype=torch.bfloat16)
        fused=copy.deepcopy(native)
        attention.configure_attention(fused,'flash2_segmented')
        calls=[]
        def counted(*args,**kwargs):
            calls.append((args[0].shape[1],args[1].shape[1],kwargs['causal']))
            return real_flash(*args,**kwargs)
        attention.require_flash2=lambda:counted
        # Interleaved text-image-text prefixes exercise non-square causal alignment.
        img_mask=[False]*17
        for _ in range(references):img_mask += [True]*16+[False]*13
        text_len=len(img_mask)
        mask=torch.ones(1,text_len,device=device,dtype=torch.bool) if padded else None
        if mask is not None:mask[0,-1]=False
        cond=dict(encoder_hidden_states=torch.randn(1,text_len,16,device=device,dtype=torch.bfloat16),
                  encoder_hidden_states_mask=mask,img_mask=torch.tensor([img_mask+[True]*16],device=device),
                  img_shapes=[[(1,8,8)]*(references+1)],
                  reference_latents=torch.randn(1,64*references,64,device=device,dtype=torch.bfloat16) if references else None)
        x=torch.randn(1,64,64,device=device);t=torch.tensor([.4],device=device)
        outputs=[];gradients=[]
        for model in (native,fused):
            xx=x.clone().requires_grad_();y=predict_velocity(model,xx,t,cond)
            y.square().mean().backward()
            outputs.append(y.detach());gradients.append((xx.grad.detach(),torch.cat([p.grad.float().flatten() for p in model.parameters() if p.grad is not None])))
        torch.testing.assert_close(outputs[0],outputs[1],rtol=.02,atol=.006)
        errors=[]
        for a,b in zip(gradients[0],gradients[1]):
            rel=((a-b).norm()/a.norm().clamp_min(1e-8)).item();errors.append(rel)
            assert rel<.02,rel
        assert bool(calls) != padded
        records.append(dict(references=references,padded=padded,max_output_difference=(outputs[0]-outputs[1]).abs().max().item(),gradient_relative_errors=errors,flash_calls=calls))
        attention.require_flash2=lambda:real_flash
        del native,fused
    print(json.dumps(records,indent=2))
if __name__=='__main__':check()
