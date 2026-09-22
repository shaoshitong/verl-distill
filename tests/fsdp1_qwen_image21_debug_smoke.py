"""Distributed debug IO audit with tiny real Qwen blocks and an explicit stub decoder.

Uses the real six-call rollout, 64-row debug event, tensor/image writers and DCP.
The stub decoder tests orchestration, not production VAE quality or GPU memory.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.distributed as dist
from PIL import Image

from verl_distill.algorithms.dmd.qwen_image21 import UpdateState
from verl_distill.data.qwen_image21 import RankCursor
from verl_distill.engine import qwen_checkpoint as cp
from verl_distill.engine.distributed import initialize_distributed, cleanup_distributed
from verl_distill.models.qwen_image21.modeling import predict_velocity, require_qwen_runtime
from verl_distill.trainers.qwen_image21 import wrap_model, optimizer_for, post_update_actions
from verl_distill.trainers.qwen_image21_debug import debug_event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--prompts', type=int, default=64)
    parser.add_argument('--comparison-steps', type=int, default=0)
    args = parser.parse_args()
    require_qwen_runtime()
    context = initialize_distributed()
    rank, world, device = context.rank, context.world_size, context.device
    root = Path(args.output)
    try:
        from diffusers import QwenImage21Transformer2DModel
        torch.manual_seed(7)
        model = wrap_model(QwenImage21Transformer2DModel(
            num_layers=2, attention_head_dim=16, num_attention_heads=2,
            context_in_dim=16, axes_dims_rope=(4, 6, 6)), context.local_rank, True)
        optimizer = optimizer_for(model, {'lr': 1e-4, 'betas': [.9,.95], 'weight_decay': .1})
        condition = dict(encoder_hidden_states=torch.randn(1,2,16,device=device,dtype=torch.bfloat16),
                         encoder_hidden_states_mask=torch.ones(1,2,device=device,dtype=torch.bool),
                         img_mask=torch.tensor([[False,False,True]],device=device),
                         img_shapes=[[(1,2,2)]], reference_latents=None)
        loss = predict_velocity(model, torch.randn(1,4,64,device=device),
                                torch.tensor([.4],device=device), condition).square().mean()
        loss.backward(); optimizer.step(); optimizer.zero_grad(set_to_none=True)
        cursor = RankCursor(32, rank, world, 42)
        cursor.next_index()
        state = UpdateState(reflow_updates=1000)
        rows = [dict(id=f'case-{i:03d}',kind='t2i',prompt='test',seed=i,width=32,height=32,
                     reference_images=[],reference_sha256=[]) for i in range(64)]
        if rank == 0:
            root.mkdir(parents=True, exist_ok=False)
            (root/'prompts.csv').write_text('test prompts snapshot\n')
        dist.barrier()
        class Store:
            def get(self, row, device): return condition
        class Schedule:
            config = {"shift_terminal": .02}
            def levels(self, h, w, steps=6, *, generator=True, device="cpu"):
                if steps == 6:
                    return torch.tensor([1.,.94,.87,.77,.63,.4,0.],device=device)
                return torch.linspace(1,0,steps+1,device=device)
        class Decoder:
            fail = False
            offloaded = False
            def decode(self, packed, height, width):
                if self.fail and rank == world-1:
                    raise RuntimeError('injected single-rank decode failure')
                torch.rand(1,device=device)  # debug must restore RNG, even after an error
                return Image.new('RGBA',(width,height),(100,120,140,255)), torch.zeros(1,4,height,width)
            def offload(self): self.offloaded = True
        decoder = Decoder()
        path = root/'checkpoint'/'reflow_step_001000'
        def save(name):
            cp.save_checkpoint(root/'checkpoint'/name, {'generator':model}, {'generator':optimizer},
                               state,cursor,{'test':True},{},rank=rank)
        def debug(output, snapshots):
            debug_event(output,state,model,snapshots,{'records':rows},Store(),Schedule(),decoder,
                        device,rank,world,root,root/'prompts.csv',{'test':True},
                        prompt_count=args.prompts, comparison_steps=args.comparison_steps)
        rng = cp.capture_rng_state()
        post_update_actions(state,'reflow',1000,3000,{'debug_after_reflow':True},save,
                            lambda:debug(root/'success',{}))
        torch.testing.assert_close(torch.get_rng_state(),rng['torch'],rtol=0,atol=0)
        torch.testing.assert_close(torch.cuda.get_rng_state(),rng['cuda_local'],rtol=0,atol=0)
        assert model.training and decoder.offloaded
        if rank == 0:
            event=root/'success/debug/reflow_step_001000'
            assert (event/'COMPLETE').exists()
            assert len(list((event/'rollout').glob('*/image.png')))==args.prompts
            if args.comparison_steps:
                comparison = event/f'rollout_{args.comparison_steps}step'
                assert len(list(comparison.glob('*/image.png')))==args.prompts
                assert not list(comparison.glob('*/steps'))
            for p in (event/'rollout').glob('*/metadata.json'):
                assert json.loads(p.read_text())['nfe']==6
        # Also exercise Fake/Real/diff tensor and image writers.
        tensors={'generator_x0':torch.ones(1,4,64), 'fake_x0':torch.ones(1,4,64)*2,
                 'real_x0':torch.ones(1,4,64)*3, 'diff_x0':-torch.ones(1,4,64)}
        snapshots={phase:[{'record':rows[rank], 'metadata':{'ga_index':i}, 'tensors':tensors}
                          for i in range(4)] for phase in ('fake_score','generator')}
        debug(root/'with_training_snapshots',snapshots)
        # One rank fails in local decode; all ranks must exit instead of hanging.
        decoder.fail=True
        rng=cp.capture_rng_state()
        try:
            debug(root/'failed_debug',{})
        except RuntimeError as exc:
            assert 'injected single-rank decode failure' in str(exc)
        else:
            raise AssertionError('Expected distributed decode failure')
        assert not (root/'failed_debug/debug/reflow_step_001000/COMPLETE').exists()
        assert model.training
        torch.testing.assert_close(torch.cuda.get_rng_state(),rng['cuda_local'],rtol=0,atol=0)
        cp.inspect_checkpoint(path,{'test':True})
        # A rank-local cursor/RNG write failure must never publish a checkpoint.
        original_save=torch.save
        def fail_sidecar(value, f, *args, **kwargs):
            if rank==world-1 and str(getattr(f,'name','')).endswith(f'rank-{rank:05d}.pt'):
                raise OSError('injected rank-local disk failure')
            return original_save(value,f,*args,**kwargs)
        torch.save=fail_sidecar
        try:
            cp.save_checkpoint(root/'failed_save',{'generator':model},{'generator':optimizer},
                               state,cursor,{'test':True},{},rank=rank)
        except RuntimeError as exc:
            assert 'injected rank-local disk failure' in str(exc)
        else:
            raise AssertionError('Expected distributed save failure')
        finally:
            torch.save=original_save
        assert not (root/'failed_save').exists()
        assert not (root/'failed_save.incomplete/COMPLETE').exists()
        dist.barrier()
        if rank==0:
            print(f'PASS: shared DCP save, {args.prompts} paired 6/{args.comparison_steps}-step rollouts, GA4 tensor dumps, single-rank decode/write failures',flush=True)
    finally:
        cleanup_distributed()


if __name__=='__main__':main()
