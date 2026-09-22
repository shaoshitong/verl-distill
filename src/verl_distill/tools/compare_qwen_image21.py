"""Paired HF / DMD Generator evaluation; launch on eight GPUs using torchrun."""
import argparse
import json
import os
import time
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

from verl_distill.data.qwen_image21 import atomic_json, canonical_hash, read_manifest, sha256
from verl_distill.engine.checkpoint import load_distributed_model_state
from verl_distill.engine.distributed import initialize_distributed, cleanup_distributed
from verl_distill.engine.qwen_checkpoint import collective_call, inspect_checkpoint
from verl_distill.models.qwen_image21.modeling import (
    model_identity, ConditionStore, QwenSchedule, QwenDecoder, load_transformer, require_qwen_runtime,
)
from verl_distill.trainers.qwen_image21 import wrap_model
from verl_distill.trainers.qwen_image21_debug import rollout_sample, save_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--checkpoint-label', default='dmd_fake45')
    parser.add_argument('--attention-backend', choices=['sdpa','flash2_segmented'])
    parser.add_argument('--training-six-only', action='store_true')
    args = parser.parse_args()
    require_qwen_runtime()
    context = initialize_distributed(timeout_seconds=1800)
    rank, device = context.rank, context.device
    out = Path(args.output)
    try:
        if context.world_size != 8:
            raise ValueError('This comparison uses 8 prompts on 8 ranks')
        config = yaml.safe_load(Path(args.config).read_text())
        if args.attention_backend:
            config['runtime']['attention_backend'] = args.attention_backend
        saved = json.loads((Path(args.checkpoint)/'state.json').read_text())
        collective_call('validate source checkpoint', lambda: inspect_checkpoint(args.checkpoint, saved['contract']))
        collective_call('reserve comparison directory', lambda: out.mkdir(parents=True, exist_ok=False) if rank == 0 else None)
        identity = collective_call('verify original HF model assets', lambda: model_identity(config['model']['pretrained_model']) if rank == 0 else None)
        objects = [identity]
        dist.broadcast_object_list(objects, src=0)
        identity = objects[0]
        if canonical_hash(identity) != saved['contract']['model_identity']:
            raise ValueError('HF assets differ from checkpoint base model')
        data = config['data']
        hashes = {k:sha256(data[k+'_manifest'] if k=='eval' else data['manifest']) for k in ['train','eval']}
        store = ConditionStore(data['condition_cache'], identity, hashes)
        evaluation = read_manifest(data['eval_manifest'])
        records = evaluation['records'][:8]
        row = records[rank]
        specs = [('official_6',6,True),('official_25',25,True),('training_6',6,False)]
        if args.training_six_only:
            specs = [('training_6',6,False)]
        collective_call('save comparison protocol', lambda: atomic_json(out/'protocol.json', {
            'checkpoint':args.checkpoint,'updates':saved['updates'],'original_model':config['model'],
            'model_identity':identity,'records':records,'samplers':specs,
            'integration':'Euler in FP32, velocity forward BF16; same inference code for all variants',
            'guidance':'conditional_only, no CFG','attention_backend':config['runtime']['attention_backend'],
            'initial_noise':'same per-prompt seed, generated BF16 then integrated FP32',
            'training_6_terminal':0.4,'created_at':time.time()}) if rank==0 else None)
        model = wrap_model(load_transformer(config['model']['pretrained_model'],config['runtime']['attention_backend']),context.local_rank,True)
        model.eval()
        schedule = QwenSchedule(config['model']['pretrained_model'],terminal=.4)
        decoder = QwenDecoder(config['model']['pretrained_model'],device)
        condition = collective_call('load evaluation conditioning',lambda:store.get(row,device))
        initial_reference = None
        for weight in ['hf_original',args.checkpoint_label]:
            if weight == args.checkpoint_label:
                load_distributed_model_state(Path(args.checkpoint)/'generator',model)
                model.eval()
            for tag,steps,official in specs:
                started = time.monotonic()
                result = rollout_sample(model,condition,row,schedule,device,steps=steps,
                                        official_schedule=official,capture_trajectory=False)
                if initial_reference is None:
                    initial_reference = result[0]
                if not torch.equal(initial_reference,result[0]):
                    raise RuntimeError('Initial noise mismatch')
                collective_call('save comparison image and latents',lambda:save_rollout(
                    out/weight/tag/row['id'],row,result,decoder,
                    {'weight_source':weight,'checkpoint':args.checkpoint if weight==args.checkpoint_label else None,
                     'schedule':tag,'scheduler_config':schedule.config,'rank':rank,
                     'seconds_before_decode':time.monotonic()-started}))
                decoder.offload()
                if rank == 0:
                    print(f'COMPLETE variant {weight}/{tag}: 8 images',flush=True)
        collective_call('commit comparison',lambda:atomic_json(out/'COMPLETE',{
            'prompts':8,'variants':2*len(specs),'images':16*len(specs),'same_initial_noise_verified':True,
            'checkpoint':args.checkpoint,'finished_at':time.time()}) if rank==0 else None)
    except BaseException:
        out.mkdir(parents=True,exist_ok=True)
        atomic_json(out/f'failure-rank-{rank:05d}.json',{'traceback':traceback.format_exc(),'time':time.time()})
        traceback.print_exc()
        os._exit(1)
    else:
        cleanup_distributed()


if __name__ == '__main__':
    main()
