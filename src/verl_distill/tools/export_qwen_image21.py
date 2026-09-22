"""Export only the Generator from a complete FSDP1 Qwen checkpoint (invoke via torchrun)."""
import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import save_file
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, FullStateDictConfig, StateDictType

from verl_distill.data.qwen_image21 import atomic_json, canonical_hash
from verl_distill.engine.checkpoint import load_distributed_model_state
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.qwen_checkpoint import collective_call
from verl_distill.models.qwen_image21.modeling import (
    load_transformer, model_identity, require_qwen_runtime,
)
from verl_distill.trainers.qwen_image21 import wrap_model


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--model", required=True, help="Original complete Qwen model directory")
    p.add_argument("--output", required=True, help="New directory for exported transformer")
    args = p.parse_args()
    require_qwen_runtime()
    context = initialize_distributed()
    if context.device.type != "cuda":
        cleanup_distributed()
        raise RuntimeError("FSDP1 export requires CUDA and torchrun")
    if not dist.is_initialized():
        dist.init_process_group("nccl", rank=context.rank, world_size=context.world_size)
    try:
        source = Path(args.checkpoint)
        if not (source / "COMPLETE").is_file():
            raise ValueError("Incomplete checkpoint")
        saved = json.loads((source / "state.json").read_text())
        if saved["world_size"] != context.world_size:
            raise ValueError("Export with the checkpoint's world size")
        def verify_assets():
            if canonical_hash(model_identity(args.model)) != saved["contract"]["model_identity"]:
                raise ValueError("Export base model differs from the training checkpoint")
        collective_call("verify export model assets", verify_assets)
        wrapped = wrap_model(load_transformer(args.model), context.local_rank, True)
        load_distributed_model_state(source / "generator", wrapped)
        with FSDP.state_dict_type(wrapped, StateDictType.FULL_STATE_DICT,
                                 FullStateDictConfig(offload_to_cpu=True, rank0_only=True)):
            state = wrapped.state_dict()
        def write():
            if context.rank != 0:
                return
            path = Path(args.output)
            staging = path.with_name(path.name + ".incomplete")
            if path.exists() or staging.exists():
                raise FileExistsError(f"Export already exists: {path}")
            staging.mkdir(parents=True)
            # Diffusers save_pretrained ignores a state_dict keyword. Serialize the
            # already gathered CPU state explicitly; rank 0 must not reenter FSDP.
            wrapped.module.save_config(staging)
            shards = split_torch_state_dict_into_shards(
                state, filename_pattern="diffusion_pytorch_model{suffix}.safetensors",
                max_shard_size="5GB")
            for filename, names in shards.filename_to_tensors.items():
                save_file({name: state[name].contiguous() for name in names},
                          staging / filename, metadata={"format": "pt"})
            if shards.is_sharded:
                atomic_json(staging / "diffusion_pytorch_model.safetensors.index.json",
                            {"metadata": shards.metadata, "weight_map": shards.tensor_to_filename})
            atomic_json(staging / "distillation.json", {"checkpoint": str(source),
                        "updates": saved["updates"], "nfe": 6,
                        "generator_shift_terminal": .4, "guidance": "conditional_only",
                        "contract": saved["contract"], "storage_dtype": "float32"})
            atomic_json(staging / "COMPLETE", {"checkpoint": str(source)})
            os.rename(staging, path)
        collective_call("export Generator weights", write)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
