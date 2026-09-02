from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist

from verl_distill.config import load_config
from verl_distill.engine.checkpoint import load_distributed_training_state
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2
from verl_distill.trainers.common import set_seed
from verl_distill.trainers.dmd import _build_models, _build_optimizer


def _stats(module: torch.nn.Module, device: torch.device) -> torch.Tensor:
    out = torch.zeros(4, device=device, dtype=torch.float64)
    for parameter in module.parameters():
        data = parameter.detach()
        if hasattr(data, "to_local"):
            data = data.to_local()
        data = data.float()
        out[0] += data.numel()
        out[1] += data.double().sum()
        out[2] += data.double().square().sum()
        out[3] += data.double().abs().sum()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out.cpu()


def main() -> None:
    checkpoint = Path(os.environ["DMD_PROBE_CHECKPOINT"])
    context = initialize_distributed("nccl")
    set_seed(42, context.rank)
    config = load_config("dmd_1000_debug")
    student, score = _build_models(config, context.device)
    student.transformer.transformer.float()
    score["real"].transformer.float()
    score["fake"].transformer.float()
    apply_zimage_fsdp2(student.transformer.transformer, param_dtype=torch.bfloat16)
    apply_zimage_fsdp2(score["real"].transformer, param_dtype=torch.bfloat16)
    apply_zimage_fsdp2(score["fake"].transformer, param_dtype=torch.bfloat16)

    generator_parameters = [
        parameter for parameter in student.transformer.parameters() if parameter.requires_grad
    ]
    fake_parameters = [
        parameter for parameter in score["fake"].parameters() if parameter.requires_grad
    ]
    generator_optimizer = _build_optimizer(generator_parameters, config["optimizer"]["generator"])
    fake_optimizer = _build_optimizer(fake_parameters, config["optimizer"]["fake_score"])
    model = torch.nn.ModuleDict(
        {
            "generator": student.transformer,
            "real_score": score["real"],
            "fake_score": score["fake"],
        }
    )

    before = {
        "generator": _stats(student.transformer, context.device),
        "fake_score": _stats(score["fake"], context.device),
        "real_score": _stats(score["real"], context.device),
    }
    step = load_distributed_training_state(
        checkpoint,
        model,
        [generator_optimizer, fake_optimizer],
    )
    after = {
        "generator": _stats(student.transformer, context.device),
        "fake_score": _stats(score["fake"], context.device),
        "real_score": _stats(score["real"], context.device),
    }
    if context.is_main_process:
        print(f"loaded_step {step}")
        for name in ("generator", "fake_score", "real_score"):
            before_stats = before[name]
            after_stats = after[name]
            print(
                name,
                f"count {int(after_stats[0].item())}",
                f"sum_delta {(after_stats[1] - before_stats[1]).item():.9e}",
                f"sqsum_delta {(after_stats[2] - before_stats[2]).item():.9e}",
                f"abssum_delta {(after_stats[3] - before_stats[3]).item():.9e}",
            )
    cleanup_distributed()


if __name__ == "__main__":
    main()
