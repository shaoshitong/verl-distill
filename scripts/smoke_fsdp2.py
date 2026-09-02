#!/usr/bin/env python3
import argparse

import torch
import torch.distributed as dist
from torch import nn

from verl_distill.engine.checkpoint import (
    load_distributed_training_state,
    save_distributed_training_state,
)
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.ema import ShardedEMA
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2, clip_grad_norm


class TinyZImageLayout(nn.Module):
    def __init__(self):
        super().__init__()
        self.noise_refiner = nn.ModuleList([nn.Linear(8, 8)])
        self.context_refiner = nn.ModuleList([nn.Linear(8, 8)])
        self.layers = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 8)])
        self.output = nn.Linear(8, 1)

    def forward(self, value):
        for blocks in (self.noise_refiner, self.context_refiner, self.layers):
            for block in blocks:
                value = torch.tanh(block(value))
        return self.output(value)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    context = initialize_distributed("nccl")
    torch.manual_seed(123)
    model = TinyZImageLayout().to(context.device)
    apply_zimage_fsdp2(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
    ema = ShardedEMA(model, decay=0.5)
    value = torch.randn(4, 8, device=context.device)
    loss = model(value).square().mean()
    loss.backward()
    clip_grad_norm(model.parameters(), 1.0)
    optimizer.step()
    ema.update(model)
    online = torch.cat(
        [parameter.detach().full_tensor().flatten() for parameter in model.parameters()]
    )
    expected_ema = {name: value.clone() for name, value in ema.state_dict()["shadows"].items()}
    save_distributed_training_state(
        args.checkpoint,
        model,
        optimizer,
        step=17,
        extra_state={"ema": ema.state_dict()},
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    restored_extra = {}
    step = load_distributed_training_state(
        args.checkpoint, model, optimizer, extra_state=restored_extra
    )
    ema.load_state_dict(restored_extra["ema"])
    after = torch.cat(
        [parameter.detach().full_tensor().flatten() for parameter in model.parameters()]
    )
    if step != 17 or not torch.equal(online, after):
        raise RuntimeError("FSDP2 checkpoint round-trip mismatch")
    for name, value in ema.state_dict()["shadows"].items():
        if not torch.equal(expected_ema[name], value):
            raise RuntimeError(f"EMA checkpoint round-trip mismatch: {name}")
    checksum = online.float().sum()
    dist.all_reduce(checksum)
    if context.is_main_process:
        print(
            f"FSDP2_SMOKE_OK world_size={context.world_size} step={step} "
            f"checksum={float(checksum):.6f}"
        )
    cleanup_distributed()


if __name__ == "__main__":
    main()
