from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0


def initialize_distributed(backend: str = "nccl", timeout_seconds: int = 1800):
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
        if backend == "nccl":
            backend = "gloo"
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(seconds=int(timeout_seconds)),
        )
    return DistributedContext(rank, local_rank, world_size, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
