from __future__ import annotations

from functools import partial

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    MixedPrecision,
)
from torch.distributed.fsdp.wrap import lambda_auto_wrap_policy


def apply_zimage_fsdp1(
    module: torch.nn.Module,
    *,
    no_split_modules: list[str] | tuple[str, ...],
    local_rank: int,
    param_dtype: torch.dtype = torch.bfloat16,
    reduce_dtype: torch.dtype = torch.float32,
    buffer_dtype: torch.dtype = torch.float32,
) -> FSDP:
    no_split = set(no_split_modules)
    module.float()
    return FSDP(
        module,
        auto_wrap_policy=partial(
            lambda_auto_wrap_policy,
            lambda_fn=lambda child: child.__class__.__name__ in no_split,
        ),
        device_id=local_rank,
        mixed_precision=MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        use_orig_params=False,
    )
