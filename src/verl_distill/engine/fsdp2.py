from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = None

ZIMAGE_BLOCK_ATTR_NAMES = ("layers", "noise_refiner", "context_refiner")


def config_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def apply_zimage_fsdp2(transformer, *, param_dtype=None, reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    for attr_name in ZIMAGE_BLOCK_ATTR_NAMES:
        blocks = getattr(transformer, attr_name, None)
        if blocks is None:
            continue
        for block in blocks:
            fully_shard(block, mp_policy=mp_policy, reshard_after_forward=True)
    for attr_name in (
        "multi_feature_discriminator_head",
        "dual_projector_multi_feature_discriminator_head",
        "dual_projector_multi_feature_discriminator_head_exit2",
    ):
        head = getattr(transformer, attr_name, None)
        if head is not None:
            fully_shard(head, mp_policy=mp_policy, reshard_after_forward=True)
    fully_shard(transformer, mp_policy=mp_policy, reshard_after_forward=False)


@torch.no_grad()
def clip_grad_norm(parameters, max_norm: float) -> float:
    """Clip regular or FSDP2-sharded gradients using one global L2 norm."""
    gradients = []
    for parameter in parameters:
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if DTensor is not None and isinstance(gradient, DTensor):
            gradient = gradient.to_local()
        gradients.append(gradient)
    if not gradients:
        return 0.0
    total_squared = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        total_squared.add_(gradient.detach().float().square().sum())
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(total_squared, op=dist.ReduceOp.SUM)
    norm = total_squared.sqrt()
    if max_norm > 0:
        coefficient = (float(max_norm) / (norm + 1e-6)).clamp(max=1.0)
        for gradient in gradients:
            gradient.mul_(coefficient.to(dtype=gradient.dtype))
    return float(norm.item())
