from typing import Dict, Tuple

import torch


def pair_diff_stats(prefix: str, fake: torch.Tensor, real: torch.Tensor) -> dict:
    diff = fake.float() - real.detach().float()
    flat = diff.flatten(1)
    mse = flat.square().mean(dim=1)
    return {
        f"{prefix}_l1": flat.abs().mean(dim=1),
        f"{prefix}_mse": mse,
        f"{prefix}_rmse": mse.sqrt(),
    }


def dual_alignment_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    mse_weight: float,
    pearson_eps: float,
    pearson_mode: str = "channel",
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    if tuple(pred.shape) != tuple(target.shape):
        raise ValueError(
            f"dual alignment shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}"
        )
    pred = pred.to(torch.float32)
    target = target.detach().to(torch.float32)
    pearson_mode = str(pearson_mode).lower()
    if pearson_mode == "flatten":
        pred_for_pearson = pred.flatten(1)
        target_for_pearson = target.flatten(1)
    elif pearson_mode == "channel":
        pred_for_pearson = pred
        target_for_pearson = target
    else:
        raise ValueError("pearson_mode must be 'channel' or 'flatten'")
    pred_centered = pred_for_pearson - pred_for_pearson.mean(dim=1, keepdim=True)
    target_centered = target_for_pearson - target_for_pearson.mean(dim=1, keepdim=True)
    pred_std = pred_centered.square().mean(dim=1, keepdim=True).sqrt()
    target_std = target_centered.square().mean(dim=1, keepdim=True).sqrt()
    pearson = (
        (
            pred_centered / (pred_std + float(pearson_eps))
            - target_centered / (target_std + float(pearson_eps))
        )
        .square()
        .mean()
    )
    mse = (pred - target).square().mean()
    total = pearson + float(mse_weight) * mse
    return total, {"pearson": pearson.detach(), "mse": mse.detach()}
