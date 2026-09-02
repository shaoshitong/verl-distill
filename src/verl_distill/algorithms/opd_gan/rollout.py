from typing import List, Optional

import torch


def broadcast_sigma(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return sigma.reshape(-1, *([1] * (x.ndim - 1)))


def cat_or_empty(
    values: List[torch.Tensor],
    reference: torch.Tensor,
) -> torch.Tensor:
    if values:
        return torch.cat(values)
    return torch.empty(0, device=reference.device, dtype=torch.float32)


def rollout_step_from_velocity(
    *,
    x_t: torch.Tensor,
    sigma_cur: torch.Tensor,
    sigma_next: torch.Tensor,
    velocity: torch.Tensor,
    noise: Optional[torch.Tensor] = None,
    rollout_stochast_ratio: float = 1.0,
) -> torch.Tensor:
    ratio = float(rollout_stochast_ratio)
    interval = sigma_cur - sigma_next
    if torch.any(interval <= 0):
        raise ValueError("student interval must have positive length")
    if ratio == 0.0:
        return x_t - broadcast_sigma(interval, x_t) * velocity

    sigma_cur_view = broadcast_sigma(sigma_cur.to(torch.float32), x_t)
    sigma_next_view = broadcast_sigma(sigma_next.to(torch.float32), x_t)
    x_float = x_t.to(torch.float32)
    velocity_float = velocity.to(torch.float32)
    x0_hat = x_float - sigma_cur_view * velocity_float
    z_hat = x_float + (1.0 - sigma_cur_view) * velocity_float
    if noise is None:
        noise = torch.randn_like(x_float, dtype=torch.float32)
    else:
        noise = noise.to(device=x_t.device, dtype=torch.float32)
    z_mix = z_hat * float(1.0 - ratio) ** 0.5 + noise * float(ratio) ** 0.5
    return (1.0 - sigma_next_view) * x0_hat + sigma_next_view * z_mix
