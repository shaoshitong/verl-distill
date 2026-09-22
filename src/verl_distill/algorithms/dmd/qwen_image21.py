"""Qwen discrete-time REFLOW and output-space DMD mathematics."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


def broadcast_sigma(sigma, reference):
    return sigma.float().reshape(-1, *([1] * (reference.ndim - 1)))


def renoise(clean, noise, sigma):
    s = broadcast_sigma(sigma, clean)
    return (1 - s) * clean.float() + s * noise.float()


def x0_from_velocity(noisy, velocity, sigma):
    return noisy.float() - broadcast_sigma(sigma, noisy) * velocity.float()


def epsilon_from_velocity(noisy, velocity, sigma):
    return noisy.float() + (1 - broadcast_sigma(sigma, noisy)) * velocity.float()


def reflow_loss(velocity, clean, noise):
    return (velocity.float() - (noise.float() - clean.float())).square().flatten(1).mean(1).mean()


def fake_score_loss(noisy, velocity, target, noise, sigma, *, max_weight=50.,
                    min_alpha=1e-4, loss_weight=1.):
    """Explicit Qwen choice: epsilon-equivalent x0 weighting, capped before reduction.

    alpha here is the NOISE coefficient sigma (not the clean coefficient 1-sigma).
    Uncapped: ((1-sigma)/sigma)^2 * MSE(x0_hat,target) == MSE(eps_hat,eps).
    The cap deliberately changes that objective at low sigma. This is not claimed
    to reproduce an unavailable attempt25 weighting function.
    """
    target = target.detach().float()
    pred = x0_from_velocity(noisy.detach(), velocity, sigma)
    eps = epsilon_from_velocity(noisy.detach(), velocity, sigma)
    s = sigma.float().flatten()
    raw_weight = ((1 - s) / s.clamp_min(min_alpha)).square()
    weight = raw_weight.clamp_max(max_weight)
    mse = (pred - target).square().flatten(1).mean(1)
    loss = (mse * weight).mean() * loss_weight
    return loss, {"fake_x0": pred, "epsilon_pred": eps, "epsilon_target": noise.detach(),
                  "x0_mse": mse.detach(), "epsilon_mse": (eps.detach() - noise).square().mean(),
                  "weight_raw": raw_weight.detach(), "weight": weight.detach(),
                  "cap_fraction": (raw_weight > max_weight).float().mean(),
                  "loss_unweighted": mse.detach().mean(), "loss_weighted": loss.detach()}


def dmd_surrogate(generated, noisy, fake_velocity, real_velocity, sigma, *,
                  loss_weight=4., normalization_eps=1e-6):
    with torch.no_grad():
        fake = x0_from_velocity(noisy, fake_velocity, sigma)
        real = x0_from_velocity(noisy, real_velocity, sigma)
        diff = fake - real
        denominator = (generated.detach().float() - real).abs().flatten(1).mean(1)
        denominator = denominator.clamp_min(normalization_eps)
        direction = diff / denominator.reshape(-1, *([1] * (diff.ndim - 1)))
        target = generated.detach().float() - direction
    unweighted = 0.5 * (generated.float() - target).square().flatten(1).mean(1).mean()
    loss = unweighted * loss_weight
    return loss, {"fake_x0": fake, "real_x0": real, "diff_x0": diff,
                  "denominator": denominator, "normalized_direction": direction,
                  "loss_unweighted": unweighted.detach(), "loss_weighted": loss.detach()}


@dataclass
class UpdateState:
    reflow_updates: int = 0
    fake_updates: int = 0
    generator_updates: int = 0
    dmd_initialized: bool = False

    @property
    def outer(self):
        return self.generator_updates

    def next_phase(self, reflow_total, fake_total, ratio=5):
        if self.reflow_updates < reflow_total:
            return "reflow"
        # The final Generator update is still due when fake_updates==fake_total.
        if self.fake_updates >= ratio * (self.generator_updates + 1):
            return "generator"
        if self.fake_updates < fake_total:
            return "fake_score"
        return "complete"

    def advance(self, phase):
        key = {"reflow": "reflow_updates", "fake_score": "fake_updates",
               "generator": "generator_updates"}[phase]
        setattr(self, key, getattr(self, key) + 1)

    def validate(self, reflow_total, fake_total, ratio=5):
        if any(type(n) is not int for n in (self.reflow_updates, self.fake_updates, self.generator_updates)):
            raise ValueError("Update counters must be integers")
        if self.generator_updates < 0:
            raise ValueError("Invalid Generator counter")
        if not 0 <= self.reflow_updates <= reflow_total:
            raise ValueError("Invalid REFLOW counter")
        if not 0 <= self.fake_updates <= fake_total:
            raise ValueError("Invalid Fake counter")
        if not 0 <= self.fake_updates - self.generator_updates * ratio <= ratio:
            raise ValueError("Inconsistent Fake/Generator counters")
        if self.fake_updates and (self.reflow_updates != reflow_total or not self.dmd_initialized):
            raise ValueError("DMD counters precede phase initialization")
        if self.dmd_initialized and self.reflow_updates != reflow_total:
            raise ValueError("DMD initialized before REFLOW completion")

    def state_dict(self):
        return asdict(self)
