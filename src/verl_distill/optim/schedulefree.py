# Copyright (c) Meta Platforms, Inc. and affiliates.
# Licensed under the MIT license in the referenced DMD implementation.
#
# This file adapts the local DMD reference implementation of Schedule-Free AdamW
# for use inside verl-distill.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch

ParamsT = Iterable[torch.Tensor] | Iterable[dict[str, Any]]


class AdamWScheduleFree(torch.optim.Optimizer):
    """Schedule-Free AdamW.

    Adapted from the local DMD reference implementation.
    Call `train()` before optimization steps and `eval()` before checkpointing or
    evaluation to materialize the averaged weights.
    """

    def __init__(
        self,
        params: ParamsT,
        lr: float | torch.Tensor = 0.0025,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        warmup_steps: int = 0,
        r: float = 0.0,
        weight_lr_power: float = 2.0,
        inner_momentum: float = 0.0,
        foreach: bool | None = hasattr(torch, "_foreach_mul_"),
    ):
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "r": r,
            "k": 0,
            "warmup_steps": warmup_steps,
            "train_mode": False,
            "weight_sum": 0.0,
            "lr_max": -1.0,
            "scheduled_lr": 0.0,
            "weight_lr_power": weight_lr_power,
            "weight_decay": weight_decay,
            "inner_momentum": inner_momentum,
            "foreach": foreach,
        }
        super().__init__(params, defaults)

    def _init_param_state(self, parameter: torch.Tensor, group: dict[str, Any]) -> None:
        state = self.state[parameter]
        if "z" in state:
            return
        beta1, _ = group["betas"]
        inner_momentum = group["inner_momentum"]
        state["z"] = torch.clone(parameter, memory_format=torch.preserve_format)
        if beta1 == 0:
            state["x"] = torch.clone(parameter, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
        if inner_momentum != 0:
            state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

    @torch.no_grad()
    def initialize_missing_state(self) -> None:
        """Materialize lazy per-parameter state before FSDP state-dict operations."""
        for group in self.param_groups:
            for parameter in group["params"]:
                self._init_param_state(parameter, group)

    @torch.no_grad()
    def eval(self):
        for group in self.param_groups:
            if not group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            if beta1 == 0:
                for parameter in group["params"]:
                    state = self.state.get(parameter)
                    if state is not None and "x" in state:
                        parameter.copy_(state["x"].to(parameter.device))
                group["train_mode"] = False
                continue
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state is not None and "z" in state:
                    parameter.lerp_(end=state["z"].to(parameter.device), weight=1 - 1 / beta1)
            group["train_mode"] = False

    @torch.no_grad()
    def train(self):
        for group in self.param_groups:
            if group["train_mode"]:
                continue
            beta1, _ = group["betas"]
            if beta1 == 0:
                for parameter in group["params"]:
                    state = self.state.get(parameter)
                    if state is not None and "z" in state:
                        parameter.copy_(state["z"].to(parameter.device))
                group["train_mode"] = True
                continue
            for parameter in group["params"]:
                state = self.state.get(parameter)
                if state is not None and "z" in state:
                    parameter.lerp_(end=state["z"].to(parameter.device), weight=1 - beta1)
            group["train_mode"] = True

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if not self.param_groups[0]["train_mode"]:
            raise RuntimeError(
                "AdamWScheduleFree.step() requires optimizer.train() before stepping"
            )

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            decay = group["weight_decay"]
            k = group["k"]
            r = group["r"]
            warmup_steps = group["warmup_steps"]
            weight_lr_power = group["weight_lr_power"]
            inner_momentum = group["inner_momentum"]

            sched = (k + 1) / warmup_steps if k < warmup_steps else 1.0
            bias_correction2 = 1 - beta2 ** (k + 1)
            if inner_momentum != 0:
                bias_correction1 = 1 - inner_momentum ** (k + 1)
            lr = group["lr"] * sched
            group["scheduled_lr"] = lr
            lr_max = group["lr_max"] = max(lr, group["lr_max"])

            weight = ((k + 1) ** r) * (lr_max**weight_lr_power)
            group["weight_sum"] = group["weight_sum"] + weight
            weight_sum = group["weight_sum"]
            ckp1 = weight / weight_sum if weight_sum else 0.0

            active_parameters = [
                parameter for parameter in group["params"] if parameter.grad is not None
            ]
            for parameter in active_parameters:
                self._init_param_state(parameter, group)

            if group["foreach"] and active_parameters:
                y, grad, exp_avg_sq, z = zip(
                    *[
                        (
                            parameter,
                            parameter.grad,
                            self.state[parameter]["exp_avg_sq"],
                            self.state[parameter]["z"],
                        )
                        for parameter in active_parameters
                    ]
                )

                torch._foreach_mul_(exp_avg_sq, beta2)
                torch._foreach_addcmul_(exp_avg_sq, grad, grad, value=1 - beta2)
                denom = torch._foreach_div(exp_avg_sq, bias_correction2)
                torch._foreach_sqrt_(denom)
                torch._foreach_add_(denom, eps)

                if inner_momentum != 0:
                    exp_avg = tuple(
                        self.state[parameter]["exp_avg"] for parameter in active_parameters
                    )
                    torch._foreach_mul_(exp_avg, inner_momentum)
                    torch._foreach_add_(exp_avg, grad, alpha=1 - inner_momentum)
                    grad_normalized = torch._foreach_div(exp_avg, bias_correction1)
                    torch._foreach_div_(grad_normalized, denom)
                else:
                    torch._foreach_div_(grad, denom)
                    grad_normalized = grad

                if decay != 0:
                    torch._foreach_add_(grad_normalized, y, alpha=decay)

                torch._foreach_lerp_(y, z, weight=ckp1)
                torch._foreach_add_(y, grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                torch._foreach_sub_(z, grad_normalized, alpha=lr)
                if beta1 == 0:
                    x_state = tuple(self.state[parameter]["x"] for parameter in active_parameters)
                    torch._foreach_lerp_(x_state, z, weight=ckp1)
            else:
                for parameter in active_parameters:
                    y = parameter
                    grad = parameter.grad
                    state = self.state[parameter]
                    z = state["z"]
                    x_state = state.get("x")
                    exp_avg_sq = state["exp_avg_sq"]

                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    denom = exp_avg_sq.div(bias_correction2).sqrt_().add_(eps)

                    if inner_momentum != 0:
                        exp_avg = state["exp_avg"]
                        exp_avg.mul_(inner_momentum).add_(grad, alpha=1 - inner_momentum)
                        grad_normalized = exp_avg.div(bias_correction1).div_(denom)
                    else:
                        grad_normalized = grad.div_(denom)

                    if decay != 0:
                        grad_normalized.add_(y, alpha=decay)

                    y.lerp_(end=z, weight=ckp1)
                    y.add_(grad_normalized, alpha=lr * (beta1 * (1 - ckp1) - 1))
                    z.sub_(grad_normalized, alpha=lr)
                    if beta1 == 0 and x_state is not None:
                        x_state.lerp_(end=z, weight=ckp1)

            group["k"] = k + 1
        return loss
