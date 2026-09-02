from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from verl_distill.engine.fsdp2 import clip_grad_norm


@dataclass
class StepResult:
    loss: torch.Tensor
    stats: dict[str, Any] = field(default_factory=dict)
    phase: str = "generator"
    updated: tuple[str, ...] = ()


def _loss_and_stats(output):
    if not isinstance(output, tuple):
        return output, {}
    return output[0], output[1]


def _backward_step(loss, optimizer, parameters, max_grad_norm: float):
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    if max_grad_norm > 0:
        clip_grad_norm(parameters, max_grad_norm)
    optimizer.step()


def opd_phase_for_step(method, step_index: int) -> str:
    if method._any_generator_objective_enabled() and not method._discriminator_adv_enabled():
        return "generator"
    switch_step = getattr(method, "phase_schedule_switch_step", None)
    pattern = getattr(method, "phase_schedule_after_pattern", None)
    if switch_step is not None and pattern and step_index >= switch_step:
        return str(pattern[(step_index - switch_step) % len(pattern)])
    discriminator_steps = int(method.discriminator_update_ratio)
    cycle = discriminator_steps + 1
    return "discriminator" if step_index % cycle < discriminator_steps else "generator"


class DMDStepRunner:
    def __init__(self, method, generator_optimizer, score_optimizer, max_grad_norm=1.0):
        self.method = method
        self.generator_optimizer = generator_optimizer
        self.score_optimizer = score_optimizer
        self.max_grad_norm = float(max_grad_norm)

    def run(self, *, step: int, generator_model, score_model, x_real, c, e):
        self.method.set_train_step(step)
        score_output = self.method.score_loss(
            generator_model=generator_model,
            score_model=score_model,
            x_real=x_real,
            c=c,
            e=e,
            latent_shape=x_real.shape,
        )
        score_loss, score_stats = _loss_and_stats(score_output)
        score_parameters = (
            score_model.parameters()
            if hasattr(score_model, "parameters")
            else score_model["fake"].parameters()
        )
        _backward_step(score_loss, self.score_optimizer, score_parameters, self.max_grad_norm)
        updated = ["score"]
        stats = dict(score_stats)
        total = score_loss.detach()
        if self.method.should_update_generator(step):
            generator_output = self.method.generator_loss(
                generator_model=generator_model,
                score_model=score_model,
                x_real=x_real,
                c=c,
                e=e,
                latent_shape=x_real.shape,
            )
            generator_loss, generator_stats = _loss_and_stats(generator_output)
            _backward_step(
                generator_loss,
                self.generator_optimizer,
                generator_model.parameters(),
                self.max_grad_norm,
            )
            updated.append("generator")
            stats.update(generator_stats)
            total = total + generator_loss.detach()
        return StepResult(total, stats, updated=tuple(updated))


class MeanFlowStepRunner:
    def __init__(self, method, optimizer, max_grad_norm=1.0):
        self.method = method
        self.optimizer = optimizer
        self.max_grad_norm = float(max_grad_norm)

    def run(self, *, step: int, student_model, teacher_model, latents, c, e):
        output = self.method.training_step(
            student_model,
            latents,
            c=c,
            e=e,
            step=step,
            teacher=teacher_model,
            return_loss_stats=True,
        )
        loss, stats = _loss_and_stats(output)
        _backward_step(loss, self.optimizer, student_model.parameters(), self.max_grad_norm)
        return StepResult(loss.detach(), stats, updated=("generator",))


class OPDGANStepRunner:
    def __init__(
        self,
        method,
        generator_optimizer,
        discriminator_optimizer,
        max_grad_norm=1.0,
    ):
        self.method = method
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.max_grad_norm = float(max_grad_norm)
        self.generator_update_index = 0

    def run(
        self,
        *,
        step: int,
        student_model,
        teacher_model,
        discriminator_model,
        latent_shape,
        c,
        initial_noise=None,
        real_image_latents=None,
    ):
        phase = opd_phase_for_step(self.method, step)
        kwargs = {}
        if phase == "generator":
            self.generator_update_index += 1
            kwargs["generator_update_index"] = self.generator_update_index
        output = self.method.training_step(
            student_model=student_model,
            teacher_model=teacher_model,
            discriminator_model=discriminator_model,
            latent_shape=tuple(latent_shape),
            c=c,
            step=step,
            initial_noise=initial_noise,
            real_image_latents=real_image_latents,
            phase=phase,
            return_loss_stats=True,
            **kwargs,
        )
        loss, stats = _loss_and_stats(output)
        if phase == "generator":
            optimizer = self.generator_optimizer
            parameters = student_model.parameters()
        else:
            optimizer = self.discriminator_optimizer
            parameters = discriminator_model.parameters()
        _backward_step(loss, optimizer, parameters, self.max_grad_norm)
        return StepResult(loss.detach(), stats, phase=phase, updated=(phase,))
