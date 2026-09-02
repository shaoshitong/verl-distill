from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn

ModelLike = Union[nn.Module, Callable]


class StandardOPD(torch.nn.Module):
    """Standard rollout-based OPD for flow-model distillation."""

    def __init__(
        self,
        transport_type: str = "Linear",
        num_train_timestep: int = 1000,
        num_student_steps: int = 4,
        teacher_micro_steps: int = 2,
        teacher_micro_step_schedule: Optional[Tuple[int, ...]] = None,
        timestep_shift: float = 3.0,
        initial_warmup_step_size: float = 0.005,
        initial_warmup_rollout: int = 1,
        loss_weight: float = 1.0,
        grad_norm_eps: float = 1e-6,
        sampling_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unknown StandardOPD config keys: {', '.join(sorted(kwargs.keys()))}")
        if transport_type != "Linear":
            raise ValueError(f"Unsupported transport_type={transport_type}; expected Linear")
        if int(num_student_steps) < 1:
            raise ValueError("num_student_steps must be >= 1")
        if int(teacher_micro_steps) < 1:
            raise ValueError("teacher_micro_steps must be >= 1")

        self.num_train_timestep = int(num_train_timestep)
        self.num_student_steps = int(num_student_steps)
        self.teacher_micro_steps = int(teacher_micro_steps)
        self.teacher_micro_step_schedule = self._normalize_teacher_micro_step_schedule(
            teacher_micro_step_schedule,
            self.num_student_steps,
            self.teacher_micro_steps,
        )
        if self.teacher_micro_step_schedule is None:
            self.teacher_micro_step_offsets = tuple(
                idx * self.teacher_micro_steps for idx in range(self.num_student_steps + 1)
            )
        else:
            offsets = [0]
            for value in self.teacher_micro_step_schedule:
                offsets.append(offsets[-1] + value)
            self.teacher_micro_step_offsets = tuple(offsets)
        self.total_teacher_micro_steps = int(self.teacher_micro_step_offsets[-1])
        self.timestep_shift = float(timestep_shift)
        self.initial_warmup_step_size = float(initial_warmup_step_size)
        self.initial_warmup_rollout = max(1, int(initial_warmup_rollout))
        self.loss_weight = float(loss_weight)
        self.grad_norm_eps = float(grad_norm_eps)
        self.default_sampler_kwargs = {
            "sampling_steps": int(sampling_steps or self.num_student_steps)
        }

    @staticmethod
    def _normalize_teacher_micro_step_schedule(
        schedule,
        num_student_steps: int,
        teacher_micro_steps: int,
    ) -> Optional[Tuple[int, ...]]:
        if schedule is None:
            return None
        if isinstance(schedule, str):
            values = tuple(int(part.strip()) for part in schedule.split(",") if part.strip())
        else:
            values = tuple(int(value) for value in schedule)
        if len(values) != int(num_student_steps):
            raise ValueError(
                "teacher_micro_step_schedule must contain num_student_steps "
                f"values, got {len(values)} for {num_student_steps} student steps"
            )
        if any(value < 1 for value in values):
            raise ValueError("teacher_micro_step_schedule values must be >= 1")
        expected_total = int(num_student_steps) * int(teacher_micro_steps)
        if sum(values) != expected_total:
            raise ValueError(
                "teacher_micro_step_schedule must sum to "
                "num_student_steps * teacher_micro_steps "
                f"({expected_total}), got {sum(values)}"
            )
        return values

    @staticmethod
    def _broadcast_sigma(sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return sigma.reshape(-1, *([1] * (x.ndim - 1)))

    def _apply_timestep_shift(self, timestep: torch.Tensor) -> torch.Tensor:
        if self.timestep_shift <= 1.0:
            return timestep

        out_dtype = timestep.dtype if torch.is_floating_point(timestep) else torch.float32
        t = timestep.to(torch.float32)
        t_norm = t / float(self.num_train_timestep)
        shifted = (
            self.timestep_shift
            * t_norm
            / (1.0 + (self.timestep_shift - 1.0) * t_norm)
            * float(self.num_train_timestep)
        )
        return shifted.clamp(0.0, float(self.num_train_timestep)).to(out_dtype)

    def _uniform_sigma_nodes(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        num_steps: int = 1,
    ) -> torch.Tensor:
        if int(num_steps) < 1:
            raise ValueError("num_steps must be >= 1")

        nodes = torch.linspace(
            float(self.num_train_timestep),
            0.0,
            int(num_steps) + 1,
            device=device,
            dtype=dtype,
        )
        return self._apply_timestep_shift(nodes) / float(self.num_train_timestep)

    def sigma_nodes(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        include_terminal_zero: bool = True,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        steps = int(num_steps or self.num_student_steps)
        if steps < 1:
            raise ValueError("num_steps must be >= 1")

        if self.teacher_micro_step_schedule is not None and steps == self.num_student_steps:
            teacher_nodes = self._uniform_sigma_nodes(
                device=device,
                dtype=dtype,
                num_steps=self.total_teacher_micro_steps,
            )
            indices = torch.tensor(
                self.teacher_micro_step_offsets,
                device=device,
                dtype=torch.long,
            )
            nodes = teacher_nodes.index_select(0, indices)
        else:
            nodes = self._uniform_sigma_nodes(
                device=device,
                dtype=dtype,
                num_steps=steps,
            )
        if not include_terminal_zero:
            nodes = nodes[:-1]
        if torch.any(nodes[:-1] <= nodes[1:]):
            raise ValueError("sigma nodes must be strictly descending")
        return nodes

    def teacher_sigma_nodes(
        self,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return self._uniform_sigma_nodes(
            device=device,
            dtype=dtype,
            num_steps=self.total_teacher_micro_steps,
        )

    def teacher_interval_micro_sigmas(
        self,
        teacher_nodes: torch.Tensor,
        student_step_idx: int,
    ) -> torch.Tensor:
        if student_step_idx < 0 or student_step_idx >= self.num_student_steps:
            raise ValueError(
                "student_step_idx must be in "
                f"[0, {self.num_student_steps - 1}], got {student_step_idx}"
            )
        start = self.teacher_micro_step_offsets[student_step_idx]
        end = self.teacher_micro_step_offsets[student_step_idx + 1]
        return teacher_nodes[start : end + 1]

    def _expand_sigma(
        self,
        sigma: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        sigma = sigma.to(device=device, dtype=torch.float32).flatten()
        if sigma.numel() == 1:
            sigma = sigma.expand(batch_size)
        if sigma.numel() != batch_size:
            raise ValueError(f"sigma must have 1 or {batch_size} values, got {sigma.numel()}")
        return sigma

    @torch.no_grad()
    def teacher_interval_target(
        self,
        teacher_model: ModelLike,
        x_t: torch.Tensor,
        sigma_cur: torch.Tensor,
        sigma_next: torch.Tensor,
        c: List[torch.Tensor],
        micro_sigmas: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x_t.shape[0]
        sigma_cur = self._expand_sigma(sigma_cur, batch_size, x_t.device)
        sigma_next = self._expand_sigma(sigma_next, batch_size, x_t.device)
        interval = sigma_cur - sigma_next
        if torch.any(interval <= 0):
            raise ValueError("teacher interval must have positive length")

        if micro_sigmas is None:
            micro_sigmas = torch.linspace(
                1.0,
                0.0,
                self.teacher_micro_steps + 1,
                device=x_t.device,
                dtype=torch.float32,
            )
            micro_sigmas = sigma_cur[:1] * micro_sigmas + sigma_next[:1] * (1.0 - micro_sigmas)
        else:
            micro_sigmas = micro_sigmas.to(device=x_t.device, dtype=torch.float32)
        micro_sigmas = micro_sigmas.flatten()
        if micro_sigmas.numel() < 2:
            raise ValueError("micro_sigmas must contain at least start and end values")
        if not torch.allclose(micro_sigmas[0].expand_as(sigma_cur), sigma_cur):
            raise ValueError("micro_sigmas must start at sigma_cur")
        if not torch.allclose(micro_sigmas[-1].expand_as(sigma_next), sigma_next):
            raise ValueError("micro_sigmas must end at sigma_next")
        if torch.any(micro_sigmas[:-1] <= micro_sigmas[1:]):
            raise ValueError("micro_sigmas must be strictly descending")

        x_start = x_t.detach()
        x_cur = x_start
        for idx in range(micro_sigmas.numel() - 1):
            t_cur = micro_sigmas[idx].expand(batch_size)
            t_next = micro_sigmas[idx + 1].expand(batch_size)
            dt = t_cur - t_next
            v_teacher = teacher_model(x_cur, t=t_cur, c=c)
            x_cur = x_cur - self._broadcast_sigma(dt, x_cur) * v_teacher

        target = x_cur.detach()
        return target, target

    @staticmethod
    def _pack_loss_stats(**values):
        stats = {}
        for key, value in values.items():
            tensor = value.detach().to(torch.float32)
            if tensor.ndim == 0:
                stats[key] = (
                    tensor,
                    torch.ones((), device=tensor.device, dtype=torch.float32),
                )
                continue

            finite = torch.isfinite(tensor)
            safe = torch.where(finite, tensor, torch.zeros_like(tensor))
            stats[key] = (
                safe.sum(),
                finite.to(torch.float32).sum().clamp_min(1.0),
            )
        return stats

    def training_step(
        self,
        student_model: ModelLike,
        teacher_model: ModelLike,
        latent_shape: Tuple[int, ...],
        c: List[torch.Tensor],
        step: Optional[int] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_loss_stats: bool = False,
        return_log_tensors: bool = False,
    ):
        del step
        device = c[0].device
        if initial_noise is None:
            x = torch.randn(tuple(latent_shape), device=device, dtype=torch.float32)
        else:
            x = initial_noise.to(device=device, dtype=torch.float32)

        batch_size = int(x.shape[0])
        nodes = self.sigma_nodes(device=device, dtype=torch.float32)
        teacher_nodes = self.teacher_sigma_nodes(device=device, dtype=torch.float32)
        if self.initial_warmup_step_size > 0.0:
            sigma_start = nodes[0].expand(batch_size)
            warmup_step_size = float(self.initial_warmup_step_size) / float(
                self.initial_warmup_rollout
            )
            with torch.no_grad():
                for _ in range(self.initial_warmup_rollout):
                    warm_velocity = student_model(x.detach(), t=sigma_start, c=c)
                    x = (x.detach() - warmup_step_size * warm_velocity).detach()

        losses = []
        sigma_cur_values = []
        sigma_next_values = []
        intervals = []
        student_velocity_abs = []
        student_next_abs = []
        teacher_next_abs = []

        for idx in range(self.num_student_steps):
            sigma_cur = nodes[idx].expand(batch_size)
            sigma_next = nodes[idx + 1].expand(batch_size)
            interval = sigma_cur - sigma_next
            if torch.any(interval <= 0):
                raise ValueError("student interval must have positive length")

            x_in = x.detach()
            v_student = student_model(x_in, t=sigma_cur, c=c)
            x_student_next = x_in - self._broadcast_sigma(interval, x_in) * v_student
            x_teacher_next, _ = self.teacher_interval_target(
                teacher_model=teacher_model,
                x_t=x_in,
                sigma_cur=sigma_cur,
                sigma_next=sigma_next,
                c=c,
                micro_sigmas=self.teacher_interval_micro_sigmas(teacher_nodes, idx),
            )
            per_step = F.mse_loss(
                x_student_next.float(),
                x_teacher_next.float(),
                reduction="mean",
            )
            losses.append(per_step)
            x = x_student_next.detach()

            sigma_cur_values.append(sigma_cur.detach())
            sigma_next_values.append(sigma_next.detach())
            intervals.append(interval.detach())
            student_velocity_abs.append(v_student.detach().abs().flatten(1).mean(dim=1))
            student_next_abs.append(x_student_next.detach().abs().flatten(1).mean(dim=1))
            teacher_next_abs.append(x_teacher_next.detach().abs().flatten(1).mean(dim=1))

        loss = torch.stack(losses).sum() * self.loss_weight
        stats = self._pack_loss_stats(
            loss_opd=loss.detach(),
            opd_student_velocity_abs=torch.cat(student_velocity_abs),
            opd_student_next_abs=torch.cat(student_next_abs),
            opd_teacher_next_abs=torch.cat(teacher_next_abs),
            opd_teacher_interval=torch.cat(intervals),
            opd_sigma_cur=torch.cat(sigma_cur_values),
            opd_sigma_next=torch.cat(sigma_next_values),
        )
        for idx, value in enumerate(losses):
            stats.update(self._pack_loss_stats(**{f"opd_step_{idx}_loss": value.detach()}))

        if return_loss_stats and return_log_tensors:
            aux = {
                "opd_sigma_cur": torch.cat(sigma_cur_values).detach(),
                "opd_sigma_next": torch.cat(sigma_next_values).detach(),
            }
            return loss, stats, aux
        if return_loss_stats:
            return loss, stats
        return loss

    @torch.no_grad()
    def sampling_loop(
        self,
        initial_noise_z: torch.FloatTensor,
        sampling_model: ModelLike,
        sampling_steps: Optional[int] = None,
        **model_kwargs,
    ):
        input_dtype = initial_noise_z.dtype
        steps = int(sampling_steps or self.num_student_steps)
        x = initial_noise_z.to(torch.float32)
        nodes = self.sigma_nodes(
            device=initial_noise_z.device,
            dtype=torch.float32,
            num_steps=steps,
        )
        if self.initial_warmup_step_size > 0.0:
            sigma_start = nodes[0].expand(x.shape[0])
            warmup_step_size = float(self.initial_warmup_step_size) / float(
                self.initial_warmup_rollout
            )
            for _ in range(self.initial_warmup_rollout):
                warm_pred = sampling_model(
                    x.to(input_dtype),
                    t=sigma_start.to(input_dtype),
                    **model_kwargs,
                )
                x = x - warmup_step_size * warm_pred.to(torch.float32)

        samples = [x.to(input_dtype).detach().cpu()]

        for idx in range(steps):
            sigma_cur = nodes[idx].expand(x.shape[0])
            sigma_next = nodes[idx + 1].expand(x.shape[0])
            dt = sigma_cur - sigma_next
            if torch.any(dt <= 0):
                raise ValueError("sampling interval must have positive length")
            pred = sampling_model(
                x.to(input_dtype),
                t=sigma_cur.to(input_dtype),
                **model_kwargs,
            )
            x = x - self._broadcast_sigma(dt, x) * pred.to(torch.float32)
            samples.append(x.to(input_dtype).detach().cpu())

        return torch.stack(samples, dim=0).to(input_dtype)
