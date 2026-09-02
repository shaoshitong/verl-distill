from contextlib import contextmanager
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

ModelLike = Union[nn.Module, Callable]
Conditioning = Optional[List[torch.Tensor]]
LossStats = Dict[str, Tuple[torch.Tensor, torch.Tensor]]


@contextmanager
def _temporary_eval(module: ModelLike):
    if not isinstance(module, nn.Module):
        yield
        return

    training_states = {submodule: submodule.training for submodule in module.modules()}
    module.eval()
    for submodule in training_states:
        submodule.training = False
    try:
        yield
    finally:
        for submodule, was_training in training_states.items():
            submodule.training = was_training


def _stat(value: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    tensor = torch.as_tensor(value).detach().float()
    return tensor, torch.ones((), device=tensor.device, dtype=torch.float32)


def _broadcast(values: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    return values.reshape(-1, *([1] * (reference.ndim - 1))).to(
        device=reference.device,
        dtype=reference.dtype,
    )


def _tensor_summary(prefix: str, value: torch.Tensor) -> LossStats:
    tensor = value.detach().float()
    finite = torch.isfinite(tensor)
    safe = torch.where(finite, tensor, torch.zeros_like(tensor))
    count = finite.float().sum().clamp_min(1.0)
    mean = safe.sum() / count
    centered = torch.where(finite, tensor - mean, torch.zeros_like(tensor))
    rms = torch.sqrt((safe.square().sum() / count).clamp_min(0.0))
    std = torch.sqrt((centered.square().sum() / count).clamp_min(0.0))
    finite_frac = finite.float().mean() if finite.numel() else torch.ones_like(mean)
    return {
        f"{prefix}_mean": (mean.detach(), torch.ones_like(mean.detach())),
        f"{prefix}_std": (std.detach(), torch.ones_like(std.detach())),
        f"{prefix}_rms": (rms.detach(), torch.ones_like(rms.detach())),
        f"{prefix}_finite_frac": (
            finite_frac.detach(),
            torch.ones_like(finite_frac.detach()),
        ),
    }


class ZImageMeanFlow(nn.Module):
    """MeanFlow training helper for Z-Image-Base style flow models."""

    def __init__(
        self,
        loss_mode: str = "finite_difference",
        align_type: str = "split_fd_fm_terminal_anchor_v3",
        fd_delta: float = 0.005,
        teacher_cfg_scale: float = 2.5,
        flow_shift: float = 3.0,
        rt_sampler: str = "gap_curriculum_split_nodiag",
        rt_curriculum_steps: int = 50000,
        gap_min: float = 0.02,
        gap_max_start: float = 1.0,
        gap_max_end: float = 0.3,
        split_gap_min: float = 1.0e-4,
        split_gap_max_start: float = 0.10,
        split_gap_max_end: float = 1.0,
        split_alpha_min: float = 0.25,
        split_alpha_max: float = 0.75,
        two_res_lambda_t: float = 1.0,
        two_res_lambda_s: float = 1.0,
        fm_lambda: float = 1.0,
        split_fd_terminal_cfg_scale_start: float = -0.5,
        split_fd_terminal_cfg_scale_end: float = 0.5,
        split_fd_terminal_cfg_scale_steps: int = 15000,
        two_res_gamma_weight_max: float = 20.0,
        sampling_steps: int = 4,
        transport_type: str = "Linear",
        rt_discrete_grid: bool = False,
        rt_grid_step: Optional[float] = None,
        loss_reduction: str = "sum",
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise ValueError(
                f"Unknown ZImageMeanFlow config keys: {', '.join(sorted(kwargs.keys()))}"
            )
        if loss_mode != "finite_difference":
            raise ValueError("ZImageMeanFlow only supports loss_mode='finite_difference'")
        if align_type != "split_fd_fm_terminal_anchor_v3":
            raise ValueError(
                "ZImageMeanFlow only supports align_type='split_fd_fm_terminal_anchor_v3'"
            )
        if rt_sampler != "gap_curriculum_split_nodiag":
            raise ValueError(
                "ZImageMeanFlow only supports rt_sampler='gap_curriculum_split_nodiag'"
            )
        if transport_type != "Linear":
            raise ValueError(f"Unsupported transport_type={transport_type}; expected Linear")
        if flow_shift < 1.0:
            raise ValueError("flow_shift must be >= 1.0")
        if fd_delta <= 0.0:
            raise ValueError("fd_delta must be positive")
        if rt_curriculum_steps < 0:
            raise ValueError("rt_curriculum_steps must be non-negative")
        if gap_min < 0.0:
            raise ValueError("gap_min must be non-negative")
        if gap_max_start <= 0.0 or gap_max_end <= 0.0:
            raise ValueError("gap_max_start/end must be positive")
        if split_gap_min < 0.0:
            raise ValueError("split_gap_min must be non-negative")
        if split_gap_max_start <= 0.0 or split_gap_max_end <= 0.0:
            raise ValueError("split_gap_max_start/end must be positive")
        if not 0.0 <= split_alpha_min <= split_alpha_max <= 1.0:
            raise ValueError("split_alpha_min/max must satisfy 0 <= min <= max <= 1")
        if two_res_lambda_t < 0.0 or two_res_lambda_s < 0.0:
            raise ValueError("two_res_lambda_t/s must be non-negative")
        if split_fd_terminal_cfg_scale_steps < 0:
            raise ValueError("split_fd_terminal_cfg_scale_steps must be non-negative")
        if two_res_gamma_weight_max <= 0.0:
            raise ValueError("two_res_gamma_weight_max must be positive")
        if sampling_steps < 1:
            raise ValueError("sampling_steps must be >= 1")
        if loss_reduction not in {"sum", "mean"}:
            raise ValueError("loss_reduction must be 'sum' or 'mean'")
        if rt_grid_step is not None and float(rt_grid_step) <= 0.0:
            raise ValueError("rt_grid_step must be positive when provided")
        if rt_discrete_grid:
            if rt_grid_step is None:
                raise ValueError("rt_grid_step is required when rt_discrete_grid=true")
            if float(rt_grid_step) > 1.0:
                raise ValueError("rt_grid_step must be <= 1.0")
            grid_intervals = round(1.0 / float(rt_grid_step))
            grid_error = abs(grid_intervals * float(rt_grid_step) - 1.0)
            if grid_intervals < 1 or grid_error > 1.0e-6:
                raise ValueError("rt_grid_step must evenly divide [0, 1]")

        self.loss_mode = loss_mode
        self.align_type = align_type
        self.fd_delta = float(fd_delta)
        self.rt_discrete_grid = bool(rt_discrete_grid)
        self.rt_grid_step = float(rt_grid_step) if rt_grid_step is not None else float(fd_delta)
        self.loss_reduction = loss_reduction
        self.teacher_cfg_scale = float(teacher_cfg_scale)
        self.flow_shift = float(flow_shift)
        self.rt_sampler = rt_sampler
        self.rt_curriculum_steps = int(rt_curriculum_steps)
        self.gap_min = float(gap_min)
        self.gap_max_start = float(gap_max_start)
        self.gap_max_end = float(gap_max_end)
        self.split_gap_min = float(split_gap_min)
        self.split_gap_max_start = float(split_gap_max_start)
        self.split_gap_max_end = float(split_gap_max_end)
        self.split_alpha_min = float(split_alpha_min)
        self.split_alpha_max = float(split_alpha_max)
        self.two_res_lambda_t = float(two_res_lambda_t)
        self.two_res_lambda_s = float(two_res_lambda_s)
        self.fm_lambda = float(fm_lambda)
        self.split_fd_terminal_cfg_scale_start = float(split_fd_terminal_cfg_scale_start)
        self.split_fd_terminal_cfg_scale_end = float(split_fd_terminal_cfg_scale_end)
        self.split_fd_terminal_cfg_scale_steps = int(split_fd_terminal_cfg_scale_steps)
        self.two_res_gamma_weight_max = float(two_res_gamma_weight_max)
        self.train_step = 0
        self.default_sampler_kwargs = {"sampling_steps": int(sampling_steps)}

    def _effective_fd_delta(self) -> float:
        if self.rt_discrete_grid:
            return self.rt_grid_step
        return self.fd_delta

    def _per_sample_error(self, error: torch.Tensor) -> torch.Tensor:
        squared = error.square().flatten(1)
        if self.loss_reduction == "mean":
            return squared.mean(dim=1)
        return squared.sum(dim=1)

    def scheduled_terminal_cfg_scale(self, step: int) -> float:
        if self.split_fd_terminal_cfg_scale_steps <= 0:
            return self.split_fd_terminal_cfg_scale_end

        frac = min(
            1.0,
            max(0.0, float(step) / float(self.split_fd_terminal_cfg_scale_steps)),
        )
        start = self.split_fd_terminal_cfg_scale_start
        end = self.split_fd_terminal_cfg_scale_end
        return start + frac * (end - start)

    def apply_flow_shift(
        self,
        sigma: torch.Tensor,
        flow_shift: Optional[float] = None,
    ) -> torch.Tensor:
        shift = self.flow_shift if flow_shift is None else float(flow_shift)
        if shift <= 1.0:
            return sigma

        dtype = sigma.dtype
        shifted = shift * sigma / (1.0 + (shift - 1.0) * sigma)
        return shifted.clamp(0.0, 1.0).to(dtype=dtype)

    def _current_gap_max(self, step: int) -> float:
        if self.rt_curriculum_steps <= 0:
            return self.gap_max_end
        frac = min(1.0, max(0.0, float(step) / float(self.rt_curriculum_steps)))
        return self.gap_max_start + frac * (self.gap_max_end - self.gap_max_start)

    def _sample_rt(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
        step: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        fd_delta = self._effective_fd_delta()
        current_gap_max = self._current_gap_max(step)
        if self.rt_discrete_grid:
            gap_max = max(fd_delta, current_gap_max)
        else:
            gap_max = max(self.gap_min + fd_delta, current_gap_max)
        gap_max = min(1.0, gap_max)
        gap_min = min(self.gap_min, gap_max - fd_delta)
        gap_min = max(fd_delta, gap_min)

        if self.rt_discrete_grid:
            grid_step = self.rt_grid_step
            grid_intervals = int(round(1.0 / grid_step))
            min_gap_steps = max(
                1,
                int(torch.ceil(torch.tensor(gap_min / grid_step)).item()),
            )
            max_gap_steps = max(
                min_gap_steps,
                int(torch.floor(torch.tensor(gap_max / grid_step)).item()),
            )
            max_gap_steps = min(grid_intervals, max_gap_steps)
            gap_steps = torch.randint(
                min_gap_steps,
                max_gap_steps + 1,
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            max_r_steps = grid_intervals - gap_steps
            r_steps = torch.floor(
                torch.rand(batch_size, device=device) * (max_r_steps + 1).float()
            ).to(torch.long)
            t_steps = r_steps + gap_steps
            r_raw = (r_steps.to(dtype=dtype) * grid_step).clamp(0.0, 1.0)
            t_raw = (t_steps.to(dtype=dtype) * grid_step).clamp(0.0, 1.0)
            r = self.apply_flow_shift(r_raw)
            t = self.apply_flow_shift(t_raw)
            return r, t, r_raw, t_raw

        gap = torch.empty(batch_size, device=device, dtype=dtype).uniform_(gap_min, gap_max)
        t_raw = gap + torch.rand(batch_size, device=device, dtype=dtype) * (1.0 - gap)
        r_raw = (t_raw - gap).clamp(0.0, 1.0)

        r = self.apply_flow_shift(r_raw)
        t = self.apply_flow_shift(t_raw)
        return r, t, r_raw, t_raw

    def _sample_split(
        self,
        r: torch.Tensor,
        t: torch.Tensor,
        r_raw: torch.Tensor,
        t_raw: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.rt_discrete_grid:
            grid_step = self.rt_grid_step
            r_steps = torch.round(r_raw.float() / grid_step).to(torch.long)
            t_steps = torch.round(t_raw.float() / grid_step).to(torch.long)
            gap_steps = (t_steps - r_steps).clamp_min(1)
            split_mask = gap_steps >= 2
            max_offsets = (gap_steps - 1).clamp_min(1)
            offsets = (
                torch.floor(torch.rand(r_raw.shape, device=r_raw.device) * max_offsets.float()).to(
                    torch.long
                )
                + 1
            )
            s_steps = torch.minimum(r_steps + offsets, t_steps)
            s_raw = (s_steps.to(dtype=r_raw.dtype) * grid_step).clamp(0.0, 1.0)
            return self.apply_flow_shift(s_raw), s_raw, split_mask

        alpha = torch.empty_like(r_raw).uniform_(
            self.split_alpha_min,
            self.split_alpha_max,
        )
        s = r + alpha * (t - r)
        s_raw = r_raw + alpha * (t_raw - r_raw)
        split_mask = torch.ones_like(r_raw, dtype=torch.bool)
        return s, s_raw, split_mask

    @staticmethod
    def _unwrap(model: ModelLike) -> ModelLike:
        return getattr(model, "module", model)

    @staticmethod
    def _as_tensor_output(output):
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (tuple, list)) and output:
            return output[0]
        if hasattr(output, "sample"):
            return output.sample
        raise TypeError(f"Expected tensor-like model output, got {type(output)!r}")

    def _call_student(
        self,
        model: ModelLike,
        z: torch.Tensor,
        r: torch.Tensor,
        t: torch.Tensor,
        c: Conditioning,
        call_tag: str,
    ) -> torch.Tensor:
        r_flat = r.flatten().to(device=z.device, dtype=z.dtype)
        t_flat = t.flatten().to(device=z.device, dtype=z.dtype)
        base_model = self._unwrap(model)
        if getattr(model, "accepts_aux_time_meta", False) or getattr(
            base_model,
            "accepts_aux_time_meta",
            False,
        ):
            output = model(
                z,
                t=t_flat,
                tt=r_flat,
                c=c,
                aux_time_meta={"step": int(self.train_step), "call_tag": call_tag},
            )
        else:
            output = model(z, t=t_flat, tt=r_flat, c=c)
        return self._as_tensor_output(output)

    @torch.no_grad()
    def _teacher_cfg_velocity(
        self,
        teacher: ModelLike,
        z: torch.Tensor,
        t: torch.Tensor,
        c: Conditioning,
        e: Conditioning,
    ) -> torch.Tensor:
        with _temporary_eval(teacher):
            t_flat = t.flatten().to(device=z.device, dtype=z.dtype)
            v_cond = self._as_tensor_output(teacher(z, t=t_flat, c=c))
            if self.teacher_cfg_scale == 1.0:
                return v_cond
            v_uncond = self._as_tensor_output(teacher(z, t=t_flat, c=e))
        return v_uncond + self.teacher_cfg_scale * (v_cond - v_uncond)

    def training_step(
        self,
        model: ModelLike,
        x: torch.Tensor,
        c: Conditioning,
        e: Conditioning,
        step: int,
        teacher: Optional[ModelLike] = None,
        return_loss_stats: bool = False,
        **kwargs,
    ):
        if kwargs:
            raise ValueError(
                f"Unknown ZImageMeanFlow training_step keys: {', '.join(sorted(kwargs.keys()))}"
            )
        if teacher is None:
            raise ValueError("ZImageMeanFlow.training_step requires a frozen teacher")

        self.train_step = int(step)
        batch_size = int(x.shape[0])
        r, t, r_raw, t_raw = self._sample_rt(
            batch_size=batch_size,
            device=x.device,
            dtype=x.dtype,
            step=int(step),
        )
        noise = torch.randn_like(x)
        h = (t - r).abs().clamp_min(torch.finfo(t.dtype).eps)
        raw_h = (t_raw - r_raw).clamp_min(torch.finfo(t_raw.dtype).eps)
        raw_delta_t = torch.minimum(
            torch.full_like(raw_h, self._effective_fd_delta()),
            raw_h,
        )
        t_minus_raw = (t_raw - raw_delta_t).clamp(0.0, 1.0)
        t_minus = self.apply_flow_shift(t_minus_raw)
        delta_t = (t - t_minus).clamp_min(torch.finfo(t.dtype).eps)
        gamma_t = (delta_t / h).clamp_min(torch.finfo(t.dtype).eps)
        z_t = (1.0 - _broadcast(t, x)) * x + _broadcast(t, x) * noise
        z_t_minus = (1.0 - _broadcast(t_minus, x)) * x + _broadcast(t_minus, x) * noise

        u = self._call_student(
            model,
            z_t,
            r=r,
            t=t,
            c=c,
            call_tag="meanflow_rt",
        )

        with torch.no_grad():
            u_terminal = self._call_student(
                model,
                z_t,
                r=t_minus,
                t=t,
                c=c,
                call_tag="meanflow_terminal_target",
            )
            v_hat = self._teacher_cfg_velocity(teacher, z_t, t, c, e)
            terminal_scale = torch.tensor(
                self.scheduled_terminal_cfg_scale(int(step)),
                device=x.device,
                dtype=x.dtype,
            )
            terminal_target = u_terminal + terminal_scale * (v_hat - u_terminal)
            u_t_prev = self._call_student(
                model,
                z_t_minus,
                r=r,
                t=t_minus,
                c=c,
                call_tag="meanflow_prev_target",
            )
            gamma_bc = _broadcast(gamma_t, x)
            target_t = (1.0 - gamma_bc) * u_t_prev + gamma_bc * terminal_target

        error_t = u - target_t.detach()
        sample_loss_t = self._per_sample_error(error_t)
        gamma_weight = (1.0 / gamma_t).clamp(max=self.two_res_gamma_weight_max)
        loss_t = (gamma_weight * sample_loss_t).mean()

        with torch.no_grad():
            s, s_raw, split_mask = self._sample_split(
                r=r,
                t=t,
                r_raw=r_raw,
                t_raw=t_raw,
            )
            pred_st = self._call_student(
                model,
                z_t,
                r=s,
                t=t,
                c=c,
                call_tag="meanflow_st",
            )
            z_s = z_t - _broadcast(t - s, x) * pred_st
            pred_rs = self._call_student(
                model,
                z_s,
                r=r,
                t=s,
                c=c,
                call_tag="meanflow_rs",
            )
            rs_weight = _broadcast((s - r) / h, x)
            st_weight = _broadcast((t - s) / h, x)
            split_target = rs_weight * pred_rs + st_weight * pred_st
        error_split = u - split_target.detach()
        split_losses = self._per_sample_error(error_split)
        split_weights = split_mask.to(
            device=split_losses.device,
            dtype=split_losses.dtype,
        )
        loss_split = (split_losses * split_weights).sum() / split_weights.sum().clamp_min(1.0)

        anchor_r_mask = torch.rand(batch_size, device=x.device) < 0.5
        anchor_time = torch.where(anchor_r_mask, r, t)
        z_anchor = (1.0 - _broadcast(anchor_time, x)) * x + _broadcast(anchor_time, x) * noise
        u_anchor = self._call_student(
            model,
            z_anchor,
            r=anchor_time,
            t=anchor_time,
            c=c,
            call_tag="meanflow_fm_anchor",
        )
        with torch.no_grad():
            v_anchor = self._teacher_cfg_velocity(teacher, z_anchor, anchor_time, c, e)
        error_fm = u_anchor - v_anchor.detach()
        loss_fm = self._per_sample_error(error_fm).mean()

        total = (
            self.two_res_lambda_t * loss_t
            + self.two_res_lambda_s * loss_split
            + self.fm_lambda * loss_fm
        )
        if not return_loss_stats:
            return total

        stats: LossStats = {
            "total": _stat(total),
            "loss_t": _stat(loss_t),
            "loss_split": _stat(loss_split),
            "loss_fm": _stat(loss_fm),
            "split_fd_terminal_cfg_scale": _stat(terminal_scale),
            "anchor_r_frac": _stat(anchor_r_mask.float().mean()),
            "split_gap_mean": _stat(h.detach().float().mean()),
            "raw_split_gap_mean": _stat(raw_h.detach().float().mean()),
            "split_eligible_frac": _stat(split_mask.float().mean()),
            "gamma_t_mean": _stat(gamma_t.detach().float().mean()),
            "delta_t_mean": _stat(delta_t.detach().float().mean()),
            "raw_delta_t_mean": _stat(raw_delta_t.detach().float().mean()),
            "effective_fd_delta": _stat(
                torch.tensor(
                    self._effective_fd_delta(),
                    device=x.device,
                    dtype=x.dtype,
                )
            ),
            "gamma_weight_mean": _stat(gamma_weight.detach().float().mean()),
            "terminal_target_mean": _stat(terminal_target.detach().float().mean()),
        }
        stats.update(_tensor_summary("r", r))
        stats.update(_tensor_summary("t", t))
        stats.update(_tensor_summary("split_s", s))
        stats.update(_tensor_summary("raw_r", r_raw))
        stats.update(_tensor_summary("raw_t", t_raw))
        stats.update(_tensor_summary("raw_split_s", s_raw))
        stats.update(_tensor_summary("teacher_cfg_velocity", v_hat))
        return total, stats

    @torch.no_grad()
    def sampling_loop(
        self,
        initial_noise_z: Optional[torch.Tensor] = None,
        sampling_model: Optional[ModelLike] = None,
        sampling_steps: int = 4,
        flow_shift: Optional[float] = None,
        inital_noise_z: Optional[torch.Tensor] = None,
        **model_kwargs,
    ) -> torch.Tensor:
        if initial_noise_z is None:
            initial_noise_z = inital_noise_z
        if initial_noise_z is None:
            raise ValueError("sampling_loop requires initial_noise_z")
        if sampling_model is None:
            raise ValueError("sampling_loop requires sampling_model")

        input_dtype = initial_noise_z.dtype
        x = initial_noise_z.to(torch.float32)
        samples = [x.cpu()]
        steps = max(1, int(sampling_steps))
        raw_sigmas = torch.linspace(
            1.0,
            0.0,
            steps + 1,
            device=initial_noise_z.device,
            dtype=torch.float32,
        )
        sigmas = self.apply_flow_shift(raw_sigmas, flow_shift=flow_shift)

        for sigma_cur, sigma_next in zip(sigmas[:-1], sigmas[1:]):
            t_cur = sigma_cur.expand(x.shape[0]).to(dtype=input_dtype)
            t_next = sigma_next.expand(x.shape[0]).to(dtype=input_dtype)
            v_pred = self._as_tensor_output(
                sampling_model(
                    x.to(input_dtype),
                    t=t_cur,
                    tt=t_next,
                    **model_kwargs,
                )
            ).to(torch.float32)
            dt = sigma_next - sigma_cur
            x = x + dt * v_pred
            samples.append(x.cpu())

        return torch.stack(samples, dim=0)
