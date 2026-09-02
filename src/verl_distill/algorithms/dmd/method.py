import logging
import math
from typing import Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

from .lora import lora_false, lora_true

logger = logging.getLogger(__name__)

ScoreBranchModel = Union[nn.Module, Callable]
ScoreModelLike = Union[ScoreBranchModel, Dict[str, ScoreBranchModel]]


class StandardDMD(torch.nn.Module):
    """Standard DMD training helper for TwinFlow models.

    The score branch supports two layouts:
    - LoRA mode: one shared base model, selecting real/fake via LoRA on/off.
    - Full mode: an explicit ``{"real": ..., "fake": ...}`` score-model dict.
    """

    def __init__(
        self,
        transport_type: str = "Linear",
        # DMD core
        num_train_timestep: int = 1000,
        min_step_percent: float = 0.02,
        max_step_percent: float = 0.98,
        num_denoising_step: int = 4,
        timestep_shift: float = 5.0,
        backward_simulation: bool = False,
        backward_simulation_mode: str = "uncorrelated",
        real_guidance_scale: float = 3.5,
        fake_guidance_scale: float = 0.0,
        dfake_gen_update_ratio: int = 5,
        dm_loss_weight: float = 1.0,
        loss_type: str = "mse",
        warmup_type: str = "none",
        warmup_iterations: int = 0,
        ode_warmup_loss_weight: float = 0.0,
        ode_warmup_pair_dir: str = "",
        ode_warmup_reward_weighting: str = "source_rank",
        ode_warmup_reward_weights: Optional[Dict] = None,
        ode_warmup_rank_weight_strength: float = 1.0,
        ode_warmup_generator_sigma_probs: Optional[List[float]] = None,
        rcgm_warmup_loss_weight: float = 0.0,
        rcgm_warmup_iterations: int = 0,
        rcgm_warmup_teacher_steps: int = 4,
        rcgm_loss_weight: float = 0.0,
        rcgm_sigma_eps: float = 0.01,
        rcgm_min_sigma: float = 1e-4,
        rcgm_teacher_steps: int = 2,
        gt_grad_loss_weight: float = 0.05,
        grad_norm_eps: float = 1e-6,
        # Score loss
        score_loss_weight: float = 1.0,
        score_use_weighting: bool = False,
        score_loss_target: str = "x0",
        fake_score_use_generator_timestep: bool = False,
        # Generator regularization (optional)
        generator_recon_weight: float = 0.0,
        # Data-free mode
        data_free: bool = False,
        # Demo sampling defaults (few-step sampling, optionally with stochastic refresh)
        sampling_steps: int = 4,
        **kwargs,
    ):
        super().__init__()
        if kwargs:
            raise ValueError(f"Unknown StandardDMD config keys: {', '.join(sorted(kwargs.keys()))}")
        if transport_type != "Linear":
            raise ValueError(
                f"Unsupported transport_type={transport_type}, only Linear is supported"
            )
        if dfake_gen_update_ratio < 1:
            raise ValueError("dfake_gen_update_ratio must be >= 1")

        self.num_train_timestep = int(num_train_timestep)
        self.min_step = max(1, int(float(min_step_percent) * self.num_train_timestep))
        self.max_step = min(
            self.num_train_timestep - 1,
            int(float(max_step_percent) * self.num_train_timestep),
        )
        self.num_denoising_step = max(1, int(num_denoising_step))
        self.timestep_shift = float(timestep_shift)
        self.backward_simulation = bool(backward_simulation)
        self.backward_simulation_mode = str(backward_simulation_mode).lower()
        if self.backward_simulation_mode not in {"uncorrelated", "deterministic"}:
            raise ValueError("backward_simulation_mode must be 'uncorrelated' or 'deterministic'")
        self.real_guidance_scale = float(real_guidance_scale)
        self.fake_guidance_scale = float(fake_guidance_scale)
        self.dfake_gen_update_ratio = int(dfake_gen_update_ratio)
        self.dm_loss_weight = float(dm_loss_weight)
        self.loss_type = str(loss_type).lower()
        if self.loss_type not in {"pearson4", "mse"}:
            raise ValueError("loss_type must be either 'pearson4' or 'mse'")
        self.warmup_type = str(warmup_type).lower()
        if self.warmup_type not in {"none", "rcgm", "ode_pair"}:
            raise ValueError("warmup_type must be one of: none, rcgm, ode_pair")
        self.warmup_iterations = max(0, int(warmup_iterations))
        self.ode_warmup_loss_weight = float(ode_warmup_loss_weight)
        self.ode_warmup_pair_dir = str(ode_warmup_pair_dir)
        self.ode_warmup_reward_weighting = str(ode_warmup_reward_weighting)
        self.ode_warmup_reward_weights = ode_warmup_reward_weights
        self.ode_warmup_rank_weight_strength = float(ode_warmup_rank_weight_strength)
        if (
            not math.isfinite(self.ode_warmup_rank_weight_strength)
            or self.ode_warmup_rank_weight_strength < 0.0
        ):
            raise ValueError("ode_warmup_rank_weight_strength must be finite and non-negative")
        self.ode_warmup_generator_sigma_probs = (
            self._normalize_sigma_probabilities(
                ode_warmup_generator_sigma_probs,
                expected_len=self.num_denoising_step,
                name="ode_warmup_generator_sigma_probs",
            )
            if ode_warmup_generator_sigma_probs is not None
            else None
        )
        self.rcgm_warmup_loss_weight = float(rcgm_warmup_loss_weight)
        self.rcgm_warmup_iterations = max(0, int(rcgm_warmup_iterations))
        if (
            self.warmup_type == "none"
            and self.rcgm_warmup_iterations > 0
            and self.rcgm_warmup_loss_weight != 0.0
        ):
            self.warmup_type = "rcgm"
            self.warmup_iterations = self.rcgm_warmup_iterations
        if self.warmup_type == "rcgm" and self.warmup_iterations <= 0:
            self.warmup_iterations = self.rcgm_warmup_iterations
        if self.warmup_type == "rcgm":
            self.rcgm_warmup_iterations = self.warmup_iterations
        if self.warmup_type == "ode_pair" and self.rcgm_warmup_loss_weight != 0.0:
            raise ValueError(
                "warmup_type=ode_pair is mutually exclusive with rcgm_warmup_loss_weight"
            )
        self.rcgm_warmup_teacher_steps = max(1, int(rcgm_warmup_teacher_steps))
        self.rcgm_loss_weight = float(rcgm_loss_weight)
        self.rcgm_sigma_eps = float(rcgm_sigma_eps)
        self.rcgm_min_sigma = float(rcgm_min_sigma)
        self.rcgm_teacher_steps = max(1, int(rcgm_teacher_steps))
        self.train_step = 0
        self.gt_grad_loss_weight = float(gt_grad_loss_weight)
        self.grad_norm_eps = float(grad_norm_eps)
        self.score_loss_weight = float(score_loss_weight)
        self.score_use_weighting = bool(score_use_weighting)
        self.score_loss_target = str(score_loss_target).lower()
        if self.score_loss_target not in {"x0", "flow"}:
            raise ValueError("score_loss_target must be either 'x0' or 'flow'")
        self.fake_score_use_generator_timestep = bool(fake_score_use_generator_timestep)
        self.generator_recon_weight = float(generator_recon_weight)
        self.data_free = bool(data_free)
        if self.data_free:
            if not self.backward_simulation:
                raise ValueError("data_free=True currently requires backward_simulation=True")
            if self.gt_grad_loss_weight != 0.0:
                raise ValueError("data_free=True requires gt_grad_loss_weight == 0")
            if self.generator_recon_weight != 0.0:
                raise ValueError("data_free=True requires generator_recon_weight == 0")
        self.default_sampler_kwargs = dict(sampling_steps=int(sampling_steps), stochast_ratio=0.0)

    # ----- Flow / sampler helpers (copied from TwinFlow, stripped down) -----
    def alpha_in(self, t):
        return t

    def gamma_in(self, t):
        return 1 - t

    def alpha_to(self, t):
        return 1

    def gamma_to(self, t):
        return -1

    def _unwrap(self, model: Union[nn.Module, Callable]):
        return getattr(model, "module", model)

    def _select_score_branch(
        self,
        score_model: ScoreModelLike,
        use_lora: Optional[bool],
    ) -> Tuple[ScoreBranchModel, Optional[bool]]:
        if isinstance(score_model, dict):
            if use_lora is None:
                raise ValueError(
                    "score_model with explicit real/fake branches requires use_lora=True/False "
                    "to select the fake/real branch."
                )
            branch = "fake" if use_lora else "real"
            if branch not in score_model:
                raise KeyError(
                    f"score_model dict is missing '{branch}' branch; got keys={tuple(score_model.keys())}"
                )
            return score_model[branch], None
        return score_model, use_lora

    def _call_model(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: List[torch.Tensor],
        target_timestep: Optional[torch.Tensor] = None,
        target_timestep_log_tag: Optional[str] = None,
    ) -> torch.Tensor:
        t_flat = t.flatten()
        if target_timestep is None:
            return model(x_t, t=t_flat, c=c)

        tt_flat = target_timestep.flatten().to(device=t_flat.device, dtype=t_flat.dtype)
        if tt_flat.numel() == 1 and t_flat.numel() != 1:
            tt_flat = tt_flat.expand_as(t_flat)
        base_model = self._unwrap(model)
        if getattr(model, "accepts_aux_time_meta", False) or getattr(
            base_model, "accepts_aux_time_meta", False
        ):
            aux_time_meta = {
                "step": int(self.train_step),
                "call_tag": target_timestep_log_tag or "fake_score",
            }
            return model(x_t, t=t_flat, c=c, tt=tt_flat, aux_time_meta=aux_time_meta)
        return model(x_t, t=t_flat, c=c, tt=tt_flat)

    def forward(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        **model_kwargs,
    ):
        dent = -1
        q = torch.ones(x_t.size(0), device=x_t.device, dtype=x_t.dtype) * (t).flatten()
        F_t = model(x_t, t=q, **model_kwargs)
        t = torch.abs(t).flatten().to(device=x_t.device, dtype=x_t.dtype)
        t_b = self._broadcast_sigma(t, x_t)
        z_hat = (x_t * self.gamma_to(t_b) - F_t * self.gamma_in(t_b)) / dent
        x_hat = (F_t * self.alpha_in(t_b) - x_t * self.alpha_to(t_b)) / dent
        return x_hat, z_hat, F_t, dent

    @torch.no_grad()
    def sampling_loop(
        self,
        inital_noise_z: torch.FloatTensor,
        sampling_model: Union[nn.Module, Callable],
        sampling_steps: int = 20,
        stochast_ratio: Union[float, str] = 0.0,
        timestep_shift: Optional[float] = None,
        **model_kwargs,
    ):
        input_dtype = inital_noise_z.dtype
        num_steps = max(1, int(sampling_steps))
        # Align demo sampling time grid with the same few-step sigma levels used in training.
        sigma_levels = self._generator_sigma_levels(
            num_steps,
            device=inital_noise_z.device,
            dtype=torch.float64,
            include_terminal_zero=False,
            timestep_shift=timestep_shift,
        )
        t_steps = torch.cat(
            [
                sigma_levels,
                torch.zeros(1, device=sigma_levels.device, dtype=sigma_levels.dtype),
            ]
        )
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z.cpu()]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(input_dtype),
                t_cur.to(input_dtype),
                **model_kwargs,
            )
            samples.append(x_hat.cpu())
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)
            if stochast_ratio == "SDE":
                step_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                step_ratio = torch.clamp(step_ratio**2, min=0.0, max=1.0)
                noise = torch.randn_like(x_cur)
            else:
                step_ratio = float(stochast_ratio)
                if step_ratio < 0.0 or step_ratio > 1.0:
                    raise ValueError("stochast_ratio must be in [0, 1] or the string 'SDE'")
                noise = torch.randn_like(x_cur) if step_ratio > 0.0 else 0.0

            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - step_ratio) ** 0.5) + noise * (step_ratio**0.5)
            )
            x_cur = x_next

        return torch.stack(samples, dim=0).to(input_dtype)

    # ----- DMD helpers -----
    def is_warmup_step(self, step: Optional[int] = None) -> bool:
        if self.warmup_type == "none" or self.warmup_iterations <= 0:
            return False
        if self.warmup_type == "ode_pair" and self.ode_warmup_loss_weight == 0.0:
            return False
        if self.warmup_type == "rcgm" and self.rcgm_warmup_loss_weight == 0.0:
            return False
        step_i = self.train_step if step is None else int(step)
        return 0 < step_i <= self.warmup_iterations

    def is_rcgm_warmup_step(self, step: Optional[int] = None) -> bool:
        if self.warmup_type != "rcgm":
            return False
        return self.is_warmup_step(step)

    def is_ode_pair_warmup_step(self, step: Optional[int] = None) -> bool:
        if self.warmup_type != "ode_pair":
            return False
        return self.is_warmup_step(step)

    def should_update_generator(self, step: int) -> bool:
        step_i = int(step)
        if self.is_warmup_step(step_i):
            return True
        return step_i % self.dfake_gen_update_ratio == 0

    def set_train_step(self, step: int):
        self.train_step = int(step)

    def _broadcast_sigma(self, sigma: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return sigma.view(-1, *([1] * (x.ndim - 1))).to(dtype=x.dtype, device=x.device)

    def _sample_timestep(self, batch_size: int, device: torch.device) -> torch.Tensor:
        t = torch.randint(
            low=self.min_step,
            high=self.max_step + 1,
            size=(batch_size,),
            device=device,
            dtype=torch.long,
        )
        return t.to(torch.float32)

    def _apply_timestep_shift(self, timestep: torch.Tensor) -> torch.Tensor:
        # Match the DMD timestep shift used in the video implementation.
        return self._apply_timestep_shift_value(timestep, self.timestep_shift)

    def _apply_timestep_shift_value(
        self, timestep: torch.Tensor, timestep_shift: float
    ) -> torch.Tensor:
        if float(timestep_shift) <= 1.0:
            return timestep
        out_dtype = timestep.dtype if torch.is_floating_point(timestep) else torch.float32
        t = timestep.to(torch.float32)
        denom_base = float(self.num_train_timestep)
        t_norm = t / denom_base
        t_shift = (
            float(timestep_shift)
            * t_norm
            / (1.0 + (float(timestep_shift) - 1.0) * t_norm)
            * denom_base
        )
        return t_shift.clamp(0.0, denom_base).to(out_dtype)

    def _sample_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        timesteps = self._sample_timestep(batch_size, device)
        # score_sigma / dm_sigma should stay on the original uniform timestep sampling,
        # i.e. no timestep_shift here. timestep_shift is only for generator few-step grid.
        return timesteps.to(torch.float32) / float(self.num_train_timestep)

    def _spatial_standardize(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=[2, 3], keepdim=True)
        std = x.std(dim=[2, 3], keepdim=True)
        return (x - mean) / (std + 1e-6)

    def _mse_loss_per_sample(self, pred: torch.Tensor, target: torch.Tensor, scale: float = 1.0):
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        return float(scale) * loss.flatten(1).mean(dim=1)

    def _generator_sigma_levels(
        self,
        num_steps: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
        include_terminal_zero: bool = False,
        timestep_shift: Optional[float] = None,
    ) -> torch.Tensor:
        # `num_steps` is the number of denoising updates, not the number of grid nodes.
        # Build nodes from T->0 and drop the terminal zero by default so training does
        # not sample sigma=0 and sampling can append the final zero exactly once.
        num_steps = max(1, int(num_steps))
        levels = torch.linspace(
            float(self.num_train_timestep),
            0,
            num_steps + 1,
            device=device,
            dtype=dtype,
        )
        levels = self._apply_timestep_shift_value(
            levels,
            self.timestep_shift if timestep_shift is None else float(timestep_shift),
        )
        if not include_terminal_zero:
            levels = levels[:-1]
        return levels / float(self.num_train_timestep)

    def _sample_generator_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        levels = self._generator_sigma_levels(
            self.num_denoising_step,
            device=device,
            dtype=torch.float32,
            include_terminal_zero=False,
        )
        idx = torch.randint(0, levels.numel(), (batch_size,), device=device)
        return levels[idx]

    def _normalize_sigma_probabilities(
        self, probs: List[float], expected_len: int, name: str
    ) -> torch.Tensor:
        prob_tensor = torch.as_tensor(probs, dtype=torch.float32)
        if prob_tensor.ndim != 1 or prob_tensor.numel() != int(expected_len):
            raise ValueError(
                f"{name} must contain exactly {int(expected_len)} values, "
                f"got shape={tuple(prob_tensor.shape)}"
            )
        if not torch.isfinite(prob_tensor).all():
            raise ValueError(f"{name} must contain only finite values")
        if (prob_tensor < 0).any():
            raise ValueError(f"{name} must be non-negative")
        total = prob_tensor.sum()
        if float(total.item()) <= 0.0:
            raise ValueError(f"{name} must have positive sum")
        return prob_tensor / total

    def _sample_ode_warmup_generator_sigmas(
        self, batch_size: int, device: torch.device
    ) -> torch.Tensor:
        if self.ode_warmup_generator_sigma_probs is None:
            return self._sample_generator_sigmas(batch_size, device)
        levels = self._generator_sigma_levels(
            self.num_denoising_step,
            device=device,
            dtype=torch.float32,
            include_terminal_zero=False,
        )
        probs = self.ode_warmup_generator_sigma_probs.to(device=device)
        idx = torch.multinomial(probs, batch_size, replacement=True)
        return levels[idx]

    def _sample_generator_step_indices(self, batch_size: int, device: torch.device) -> torch.Tensor:
        num_steps = max(1, int(self.num_denoising_step))
        if not self.backward_simulation:
            return torch.randint(0, num_steps, (batch_size,), device=device, dtype=torch.long)

        # Match DMD2's backward simulation: choose one denoising step and share it
        # across the whole batch (and across ranks when distributed is active).
        shared_step = torch.randint(0, num_steps, (1,), device=device, dtype=torch.long)
        if dist.is_available() and dist.is_initialized():
            dist.broadcast(shared_step, src=0)
        return shared_step.expand(batch_size)

    def _generator_sigma_levels_f32(self, device: torch.device) -> torch.Tensor:
        return self._generator_sigma_levels(
            self.num_denoising_step,
            device=device,
            dtype=torch.float32,
            include_terminal_zero=False,
        )

    def _next_generator_sigma_level(self, sigma: torch.Tensor) -> torch.Tensor:
        levels = self._generator_sigma_levels(
            self.num_denoising_step,
            device=sigma.device,
            dtype=torch.float32,
            include_terminal_zero=True,
        )
        sigma_f = sigma.to(device=sigma.device, dtype=torch.float32).flatten()
        nearest_idx = (sigma_f[:, None] - levels[None, :]).abs().argmin(dim=1)
        next_idx = (nearest_idx + 1).clamp(max=levels.numel() - 1)
        return levels[next_idx].to(device=sigma.device, dtype=torch.float32)

    def _step_condition_list(
        self, c: List[torch.Tensor], index: torch.Tensor
    ) -> List[torch.Tensor]:
        return [ci[index] for ci in c]

    @torch.no_grad()
    def _simulate_backward_generator_inputs(
        self,
        generator_model: Union[nn.Module, Callable],
        x_shape: torch.Size,
        c: List[torch.Tensor],
        step_indices: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Simulate inference-time intermediate states for few-step training.

        DMD2 backward simulation trains the current denoising step on synthetic
        intermediate states produced by earlier generator steps rather than on
        noisy real images. We mirror that logic here by rolling a pure-noise
        batch forward along the generator sigma grid up to the requested step.

        When ``backward_simulation_mode='uncorrelated'`` we re-noise the
        predicted clean sample with fresh Gaussian noise at each intermediate
        step, matching the official DMD2 implementation. ``deterministic``
        keeps the current sampler's correlated update rule.
        """
        device = step_indices.device
        levels = self._generator_sigma_levels_f32(device)
        max_step = int(step_indices.max().item())
        if initial_noise is None:
            x_cur = torch.randn(x_shape, device=device, dtype=torch.float32)
        else:
            x_cur = initial_noise.to(device=device, dtype=torch.float32)
            if tuple(x_cur.shape) != tuple(x_shape):
                raise ValueError(
                    f"initial_noise shape mismatch: {tuple(x_cur.shape)} vs {tuple(x_shape)}"
                )

        if max_step <= 0:
            return x_cur

        for step in range(max_step):
            active = torch.nonzero(step_indices > step, as_tuple=False).flatten()
            if active.numel() == 0:
                continue

            sigma_cur = torch.full(
                (active.numel(),),
                float(levels[step].item()),
                device=device,
                dtype=torch.float32,
            )
            sigma_next = float(levels[step + 1].item())
            x_step = x_cur[active]
            c_step = self._step_condition_list(c, active)
            x_hat, z_hat, _, _ = self.forward(
                generator_model,
                x_step,
                sigma_cur,
                c=c_step,
            )

            if self.backward_simulation_mode == "uncorrelated":
                fresh_noise = torch.randn_like(x_hat)
                sigma_next_b = self._broadcast_sigma(
                    torch.full(
                        (active.numel(),),
                        sigma_next,
                        device=device,
                        dtype=torch.float32,
                    ),
                    x_hat,
                )
                x_next = sigma_next_b * fresh_noise + (1.0 - sigma_next_b) * x_hat
            else:
                sigma_next_t = torch.full(
                    (active.numel(),),
                    sigma_next,
                    device=device,
                    dtype=torch.float32,
                )
                x_next = self.gamma_in(sigma_next_t) * x_hat + self.alpha_in(sigma_next_t) * z_hat

            x_cur[active] = x_next.to(dtype=x_cur.dtype)

        return x_cur

    def _weighted_mse_target(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        sigma: torch.Tensor,
        return_meta: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        loss_pre = F.mse_loss(pred.float(), target.float(), reduction="none")
        loss_pre = loss_pre.flatten(1).mean(dim=1)
        if self.score_use_weighting:
            # Simple SD3-style proxy weighting: emphasize mid/late noise.
            weight = torch.clamp(sigma, min=1e-4) / torch.clamp(1.0 - sigma, min=1e-4)
            loss_post = loss_pre * weight
        else:
            weight = torch.ones_like(loss_pre)
            loss_post = loss_pre
        loss_mean = loss_post.mean()
        if return_meta:
            return loss_mean, {
                "score_loss_pre_weight": loss_pre.detach(),
                "score_loss_post_weight": loss_post.detach(),
                "score_loss_weight": weight.detach(),
            }
        return loss_mean

    def _predict_x0_from_flow(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]] = None,
        guidance_scale: float = 0.0,
        use_lora: Optional[bool] = None,
        target_timestep: Optional[torch.Tensor] = None,
        target_timestep_log_tag: Optional[str] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        branch_model, branch_use_lora = self._select_score_branch(model, use_lora)
        base_model = self._unwrap(branch_model)
        if branch_use_lora is not None:
            if branch_use_lora:
                lora_true(base_model)
            else:
                lora_false(base_model)

        F_cond = self._call_model(
            branch_model,
            x_t,
            sigma,
            c,
            target_timestep=target_timestep,
            target_timestep_log_tag=target_timestep_log_tag,
        )
        if guidance_scale != 0.0 and e is not None:
            F_uncond = self._call_model(
                branch_model,
                x_t,
                sigma,
                e,
                target_timestep=target_timestep,
                target_timestep_log_tag=(
                    f"{target_timestep_log_tag}_uncond"
                    if target_timestep_log_tag
                    else "fake_score_uncond"
                ),
            )
            F_pred = F_cond + (F_cond - F_uncond) * guidance_scale
        else:
            F_pred = F_cond
        x0 = x_t - self._broadcast_sigma(sigma, x_t) * F_pred
        return x0, F_pred

    @torch.no_grad()
    def _multi_fwd_avg_vector(
        self,
        model: Union[nn.Module, Callable],
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        num_steps: int = 2,
        guidance_scale: float = 0.0,
        use_lora: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Average denoising vector over [tt, t] using N sub-steps."""
        num_steps = max(1, int(num_steps))
        ts = [t * (1.0 - i / num_steps) + tt * (i / num_steps) for i in range(num_steps + 1)]
        x_cur = x_t
        pred = torch.zeros_like(x_t)
        total = torch.zeros_like(t)
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            _, F_c = self._predict_x0_from_flow(
                model,
                x_cur,
                t_c,
                c=c,
                e=e,
                guidance_scale=guidance_scale,
                use_lora=use_lora,
            )
            dt = (t_c - t_n).clamp_min(0.0)
            dt_b = self._broadcast_sigma(dt, x_cur)
            x_cur = x_cur - dt_b * F_c
            pred = pred + dt_b * F_c
            total = total + dt
        total_safe = total.clamp_min(self.grad_norm_eps)
        avg = pred / self._broadcast_sigma(total_safe, pred)
        return avg, total

    def _rcgm_regularizer_loss(
        self,
        score_model: ScoreModelLike,
        x_fake: torch.Tensor,
        gen_input_sigma: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        return_debug_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        """RCGM-style vector alignment on generated samples for all batch elements.

        For each sample:
        1) re-noise x_fake to sigma_s < sigma
        2) run real score model (LoRA off) for a short multi-step denoising rollout
        3) align the generator-implied vector at sigma_s with the rollout average vector
        """
        sigma = gen_input_sigma.detach().to(device=x_fake.device, dtype=torch.float32).flatten()
        sigma_s = torch.clamp(sigma - self.rcgm_sigma_eps, min=self.rcgm_min_sigma)
        # Random denoising length in [0, sigma_s], with positive target sigma.
        sigma_t = sigma_s - torch.rand_like(sigma_s) * sigma_s
        sigma_t = sigma_t.clamp(min=self.rcgm_min_sigma)

        noise = torch.randn_like(x_fake)
        sigma_s_b = self._broadcast_sigma(sigma_s, x_fake)
        x_s = sigma_s_b * noise + (1.0 - sigma_s_b) * x_fake

        # Generator-implied one-step vector at sigma_s toward x_fake.
        vec_gen = (x_s - x_fake) / sigma_s_b.clamp_min(self.grad_norm_eps)

        with torch.no_grad():
            vec_teacher, interval = self._multi_fwd_avg_vector(
                score_model,
                x_t=x_s.detach(),
                t=sigma_s,
                tt=sigma_t,
                c=c,
                e=e,
                num_steps=self.rcgm_teacher_steps,
                guidance_scale=self.real_guidance_scale,
                use_lora=False,
            )

        loss = F.mse_loss(vec_gen.float(), vec_teacher.float(), reduction="none")
        loss = loss.flatten(1).mean(dim=1).mean()
        stats = {
            "rcgm_sigma_s": sigma_s.detach(),
            "rcgm_sigma_t": sigma_t.detach(),
            "rcgm_interval": interval.detach(),
            "rcgm_vec_gen_abs": vec_gen.detach().abs().flatten(1).mean(dim=1),
            "rcgm_vec_teacher_abs": vec_teacher.detach().abs().flatten(1).mean(dim=1),
        }
        if return_debug_tensors:
            x0_from_vec_gen = x_s - sigma_s_b * vec_gen
            x0_from_vec_teacher = x_s - sigma_s_b * vec_teacher
            debug = {
                "rcgm_x0_from_vec_gen": x0_from_vec_gen.detach(),
                "rcgm_x0_from_vec_teacher": x0_from_vec_teacher.detach(),
            }
            return loss, stats, debug
        return loss, stats

    @torch.no_grad()
    def _rcgm_warmup_mixed_avg_vector(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_t: torch.Tensor,
        t: torch.Tensor,
        tt: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        num_steps: int = 4,
        guidance_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Warmup teacher: real-score CFG to next few-step node, then generator to x0."""
        num_steps = max(1, int(num_steps))
        real_steps = max(1, num_steps - 1)
        ts = [t * (1.0 - i / real_steps) + tt * (i / real_steps) for i in range(real_steps + 1)]
        score_branch, score_use_lora = self._select_score_branch(score_model, False)
        score_base = self._unwrap(score_branch)
        if score_use_lora is not None:
            if score_use_lora:
                lora_true(score_base)
            else:
                lora_false(score_base)

        x_cur = x_t
        pred = torch.zeros_like(x_t)
        generator_steps = 0
        real_cfg_steps = 0
        for t_c, t_n in zip(ts[:-1], ts[1:]):
            F_cond = self._call_model(score_branch, x_cur, t_c, c)
            if e is None:
                raise ValueError("RCGM warmup CFG teacher requires unconditional embeddings.")
            F_uncond = self._call_model(score_branch, x_cur, t_c, e)
            F_c = F_uncond + (F_cond - F_uncond) * guidance_scale
            real_cfg_steps += 1
            dt = (t_c - t_n).clamp_min(0.0)
            dt_b = self._broadcast_sigma(dt, x_cur)
            x_cur = x_cur - dt_b * F_c
            pred = pred + F_c
        F_gen = self._call_model(generator_model, x_cur, tt, c)
        pred = pred + F_gen
        generator_steps += 1
        total = torch.full_like(t, float(real_cfg_steps + generator_steps))
        avg = pred / float(real_cfg_steps + generator_steps)
        meta = {
            "rcgm_teacher_total_steps": torch.full_like(t, float(num_steps)),
            "rcgm_teacher_generator_steps": torch.full_like(t, float(generator_steps)),
            "rcgm_teacher_real_cfg_steps": torch.full_like(t, float(real_cfg_steps)),
            "rcgm_teacher_cfg_scale": torch.full_like(t, float(guidance_scale)),
            "rcgm_teacher_target_sigma": tt.detach().to(torch.float32),
        }
        return avg, total, meta

    def _rcgm_warmup_regularizer_loss(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_fake: torch.Tensor,
        gen_input_sigma: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        return_debug_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        """Warmup RCGM loss using the generator's conditional velocity output."""
        sigma_s = (
            gen_input_sigma.detach()
            .to(device=x_fake.device, dtype=torch.float32)
            .flatten()
            .clamp(min=self.rcgm_min_sigma)
        )
        sigma_t = self._next_generator_sigma_level(sigma_s)

        x_fake_base = x_fake.detach()
        noise = torch.randn_like(x_fake_base)
        sigma_s_b = self._broadcast_sigma(sigma_s, x_fake_base)
        x_s = sigma_s_b * noise + (1.0 - sigma_s_b) * x_fake_base

        vec_gen = self._call_model(generator_model, x_s, sigma_s, c)

        with torch.no_grad():
            vec_teacher, interval, teacher_meta = self._rcgm_warmup_mixed_avg_vector(
                generator_model,
                score_model,
                x_t=x_s.detach(),
                t=sigma_s,
                tt=sigma_t,
                c=c,
                e=e,
                num_steps=self.rcgm_warmup_teacher_steps,
                guidance_scale=self.real_guidance_scale,
            )

        loss = F.mse_loss(vec_gen.float(), vec_teacher.float(), reduction="none")
        loss = loss.flatten(1).mean(dim=1).mean()
        stats = {
            "rcgm_sigma_s": sigma_s.detach(),
            "rcgm_sigma_t": sigma_t.detach(),
            "rcgm_interval": interval.detach(),
            "rcgm_vec_gen_abs": vec_gen.detach().abs().flatten(1).mean(dim=1),
            "rcgm_vec_teacher_abs": vec_teacher.detach().abs().flatten(1).mean(dim=1),
        }
        stats.update({k: v.detach() for k, v in teacher_meta.items()})
        if return_debug_tensors:
            x0_from_vec_gen = x_s - sigma_s_b * vec_gen
            x0_from_vec_teacher = x_s - sigma_s_b * vec_teacher
            debug = {
                "rcgm_x0_from_vec_gen": x0_from_vec_gen.detach(),
                "rcgm_x0_from_vec_teacher": x0_from_vec_teacher.detach(),
            }
            return loss, stats, debug
        return loss, stats

    def generate_one_step_latents(
        self,
        generator_model: Union[nn.Module, Callable],
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        sigma_override: Optional[torch.Tensor] = None,
        noise_override: Optional[torch.Tensor] = None,
        return_internal: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x_real is not None:
            batch_size = x_real.shape[0]
            device = x_real.device
            x_shape = tuple(x_real.shape)
            target_dtype = x_real.dtype
        elif initial_noise is not None:
            batch_size = initial_noise.shape[0]
            device = initial_noise.device
            x_shape = tuple(initial_noise.shape)
            target_dtype = initial_noise.dtype
        elif latent_shape is not None:
            batch_size = int(latent_shape[0])
            device = c[0].device
            x_shape = tuple(latent_shape)
            target_dtype = torch.float32
        else:
            raise ValueError(
                "generate_one_step_latents requires x_real, initial_noise, or latent_shape"
            )
        allow_backward_sim = (
            self.backward_simulation and sigma_override is None and noise_override is None
        )
        if allow_backward_sim:
            step_indices = self._sample_generator_step_indices(batch_size, device)
            sigma_levels = self._generator_sigma_levels_f32(device)
            sigma = sigma_levels[step_indices]
            x_t = self._simulate_backward_generator_inputs(
                generator_model=generator_model,
                x_shape=torch.Size(x_shape),
                c=c,
                step_indices=step_indices,
                initial_noise=initial_noise,
            ).to(dtype=target_dtype)
            noise = torch.empty(x_shape, device=device, dtype=target_dtype)
        else:
            if sigma_override is None:
                sigma = self._sample_generator_sigmas(batch_size, device)
            else:
                sigma = sigma_override.detach().to(device=device, dtype=torch.float32).flatten()
                if sigma.numel() != batch_size:
                    raise ValueError(
                        f"sigma_override must have {batch_size} elements, got {sigma.numel()}"
                    )
            if x_real is None:
                raise ValueError(
                    "generate_one_step_latents requires x_real when backward_simulation is disabled"
                )
            if noise_override is None:
                noise = torch.randn_like(x_real)
            else:
                noise = noise_override.to(device=device, dtype=x_real.dtype)
                if noise.shape != x_real.shape:
                    raise ValueError(
                        f"noise_override shape mismatch: {tuple(noise.shape)} vs {tuple(x_real.shape)}"
                    )
            sigma_b = self._broadcast_sigma(sigma, x_real)
            x_t = sigma_b * noise + (1.0 - sigma_b) * x_real

        x_fake, gen_flow = self._predict_x0_from_flow(
            generator_model,
            x_t,
            sigma,
            c=c,
            e=None,
            guidance_scale=0.0,
        )
        meta = {
            "gen_input_sigma": sigma.detach(),
            "gen_backward_simulation": torch.full(
                (batch_size,),
                1.0 if allow_backward_sim else 0.0,
                device=device,
                dtype=torch.float32,
            ),
        }
        if allow_backward_sim:
            meta["gen_step_index"] = step_indices.detach()
        if return_internal:
            meta["gen_noise"] = noise.detach()
            meta["gen_x_t"] = x_t.detach()
            meta["gen_flow"] = gen_flow.detach()
        return x_fake, meta

    def _compute_dmd_grad(
        self,
        score_model: ScoreModelLike,
        x_fake: torch.Tensor,
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        gen_input_sigma: Optional[torch.Tensor] = None,
        return_extra: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, Dict[str, torch.Tensor], Dict[str, torch.Tensor]],
    ]:
        fake_score_tt = None
        if self.fake_score_use_generator_timestep:
            if not isinstance(score_model, dict):
                raise ValueError(
                    "fake_score_use_generator_timestep=True requires explicit "
                    "score_model dict with fake and real branches"
                )
            if gen_input_sigma is None:
                raise ValueError("fake_score_use_generator_timestep=True requires gen_input_sigma")
            fake_score_tt = (
                gen_input_sigma.detach().to(device=x_fake.device, dtype=torch.float32).flatten()
            )
        batch_size = x_fake.shape[0]
        sigma = self._sample_sigmas(batch_size, x_fake.device)
        noise = torch.randn_like(x_fake)
        sigma_b = self._broadcast_sigma(sigma, x_fake)
        noisy = sigma_b * noise + (1.0 - sigma_b) * x_fake

        pred_fake_x0, fake_score_flow = (
            self._predict_x0_from_flow(  # Fake score，代表 当前生成图的方向。pred_real_x0 告诉generator ：真实/teacher 分布在这个 noisy 点附近，应该往哪去
                score_model,
                noisy,
                sigma,
                c=c,
                e=e,
                guidance_scale=self.fake_guidance_scale,
                use_lora=True,
                target_timestep=fake_score_tt,
                target_timestep_log_tag="generator_dmd_fake_score",
            )
        )
        pred_real_x0, real_score_flow = (
            self._predict_x0_from_flow(  # Real Score from Teacher (without Lora) 代表真实好图的方向。pred_fake_x0  告诉generator ：你当前这个少步 generator 生成分布，在这个 noisy 点附近，实际在往哪去
                score_model,
                noisy,
                sigma,
                c=c,
                e=e,
                guidance_scale=self.real_guidance_scale,
                use_lora=False,
            )
        )

        grad = (
            pred_fake_x0 - pred_real_x0
        )  # 你这个少步模型当前的分布方向，和 teacher 分布方向还差多少。
        denom = (x_fake - pred_real_x0).abs().flatten(1).mean(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=self.grad_norm_eps)
        grad = grad / denom.view(-1, *([1] * (x_fake.ndim - 1)))
        grad = torch.nan_to_num(grad)

        stats = {
            "dm_sigma": sigma.detach(),
            "dm_grad_abs": grad.detach().abs().flatten(1).mean(dim=1),
        }
        if fake_score_tt is not None:
            stats["dmd_fake_score_t"] = sigma.detach()
            stats["dmd_fake_score_tt"] = fake_score_tt.detach()
            stats["dmd_fake_score_has_tt"] = torch.ones_like(sigma.detach(), dtype=torch.float32)
        if return_extra:
            extra = {
                "dmd_noisy": noisy.detach(),
                "dmd_noise": noise.detach(),
                "pred_fake_x0": pred_fake_x0.detach(),
                "pred_real_x0": pred_real_x0.detach(),
                "fake_score_flow": fake_score_flow.detach(),
                "real_score_flow": real_score_flow.detach(),
            }
            if fake_score_tt is not None:
                extra["dmd_fake_score_t"] = sigma.detach()
                extra["dmd_fake_score_tt"] = fake_score_tt.detach()
                extra["dmd_fake_score_has_tt"] = torch.ones_like(
                    sigma.detach(), dtype=torch.float32
                )
            return grad, stats, extra
        return grad, stats

    def _generator_dmd_loss_impl(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        sigma_override: Optional[torch.Tensor] = None,
        noise_override: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[
            torch.Tensor,
            Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
    ]:
        if latent_shape is None and initial_noise is None:
            x_fake, gen_meta = self.generate_one_step_latents(
                generator_model,
                x_real,
                c,
                return_internal=return_debug_tensors,
                sigma_override=sigma_override,
                noise_override=noise_override,
            )
        else:
            x_fake, gen_meta = self.generate_one_step_latents(
                generator_model,
                x_real,
                c,
                latent_shape=latent_shape,
                initial_noise=initial_noise,
                return_internal=return_debug_tensors,
                sigma_override=sigma_override,
                noise_override=noise_override,
            )
        with torch.no_grad():
            dmd_grad, dm_meta, dmd_extra = self._compute_dmd_grad(
                score_model,
                x_fake,
                c,
                e,
                gen_input_sigma=gen_meta["gen_input_sigma"],
                return_extra=True,
            )
        target = (x_fake - dmd_grad).detach()
        raw_mse_loss_per_sample = self._mse_loss_per_sample(x_fake, target, scale=0.5)
        pearson_loss_per_sample = torch.zeros_like(raw_mse_loss_per_sample)
        pearson4_pearson_weight = torch.zeros_like(raw_mse_loss_per_sample)
        pearson4_mse_weight = torch.ones_like(raw_mse_loss_per_sample)
        if self.loss_type == "pearson4":
            dmd_x_fake_for_loss = self._spatial_standardize(x_fake)
            dmd_target_for_loss = self._spatial_standardize(target).detach()
            pearson_loss_per_sample = self._mse_loss_per_sample(
                dmd_x_fake_for_loss,
                dmd_target_for_loss,
                scale=0.5,
            )
            pearson4_pearson_weight = (
                gen_meta["gen_input_sigma"]
                .detach()
                .to(device=x_fake.device, dtype=torch.float32)
                .flatten()
                .clamp(0.0, 1.0)
            )
            if pearson4_pearson_weight.numel() != x_fake.shape[0]:
                raise ValueError("gen_input_sigma must have one value per sample for pearson4")
            dm_loss = (
                pearson4_pearson_weight * pearson_loss_per_sample
                + pearson4_mse_weight * raw_mse_loss_per_sample
            ).mean()
        else:
            dm_loss = raw_mse_loss_per_sample.mean()
        loss = dm_loss * self.dm_loss_weight
        gt_anchor_loss = torch.tensor(0.0, device=x_fake.device)
        rcgm_loss = torch.tensor(0.0, device=x_fake.device)
        recon_loss = torch.tensor(0.0, device=x_fake.device)

        rcgm_meta = None
        rcgm_debug = None
        if self.rcgm_loss_weight > 0 and not self.is_warmup_step():
            rcgm_out = self._rcgm_regularizer_loss(
                score_model=score_model,
                x_fake=x_fake,
                gen_input_sigma=gen_meta["gen_input_sigma"],
                c=c,
                e=e,
                return_debug_tensors=return_debug_tensors,
            )
            if return_debug_tensors:
                rcgm_loss, rcgm_meta, rcgm_debug = rcgm_out
            else:
                rcgm_loss, rcgm_meta = rcgm_out
            loss = loss + self.rcgm_loss_weight * rcgm_loss

        if self.gt_grad_loss_weight > 0:
            # Optional anchor term from the video implementation to reduce early drift.
            gt_grad = dmd_extra["pred_fake_x0"] - x_real
            gt_denom = (
                gt_grad.abs().flatten(1).mean(dim=1, keepdim=True).clamp(min=self.grad_norm_eps)
            )
            gt_grad = gt_grad / gt_denom.view(-1, *([1] * (x_fake.ndim - 1)))
            gt_target = (x_fake - gt_grad).detach()
            gt_anchor_loss = 0.5 * F.mse_loss(x_fake.float(), gt_target.float(), reduction="mean")
            loss = loss + self.gt_grad_loss_weight * gt_anchor_loss

        if self.generator_recon_weight > 0:
            recon_loss = F.mse_loss(x_fake.float(), x_real.float(), reduction="mean")
            loss = loss + self.generator_recon_weight * recon_loss

        stats = self._pack_loss_stats(
            loss_gen=loss.detach(),
            loss_gen_dm=(dm_loss.detach() * self.dm_loss_weight),
            loss_gen_rcgm=(rcgm_loss.detach() * self.rcgm_loss_weight),
            loss_gen_rcgm_warmup=torch.tensor(0.0, device=x_fake.device),
            loss_gen_gt_anchor=(gt_anchor_loss.detach() * self.gt_grad_loss_weight),
            loss_gen_recon=(recon_loss.detach() * self.generator_recon_weight),
            gen_input_sigma=gen_meta["gen_input_sigma"],
            gen_backward_simulation=gen_meta["gen_backward_simulation"],
            dm_sigma=dm_meta["dm_sigma"],
            dm_grad_abs=dm_meta["dm_grad_abs"],
            x_fake_abs=x_fake.detach().abs().flatten(1).mean(dim=1),
        )
        if rcgm_meta is not None:
            stats.update(self._pack_loss_stats(**rcgm_meta))
        extra_dm_stats = {k: v for k, v in dm_meta.items() if k not in {"dm_sigma", "dm_grad_abs"}}
        if extra_dm_stats:
            stats.update(self._pack_loss_stats(**extra_dm_stats))
        stats.update(
            self._pack_loss_stats(
                dmd_pearson4_pearson_loss=pearson_loss_per_sample.detach(),
                dmd_pearson4_raw_mse_loss=raw_mse_loss_per_sample.detach(),
                dmd_pearson4_pearson_weight=pearson4_pearson_weight.detach(),
                dmd_pearson4_mse_weight=pearson4_mse_weight.detach(),
            )
        )
        if return_debug_tensors or return_log_tensors:
            aux = {
                "gen_input_sigma": gen_meta["gen_input_sigma"].detach(),
                "gen_backward_simulation": gen_meta["gen_backward_simulation"].detach(),
                "dm_sigma": dm_meta["dm_sigma"].detach(),
                "x_fake_live": x_fake,
                "dmd_pearson4_raw_target": target.detach(),
                "dmd_pearson4_pearson_loss": pearson_loss_per_sample.detach(),
                "dmd_pearson4_raw_mse_loss": raw_mse_loss_per_sample.detach(),
                "dmd_pearson4_pearson_weight": pearson4_pearson_weight.detach(),
                "dmd_pearson4_mse_weight": pearson4_mse_weight.detach(),
            }
            if "gen_step_index" in gen_meta:
                aux["gen_step_index"] = gen_meta["gen_step_index"].detach()
            if rcgm_meta is not None:
                aux["rcgm_sigma_s"] = rcgm_meta["rcgm_sigma_s"].detach()
                aux["rcgm_sigma_t"] = rcgm_meta["rcgm_sigma_t"].detach()
            if "dmd_fake_score_t" in dmd_extra:
                aux["dmd_fake_score_t"] = dmd_extra["dmd_fake_score_t"].detach()
            if "dmd_fake_score_tt" in dmd_extra:
                aux["dmd_fake_score_tt"] = dmd_extra["dmd_fake_score_tt"].detach()
            if "dmd_fake_score_has_tt" in dmd_extra:
                aux["dmd_fake_score_has_tt"] = dmd_extra["dmd_fake_score_has_tt"].detach()
            if return_debug_tensors:
                if "gen_x_t" in gen_meta:
                    aux["gen_x_t"] = gen_meta["gen_x_t"].detach()
                if "gen_noise" in gen_meta:
                    aux["gen_noise"] = gen_meta["gen_noise"].detach()
                if "gen_flow" in gen_meta:
                    aux["gen_flow"] = gen_meta["gen_flow"].detach()
                if "dmd_noisy" in dmd_extra:
                    aux["dmd_noisy"] = dmd_extra["dmd_noisy"].detach()
                if "dmd_noise" in dmd_extra:
                    aux["dmd_noise"] = dmd_extra["dmd_noise"].detach()
                aux["x_fake"] = x_fake.detach()
                aux["pred_x0"] = dmd_extra["pred_fake_x0"].detach()
                aux["pred_fake_x0"] = dmd_extra["pred_fake_x0"].detach()
                aux["pred_real_x0"] = dmd_extra["pred_real_x0"].detach()
                aux["fake_score_flow"] = dmd_extra["fake_score_flow"].detach()
                aux["real_score_flow"] = dmd_extra["real_score_flow"].detach()
                if rcgm_debug is not None:
                    aux["rcgm_x0_from_vec_gen"] = rcgm_debug["rcgm_x0_from_vec_gen"]
                    aux["rcgm_x0_from_vec_teacher"] = rcgm_debug["rcgm_x0_from_vec_teacher"]
            elif return_log_tensors:
                aux["x_fake"] = x_fake.detach()
            return loss, stats, aux
        return loss, stats

    def _generator_rcgm_warmup_loss_impl(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[
            torch.Tensor,
            Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
    ]:
        with torch.no_grad():
            if latent_shape is None and initial_noise is None:
                x_fake, gen_meta = self.generate_one_step_latents(
                    generator_model,
                    x_real,
                    c,
                )
            else:
                x_fake, gen_meta = self.generate_one_step_latents(
                    generator_model,
                    x_real,
                    c,
                    latent_shape=latent_shape,
                    initial_noise=initial_noise,
                )

        rcgm_out = self._rcgm_warmup_regularizer_loss(
            generator_model=generator_model,
            score_model=score_model,
            x_fake=x_fake,
            gen_input_sigma=gen_meta["gen_input_sigma"],
            c=c,
            e=e,
            return_debug_tensors=return_debug_tensors,
        )
        if return_debug_tensors:
            rcgm_loss, rcgm_meta, rcgm_debug = rcgm_out
        else:
            rcgm_loss, rcgm_meta = rcgm_out
            rcgm_debug = None

        loss = self.rcgm_warmup_loss_weight * rcgm_loss
        stats = self._pack_loss_stats(
            loss_gen=loss.detach(),
            loss_gen_rcgm_warmup=loss.detach(),
            gen_input_sigma=gen_meta["gen_input_sigma"],
            gen_backward_simulation=gen_meta["gen_backward_simulation"],
            x_fake_abs=x_fake.detach().abs().flatten(1).mean(dim=1),
        )
        stats.update(self._pack_loss_stats(**rcgm_meta))

        if return_debug_tensors or return_log_tensors:
            aux = {
                "gen_input_sigma": gen_meta["gen_input_sigma"].detach(),
                "gen_backward_simulation": gen_meta["gen_backward_simulation"].detach(),
                "dm_sigma": rcgm_meta["rcgm_sigma_t"].detach(),
                "rcgm_sigma_s": rcgm_meta["rcgm_sigma_s"].detach(),
                "rcgm_sigma_t": rcgm_meta["rcgm_sigma_t"].detach(),
                "x_fake_live": x_fake,
            }
            if "gen_step_index" in gen_meta:
                aux["gen_step_index"] = gen_meta["gen_step_index"].detach()
            if return_debug_tensors:
                aux["x_fake"] = x_fake.detach()
                aux["pred_x0"] = x_fake.detach()
                if rcgm_debug is not None:
                    aux["rcgm_x0_from_vec_gen"] = rcgm_debug["rcgm_x0_from_vec_gen"]
                    aux["rcgm_x0_from_vec_teacher"] = rcgm_debug["rcgm_x0_from_vec_teacher"]
            elif return_log_tensors:
                aux["x_fake"] = x_fake.detach()
            return loss, stats, aux
        return loss, stats

    def _generator_ode_pair_warmup_loss_impl(
        self,
        generator_model: Union[nn.Module, Callable],
        x_real: torch.Tensor,
        initial_noise: torch.Tensor,
        c: List[torch.Tensor],
        ode_weight: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ):
        if x_real is None or initial_noise is None:
            raise ValueError("ODE pair warmup requires x_real clean latents and initial_noise")
        clean = x_real.to(torch.float32)
        noise = initial_noise.to(device=clean.device, dtype=clean.dtype)
        if clean.shape != noise.shape:
            raise ValueError(
                f"ODE pair shape mismatch: clean={tuple(clean.shape)} noise={tuple(noise.shape)}"
            )
        batch_size = clean.shape[0]
        sigma = self._sample_ode_warmup_generator_sigmas(batch_size, clean.device)
        sigma_b = self._broadcast_sigma(sigma, clean)
        x_t = clean * (1.0 - sigma_b) + noise * sigma_b
        target_flow = noise - clean
        pred_flow = self._call_model(generator_model, x_t, sigma, c)
        pred_clean = x_t - sigma_b * pred_flow.float()
        per_sample_loss = pred_flow.float().sub(target_flow.float()).square().flatten(1).mean(dim=1)
        if ode_weight is None:
            weight = torch.ones(batch_size, device=clean.device, dtype=torch.float32)
        else:
            weight = ode_weight.to(device=clean.device, dtype=torch.float32).flatten()
            if weight.shape[0] != batch_size:
                raise ValueError(
                    f"ODE weight shape mismatch: weights={tuple(weight.shape)} batch={batch_size}"
                )
        fm_loss_unweighted = per_sample_loss.mean()
        fm_loss = (per_sample_loss * weight).mean()
        loss = fm_loss * self.ode_warmup_loss_weight
        stats = self._pack_loss_stats(
            loss_gen=loss.detach(),
            loss_gen_ode_warmup=loss.detach(),
            loss_gen_ode_warmup_unweighted=(
                fm_loss_unweighted.detach() * self.ode_warmup_loss_weight
            ),
            gen_input_sigma=sigma.detach(),
            ode_weight_mean=weight.detach(),
            ode_loss_weight=weight.detach(),
            ode_loss_weight_mean=weight.detach(),
            ode_target_flow_abs=target_flow.detach().abs().flatten(1).mean(dim=1),
            ode_pred_flow_abs=pred_flow.detach().abs().flatten(1).mean(dim=1),
        )
        if return_debug_tensors or return_log_tensors:
            aux = {
                "gen_input_sigma": sigma.detach(),
                "dm_sigma": sigma.detach(),
                "gen_backward_simulation": torch.zeros(
                    batch_size, device=clean.device, dtype=torch.float32
                ),
                "ode_x_t": x_t.detach(),
                "ode_target_flow": target_flow.detach(),
                "ode_pred_flow": pred_flow.detach(),
                "ode_weight": weight.detach(),
                "x_fake_live": clean,
                "x_fake": clean.detach(),
            }
            if return_debug_tensors:
                aux["pred_x0"] = pred_clean.detach()
            return loss, stats, aux
        return loss, stats

    def generator_loss(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        ode_weight: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[
            torch.Tensor,
            Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
    ]:
        if self.is_ode_pair_warmup_step():
            return self._generator_ode_pair_warmup_loss_impl(
                generator_model=generator_model,
                x_real=x_real,
                initial_noise=initial_noise,
                c=c,
                ode_weight=ode_weight,
                return_debug_tensors=return_debug_tensors,
                return_log_tensors=return_log_tensors,
            )
        if self.is_rcgm_warmup_step():
            return self._generator_rcgm_warmup_loss_impl(
                generator_model=generator_model,
                score_model=score_model,
                x_real=x_real,
                c=c,
                e=e,
                latent_shape=latent_shape,
                initial_noise=initial_noise,
                return_debug_tensors=return_debug_tensors,
                return_log_tensors=return_log_tensors,
            )
        return self._generator_dmd_loss_impl(
            generator_model=generator_model,
            score_model=score_model,
            x_real=x_real,
            c=c,
            e=e,
            latent_shape=latent_shape,
            initial_noise=initial_noise,
            sigma_override=None,
            noise_override=None,
            return_debug_tensors=return_debug_tensors,
            return_log_tensors=return_log_tensors,
        )

    def generator_dmd_loss(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        sigma_override: Optional[torch.Tensor] = None,
        noise_override: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ):
        return self._generator_dmd_loss_impl(
            generator_model=generator_model,
            score_model=score_model,
            x_real=x_real,
            c=c,
            e=e,
            latent_shape=latent_shape,
            initial_noise=initial_noise,
            sigma_override=sigma_override,
            noise_override=noise_override,
            return_debug_tensors=return_debug_tensors,
            return_log_tensors=return_log_tensors,
        )

    def score_loss(
        self,
        generator_model: Union[nn.Module, Callable],
        score_model: ScoreModelLike,
        x_real: Optional[torch.Tensor],
        c: List[torch.Tensor],
        e: Optional[List[torch.Tensor]],
        latent_shape: Optional[torch.Size] = None,
        initial_noise: Optional[torch.Tensor] = None,
        return_debug_tensors: bool = False,
        return_log_tensors: bool = False,
    ) -> Union[
        Tuple[torch.Tensor, Dict[str, Tuple[torch.Tensor, torch.Tensor]]],
        Tuple[
            torch.Tensor,
            Dict[str, Tuple[torch.Tensor, torch.Tensor]],
            Dict[str, torch.Tensor],
        ],
    ]:
        def _finite_ratio(x: torch.Tensor) -> torch.Tensor:
            return torch.isfinite(x.detach()).to(torch.float32).mean()

        def _safe_absmax(x: torch.Tensor) -> torch.Tensor:
            x_det = x.detach()
            if x_det.numel() == 0:
                return torch.tensor(0.0, device=x_det.device)
            x_safe = torch.nan_to_num(x_det, nan=0.0, posinf=0.0, neginf=0.0)
            return x_safe.abs().amax().to(torch.float32)

        def _skip_nonfinite(
            stage_code: float,
            x_fake_ratio: torch.Tensor,
            noisy_ratio: torch.Tensor,
            pred_x0_ratio: torch.Tensor,
            x_fake_absmax: torch.Tensor,
            noisy_absmax: torch.Tensor,
            pred_x0_absmax: torch.Tensor,
            x_fake_dbg: torch.Tensor,
            pred_x0_dbg: Optional[torch.Tensor] = None,
        ):
            # Keep backward/optimizer flow valid while skipping this batch update.
            loss = torch.zeros(
                (), device=x_fake_dbg.device, dtype=torch.float32, requires_grad=True
            )
            stats = self._pack_loss_stats(
                loss_score=loss.detach(),
                score_loss_pre_weight=loss.detach(),
                score_loss_post_weight=loss.detach(),
                score_loss_weight=torch.tensor(1.0, device=x_fake_dbg.device),
                score_sigma=torch.zeros(
                    x_fake_dbg.shape[0], device=x_fake_dbg.device, dtype=torch.float32
                ),
                score_pred_err=torch.zeros(
                    x_fake_dbg.shape[0], device=x_fake_dbg.device, dtype=torch.float32
                ),
                score_x_fake_finite_ratio=x_fake_ratio,
                score_noisy_finite_ratio=noisy_ratio,
                score_pred_x0_finite_ratio=pred_x0_ratio,
                score_x_fake_absmax=x_fake_absmax,
                score_noisy_absmax=noisy_absmax,
                score_pred_x0_absmax=pred_x0_absmax,
                score_skipped_nonfinite=torch.tensor(1.0, device=x_fake_dbg.device),
                score_nonfinite_stage=torch.tensor(stage_code, device=x_fake_dbg.device),
            )
            if return_debug_tensors or return_log_tensors:
                aux = {
                    "score_sigma": torch.zeros(
                        x_fake_dbg.shape[0],
                        device=x_fake_dbg.device,
                        dtype=torch.float32,
                    )
                }
                if return_debug_tensors:
                    aux["x_fake"] = x_fake_dbg.detach()
                    if pred_x0_dbg is not None:
                        aux["pred_x0"] = pred_x0_dbg.detach()
                return loss, stats, aux
            return loss, stats

        with torch.no_grad():
            if latent_shape is None and initial_noise is None:
                x_fake, gen_meta = self.generate_one_step_latents(generator_model, x_real, c)
            else:
                x_fake, gen_meta = self.generate_one_step_latents(
                    generator_model,
                    x_real,
                    c,
                    latent_shape=latent_shape,
                    initial_noise=initial_noise,
                )
        x_fake_finite_ratio = _finite_ratio(x_fake)
        x_fake_absmax = _safe_absmax(x_fake)
        if x_fake_finite_ratio.item() < 1.0:
            zero_ratio = torch.tensor(0.0, device=x_fake.device)
            return _skip_nonfinite(
                stage_code=1.0,  # x_fake
                x_fake_ratio=x_fake_finite_ratio,
                noisy_ratio=zero_ratio,
                pred_x0_ratio=zero_ratio,
                x_fake_absmax=x_fake_absmax,
                noisy_absmax=torch.tensor(0.0, device=x_fake.device),
                pred_x0_absmax=torch.tensor(0.0, device=x_fake.device),
                x_fake_dbg=x_fake,
            )

        batch_size = x_fake.shape[0]
        sigma = self._sample_sigmas(batch_size, x_fake.device)
        sigma_b = self._broadcast_sigma(sigma, x_fake)
        noise = torch.randn_like(x_fake)
        noisy = sigma_b * noise + (1.0 - sigma_b) * x_fake
        noisy_finite_ratio = _finite_ratio(noisy)
        noisy_absmax = _safe_absmax(noisy)
        if noisy_finite_ratio.item() < 1.0:
            zero_ratio = torch.tensor(0.0, device=x_fake.device)
            return _skip_nonfinite(
                stage_code=2.0,  # noisy
                x_fake_ratio=x_fake_finite_ratio,
                noisy_ratio=noisy_finite_ratio,
                pred_x0_ratio=zero_ratio,
                x_fake_absmax=x_fake_absmax,
                noisy_absmax=noisy_absmax,
                pred_x0_absmax=torch.tensor(0.0, device=x_fake.device),
                x_fake_dbg=x_fake,
            )

        pred_x0, pred_flow = self._predict_x0_from_flow(
            score_model,
            noisy,
            sigma,
            c=c,
            e=e,
            guidance_scale=self.fake_guidance_scale,
            use_lora=True,
        )
        pred_x0_finite_ratio = _finite_ratio(pred_x0)
        pred_x0_absmax = _safe_absmax(pred_x0)
        if pred_x0_finite_ratio.item() < 1.0:
            return _skip_nonfinite(
                stage_code=3.0,  # pred_x0
                x_fake_ratio=x_fake_finite_ratio,
                noisy_ratio=noisy_finite_ratio,
                pred_x0_ratio=pred_x0_finite_ratio,
                x_fake_absmax=x_fake_absmax,
                noisy_absmax=noisy_absmax,
                pred_x0_absmax=pred_x0_absmax,
                x_fake_dbg=x_fake,
                pred_x0_dbg=pred_x0,
            )

        if self.score_loss_target == "flow":
            target_score = noise - x_fake
            pred_score = pred_flow
        else:
            target_score = x_fake
            pred_score = pred_x0

        score_mse, score_meta = self._weighted_mse_target(
            pred_score, target_score, sigma, return_meta=True
        )
        loss = score_mse * self.score_loss_weight
        stats = self._pack_loss_stats(
            loss_score=loss.detach(),
            score_loss_pre_weight=score_meta["score_loss_pre_weight"],
            score_loss_post_weight=(score_meta["score_loss_post_weight"] * self.score_loss_weight),
            score_loss_weight=score_meta["score_loss_weight"],
            score_gen_backward_simulation=gen_meta["gen_backward_simulation"],
            score_sigma=sigma.detach(),
            score_pred_err=(pred_score.detach() - target_score.detach())
            .abs()
            .flatten(1)
            .mean(dim=1),
            score_x_fake_finite_ratio=x_fake_finite_ratio,
            score_noisy_finite_ratio=noisy_finite_ratio,
            score_pred_x0_finite_ratio=pred_x0_finite_ratio,
            score_x_fake_absmax=x_fake_absmax,
            score_noisy_absmax=noisy_absmax,
            score_pred_x0_absmax=pred_x0_absmax,
            score_skipped_nonfinite=torch.tensor(0.0, device=x_fake.device),
            score_nonfinite_stage=torch.tensor(0.0, device=x_fake.device),
        )
        if return_debug_tensors or return_log_tensors:
            aux = {
                "score_sigma": sigma.detach(),
                "gen_backward_simulation": gen_meta["gen_backward_simulation"].detach(),
            }
            if "gen_step_index" in gen_meta:
                aux["gen_step_index"] = gen_meta["gen_step_index"].detach()
            if return_debug_tensors:
                aux["x_fake"] = x_fake.detach()
                aux["pred_x0"] = pred_x0.detach()
            return loss, stats, aux
        return loss, stats

    def _pack_loss_stats(
        self, **scalars: torch.Tensor
    ) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
        out: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        for key, value in scalars.items():
            if value.ndim == 0:
                out[key] = (
                    value.detach().to(torch.float64),
                    torch.tensor(1.0, device=value.device, dtype=torch.float64),
                )
            else:
                out[key] = (
                    value.detach().to(torch.float64).sum(),
                    torch.tensor(float(value.numel()), device=value.device, dtype=torch.float64),
                )
        return out
