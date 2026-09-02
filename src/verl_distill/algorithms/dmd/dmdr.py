from typing import Any, Dict, Optional

import torch

from .lora import lora_false, weak_lora
from .method import ScoreModelLike, StandardDMD


class DMDR(StandardDMD):
    """DMDR variant aligned to the public official demo semantics.

    This keeps the standard DMD score/generator structure, and adds:
    - direct reward loss gating via ``cold_start_iter``
    - DynaDG real-branch weak-LoRA decay
    - DynaRS biased sigma sampling that anneals to uniform
    """

    def __init__(
        self,
        reward_rl: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.reward_rl_cfg = self._build_reward_rl_cfg(reward_rl)
        self.reward_rl_enable = bool(self.reward_rl_cfg.get("enable", False))
        self.current_train_step = 0

    def _build_reward_rl_cfg(self, reward_rl: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        cfg = dict(reward_rl or {})
        cfg.update(
            {
                "enable": bool(cfg.get("enable", False)),
                "backend": str(cfg.get("backend", "none")),
                "model_path": cfg.get("model_path", None),
                "weight": float(cfg.get("weight", 1.0)),
                "cold_start_iter": max(0, int(cfg.get("cold_start_iter", 0))),
                "dynamic_step": max(0, int(cfg.get("dynamic_step", 0))),
                "gen_a": float(cfg.get("gen_a", 1.0)),
                "gen_b": float(cfg.get("gen_b", 1.0)),
                "s_type_gen": str(cfg.get("s_type_gen", "uniform")).lower(),
                "gui_a": float(cfg.get("gui_a", 1.0)),
                "gui_b": float(cfg.get("gui_b", 1.0)),
                "s_type_gui": str(cfg.get("s_type_gui", "uniform")).lower(),
                "lora_scale_r": float(cfg.get("lora_scale_r", 0.0)),
            }
        )
        return cfg

    def set_train_step(self, step: int):
        super().set_train_step(step)
        self.current_train_step = max(0, int(step))

    def reward_weight(self) -> float:
        return float(self.reward_rl_cfg.get("weight", 1.0))

    def should_apply_reward(self, global_step: Optional[int] = None) -> bool:
        if not self.reward_rl_enable:
            return False
        step = self.current_train_step if global_step is None else int(global_step)
        return step >= int(self.reward_rl_cfg.get("cold_start_iter", 0))

    def _annealed_beta_params(
        self,
        alpha: float,
        beta: float,
        step: Optional[int] = None,
        dynamic_step: Optional[int] = None,
    ):
        if step is None:
            step = self.current_train_step
        if dynamic_step is None:
            dynamic_step = int(self.reward_rl_cfg.get("dynamic_step", 0))
        if dynamic_step <= 0:
            return float(alpha), float(beta)
        progress = min(max(float(step), 0.0) / float(dynamic_step), 1.0)
        cosine_decay = 0.5 * (1.0 + torch.cos(torch.tensor(progress * torch.pi))).item()
        alpha_out = 1.0 + (float(alpha) - 1.0) * cosine_decay
        beta_out = 1.0 + (float(beta) - 1.0) * cosine_decay
        return float(alpha_out), float(beta_out)

    def _sample_beta_continuous(
        self,
        batch_size: int,
        device: torch.device,
        alpha: float,
        beta: float,
        s_type: str,
    ) -> torch.Tensor:
        s_type = str(s_type).lower()
        if s_type == "uniform":
            return torch.rand(batch_size, device=device, dtype=torch.float32).clamp_min(1e-3)
        if s_type != "logit_normal":
            raise ValueError(
                f"Unsupported DMDR sigma sampler type '{s_type}'. "
                "Choose from {'uniform', 'logit_normal'}."
            )
        alpha_dyn, beta_dyn = self._annealed_beta_params(alpha, beta)
        dist = torch.distributions.Beta(alpha_dyn, beta_dyn)
        return dist.sample((batch_size,)).to(device=device, dtype=torch.float32)

    def _sample_beta_discrete_levels(
        self,
        levels: torch.Tensor,
        batch_size: int,
        alpha: float,
        beta: float,
        s_type: str,
    ) -> torch.Tensor:
        s_type = str(s_type).lower()
        levels_f = levels.to(dtype=torch.float32)
        if s_type == "uniform":
            idx = torch.randint(
                0,
                levels_f.numel(),
                (batch_size,),
                device=levels_f.device,
            )
            return levels_f[idx]
        if s_type != "logit_normal":
            raise ValueError(
                f"Unsupported DMDR sigma sampler type '{s_type}'. "
                "Choose from {'uniform', 'logit_normal'}."
            )
        alpha_dyn, beta_dyn = self._annealed_beta_params(alpha, beta)
        dist = torch.distributions.Beta(alpha_dyn, beta_dyn)
        t = dist.sample((batch_size,)).to(device=levels_f.device, dtype=torch.float32)
        distances = torch.abs(t.unsqueeze(-1) - levels_f.view(1, -1))
        idx = distances.argmin(dim=-1)
        return levels_f[idx]

    def _sample_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        cfg = self.reward_rl_cfg
        return self._sample_beta_continuous(
            batch_size=batch_size,
            device=device,
            alpha=float(cfg["gui_a"]),
            beta=float(cfg["gui_b"]),
            s_type=str(cfg["s_type_gui"]),
        )

    def _sample_generator_sigmas(self, batch_size: int, device: torch.device) -> torch.Tensor:
        levels = self._generator_sigma_levels(
            self.num_denoising_step,
            device=device,
            dtype=torch.float32,
            include_terminal_zero=False,
        )
        cfg = self.reward_rl_cfg
        return self._sample_beta_discrete_levels(
            levels=levels,
            batch_size=batch_size,
            alpha=float(cfg["gen_a"]),
            beta=float(cfg["gen_b"]),
            s_type=str(cfg["s_type_gen"]),
        )

    def current_real_lora_scale(self, global_step: Optional[int] = None) -> float:
        cfg = self.reward_rl_cfg
        base_scale = float(cfg.get("lora_scale_r", 0.0))
        dynamic_step = int(cfg.get("dynamic_step", 0))
        if base_scale <= 0.0:
            return 0.0
        step = self.current_train_step if global_step is None else int(global_step)
        if dynamic_step <= 0:
            return base_scale
        if step >= dynamic_step:
            return 0.0
        cosine_factor = (
            0.5
            * (1.0 + torch.cos(torch.tensor(float(step) * torch.pi / float(dynamic_step)))).item()
        )
        return base_scale * cosine_factor

    def _predict_real_x0_with_dynadg(
        self,
        score_model: ScoreModelLike,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        c,
        e,
    ):
        branch_model, branch_use_lora = self._select_score_branch(score_model, False)
        if branch_use_lora is None:
            pred_x0, pred_flow = self._predict_x0_from_flow(
                score_model,
                x_t,
                sigma,
                c=c,
                e=e,
                guidance_scale=self.real_guidance_scale,
                use_lora=False,
            )
            return pred_x0, pred_flow, 0.0

        base_model = self._unwrap(branch_model)
        real_lora_scale = float(self.current_real_lora_scale())
        if real_lora_scale > 0.0:
            weak_lora(base_model, alpha=real_lora_scale)
        else:
            lora_false(base_model)
        try:
            F_cond = self._call_model(branch_model, x_t, sigma, c)
            if self.real_guidance_scale != 0.0 and e is not None:
                F_uncond = self._call_model(branch_model, x_t, sigma, e)
                F_pred = F_cond + (F_cond - F_uncond) * self.real_guidance_scale
            else:
                F_pred = F_cond
            x0 = x_t - self._broadcast_sigma(sigma, x_t) * F_pred
        finally:
            lora_false(base_model)
        return x0, F_pred, real_lora_scale

    def _compute_dmd_grad(
        self,
        score_model: ScoreModelLike,
        x_fake: torch.Tensor,
        c,
        e,
        gen_input_sigma: Optional[torch.Tensor] = None,
        return_extra: bool = False,
    ):
        del gen_input_sigma
        batch_size = x_fake.shape[0]
        sigma = self._sample_sigmas(batch_size, x_fake.device)
        noise = torch.randn_like(x_fake)
        sigma_b = self._broadcast_sigma(sigma, x_fake)
        noisy = sigma_b * noise + (1.0 - sigma_b) * x_fake

        pred_fake_x0, _ = self._predict_x0_from_flow(
            score_model,
            noisy,
            sigma,
            c=c,
            e=e,
            guidance_scale=self.fake_guidance_scale,
            use_lora=True,
        )
        pred_real_x0, _, real_lora_scale = self._predict_real_x0_with_dynadg(
            score_model,
            noisy,
            sigma,
            c=c,
            e=e,
        )

        grad = pred_fake_x0 - pred_real_x0
        denom = (x_fake - pred_real_x0).abs().flatten(1).mean(dim=1, keepdim=True)
        denom = torch.clamp(denom, min=self.grad_norm_eps)
        grad = grad / denom.view(-1, *([1] * (x_fake.ndim - 1)))
        grad = torch.nan_to_num(grad)

        stats = {
            "dm_sigma": sigma.detach(),
            "dm_grad_abs": grad.detach().abs().flatten(1).mean(dim=1),
            "dm_real_lora_scale": torch.full(
                (batch_size,),
                float(real_lora_scale),
                device=x_fake.device,
                dtype=torch.float32,
            ),
        }
        if return_extra:
            return (
                grad,
                stats,
                {
                    "pred_fake_x0": pred_fake_x0.detach(),
                    "pred_real_x0": pred_real_x0.detach(),
                },
            )
        return grad, stats
