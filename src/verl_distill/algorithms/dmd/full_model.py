from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from .method import ScoreModelLike, StandardDMD


class FullModelDMD(StandardDMD):
    """DMD with independent full models for fake/real score -- no LoRA.

    Expects ``score_model`` to be a dict ``{"real": nn.Module, "fake": nn.Module}``.
    The parent class's ``_select_score_branch()`` already handles dict routing
    for ``_compute_dmd_grad``, ``_predict_x0_from_flow``, ``generator_loss``, etc.
    ODE warmup generator sampling is also inherited from ``StandardDMD``. Only
    ``score_loss()`` is overridden to remove LoRA toggle calls.
    """

    def score_loss(
        self,
        generator_model: Union[nn.Module, callable],
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
            loss = torch.zeros(
                (), device=x_fake_dbg.device, dtype=torch.float32, requires_grad=True
            )
            stats = self._pack_loss_stats(
                loss_score=loss.detach(),
                score_loss_pre_weight=loss.detach(),
                score_loss_post_weight=loss.detach(),
                score_loss_weight=torch.tensor(1.0, device=x_fake_dbg.device),
                score_sigma=torch.zeros(
                    x_fake_dbg.shape[0],
                    device=x_fake_dbg.device,
                    dtype=torch.float32,
                ),
                score_pred_err=torch.zeros(
                    x_fake_dbg.shape[0],
                    device=x_fake_dbg.device,
                    dtype=torch.float32,
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
        fake_score_tt = None
        if self.fake_score_use_generator_timestep:
            if not isinstance(score_model, dict):
                raise ValueError(
                    "fake_score_use_generator_timestep=True requires explicit "
                    "score_model dict with fake and real branches"
                )
            if "gen_input_sigma" not in gen_meta:
                raise ValueError("fake_score_use_generator_timestep=True requires gen_input_sigma")
            fake_score_tt = (
                gen_meta["gen_input_sigma"]
                .detach()
                .to(device=x_fake.device, dtype=torch.float32)
                .flatten()
            )
        x_fake_finite_ratio = _finite_ratio(x_fake)
        x_fake_absmax = _safe_absmax(x_fake)
        if x_fake_finite_ratio.item() < 1.0:
            zero_ratio = torch.tensor(0.0, device=x_fake.device)
            return _skip_nonfinite(
                stage_code=1.0,
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
                stage_code=2.0,
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
            target_timestep=fake_score_tt,
            target_timestep_log_tag="score_loss_fake_score",
        )
        pred_x0_finite_ratio = _finite_ratio(pred_x0)
        pred_x0_absmax = _safe_absmax(pred_x0)
        if pred_x0_finite_ratio.item() < 1.0:
            return _skip_nonfinite(
                stage_code=3.0,
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
        if fake_score_tt is not None:
            stats.update(
                self._pack_loss_stats(
                    score_loss_fake_score_t=sigma.detach(),
                    score_loss_fake_score_tt=fake_score_tt.detach(),
                    score_loss_fake_score_has_tt=torch.ones_like(
                        sigma.detach(), dtype=torch.float32
                    ),
                )
            )
        if return_debug_tensors or return_log_tensors:
            aux = {
                "score_sigma": sigma.detach(),
                "gen_backward_simulation": gen_meta["gen_backward_simulation"].detach(),
            }
            if fake_score_tt is not None:
                aux["score_fake_score_t"] = sigma.detach()
                aux["score_fake_score_tt"] = fake_score_tt.detach()
                aux["score_fake_score_has_tt"] = torch.ones_like(
                    sigma.detach(), dtype=torch.float32
                )
            if "gen_step_index" in gen_meta:
                aux["gen_step_index"] = gen_meta["gen_step_index"].detach()
            if return_debug_tensors:
                aux["x_fake"] = x_fake.detach()
                aux["pred_x0"] = pred_x0.detach()
            return loss, stats, aux
        return loss, stats
