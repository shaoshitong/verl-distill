import inspect
import math
from contextlib import contextmanager
from typing import List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

from verl_distill.algorithms.dmd.lora import LoRALayer

from .dual_config import (
    normalize_exit_point_choices,
    normalize_exit_point_weights,
)
from .losses import (
    dual_alignment_loss,
    pair_diff_stats,
)
from .method import ModelLike, StandardOPD
from .rollout import (
    cat_or_empty,
    rollout_step_from_velocity,
)


class DualDistilledDiscriminatorOPD(StandardOPD):
    """Dual-DINO-distilled GAN objective with frozen alignment target A and trainable discriminator B."""

    FULL_TEACHER_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE = (
        "dual_distilled_early_exit_x0_full_teacher_real_separate_discs"
    )
    DATA_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE = (
        "dual_distilled_early_exit_x0_data_real_separate_discs"
    )
    CURRICULUM_SEPARATE_EXIT_DISCS_LOSS_MODE = (
        "dual_distilled_early_exit_x0_curriculum_separate_discs"
    )
    SEPARATE_EXIT_DISCS_LOSS_MODE = FULL_TEACHER_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE
    SEPARATE_EXIT_DISCS_LOSS_MODES = frozenset(
        {
            FULL_TEACHER_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE,
            DATA_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE,
            CURRICULUM_SEPARATE_EXIT_DISCS_LOSS_MODE,
        }
    )
    _LOSS_MODES = {
        "dual_distilled_early_exit_xt",
        "dual_distilled_early_exit_x0",
        "dual_distilled_early_exit_x0_full_teacher_real",
        *SEPARATE_EXIT_DISCS_LOSS_MODES,
    }
    _ROLLOUT_MODES = {"random_uniform", "trajectory_bernoulli", "student"}
    _GENERATOR_REGULARIZER_TYPES = {"none", "flow_matching", "dmd"}
    _GENERATOR_REFL_MODES = {"refl", "reward_gan"}
    _DMD_SCORE_LOSS_TARGETS = {"x0"}
    _DMD_FAKE_SCORE_TT_MODES = {"exit_sigma"}

    def __init__(
        self,
        discriminator_update_ratio: int = 3,
        phase_schedule_switch_step: Optional[int] = None,
        phase_schedule_after_pattern: Optional[Tuple[str, ...]] = None,
        discriminator_t_min: float = 0.02,
        discriminator_t_max: float = 0.98,
        discriminator_loss_weight: float = 1.0,
        discriminator_loss_weight_by_exit: Optional[Tuple[float, ...]] = None,
        generator_loss_weight: float = 1.0,
        generator_adv_loss_max: Optional[float] = None,
        generator_adv_loss_min: Optional[float] = None,
        generator_gan_loss_weight_by_exit: Optional[Tuple[float, ...]] = None,
        rollout_mode: str = "random_uniform",
        loss_mode: str = "dual_distilled_early_exit_xt",
        exit_step_mode: str = "random",
        fixed_exit_step: Optional[int] = None,
        exit_point_choices: Optional[Tuple[int, ...]] = None,
        exit_point_weights: Optional[Tuple[float, ...]] = None,
        rollout_stochast_ratio: float = 1.0,
        dual_align_loss_weight: float = 1.0,
        dual_align_mse_weight: float = 0.5,
        dual_align_pearson_eps: float = 1.0e-6,
        dual_align_pearson_mode: str = "channel",
        dual_frozen_discriminator_t: float = 0.005,
        x0_noise_sigma_by_exit: Optional[Tuple[float, ...]] = None,
        x0_noise_share_fake_real: bool = True,
        generator_dual_align_loss_weight: float = 0.0,
        generator_dual_align_mse_weight: float = 0.5,
        generator_dual_align_pearson_eps: float = 1.0e-6,
        generator_dual_align_pearson_mode: str = "channel",
        generator_flow_matching_weight: float = 0.0,
        generator_flow_matching_t_min: float = 0.02,
        generator_flow_matching_t_max: float = 0.6,
        generator_regularizer_type: str = "flow_matching",
        generator_dmd_regularizer_weight: float = 0.0,
        generator_refl_mode: str = "refl",
        generator_refl_weight: float = 0.0,
        reward_gan_discriminator_weight: float = 0.0,
        reward_gan_generator_weight: float = 0.0,
        generator_refl_warmup_steps: int = 0,
        generator_refl_clip_score: Optional[float] = None,
        dmd_fake_score_loss_weight: float = 0.0,
        dmd_generator_update_ratio: int = 5,
        dmd_score_t_min: float = 0.02,
        dmd_score_t_max: float = 0.98,
        dmd_grad_norm_eps: float = 1.0e-6,
        dmd_fake_x0_smooth_num_samples: int = 5,
        dmd_fake_x0_smooth_noise_scale: float = 0.01,
        dmd_real_score_rollout_steps: int = 1,
        dmd_score_loss_target: str = "x0",
        dmd_fake_score_tt_mode: str = "exit_sigma",
        real_data_curriculum_start_step: int = 0,
        real_data_curriculum_end_step: Optional[int] = None,
        real_data_curriculum_schedule: str = "linear",
        exit_gap_temperature_target_by_exit: Optional[Tuple[float, ...]] = None,
        apt_r1_weight: float = 0.0,
        apt_r1_sigma: Optional[Tuple[float, ...]] = None,
        apt_r1_sigma_min: Optional[float] = None,
        apt_r1_sigma_max: Optional[float] = None,
        apt_r2_weight: float = 0.0,
        feature_matching_weight: float = 0.0,
        perception_loss_weight: float = 0.0,
        pf_vjp_loss_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.discriminator_update_ratio = int(discriminator_update_ratio)
        if self.discriminator_update_ratio < 0:
            raise ValueError("discriminator_update_ratio must be >= 0")
        self.phase_schedule_switch_step = (
            None if phase_schedule_switch_step is None else int(phase_schedule_switch_step)
        )
        self.phase_schedule_after_pattern = self._normalize_phase_schedule_pattern(
            phase_schedule_after_pattern
        )
        if (self.phase_schedule_switch_step is None) != (self.phase_schedule_after_pattern is None):
            raise ValueError(
                "phase_schedule_switch_step and phase_schedule_after_pattern must be set together"
            )
        if self.phase_schedule_switch_step is not None and self.phase_schedule_switch_step < 0:
            raise ValueError("phase_schedule_switch_step must be >= 0")

        self.discriminator_t_min = float(discriminator_t_min)
        self.discriminator_t_max = float(discriminator_t_max)
        if not 0.0 <= self.discriminator_t_min <= self.discriminator_t_max <= 1.0:
            raise ValueError(
                "discriminator_t_min/discriminator_t_max must satisfy 0 <= min <= max <= 1"
            )
        self.discriminator_loss_weight = float(discriminator_loss_weight)
        self.generator_loss_weight = float(generator_loss_weight)
        if self.discriminator_loss_weight < 0.0:
            raise ValueError("discriminator_loss_weight must be >= 0")
        if self.generator_loss_weight < 0.0:
            raise ValueError("generator_loss_weight must be >= 0")
        self.discriminator_loss_weight_by_exit = self._normalize_loss_weight_by_exit(
            discriminator_loss_weight_by_exit,
            self.num_student_steps,
            "discriminator_loss_weight_by_exit",
        )
        self.generator_adv_loss_max = generator_adv_loss_max
        if self.generator_adv_loss_max is not None and float(self.generator_adv_loss_max) <= 0.0:
            raise ValueError("generator_adv_loss_max must be > 0 when set")
        self.generator_adv_loss_min = generator_adv_loss_min
        if self.generator_adv_loss_min is not None and float(self.generator_adv_loss_min) < 0.0:
            raise ValueError("generator_adv_loss_min must be >= 0 when set")
        self.generator_gan_loss_weight_by_exit = self._normalize_loss_weight_by_exit(
            generator_gan_loss_weight_by_exit,
            self.num_student_steps,
            "generator_gan_loss_weight_by_exit",
        )

        self.loss_mode = str(loss_mode)
        if self.loss_mode not in self._LOSS_MODES:
            raise ValueError(
                f"loss_mode must be one of {sorted(self._LOSS_MODES)}, got {self.loss_mode!r}"
            )
        self.rollout_mode = str(rollout_mode)
        if self.rollout_mode not in self._ROLLOUT_MODES:
            raise ValueError(
                "rollout_mode must be one of "
                f"{sorted(self._ROLLOUT_MODES)} for DualDistilledDiscriminatorOPD"
            )
        self.exit_step_mode = str(exit_step_mode)
        if self.exit_step_mode not in {"random", "fixed"}:
            raise ValueError("exit_step_mode must be 'random' or 'fixed'")
        self.exit_point_choices, self._allowed_exit_steps = normalize_exit_point_choices(
            exit_point_choices,
            self.num_student_steps,
        )
        self.exit_point_weights = normalize_exit_point_weights(
            exit_point_weights,
            self.exit_point_choices,
        )
        if self.exit_step_mode == "fixed":
            if fixed_exit_step is None:
                raise ValueError("fixed_exit_step is required when exit_step_mode='fixed'")
            fixed_exit_step = int(fixed_exit_step)
            if fixed_exit_step < 1 or fixed_exit_step > self.num_student_steps:
                raise ValueError(
                    "fixed_exit_step must be in "
                    f"[1, {self.num_student_steps}], got {fixed_exit_step}"
                )
            self.fixed_exit_step = fixed_exit_step
        else:
            self.fixed_exit_step = None if fixed_exit_step is None else int(fixed_exit_step)
        self.rollout_stochast_ratio = float(rollout_stochast_ratio)
        if not 0.0 <= self.rollout_stochast_ratio <= 1.0:
            raise ValueError("rollout_stochast_ratio must be in [0, 1]")

        self.dual_align_loss_weight = float(dual_align_loss_weight)
        self.dual_align_mse_weight = float(dual_align_mse_weight)
        self.dual_align_pearson_eps = float(dual_align_pearson_eps)
        self.dual_align_pearson_mode = str(dual_align_pearson_mode).lower()
        self.generator_dual_align_loss_weight = float(generator_dual_align_loss_weight)
        self.generator_dual_align_mse_weight = float(generator_dual_align_mse_weight)
        self.generator_dual_align_pearson_eps = float(generator_dual_align_pearson_eps)
        self.generator_dual_align_pearson_mode = str(generator_dual_align_pearson_mode).lower()
        self.generator_flow_matching_weight = float(generator_flow_matching_weight)
        self.generator_flow_matching_t_min = float(generator_flow_matching_t_min)
        self.generator_flow_matching_t_max = float(generator_flow_matching_t_max)
        self.generator_regularizer_type = str(generator_regularizer_type).lower()
        if self.generator_regularizer_type not in self._GENERATOR_REGULARIZER_TYPES:
            raise ValueError(
                "generator_regularizer_type must be one of "
                f"{sorted(self._GENERATOR_REGULARIZER_TYPES)}, "
                f"got {self.generator_regularizer_type!r}"
            )
        self.generator_dmd_regularizer_weight = float(generator_dmd_regularizer_weight)
        self.generator_refl_mode = str(generator_refl_mode).lower()
        self.generator_refl_weight = float(generator_refl_weight)
        self.reward_gan_discriminator_weight = float(reward_gan_discriminator_weight)
        self.reward_gan_generator_weight = float(reward_gan_generator_weight)
        self.generator_refl_warmup_steps = int(generator_refl_warmup_steps)
        self.generator_refl_clip_score = (
            None if generator_refl_clip_score is None else float(generator_refl_clip_score)
        )
        self.dmd_fake_score_loss_weight = float(dmd_fake_score_loss_weight)
        self.dmd_generator_update_ratio = int(dmd_generator_update_ratio)
        self.dmd_score_t_min = float(dmd_score_t_min)
        self.dmd_score_t_max = float(dmd_score_t_max)
        self.dmd_grad_norm_eps = float(dmd_grad_norm_eps)
        self.dmd_fake_x0_smooth_num_samples = int(dmd_fake_x0_smooth_num_samples)
        self.dmd_fake_x0_smooth_noise_scale = float(dmd_fake_x0_smooth_noise_scale)
        self.dmd_real_score_rollout_steps = int(dmd_real_score_rollout_steps)
        self.dmd_score_loss_target = str(dmd_score_loss_target).lower()
        self.dmd_fake_score_tt_mode = str(dmd_fake_score_tt_mode).lower()
        self.real_data_curriculum_start_step = int(real_data_curriculum_start_step)
        self.real_data_curriculum_end_step = (
            None if real_data_curriculum_end_step is None else int(real_data_curriculum_end_step)
        )
        self.real_data_curriculum_schedule = str(real_data_curriculum_schedule).lower()
        self.dual_frozen_discriminator_t = float(dual_frozen_discriminator_t)
        if self.dual_align_loss_weight < 0.0:
            raise ValueError("dual_align_loss_weight must be >= 0")
        if self.dual_align_mse_weight < 0.0:
            raise ValueError("dual_align_mse_weight must be >= 0")
        if self.dual_align_pearson_eps < 0.0:
            raise ValueError("dual_align_pearson_eps must be >= 0")
        if self.dual_align_pearson_mode not in {"channel", "flatten"}:
            raise ValueError("dual_align_pearson_mode must be 'channel' or 'flatten'")
        if self.generator_dual_align_pearson_mode not in {"channel", "flatten"}:
            raise ValueError("generator_dual_align_pearson_mode must be 'channel' or 'flatten'")
        if self.dual_frozen_discriminator_t < 0.0:
            raise ValueError("dual_frozen_discriminator_t must be >= 0")
        if self.generator_flow_matching_weight < 0.0:
            raise ValueError("generator_flow_matching_weight must be >= 0")
        if self.generator_dmd_regularizer_weight < 0.0:
            raise ValueError("generator_dmd_regularizer_weight must be >= 0")
        if self.generator_refl_mode not in self._GENERATOR_REFL_MODES:
            raise ValueError(
                "generator_refl_mode must be one of "
                f"{sorted(self._GENERATOR_REFL_MODES)}, "
                f"got {self.generator_refl_mode!r}"
            )
        if self.generator_refl_weight < 0.0:
            raise ValueError("generator_refl_weight must be >= 0")
        if self.reward_gan_discriminator_weight < 0.0:
            raise ValueError("reward_gan_discriminator_weight must be >= 0")
        if self.reward_gan_generator_weight < 0.0:
            raise ValueError("reward_gan_generator_weight must be >= 0")
        if self.generator_refl_warmup_steps < 0:
            raise ValueError("generator_refl_warmup_steps must be >= 0")
        if self.generator_refl_clip_score is not None and not math.isfinite(
            float(self.generator_refl_clip_score)
        ):
            raise ValueError("generator_refl_clip_score must be finite when set")
        if self.dmd_fake_score_loss_weight < 0.0:
            raise ValueError("dmd_fake_score_loss_weight must be >= 0")
        if self.generator_regularizer_type == "dmd":
            if self.generator_dmd_regularizer_weight <= 0.0:
                raise ValueError(
                    "generator_dmd_regularizer_weight must be > 0 when "
                    "generator_regularizer_type='dmd'"
                )
            if self.dmd_fake_score_loss_weight <= 0.0:
                raise ValueError(
                    "dmd_fake_score_loss_weight must be > 0 when generator_regularizer_type='dmd'"
                )
        if self.dmd_generator_update_ratio < 1:
            raise ValueError("dmd_generator_update_ratio must be >= 1")
        if not 0.0 <= self.dmd_score_t_min <= self.dmd_score_t_max <= 1.0:
            raise ValueError("dmd_score_t_min/t_max must satisfy 0 <= min <= max <= 1")
        if self.dmd_grad_norm_eps < 0.0:
            raise ValueError("dmd_grad_norm_eps must be >= 0")
        if self.dmd_fake_x0_smooth_num_samples < 1:
            raise ValueError("dmd_fake_x0_smooth_num_samples must be >= 1")
        if self.dmd_fake_x0_smooth_noise_scale < 0.0:
            raise ValueError("dmd_fake_x0_smooth_noise_scale must be >= 0")
        if self.dmd_real_score_rollout_steps < 1:
            raise ValueError("dmd_real_score_rollout_steps must be >= 1")
        if self.dmd_score_loss_target not in self._DMD_SCORE_LOSS_TARGETS:
            raise ValueError(
                "dmd_score_loss_target must be one of "
                f"{sorted(self._DMD_SCORE_LOSS_TARGETS)}, "
                f"got {self.dmd_score_loss_target!r}"
            )
        if self.dmd_fake_score_tt_mode not in self._DMD_FAKE_SCORE_TT_MODES:
            raise ValueError(
                "dmd_fake_score_tt_mode must be one of "
                f"{sorted(self._DMD_FAKE_SCORE_TT_MODES)}, "
                f"got {self.dmd_fake_score_tt_mode!r}"
            )
        if self.real_data_curriculum_start_step < 0:
            raise ValueError("real_data_curriculum_start_step must be >= 0")
        if (
            self.real_data_curriculum_end_step is not None
            and self.real_data_curriculum_end_step <= self.real_data_curriculum_start_step
        ):
            raise ValueError(
                "real_data_curriculum_end_step must be greater than real_data_curriculum_start_step"
            )
        if self.real_data_curriculum_schedule != "linear":
            raise ValueError("real_data_curriculum_schedule must be 'linear'")
        if not (
            0.0 <= self.generator_flow_matching_t_min <= self.generator_flow_matching_t_max <= 1.0
        ):
            raise ValueError(
                "generator_flow_matching_t_min/t_max must satisfy 0 <= min <= max <= 1"
            )
        if self.generator_flow_matching_weight != 0.0 and self.loss_mode not in {
            "dual_distilled_early_exit_x0",
            "dual_distilled_early_exit_x0_full_teacher_real",
            *self.SEPARATE_EXIT_DISCS_LOSS_MODES,
        }:
            raise ValueError("generator_flow_matching_weight is only supported for x0 loss modes")
        self.x0_noise_sigma_by_exit = self._normalize_x0_noise_sigma_by_exit(
            x0_noise_sigma_by_exit,
            self.num_student_steps,
        )
        self.x0_noise_share_fake_real = self._normalize_bool(
            x0_noise_share_fake_real,
            "x0_noise_share_fake_real",
        )
        self.exit_gap_temperature_target_by_exit = (
            self._normalize_exit_gap_temperature_target_by_exit(
                exit_gap_temperature_target_by_exit,
                self.num_student_steps,
            )
        )

        self.apt_r1_weight = float(apt_r1_weight)
        if self.apt_r1_weight < 0.0:
            raise ValueError("apt_r1_weight must be >= 0")
        self.apt_r1_sigma_min, self.apt_r1_sigma_max = self._normalize_apt_r1_sigma_range(
            apt_r1_sigma,
            apt_r1_sigma_min,
            apt_r1_sigma_max,
        )
        disabled_options = {
            "apt_r2_weight": apt_r2_weight,
            "feature_matching_weight": feature_matching_weight,
            "perception_loss_weight": perception_loss_weight,
            "pf_vjp_loss_weight": pf_vjp_loss_weight,
        }
        for name, value in disabled_options.items():
            if float(value) != 0.0:
                raise ValueError(f"{name} must be 0 for DualDistilledDiscriminatorOPD")

        self._cached_exit_step_key = None
        self._cached_exit_step_value = None
        self._forced_exit_step = None

    @staticmethod
    def _normalize_apt_r1_sigma_range(
        apt_r1_sigma,
        apt_r1_sigma_min: Optional[float],
        apt_r1_sigma_max: Optional[float],
    ) -> Tuple[float, float]:
        if apt_r1_sigma is not None:
            if apt_r1_sigma_min is not None or apt_r1_sigma_max is not None:
                raise ValueError(
                    "apt_r1_sigma cannot be combined with apt_r1_sigma_min/apt_r1_sigma_max"
                )
            if isinstance(apt_r1_sigma, str):
                values = [
                    float(part.strip())
                    for part in apt_r1_sigma.strip().strip("[]").split(",")
                    if part.strip()
                ]
            elif isinstance(apt_r1_sigma, (list, tuple)):
                values = [float(value) for value in apt_r1_sigma]
            else:
                values = [float(apt_r1_sigma)]
            if len(values) == 1:
                sigma_min = sigma_max = values[0]
            elif len(values) == 2:
                sigma_min, sigma_max = values
            else:
                raise ValueError("apt_r1_sigma must be a scalar or two values")
        else:
            sigma_min = 0.0 if apt_r1_sigma_min is None else float(apt_r1_sigma_min)
            sigma_max = sigma_min if apt_r1_sigma_max is None else float(apt_r1_sigma_max)
        if not math.isfinite(sigma_min) or not math.isfinite(sigma_max):
            raise ValueError("apt_r1_sigma values must be finite")
        if sigma_min < 0.0 or sigma_max < 0.0:
            raise ValueError("apt_r1_sigma values must be non-negative")
        if sigma_max < sigma_min:
            raise ValueError("apt_r1_sigma_max must be >= apt_r1_sigma_min")
        return sigma_min, sigma_max

    def _separate_exit_discriminators_enabled(self) -> bool:
        return self.loss_mode in self.SEPARATE_EXIT_DISCS_LOSS_MODES

    def _discriminator_head_for_exit(self, exit_step_value: int) -> Optional[str]:
        if not self._separate_exit_discriminators_enabled():
            return None
        if int(exit_step_value) == 1:
            return "exit1"
        if int(exit_step_value) == 2:
            return "exit2"
        raise ValueError(
            f"separate exit discriminators require exit_step to be 1 or 2, got {exit_step_value}"
        )

    def _resolve_real_source_is_data(self, real_source: Optional[str]) -> bool:
        if self.loss_mode == self.DATA_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE:
            if real_source is not None and str(real_source).strip().lower() != "data":
                raise ValueError(
                    "dual_distilled_early_exit_x0_data_real_separate_discs "
                    "requires real_source='data'"
                )
            return True
        if self.loss_mode == self.CURRICULUM_SEPARATE_EXIT_DISCS_LOSS_MODE:
            if real_source is None:
                return False
            normalized = str(real_source).strip().lower()
            if normalized == "data":
                return True
            if normalized == "teacher":
                return False
            raise ValueError("real_source must be 'teacher' or 'data'")
        if real_source is not None and str(real_source).strip().lower() not in {
            "teacher",
            "",
        }:
            raise ValueError("real_source is only supported as 'teacher' for this loss_mode")
        return False

    @staticmethod
    def _normalize_phase_schedule_pattern(pattern) -> Optional[Tuple[str, ...]]:
        if pattern is None:
            return None
        if isinstance(pattern, str):
            values = tuple(part.strip() for part in pattern.split(",") if part.strip())
        else:
            try:
                values = tuple(str(part).strip() for part in pattern)
            except TypeError as exc:
                raise ValueError(
                    "phase_schedule_after_pattern must be a comma-separated string or sequence"
                ) from exc
        if not values:
            raise ValueError("phase_schedule_after_pattern must not be empty")
        valid_values = {"generator", "discriminator"}
        invalid_values = [value for value in values if value not in valid_values]
        if invalid_values:
            raise ValueError(
                "phase_schedule_after_pattern values must be 'generator' or "
                f"'discriminator', got {invalid_values}"
            )
        return values

    def set_forced_exit_step(self, exit_step: int) -> None:
        exit_step = int(exit_step)
        if exit_step < 1 or exit_step > self.num_student_steps:
            raise ValueError(
                f"forced exit_step must be in [1, {self.num_student_steps}], got {exit_step}"
            )
        self._forced_exit_step = exit_step

    def clear_forced_exit_step(self) -> None:
        self._forced_exit_step = None

    @staticmethod
    def _normalize_exit_point_choices(
        exit_point_choices,
        num_student_steps: int,
    ) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
        return normalize_exit_point_choices(exit_point_choices, num_student_steps)

    @staticmethod
    def _normalize_exit_point_weights(
        exit_point_weights,
        exit_point_choices: Tuple[int, ...],
    ) -> Optional[Tuple[float, ...]]:
        return normalize_exit_point_weights(exit_point_weights, exit_point_choices)

    @staticmethod
    def _normalize_loss_weight_by_exit(
        values,
        num_student_steps: int,
        name: str,
    ) -> Optional[Tuple[float, ...]]:
        if values is None:
            return None
        if isinstance(values, str):
            stripped = values.strip().strip("[]")
            weights = tuple(float(part.strip()) for part in stripped.split(",") if part.strip())
        else:
            weights = tuple(float(value) for value in values)
        num_steps = int(num_student_steps)
        if len(weights) != num_steps:
            raise ValueError(
                f"{name} must contain "
                f"num_student_steps values, got {len(weights)} for "
                f"{num_steps} student steps"
            )
        if any(not math.isfinite(weight) for weight in weights):
            raise ValueError(f"{name} must be finite")
        if any(weight < 0.0 for weight in weights):
            raise ValueError(f"{name} must be non-negative")
        return weights

    def _discriminator_loss_weight_for_exit(
        self,
        selected_exit_step: Optional[int],
    ) -> float:
        if self.discriminator_loss_weight_by_exit is None:
            return 1.0
        if selected_exit_step is None:
            raise ValueError(
                "selected_exit_step is required when discriminator_loss_weight_by_exit is set"
            )
        exit_step = int(selected_exit_step)
        if exit_step < 1 or exit_step > self.num_student_steps:
            raise ValueError(
                f"selected_exit_step must be in [1, {self.num_student_steps}], got {exit_step}"
            )
        return float(self.discriminator_loss_weight_by_exit[exit_step - 1])

    def _generator_gan_loss_weight_for_exit(
        self,
        selected_exit_step: Optional[int],
    ) -> float:
        if self.generator_gan_loss_weight_by_exit is None:
            return 1.0
        if selected_exit_step is None:
            raise ValueError(
                "selected_exit_step is required when generator_gan_loss_weight_by_exit is set"
            )
        exit_step = int(selected_exit_step)
        if exit_step < 1 or exit_step > self.num_student_steps:
            raise ValueError(
                f"selected_exit_step must be in [1, {self.num_student_steps}], got {exit_step}"
            )
        return float(self.generator_gan_loss_weight_by_exit[exit_step - 1])

    @staticmethod
    def _normalize_x0_noise_sigma_by_exit(
        values,
        num_student_steps: int,
    ) -> Tuple[float, ...]:
        if values is None:
            parsed = (0.0,) * int(num_student_steps)
        elif isinstance(values, str):
            parsed = tuple(float(part.strip()) for part in values.split(",") if part.strip())
        elif isinstance(values, (int, float)):
            parsed = (float(values),) * int(num_student_steps)
        else:
            parsed = tuple(float(value) for value in values)
        if len(parsed) == 1 and int(num_student_steps) != 1:
            parsed = parsed * int(num_student_steps)
        if len(parsed) != int(num_student_steps):
            raise ValueError(
                "x0_noise_sigma_by_exit must contain num_student_steps values, "
                f"got {len(parsed)} for {num_student_steps} student steps"
            )
        if any(value < 0.0 or value > 1.0 for value in parsed):
            raise ValueError("x0_noise_sigma_by_exit values must be in [0, 1]")
        return parsed

    @staticmethod
    def _normalize_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "y", "on"}:
                return True
            if lowered in {"false", "0", "no", "n", "off"}:
                return False
        raise ValueError(f"{name} must be a bool")

    @staticmethod
    def _set_lora_mode(model: ModelLike, use_lora: bool):
        if not hasattr(model, "modules"):
            return
        for module in model.modules():
            if isinstance(module, LoRALayer):
                module.use_lora = bool(use_lora)
                module.weak_lora = False

    @staticmethod
    def _cat_or_empty(
        values: List[torch.Tensor],
        reference: torch.Tensor,
    ) -> torch.Tensor:
        return cat_or_empty(values, reference)

    @staticmethod
    def _pair_diff_stats(prefix: str, fake: torch.Tensor, real: torch.Tensor) -> dict:
        return pair_diff_stats(prefix, fake, real)

    @staticmethod
    def _early_exit_stat_tensors(aux: dict) -> dict:
        return {
            "opd_exit_step": aux["opd_exit_step"],
            "opd_exit_sigma": aux["opd_exit_sigma"],
            "opd_exit_rollout_steps": aux["opd_exit_rollout_steps"],
            "opd_student_next_abs": aux["opd_student_next_abs"],
            "opd_teacher_next_abs": aux["opd_teacher_next_abs"],
            "opd_alpha_roll": aux["opd_alpha_roll"],
            "opd_alpha_roll_prob": aux["opd_alpha_roll_prob"],
            "opd_rollout_stochast_ratio": aux["opd_rollout_stochast_ratio"],
            "opd_x0_noise_sigma": aux["opd_x0_noise_sigma"],
        }

    def _x0_noise_sigma_for_exit(
        self,
        exit_step: int,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        value = float(self.x0_noise_sigma_by_exit[int(exit_step) - 1])
        return torch.full((batch_size,), value, device=device, dtype=torch.float32)

    @staticmethod
    def _normalize_exit_gap_temperature_target_by_exit(
        values,
        num_student_steps: int,
    ) -> Tuple[float, ...]:
        if values is None:
            parsed = (0.0,) * int(num_student_steps)
        elif isinstance(values, str):
            parsed = tuple(float(part.strip()) for part in values.split(",") if part.strip())
        else:
            parsed = tuple(float(value) for value in values)
        if len(parsed) != int(num_student_steps):
            raise ValueError(
                "exit_gap_temperature_target_by_exit must contain "
                "num_student_steps values, "
                f"got {len(parsed)} for {num_student_steps} student steps"
            )
        if any(value < 0.0 for value in parsed):
            raise ValueError("exit_gap_temperature_target_by_exit values must be >= 0")
        return parsed

    def _gap_temperature_target_for_exit(self, exit_step: int) -> float:
        exit_step = int(exit_step)
        if exit_step < 1 or exit_step > len(self.exit_gap_temperature_target_by_exit):
            return 0.0
        return float(self.exit_gap_temperature_target_by_exit[exit_step - 1])

    def _disc_gap_temperature(
        self,
        pred_fake: torch.Tensor,
        pred_real: torch.Tensor,
        exit_step: int,
    ) -> Tuple[float, float, float]:
        target = self._gap_temperature_target_for_exit(exit_step)
        if target <= 0.0:
            return 1.0, 0.0, 0.0
        gap = float((pred_real.detach().mean() - pred_fake.detach().mean()).item())
        temperature = max(1.0, gap / target)
        return float(temperature), float(gap), float(target)

    def _gen_gap_temperature(
        self,
        pred_fake: torch.Tensor,
        exit_step: int,
    ) -> Tuple[float, float, float]:
        target = self._gap_temperature_target_for_exit(exit_step)
        if target <= 0.0:
            return 1.0, 0.0, 0.0
        gap_proxy = max(0.0, float((-pred_fake.detach().mean()).item()))
        temperature = max(1.0, gap_proxy / target)
        return float(temperature), float(gap_proxy), float(target)

    def _apply_x0_noise(
        self,
        x0: torch.Tensor,
        sigma: torch.Tensor,
        *,
        noise: Optional[torch.Tensor] = None,
        return_noise: bool = False,
    ):
        if noise is not None:
            noise = noise.to(device=x0.device, dtype=torch.float32)
        elif return_noise:
            noise = torch.randn_like(x0, dtype=torch.float32)
        if torch.all(sigma == 0):
            if return_noise:
                return x0, noise
            return x0
        sigma_view = self._broadcast_sigma(sigma, x0)
        if noise is None:
            noise = torch.randn_like(x0, dtype=torch.float32)
        noised = (1.0 - sigma_view) * x0.to(torch.float32) + sigma_view * noise
        if return_noise:
            return noised, noise
        return noised

    def _generator_flow_matching_enabled(self) -> bool:
        return (
            self.generator_regularizer_type == "flow_matching"
            and float(self.generator_flow_matching_weight) != 0.0
        )

    def _generator_adv_enabled(self) -> bool:
        return float(self.generator_loss_weight) != 0.0 and float(self.loss_weight) != 0.0

    def _discriminator_adv_enabled(self) -> bool:
        return float(self.discriminator_loss_weight) != 0.0 and float(self.loss_weight) != 0.0

    def _any_generator_objective_enabled(self) -> bool:
        return (
            self._generator_adv_enabled()
            or self._generator_flow_matching_enabled()
            or self._dmd_generator_regularizer_enabled()
            or self._generator_refl_enabled()
            or self._reward_gan_enabled()
        )

    def dmd_regularizer_enabled(self) -> bool:
        return self.generator_regularizer_type == "dmd"

    def _dmd_score_loss_enabled(self) -> bool:
        return self.dmd_regularizer_enabled() and float(self.dmd_fake_score_loss_weight) != 0.0

    def _dmd_generator_regularizer_enabled(self) -> bool:
        return (
            self.dmd_regularizer_enabled() and float(self.generator_dmd_regularizer_weight) != 0.0
        )

    def _generator_refl_enabled(self) -> bool:
        return self.generator_refl_mode == "refl" and float(self.generator_refl_weight) != 0.0

    def _reward_gan_enabled(self) -> bool:
        return self.generator_refl_mode == "reward_gan" and (
            float(self.reward_gan_discriminator_weight) != 0.0
            or float(self.reward_gan_generator_weight) != 0.0
        )

    def _generator_refl_active(self, step: Optional[int] = None) -> bool:
        if not self._generator_refl_enabled():
            return False
        if step is None:
            return True
        return int(step) >= int(self.generator_refl_warmup_steps)

    def _sample_generator_flow_matching_t(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_min = float(self.generator_flow_matching_t_min)
        t_max = float(self.generator_flow_matching_t_max)
        if t_min == t_max:
            return torch.full((batch_size,), t_min, device=device, dtype=torch.float32)
        return torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
            t_min,
            t_max,
        )

    def _sample_dmd_score_t(
        self,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        t_min = float(self.dmd_score_t_min)
        t_max = float(self.dmd_score_t_max)
        if t_min == t_max:
            return torch.full((batch_size,), t_min, device=device, dtype=torch.float32)
        return torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
            t_min,
            t_max,
        )

    def _dmd_fake_score_tt(self, exit_aux: dict, x: torch.Tensor) -> torch.Tensor:
        if self.dmd_fake_score_tt_mode != "exit_sigma":
            raise RuntimeError(
                f"Unsupported dmd_fake_score_tt_mode={self.dmd_fake_score_tt_mode!r}"
            )
        tt = exit_aux.get("opd_exit_sigma")
        if tt is None:
            raise RuntimeError("DMD fake score requires opd_exit_sigma")
        batch_size = int(x.shape[0])
        tt = tt.detach().to(device=x.device, dtype=torch.float32).flatten()
        if tt.numel() == 1 and batch_size != 1:
            tt = tt.expand(batch_size)
        if tt.numel() != batch_size:
            raise RuntimeError(f"DMD fake score tt must have {batch_size} values, got {tt.numel()}")
        return tt

    def _call_score_velocity(
        self,
        score_model: ModelLike,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: List[torch.Tensor],
        tt: Optional[torch.Tensor] = None,
        *,
        use_lora: bool,
    ) -> torch.Tensor:
        self._set_lora_mode(score_model, use_lora=use_lora)
        kwargs = {"t": t, "c": c}
        if tt is not None:
            kwargs["tt"] = tt
        output = score_model(x_t, **kwargs)
        if isinstance(output, tuple) and len(output) == 1:
            output = output[0]
        if hasattr(output, "sample"):
            output = output.sample
        return output.to(torch.float32)

    def _predict_x0_from_score_velocity(
        self,
        score_model: ModelLike,
        x_t: torch.Tensor,
        sigma: torch.Tensor,
        c: List[torch.Tensor],
        tt: Optional[torch.Tensor] = None,
        *,
        use_lora: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        velocity = self._call_score_velocity(
            score_model=score_model,
            x_t=x_t,
            t=sigma,
            c=c,
            tt=tt,
            use_lora=use_lora,
        )
        x0 = x_t.to(torch.float32) - self._broadcast_sigma(sigma, x_t) * velocity
        return x0, velocity

    def _dmd_fake_score_loss(
        self,
        discriminator_model: ModelLike,
        fake_x0_clean: torch.Tensor,
        c: List[torch.Tensor],
        fake_score_tt: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        batch_size = int(fake_x0_clean.shape[0])
        sigma = self._sample_dmd_score_t(batch_size, fake_x0_clean.device)
        noise = torch.randn_like(fake_x0_clean, dtype=torch.float32)
        sigma_b = self._broadcast_sigma(sigma, fake_x0_clean)
        noisy = sigma_b * noise + (1.0 - sigma_b) * fake_x0_clean.detach().to(torch.float32)
        pred_x0, pred_velocity = self._predict_x0_from_score_velocity(
            score_model=discriminator_model,
            x_t=noisy,
            sigma=sigma,
            c=c,
            tt=fake_score_tt,
            use_lora=True,
        )
        target = fake_x0_clean.detach().to(torch.float32)
        loss = F.mse_loss(pred_x0, target, reduction="mean")
        stats = {
            "opd_dmd_score_loss": loss.detach().view(1),
            "opd_dmd_score_sigma": sigma.detach(),
            "opd_dmd_score_fake_score_tt": fake_score_tt.detach(),
            "opd_dmd_score_pred_err": (pred_x0.detach() - target).abs().flatten(1).mean(dim=1),
        }
        aux = {
            "opd_debug_dmd_score_x0_hat": target.detach(),
            "opd_debug_dmd_score_x_t": noisy.detach(),
            "opd_debug_dmd_score_v_pred": pred_velocity.detach(),
            "opd_debug_dmd_score_pred_x0": pred_x0.detach(),
            "opd_debug_dmd_score_sigma": sigma.detach(),
            "opd_debug_dmd_score_fake_score_tt": fake_score_tt.detach(),
        }
        return loss, stats, aux

    def _dmd_predict_fake_x0_smoothed(
        self,
        discriminator_model: ModelLike,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        c: List[torch.Tensor],
        fake_score_tt: torch.Tensor,
    ) -> torch.Tensor:
        samples = max(1, int(self.dmd_fake_x0_smooth_num_samples))
        noise_scale = float(self.dmd_fake_x0_smooth_noise_scale)
        if samples == 1 or noise_scale == 0.0:
            pred_x0, _ = self._predict_x0_from_score_velocity(
                score_model=discriminator_model,
                x_t=noisy,
                sigma=sigma,
                c=c,
                tt=fake_score_tt,
                use_lora=True,
            )
            return pred_x0
        preds = []
        for _ in range(samples):
            perturbed = noisy + noise_scale * torch.randn_like(noisy, dtype=torch.float32)
            pred_x0, _ = self._predict_x0_from_score_velocity(
                score_model=discriminator_model,
                x_t=perturbed,
                sigma=sigma,
                c=c,
                tt=fake_score_tt,
                use_lora=True,
            )
            preds.append(pred_x0)
        return torch.stack(preds, dim=0).mean(dim=0)

    def _dmd_predict_real_x0(
        self,
        discriminator_model: ModelLike,
        noisy: torch.Tensor,
        sigma: torch.Tensor,
        c: List[torch.Tensor],
    ) -> torch.Tensor:
        steps = max(1, int(self.dmd_real_score_rollout_steps))
        if steps == 1:
            pred_real_x0, _ = self._predict_x0_from_score_velocity(
                score_model=discriminator_model,
                x_t=noisy,
                sigma=sigma,
                c=c,
                tt=None,
                use_lora=False,
            )
            return pred_real_x0

        x = noisy.to(torch.float32)
        sigma_start = sigma.to(device=x.device, dtype=torch.float32).flatten()
        if sigma_start.numel() == 1 and int(x.shape[0]) != 1:
            sigma_start = sigma_start.expand(int(x.shape[0]))
        if sigma_start.numel() != int(x.shape[0]):
            raise RuntimeError("DMD real-score rollout sigma must have one value per sample")
        for idx in range(steps):
            t_cur = sigma_start * float(steps - idx) / float(steps)
            t_next = sigma_start * float(steps - idx - 1) / float(steps)
            velocity = self._call_score_velocity(
                score_model=discriminator_model,
                x_t=x,
                t=t_cur,
                c=c,
                tt=None,
                use_lora=False,
            )
            x = x - self._broadcast_sigma(t_cur - t_next, x) * velocity
        return x

    def _dmd_generator_regularizer_loss(
        self,
        discriminator_model: ModelLike,
        fake_x0_clean_live: torch.Tensor,
        c: List[torch.Tensor],
        fake_score_tt: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict, dict]:
        x_fake = fake_x0_clean_live.to(torch.float32)
        batch_size = int(x_fake.shape[0])
        with torch.no_grad():
            sigma = self._sample_dmd_score_t(batch_size, x_fake.device)
            noise = torch.randn_like(x_fake, dtype=torch.float32)
            sigma_b = self._broadcast_sigma(sigma, x_fake)
            noisy = sigma_b * noise + (1.0 - sigma_b) * x_fake
            pred_fake_x0 = self._dmd_predict_fake_x0_smoothed(
                discriminator_model=discriminator_model,
                noisy=noisy,
                sigma=sigma,
                c=c,
                fake_score_tt=fake_score_tt,
            )
            pred_real_x0 = self._dmd_predict_real_x0(
                discriminator_model=discriminator_model,
                noisy=noisy,
                sigma=sigma,
                c=c,
            )
            raw_dmd_delta = pred_fake_x0 - pred_real_x0
            raw_dmd_delta_flat = raw_dmd_delta.detach().flatten(1)
            denom_raw = (
                (x_fake - pred_real_x0)
                .abs()
                .flatten(1)
                .mean(
                    dim=1,
                    keepdim=True,
                )
            )
            denom = torch.clamp(denom_raw, min=float(self.dmd_grad_norm_eps))
            dmd_grad = raw_dmd_delta / denom.view(-1, *([1] * (x_fake.ndim - 1)))
            dmd_grad = torch.nan_to_num(dmd_grad)
            target = (x_fake - dmd_grad).detach()
        dm_loss = 0.5 * F.mse_loss(x_fake, target, reduction="mean")
        dmd_grad_flat = dmd_grad.detach().flatten(1)
        stats = {
            "opd_gen_dmd_loss": dm_loss.detach().view(1),
            "opd_gen_dmd_sigma": sigma.detach(),
            "opd_gen_dmd_denom_raw": denom_raw.detach().flatten(),
            "opd_gen_dmd_denom_clamped": denom.detach().flatten(),
            "opd_gen_dmd_raw_delta_abs": raw_dmd_delta_flat.abs().mean(dim=1),
            "opd_gen_dmd_raw_delta_rms": raw_dmd_delta_flat.pow(2).mean(dim=1).sqrt(),
            "opd_gen_dmd_grad_abs": dmd_grad_flat.abs().mean(dim=1),
            "opd_gen_dmd_grad_rms": dmd_grad_flat.pow(2).mean(dim=1).sqrt(),
            "opd_gen_dmd_fake_score_tt": fake_score_tt.detach(),
            "opd_gen_dmd_fake_score_has_tt": torch.ones_like(sigma.detach(), dtype=torch.float32),
            "opd_gen_dmd_fake_smooth_samples": torch.full(
                (1,),
                float(self.dmd_fake_x0_smooth_num_samples),
                device=x_fake.device,
                dtype=torch.float32,
            ),
            "opd_gen_dmd_fake_smooth_noise_scale": torch.full(
                (1,),
                float(self.dmd_fake_x0_smooth_noise_scale),
                device=x_fake.device,
                dtype=torch.float32,
            ),
            "opd_gen_dmd_real_rollout_steps": torch.full(
                (1,),
                float(self.dmd_real_score_rollout_steps),
                device=x_fake.device,
                dtype=torch.float32,
            ),
        }
        aux = {
            "opd_debug_dmd_x_fake": x_fake.detach(),
            "opd_debug_dmd_noisy": noisy.detach(),
            "opd_debug_dmd_pred_fake_x0": pred_fake_x0.detach(),
            "opd_debug_dmd_pred_real_x0": pred_real_x0.detach(),
            "opd_debug_dmd_sigma": sigma.detach(),
            "opd_debug_dmd_fake_score_tt": fake_score_tt.detach(),
        }
        return dm_loss, stats, aux

    def _dmd_generator_regularizer_active(
        self,
        generator_update_index: Optional[int],
    ) -> bool:
        if not self._dmd_generator_regularizer_enabled():
            return False
        if generator_update_index is None:
            return True
        return int(generator_update_index) % int(self.dmd_generator_update_ratio) == 0

    def _generator_refl_loss(
        self,
        reward_adapter,
        wrapped_model,
        latents: torch.Tensor,
        text=None,
        prompt_embeds=None,
        prompt_mask=None,
        train_step: Optional[int] = None,
    ) -> Tuple[torch.Tensor, dict]:
        if reward_adapter is None:
            raise ValueError("reward_adapter is required when generator_refl_weight > 0")
        if wrapped_model is None:
            raise ValueError("wrapped_model is required when generator_refl_weight > 0")
        kwargs = {
            "text": text,
            "prompt_embeds": prompt_embeds,
        }
        try:
            signature = inspect.signature(reward_adapter.score_from_latents)
            accepts_kwargs = any(
                param.kind == inspect.Parameter.VAR_KEYWORD
                for param in signature.parameters.values()
            )
            if "prompt_mask" in signature.parameters or accepts_kwargs:
                kwargs["prompt_mask"] = prompt_mask
            if "train_step" in signature.parameters or accepts_kwargs:
                kwargs["train_step"] = train_step
        except (TypeError, ValueError):
            kwargs["prompt_mask"] = prompt_mask
            kwargs["train_step"] = train_step
        result = reward_adapter.score_from_latents(wrapped_model, latents, **kwargs)
        if isinstance(result, tuple):
            scores, reward_stats = result
        else:
            scores, reward_stats = result, {}
        if self.generator_refl_clip_score is None:
            loss = -scores.mean()
            clip_score = torch.full(
                (1,),
                float("nan"),
                device=scores.device,
                dtype=torch.float32,
            )
            keep_ratio = torch.ones(1, device=scores.device, dtype=torch.float32)
        else:
            score_mean = scores.mean()
            clip_score = torch.full(
                (1,),
                float(self.generator_refl_clip_score),
                device=scores.device,
                dtype=torch.float32,
            )
            under_clip = score_mean <= float(self.generator_refl_clip_score)
            if bool(under_clip.detach().item()):
                loss = -score_mean
                keep_ratio = torch.ones(1, device=scores.device, dtype=torch.float32)
            else:
                loss = scores.sum() * 0.0
                keep_ratio = torch.zeros(1, device=scores.device, dtype=torch.float32)
        stats = {
            "opd_gen_refl_score": scores.detach(),
            "opd_gen_refl_loss": loss.detach().view(1),
            "opd_gen_refl_clip_score": clip_score.detach(),
            "opd_gen_refl_clip_keep_ratio": keep_ratio.detach(),
        }
        for key, value in dict(reward_stats).items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("reward_"):
                stats[f"opd_gen_refl_{key.removeprefix('reward_')}"] = value.detach()
        return loss, stats

    def _reward_gan_disc_loss(
        self,
        reward_adapter,
        wrapped_model,
        fake_latents: torch.Tensor,
        real_latents: torch.Tensor,
        text=None,
        prompt_embeds=None,
    ) -> Tuple[torch.Tensor, dict]:
        if reward_adapter is None:
            raise ValueError("reward_adapter is required when reward_gan_discriminator_weight > 0")
        if wrapped_model is None:
            raise ValueError("wrapped_model is required when reward_gan_discriminator_weight > 0")
        fake_result = reward_adapter.reward_gan_logits_from_latents(
            wrapped_model,
            fake_latents,
            text=text,
            prompt_embeds=prompt_embeds,
        )
        real_result = reward_adapter.reward_gan_logits_from_latents(
            wrapped_model,
            real_latents,
            text=text,
            prompt_embeds=prompt_embeds,
        )
        fake_logits = fake_result[0] if isinstance(fake_result, tuple) else fake_result
        real_logits = real_result[0] if isinstance(real_result, tuple) else real_result
        fake_logits = fake_logits.reshape(-1).to(torch.float32)
        real_logits = real_logits.reshape(-1).to(torch.float32)
        loss = F.softplus(fake_logits).mean() + F.softplus(-real_logits).mean()
        stats = {
            "opd_reward_gan_disc_fake_logit": fake_logits.detach(),
            "opd_reward_gan_disc_real_logit": real_logits.detach(),
        }
        return loss, stats

    def _reward_gan_gen_loss(
        self,
        reward_adapter,
        wrapped_model,
        fake_latents: torch.Tensor,
        text=None,
        prompt_embeds=None,
    ) -> Tuple[torch.Tensor, dict]:
        if reward_adapter is None:
            raise ValueError("reward_adapter is required when reward_gan_generator_weight > 0")
        if wrapped_model is None:
            raise ValueError("wrapped_model is required when reward_gan_generator_weight > 0")
        result = reward_adapter.reward_gan_logits_from_latents(
            wrapped_model,
            fake_latents,
            text=text,
            prompt_embeds=prompt_embeds,
        )
        fake_logits = result[0] if isinstance(result, tuple) else result
        fake_logits = fake_logits.reshape(-1).to(torch.float32)
        loss = F.softplus(-fake_logits).mean()
        stats = {"opd_reward_gan_gen_fake_logit": fake_logits.detach()}
        return loss, stats

    def _initial_rollout_state(
        self,
        student_model: ModelLike,
        latent_shape: Tuple[int, ...],
        c: List[torch.Tensor],
        initial_noise: Optional[torch.Tensor],
    ) -> torch.Tensor:
        device = c[0].device
        if initial_noise is None:
            x = torch.randn(tuple(latent_shape), device=device, dtype=torch.float32)
        else:
            x = initial_noise.to(device=device, dtype=torch.float32)

        if self.initial_warmup_step_size <= 0.0:
            return x

        batch_size = int(x.shape[0])
        nodes = self.sigma_nodes(device=device, dtype=torch.float32)
        sigma_start = nodes[0].expand(batch_size)
        with torch.no_grad():
            warm_velocity = student_model(x.detach(), t=sigma_start, c=c)
        return (x.detach() - float(self.initial_warmup_step_size) * warm_velocity).detach()

    def _sample_exit_step(
        self,
        device: torch.device,
        step: Optional[int],
    ) -> int:
        if self._forced_exit_step is not None:
            return int(self._forced_exit_step)
        if self.exit_step_mode == "fixed":
            return int(self.fixed_exit_step)
        cache_key = int(step) if step is not None else None
        if (
            cache_key is not None
            and self._cached_exit_step_key == cache_key
            and self._cached_exit_step_value is not None
        ):
            return int(self._cached_exit_step_value)

        full_range = tuple(range(1, self.num_student_steps + 1))
        if self.exit_point_weights is not None:
            weights = torch.tensor(
                self.exit_point_weights,
                device=device,
                dtype=torch.float32,
            )
            choice_idx = torch.multinomial(weights, 1)
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(choice_idx, src=0)
            value = int(self._allowed_exit_steps[int(choice_idx.item())])
        elif self._allowed_exit_steps == full_range:
            exit_step = torch.randint(
                1,
                self.num_student_steps + 1,
                (1,),
                device=device,
                dtype=torch.int64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(exit_step, src=0)
            value = int(exit_step.item())
        else:
            choice_idx = torch.randint(
                0,
                len(self._allowed_exit_steps),
                (1,),
                device=device,
                dtype=torch.int64,
            )
            if dist.is_available() and dist.is_initialized():
                dist.broadcast(choice_idx, src=0)
            value = int(self._allowed_exit_steps[int(choice_idx.item())])
        if cache_key is not None:
            self._cached_exit_step_key = cache_key
            self._cached_exit_step_value = value
        return value

    def _rollout_step_from_velocity(
        self,
        x_t: torch.Tensor,
        sigma_cur: torch.Tensor,
        sigma_next: torch.Tensor,
        velocity: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        rollout_stochast_ratio: Optional[float] = None,
    ) -> torch.Tensor:
        ratio = (
            float(self.rollout_stochast_ratio)
            if rollout_stochast_ratio is None
            else float(rollout_stochast_ratio)
        )
        return rollout_step_from_velocity(
            x_t=x_t,
            sigma_cur=sigma_cur,
            sigma_next=sigma_next,
            velocity=velocity,
            noise=noise,
            rollout_stochast_ratio=ratio,
        )

    @torch.no_grad()
    def _teacher_interval_rollout_target(
        self,
        teacher_model: ModelLike,
        x_t: torch.Tensor,
        sigma_cur: torch.Tensor,
        sigma_next: torch.Tensor,
        c: List[torch.Tensor],
        micro_sigmas: torch.Tensor,
        terminal_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        sigma_cur = self._expand_sigma(sigma_cur, batch_size, x_t.device)
        sigma_next = self._expand_sigma(sigma_next, batch_size, x_t.device)
        interval = sigma_cur - sigma_next
        if torch.any(interval <= 0):
            raise ValueError("teacher interval must have positive length")

        micro_sigmas = micro_sigmas.to(device=x_t.device, dtype=torch.float32).flatten()
        if micro_sigmas.numel() < 2:
            raise ValueError("micro_sigmas must contain at least start and end values")
        if not torch.allclose(micro_sigmas[0].expand_as(sigma_cur), sigma_cur):
            raise ValueError("micro_sigmas must start at sigma_cur")
        if not torch.allclose(micro_sigmas[-1].expand_as(sigma_next), sigma_next):
            raise ValueError("micro_sigmas must end at sigma_next")
        if torch.any(micro_sigmas[:-1] <= micro_sigmas[1:]):
            raise ValueError("micro_sigmas must be strictly descending")

        ode_target, _ = self.teacher_interval_target(
            teacher_model=teacher_model,
            x_t=x_t,
            sigma_cur=sigma_cur,
            sigma_next=sigma_next,
            c=c,
            micro_sigmas=micro_sigmas,
        )
        if float(self.rollout_stochast_ratio) == 0.0:
            return ode_target.detach()

        velocity = (
            x_t.detach().to(torch.float32) - ode_target.to(torch.float32)
        ) / self._broadcast_sigma(interval, x_t)
        noise = (
            terminal_noise.to(device=x_t.device, dtype=torch.float32)
            if terminal_noise is not None
            else torch.randn_like(x_t, dtype=torch.float32)
        )
        return self._rollout_step_from_velocity(
            x_t=x_t.detach(),
            sigma_cur=sigma_cur,
            sigma_next=sigma_next,
            velocity=velocity,
            noise=noise,
        ).detach()

    @torch.no_grad()
    def _teacher_to_zero_target(
        self,
        teacher_model: ModelLike,
        x_t: torch.Tensor,
        sigma_cur: torch.Tensor,
        c: List[torch.Tensor],
        micro_sigmas: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        sigma_zero = torch.zeros_like(sigma_cur)
        target, _ = self.teacher_interval_target(
            teacher_model=teacher_model,
            x_t=x_t,
            sigma_cur=sigma_cur,
            sigma_next=sigma_zero,
            c=c,
            micro_sigmas=micro_sigmas,
        )
        return target.detach()

    @torch.no_grad()
    def _teacher_interval_average_to_zero_target(
        self,
        teacher_model: ModelLike,
        x_t: torch.Tensor,
        sigma_cur: torch.Tensor,
        sigma_next: torch.Tensor,
        c: List[torch.Tensor],
        micro_sigmas: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x_t.shape[0]
        sigma_cur = self._expand_sigma(sigma_cur, batch_size, x_t.device)
        sigma_next = self._expand_sigma(sigma_next, batch_size, x_t.device)
        interval = sigma_cur - sigma_next
        if torch.any(interval <= 0):
            raise ValueError("teacher interval must have positive length")
        micro_sigmas = micro_sigmas.to(device=x_t.device, dtype=torch.float32).flatten()
        if micro_sigmas.numel() < 2:
            raise ValueError("micro_sigmas must contain at least start and end values")
        if not torch.allclose(micro_sigmas[0].expand_as(sigma_cur), sigma_cur):
            raise ValueError("micro_sigmas must start at sigma_cur")
        if not torch.allclose(micro_sigmas[-1].expand_as(sigma_next), sigma_next):
            raise ValueError("micro_sigmas must end at sigma_next")
        if torch.any(micro_sigmas[:-1] <= micro_sigmas[1:]):
            raise ValueError("micro_sigmas must be strictly descending")

        interval_target = x_t.detach().to(torch.float32)
        for cur_sigma, next_sigma in zip(micro_sigmas[:-1], micro_sigmas[1:]):
            cur = torch.full(
                (batch_size,),
                float(cur_sigma.item()),
                device=x_t.device,
                dtype=torch.float32,
            )
            nxt = torch.full(
                (batch_size,),
                float(next_sigma.item()),
                device=x_t.device,
                dtype=torch.float32,
            )
            velocity = teacher_model(interval_target, t=cur, c=c).to(torch.float32)
            interval_target = (
                interval_target
                - self._broadcast_sigma(
                    cur - nxt,
                    interval_target,
                )
                * velocity
            )
        avg_velocity = (x_t.detach().to(torch.float32) - interval_target) / (
            self._broadcast_sigma(interval, x_t)
        )
        return (
            x_t.detach().to(torch.float32) - self._broadcast_sigma(sigma_cur, x_t) * avg_velocity
        ).detach()

    def _early_exit_xt_pair(
        self,
        student_model: ModelLike,
        teacher_model: ModelLike,
        latent_shape: Tuple[int, ...],
        c: List[torch.Tensor],
        initial_noise: Optional[torch.Tensor],
        step: Optional[int],
        grad_enabled: bool,
        include_real: bool,
        real_image_latents: Optional[torch.Tensor] = None,
        real_source_is_data: bool = False,
    ):
        self._set_lora_mode(teacher_model, use_lora=False)
        x = self._initial_rollout_state(
            student_model=student_model,
            latent_shape=latent_shape,
            c=c,
            initial_noise=initial_noise,
        )
        x_root = x.detach()
        batch_size = int(x.shape[0])
        nodes = self.sigma_nodes(device=x.device, dtype=torch.float32)
        teacher_nodes = self.teacher_sigma_nodes(device=x.device, dtype=torch.float32)
        exit_step = self._sample_exit_step(x.device, step=step)
        rollout_steps = exit_step - 1

        student_next_abs = []
        teacher_next_abs = []
        alpha_roll_values = []
        alpha_prob_values = []
        trajectory_alpha_roll = None
        if self.rollout_mode == "trajectory_bernoulli" and rollout_steps > 0:
            trajectory_alpha_roll = (
                torch.rand((batch_size,), device=x.device, dtype=torch.float32) < 0.5
            ).to(torch.float32)
        for idx in range(rollout_steps):
            sigma_cur = nodes[idx].expand(batch_size)
            sigma_next = nodes[idx + 1].expand(batch_size)
            x_in = x.detach()
            transition_noise = (
                torch.randn_like(x_in, dtype=torch.float32)
                if self.rollout_stochast_ratio > 0.0
                else None
            )
            with torch.no_grad():
                v_student = student_model(x_in, t=sigma_cur, c=c)
                x_student_next = self._rollout_step_from_velocity(
                    x_t=x_in,
                    sigma_cur=sigma_cur,
                    sigma_next=sigma_next,
                    velocity=v_student,
                    noise=transition_noise,
                )
                x_teacher_next = None
                if self.rollout_mode != "student":
                    x_teacher_next = self._teacher_interval_rollout_target(
                        teacher_model=teacher_model,
                        x_t=x_in,
                        sigma_cur=sigma_cur,
                        sigma_next=sigma_next,
                        c=c,
                        micro_sigmas=self.teacher_interval_micro_sigmas(
                            teacher_nodes,
                            idx,
                        ),
                        terminal_noise=transition_noise,
                    )

            student_next_abs.append(x_student_next.detach().abs().flatten(1).mean(dim=1))
            if x_teacher_next is not None:
                teacher_next_abs.append(x_teacher_next.detach().abs().flatten(1).mean(dim=1))
            if self.rollout_mode == "student":
                alpha_roll = torch.ones(
                    (batch_size,),
                    device=x.device,
                    dtype=torch.float32,
                )
                alpha_prob = torch.ones_like(alpha_roll)
                x = x_student_next.detach()
            elif self.rollout_mode == "trajectory_bernoulli":
                alpha_roll = trajectory_alpha_roll
                alpha_prob = torch.full_like(alpha_roll, 0.5)
                alpha_view = self._broadcast_sigma(alpha_roll, x_student_next)
                x = (
                    alpha_view * x_student_next.detach()
                    + (1.0 - alpha_view) * x_teacher_next.detach()
                ).detach()
            else:
                alpha_roll = torch.rand(
                    (batch_size,),
                    device=x.device,
                    dtype=torch.float32,
                )
                alpha_prob = torch.full_like(alpha_roll, 0.5)
                alpha_view = self._broadcast_sigma(alpha_roll, x_student_next)
                x = (
                    alpha_view * x_student_next.detach()
                    + (1.0 - alpha_view) * x_teacher_next.detach()
                ).detach()
            alpha_roll_values.append(alpha_roll.detach())
            alpha_prob_values.append(alpha_prob.detach())

        sigma_cur = nodes[exit_step - 1].expand(batch_size)
        sigma_next = nodes[exit_step].expand(batch_size)
        x_exit = x.detach()
        if grad_enabled:
            x_exit.requires_grad_(True)
        transition_noise = (
            torch.randn_like(x_exit, dtype=torch.float32)
            if self.rollout_stochast_ratio > 0.0
            else None
        )
        if grad_enabled:
            v_student_exit = student_model(x_exit, t=sigma_cur, c=c)
            fake_xt = self._rollout_step_from_velocity(
                x_t=x_exit,
                sigma_cur=sigma_cur,
                sigma_next=sigma_next,
                velocity=v_student_exit,
                noise=transition_noise,
            )
        else:
            with torch.no_grad():
                v_student_exit = student_model(x_exit, t=sigma_cur, c=c)
                fake_xt = self._rollout_step_from_velocity(
                    x_t=x_exit,
                    sigma_cur=sigma_cur,
                    sigma_next=sigma_next,
                    velocity=v_student_exit,
                    noise=transition_noise,
                )
        fake_x0_clean = x_exit - self._broadcast_sigma(sigma_cur, x_exit) * v_student_exit

        real_xt = None
        real_x0_clean = None
        if include_real:
            if real_source_is_data:
                if real_image_latents is None:
                    raise ValueError(
                        "real_source='data' requires real_image_latents during discriminator phases"
                    )
                if tuple(real_image_latents.shape) != tuple(x_exit.shape):
                    raise ValueError(
                        "real_image_latents shape mismatch: "
                        f"{tuple(real_image_latents.shape)} vs expected "
                        f"{tuple(x_exit.shape)}"
                    )
                real_x0_clean = real_image_latents.detach().to(
                    device=x_exit.device,
                    dtype=torch.float32,
                )
                real_xt = real_x0_clean
                teacher_next_abs.append(real_x0_clean.detach().abs().flatten(1).mean(dim=1))
            else:
                real_interval_micro_sigmas = self.teacher_interval_micro_sigmas(
                    teacher_nodes,
                    exit_step - 1,
                )
                real_xt = self._teacher_interval_rollout_target(
                    teacher_model=teacher_model,
                    x_t=x_exit,
                    sigma_cur=sigma_cur,
                    sigma_next=sigma_next,
                    c=c,
                    micro_sigmas=real_interval_micro_sigmas,
                    terminal_noise=transition_noise,
                )
                real_x0_clean = self._teacher_interval_average_to_zero_target(
                    teacher_model=teacher_model,
                    x_t=x_exit,
                    sigma_cur=sigma_cur,
                    sigma_next=sigma_next,
                    c=c,
                    micro_sigmas=real_interval_micro_sigmas,
                )
                if self.loss_mode in {
                    "dual_distilled_early_exit_x0_full_teacher_real",
                    self.FULL_TEACHER_REAL_SEPARATE_EXIT_DISCS_LOSS_MODE,
                    self.CURRICULUM_SEPARATE_EXIT_DISCS_LOSS_MODE,
                }:
                    root_sigma = nodes[0].expand(batch_size)
                    real_x0_clean = self._teacher_to_zero_target(
                        teacher_model=teacher_model,
                        x_t=x_root,
                        sigma_cur=root_sigma,
                        c=c,
                        micro_sigmas=teacher_nodes,
                    )
                teacher_next_abs.append(real_xt.detach().abs().flatten(1).mean(dim=1))
        student_next_abs.append(fake_xt.detach().abs().flatten(1).mean(dim=1))
        x0_noise_sigma = self._x0_noise_sigma_for_exit(
            exit_step,
            batch_size,
            x_exit.device,
        )
        fake_x0_noise = None
        real_x0_noise = None
        if real_x0_clean is not None:
            fake_x0_noise = torch.randn_like(fake_x0_clean, dtype=torch.float32)
            real_x0_noise = (
                fake_x0_noise
                if self.x0_noise_share_fake_real
                else torch.randn_like(real_x0_clean, dtype=torch.float32)
            )
        fake_x0 = self._apply_x0_noise(
            fake_x0_clean,
            x0_noise_sigma,
            noise=fake_x0_noise,
        )
        real_x0 = (
            self._apply_x0_noise(real_x0_clean, x0_noise_sigma, noise=real_x0_noise)
            if real_x0_clean is not None
            else None
        )

        aux = {
            "opd_exit_step": torch.full(
                (batch_size,),
                float(exit_step),
                device=x.device,
                dtype=torch.float32,
            ),
            "opd_exit_sigma": sigma_cur.detach(),
            "opd_exit_rollout_steps": torch.full(
                (batch_size,),
                float(rollout_steps),
                device=x.device,
                dtype=torch.float32,
            ),
            "opd_student_next_abs": self._cat_or_empty(student_next_abs, x_exit),
            "opd_teacher_next_abs": self._cat_or_empty(teacher_next_abs, x_exit),
            "opd_alpha_roll": self._cat_or_empty(alpha_roll_values, x_exit),
            "opd_alpha_roll_prob": self._cat_or_empty(alpha_prob_values, x_exit),
            "opd_rollout_stochast_ratio": torch.full(
                (batch_size,),
                float(self.rollout_stochast_ratio),
                device=x.device,
                dtype=torch.float32,
            ),
            "opd_x0_noise_sigma": x0_noise_sigma.detach(),
            "opd_debug_student_x0_clean": fake_x0_clean.detach(),
            "opd_debug_initial_noise": x_root.detach(),
        }
        if grad_enabled:
            aux["_opd_live_student_x0_clean"] = fake_x0_clean
        if real_x0_clean is not None:
            aux["opd_debug_teacher_x0_clean"] = real_x0_clean.detach()
        return (
            fake_xt if grad_enabled else fake_xt.detach(),
            real_xt,
            fake_x0 if grad_enabled else fake_x0.detach(),
            real_x0,
            sigma_next.detach(),
            sigma_cur.detach(),
            aux,
        )

    @staticmethod
    def _invoke_discriminator(
        discriminator_model: ModelLike,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: List[torch.Tensor],
        tt: torch.Tensor,
        *,
        return_raw: bool,
        discriminator_output: Optional[str] = None,
        discriminator_head: Optional[str] = None,
        skip_aux_time: bool = False,
    ):
        kwargs = {
            "t": t,
            "c": c,
            "tt": tt,
            "return_raw": return_raw,
            "skip_aux_time": skip_aux_time,
        }
        if discriminator_output is not None:
            kwargs["discriminator_output"] = discriminator_output
        if discriminator_head is not None:
            kwargs["discriminator_head"] = discriminator_head
        if hasattr(discriminator_model, "discriminate"):
            kwargs["disable_separate_r_modulation"] = True
            return discriminator_model.discriminate(x_t, **kwargs)
        kwargs["discriminator_mode"] = True
        kwargs["disable_separate_r_modulation"] = True
        return discriminator_model(x_t, **kwargs)

    def _call_frozen_discriminator(
        self,
        frozen_discriminator_model: ModelLike,
        x0: torch.Tensor,
        c: List[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = int(x0.shape[0])
        t = torch.full(
            (batch_size,),
            float(self.dual_frozen_discriminator_t),
            device=x0.device,
            dtype=torch.float32,
        )
        with torch.no_grad():
            output = self._invoke_discriminator(
                discriminator_model=frozen_discriminator_model,
                x_t=x0.detach(),
                t=t,
                c=c,
                tt=t,
                return_raw=True,
                skip_aux_time=True,
            )
        if isinstance(output, tuple) and len(output) == 1:
            output = output[0]
        return output.detach().to(torch.float32)

    def _split_dual_output(self, output):
        if isinstance(output, dict):
            if "align" not in output or "logits" not in output:
                raise ValueError("dual discriminator dict output must contain 'align' and 'logits'")
            return output["align"], output["logits"]
        if hasattr(output, "align") and hasattr(output, "logits"):
            return output.align, output.logits
        if isinstance(output, (tuple, list)) and len(output) == 2:
            return output[0], output[1]
        raise ValueError(
            "dual discriminator output must be a dict, namedtuple, or (align, logits) pair"
        )

    def _call_trainable_discriminator(
        self,
        discriminator_model: ModelLike,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: List[torch.Tensor],
        tt: torch.Tensor,
        *,
        output: str,
        discriminator_head: Optional[str] = None,
    ):
        self._set_lora_mode(discriminator_model, use_lora=True)
        raw = self._invoke_discriminator(
            discriminator_model=discriminator_model,
            x_t=x_t,
            t=t,
            c=c,
            tt=tt,
            return_raw=True,
            discriminator_output=output,
            discriminator_head=discriminator_head,
        )
        if output == "gan":
            if isinstance(raw, tuple) and len(raw) == 1:
                raw = raw[0]
            return raw.reshape(-1).to(torch.float32)
        if output == "align":
            if isinstance(raw, tuple) and len(raw) == 1:
                raw = raw[0]
            return raw.to(torch.float32)
        align, logits = self._split_dual_output(raw)
        return align.to(torch.float32), logits.reshape(-1).to(torch.float32)

    def _call_trainable_velocity(
        self,
        discriminator_model: ModelLike,
        x_t: torch.Tensor,
        t: torch.Tensor,
        c: List[torch.Tensor],
        tt: torch.Tensor,
    ) -> torch.Tensor:
        return self._call_score_velocity(
            score_model=discriminator_model,
            x_t=x_t,
            t=t,
            c=c,
            tt=tt,
            use_lora=True,
        )

    def _sample_apt_r1_sigma(self, reference: torch.Tensor) -> torch.Tensor:
        sigma_min = float(self.apt_r1_sigma_min)
        sigma_max = float(self.apt_r1_sigma_max)
        if sigma_max == sigma_min:
            return torch.full(
                (1,),
                sigma_min,
                device=reference.device,
                dtype=torch.float32,
            )
        return torch.empty(1, device=reference.device, dtype=torch.float32).uniform_(
            sigma_min,
            sigma_max,
        )

    def _apt_r1_real_consistency_loss(
        self,
        *,
        discriminator_model: ModelLike,
        real_disc_x: torch.Tensor,
        pred_real: torch.Tensor,
        disc_t: torch.Tensor,
        c: List[torch.Tensor],
        disc_tt: torch.Tensor,
        discriminator_head: Optional[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sigma = self._sample_apt_r1_sigma(real_disc_x)
        sigma_view = sigma.reshape(-1, *([1] * (real_disc_x.ndim - 1))).to(
            dtype=real_disc_x.dtype,
        )
        noisy_real = (
            real_disc_x.detach()
            + torch.randn_like(
                real_disc_x,
                dtype=torch.float32,
            )
            * sigma_view
        )
        pred_real_perturbed = self._call_trainable_discriminator(
            discriminator_model=discriminator_model,
            x_t=noisy_real.detach(),
            t=disc_t,
            c=c,
            tt=disc_tt,
            output="gan",
            discriminator_head=discriminator_head,
        )
        r1_loss = F.mse_loss(pred_real, pred_real_perturbed)
        return r1_loss, sigma.detach(), pred_real_perturbed.detach()

    def _discriminator_inputs_for_loss_mode(
        self,
        fake_xt: torch.Tensor,
        real_xt: Optional[torch.Tensor],
        fake_x0: torch.Tensor,
        real_x0: Optional[torch.Tensor],
        t_disc: torch.Tensor,
        tt: torch.Tensor,
        x0_noise_sigma: Optional[torch.Tensor] = None,
    ):
        if self.loss_mode == "dual_distilled_early_exit_xt":
            return fake_xt, real_xt, t_disc, tt
        if self.loss_mode in {
            "dual_distilled_early_exit_x0",
            "dual_distilled_early_exit_x0_full_teacher_real",
            *self.SEPARATE_EXIT_DISCS_LOSS_MODES,
        }:
            batch_size = int(fake_x0.shape[0])
            x0_t = torch.full(
                (batch_size,),
                float(self.dual_frozen_discriminator_t),
                device=fake_x0.device,
                dtype=torch.float32,
            )
            return fake_x0, real_x0, x0_t, x0_t
        raise RuntimeError(f"Unsupported loss_mode={self.loss_mode!r}")

    @staticmethod
    @contextmanager
    def _temporarily_requires_grad(model: ModelLike, requires_grad: bool):
        if not hasattr(model, "parameters"):
            yield
            return
        params = list(model.parameters())
        original_flags = [param.requires_grad for param in params]
        try:
            for param in params:
                param.requires_grad_(requires_grad)
            yield
        finally:
            for param, original_flag in zip(params, original_flags):
                param.requires_grad_(original_flag)

    def dual_alignment_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, dict]:
        return dual_alignment_loss(
            pred,
            target,
            mse_weight=float(self.dual_align_mse_weight),
            pearson_eps=float(self.dual_align_pearson_eps),
            pearson_mode=self.dual_align_pearson_mode,
        )

    def _generator_adv_loss(
        self,
        pred_fake: torch.Tensor,
        selected_exit_step: Optional[int] = None,
        return_raw: bool = False,
    ):
        loss = (
            F.softplus(-pred_fake).mean()
            * float(self.loss_weight)
            * float(self.generator_loss_weight)
        )
        raw_loss = loss
        if self.generator_adv_loss_max is not None:
            loss = loss.clamp(max=float(self.generator_adv_loss_max))
        if self.generator_adv_loss_min is not None:
            keep_loss = raw_loss.detach() >= float(self.generator_adv_loss_min)
            loss = torch.where(keep_loss, loss, raw_loss * 0.0)
        loss = loss * self._generator_gan_loss_weight_for_exit(selected_exit_step)
        if return_raw:
            return loss, raw_loss
        return loss

    @staticmethod
    def _format_return(loss, stats, aux, return_loss_stats, return_log_tensors):
        if return_loss_stats and return_log_tensors:
            return loss, stats, aux
        if return_loss_stats:
            return loss, stats
        return loss

    def _discriminator_phase_step(
        self,
        student_model: ModelLike,
        teacher_model: ModelLike,
        latent_shape: Tuple[int, ...],
        c: List[torch.Tensor],
        step: Optional[int],
        initial_noise: Optional[torch.Tensor],
        real_image_latents: Optional[torch.Tensor],
        real_source_is_data: bool,
        real_source_data_prob: Optional[float],
        discriminator_model: ModelLike,
        frozen_discriminator_model: Optional[ModelLike],
        return_loss_stats: bool,
        return_log_tensors: bool,
        reward_adapter=None,
        wrapped_model=None,
        text=None,
        prompt_embeds=None,
    ):
        if frozen_discriminator_model is None:
            raise ValueError("frozen_discriminator_model is required for discriminator phase")
        (
            fake_xt,
            real_xt,
            fake_x0,
            real_x0,
            t_disc,
            tt,
            exit_aux,
        ) = self._early_exit_xt_pair(
            student_model=student_model,
            teacher_model=teacher_model,
            latent_shape=latent_shape,
            c=c,
            initial_noise=initial_noise,
            step=step,
            grad_enabled=False,
            include_real=True,
            real_image_latents=real_image_latents,
            real_source_is_data=real_source_is_data,
        )
        if real_xt is None or real_x0 is None:
            raise RuntimeError("D phase requires real xt and x0 targets")
        frozen_disc_fake = exit_aux.get("opd_debug_student_x0_clean")
        frozen_disc_real = exit_aux.get("opd_debug_teacher_x0_clean")
        if frozen_disc_fake is None or frozen_disc_real is None:
            raise RuntimeError("D phase requires clean x0 targets")
        debug_student_x0 = frozen_disc_fake
        debug_teacher_x0 = frozen_disc_real
        fake_disc_x, real_disc_x, disc_t, disc_tt = self._discriminator_inputs_for_loss_mode(
            fake_xt=fake_xt,
            real_xt=real_xt,
            fake_x0=fake_x0,
            real_x0=real_x0,
            t_disc=t_disc,
            tt=tt,
            x0_noise_sigma=exit_aux.get("opd_x0_noise_sigma"),
        )
        if real_disc_x is None:
            raise RuntimeError("D phase requires real discriminator targets")
        exit_step_value = int(round(float(exit_aux["opd_exit_step"].flatten()[0].item())))
        discriminator_head = self._discriminator_head_for_exit(exit_step_value)

        fake_target = self._call_frozen_discriminator(
            frozen_discriminator_model=frozen_discriminator_model,
            x0=frozen_disc_fake.detach(),
            c=c,
        )
        real_target = self._call_frozen_discriminator(
            frozen_discriminator_model=frozen_discriminator_model,
            x0=frozen_disc_real.detach(),
            c=c,
        )
        fake_align, pred_fake = self._call_trainable_discriminator(
            discriminator_model=discriminator_model,
            x_t=fake_disc_x.detach(),
            t=disc_t,
            c=c,
            tt=disc_tt,
            output="both",
            discriminator_head=discriminator_head,
        )
        real_align, pred_real = self._call_trainable_discriminator(
            discriminator_model=discriminator_model,
            x_t=real_disc_x.detach(),
            t=disc_t,
            c=c,
            tt=disc_tt,
            output="both",
            discriminator_head=discriminator_head,
        )

        fake_align_loss, fake_align_aux = self.dual_alignment_loss(
            fake_align,
            fake_target,
        )
        real_align_loss, real_align_aux = self.dual_alignment_loss(
            real_align,
            real_target,
        )
        align_loss = 0.5 * (fake_align_loss + real_align_loss)
        gap_temperature, gap_value, gap_target = self._disc_gap_temperature(
            pred_fake,
            pred_real,
            exit_step_value,
        )
        pred_fake_for_loss = pred_fake / gap_temperature
        pred_real_for_loss = pred_real / gap_temperature
        disc_gan_loss = (
            F.softplus(pred_fake_for_loss).mean() + F.softplus(-pred_real_for_loss).mean()
        )
        dmd_score_loss = torch.zeros_like(disc_gan_loss)
        dmd_score_stats = {}
        dmd_score_debug_aux = {}
        if self._dmd_score_loss_enabled():
            fake_score_tt = self._dmd_fake_score_tt(exit_aux, frozen_disc_fake)
            dmd_score_loss, dmd_score_stats, dmd_score_debug_aux = self._dmd_fake_score_loss(
                discriminator_model=discriminator_model,
                fake_x0_clean=frozen_disc_fake.detach(),
                c=c,
                fake_score_tt=fake_score_tt,
            )
        dmd_score_weighted_loss = float(self.dmd_fake_score_loss_weight) * dmd_score_loss
        reward_gan_disc_loss = torch.zeros_like(disc_gan_loss)
        reward_gan_disc_stats = {}
        reward_gan_disc_active = (
            self._reward_gan_enabled() and float(self.reward_gan_discriminator_weight) != 0.0
        )
        if reward_gan_disc_active:
            reward_gan_disc_loss, reward_gan_disc_stats = self._reward_gan_disc_loss(
                reward_adapter=reward_adapter,
                wrapped_model=wrapped_model,
                fake_latents=frozen_disc_fake.detach(),
                real_latents=frozen_disc_real.detach(),
                text=text,
                prompt_embeds=prompt_embeds,
            )
        reward_gan_disc_weighted_loss = (
            float(self.reward_gan_discriminator_weight) * reward_gan_disc_loss
        )
        apt_r1_loss = torch.zeros_like(disc_gan_loss)
        apt_r1_sigma = torch.zeros(1, device=fake_xt.device, dtype=torch.float32)
        apt_r1_real_perturbed_logit = torch.zeros_like(pred_real.detach())
        if float(self.apt_r1_weight) != 0.0:
            (
                apt_r1_loss,
                apt_r1_sigma,
                apt_r1_real_perturbed_logit,
            ) = self._apt_r1_real_consistency_loss(
                discriminator_model=discriminator_model,
                real_disc_x=real_disc_x,
                pred_real=pred_real,
                disc_t=disc_t,
                c=c,
                disc_tt=disc_tt,
                discriminator_head=discriminator_head,
            )
        apt_r1_weighted_loss = float(self.apt_r1_weight) * apt_r1_loss
        disc_loss_weight_for_exit = self._discriminator_loss_weight_for_exit(exit_step_value)
        weighted_disc_terms = (
            disc_gan_loss
            + float(self.dual_align_loss_weight) * align_loss
            + dmd_score_weighted_loss
            + reward_gan_disc_weighted_loss
        )
        loss = (
            float(self.loss_weight)
            * float(self.discriminator_loss_weight)
            * (float(disc_loss_weight_for_exit) * weighted_disc_terms + apt_r1_weighted_loss)
        )
        exit_aux.update(self._pair_diff_stats("opd_xt_diff", fake_xt, real_xt))
        exit_aux.update(
            self._pair_diff_stats(
                "opd_x0_diff",
                debug_student_x0,
                debug_teacher_x0,
            )
        )
        stats = self._pack_loss_stats(
            loss_opd=loss.detach(),
            opd_disc_loss=loss.detach(),
            opd_disc_gan_loss=disc_gan_loss.detach().view(1),
            opd_disc_loss_weight=torch.full(
                (1,),
                float(disc_loss_weight_for_exit),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_dual_align_loss=align_loss.detach().view(1),
            opd_dual_align_fake_loss=fake_align_loss.detach().view(1),
            opd_dual_align_real_loss=real_align_loss.detach().view(1),
            opd_dual_align_fake_pearson=fake_align_aux["pearson"].view(1),
            opd_dual_align_fake_mse=fake_align_aux["mse"].view(1),
            opd_dual_align_real_pearson=real_align_aux["pearson"].view(1),
            opd_dual_align_real_mse=real_align_aux["mse"].view(1),
            opd_dmd_score_loss=dmd_score_loss.detach().view(1),
            opd_dmd_score_weight=torch.full(
                (1,),
                float(self.dmd_fake_score_loss_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_dmd_score_weighted_loss=dmd_score_weighted_loss.detach().view(1),
            opd_apt_r1_loss=apt_r1_loss.detach().view(1),
            opd_apt_r1_weight=torch.full(
                (1,),
                float(self.apt_r1_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_apt_r1_sigma=apt_r1_sigma.detach().view(1),
            opd_apt_r1_real_perturbed_logit=apt_r1_real_perturbed_logit.detach(),
            opd_reward_gan_disc_loss=reward_gan_disc_loss.detach().view(1),
            opd_reward_gan_disc_weighted_loss=reward_gan_disc_weighted_loss.detach().view(1),
            opd_reward_gan_disc_fake_logit=reward_gan_disc_stats.get(
                "opd_reward_gan_disc_fake_logit",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_reward_gan_disc_real_logit=reward_gan_disc_stats.get(
                "opd_reward_gan_disc_real_logit",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_reward_gan_gen_loss=torch.zeros(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_gen_weighted_loss=torch.zeros(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_gen_fake_logit=torch.full(
                (1,),
                float("nan"),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_discriminator_weight=torch.full(
                (1,),
                float(self.reward_gan_discriminator_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_generator_weight=torch.full(
                (1,),
                float(self.reward_gan_generator_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_active=torch.full(
                (1,),
                1.0 if reward_gan_disc_active else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_dmd_score_sigma=dmd_score_stats.get(
                "opd_dmd_score_sigma",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_dmd_score_fake_score_tt=dmd_score_stats.get(
                "opd_dmd_score_fake_score_tt",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_dmd_score_pred_err=dmd_score_stats.get(
                "opd_dmd_score_pred_err",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_disc_fake_logit=pred_fake.detach(),
            opd_disc_real_logit=pred_real.detach(),
            opd_active_discriminator_exit=torch.full(
                (1,),
                float(exit_step_value if discriminator_head is not None else 0.0),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head=torch.full(
                (1,),
                float(exit_step_value if discriminator_head is not None else 0.0),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head_exit1=torch.full(
                (1,),
                1.0 if discriminator_head == "exit1" else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head_exit2=torch.full(
                (1,),
                1.0 if discriminator_head == "exit2" else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_real_source_is_data=torch.full(
                (1,),
                float(real_source_is_data),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_real_source_data_prob=torch.full(
                (1,),
                (
                    float(real_source_data_prob)
                    if real_source_data_prob is not None
                    else float(real_source_is_data)
                ),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_exit_gap_temperature=torch.full(
                (1,),
                float(gap_temperature),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_exit_gap_value=torch.full(
                (1,),
                float(gap_value),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_exit_gap_target=torch.full(
                (1,),
                float(gap_target),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_disc_t=disc_t.detach(),
            opd_disc_tt=disc_tt.detach(),
            opd_phase_is_discriminator=torch.ones(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            **self._early_exit_stat_tensors(exit_aux),
        )
        aux = {
            "opd_disc_t": disc_t.detach(),
            "opd_disc_tt": disc_tt.detach(),
            "opd_disc_input_t": disc_t.detach(),
            "opd_disc_input_tt": disc_tt.detach(),
            "opd_disc_fake_logit": pred_fake.detach(),
            "opd_disc_real_logit": pred_real.detach(),
            "opd_dual_fake_target_abs": fake_target.detach().abs().flatten(1).mean(dim=1),
            "opd_dual_real_target_abs": real_target.detach().abs().flatten(1).mean(dim=1),
            "opd_debug_student_xt": fake_xt.detach(),
            "opd_debug_teacher_xt": real_xt.detach(),
            "opd_debug_student_x0": debug_student_x0.detach(),
            "opd_debug_teacher_x0": debug_teacher_x0.detach(),
            **exit_aux,
            **dmd_score_debug_aux,
        }
        return self._format_return(
            loss,
            stats,
            aux,
            return_loss_stats,
            return_log_tensors,
        )

    def _generator_phase_step(
        self,
        student_model: ModelLike,
        teacher_model: ModelLike,
        latent_shape: Tuple[int, ...],
        c: List[torch.Tensor],
        step: Optional[int],
        initial_noise: Optional[torch.Tensor],
        discriminator_model: ModelLike,
        return_loss_stats: bool,
        return_log_tensors: bool,
        generator_update_index: Optional[int] = None,
        real_source_is_data: bool = False,
        real_source_data_prob: Optional[float] = None,
        reward_adapter=None,
        wrapped_model=None,
        text=None,
        prompt_embeds=None,
        prompt_mask=None,
    ):
        (
            fake_xt,
            _real_xt,
            fake_x0,
            _real_x0,
            t_disc,
            tt,
            exit_aux,
        ) = self._early_exit_xt_pair(
            student_model=student_model,
            teacher_model=teacher_model,
            latent_shape=latent_shape,
            c=c,
            initial_noise=initial_noise,
            step=step,
            grad_enabled=True,
            include_real=False,
        )
        fake_x0_clean_live = exit_aux.pop("_opd_live_student_x0_clean", None)
        fake_disc_x, _real_disc_x, disc_t, disc_tt = self._discriminator_inputs_for_loss_mode(
            fake_xt=fake_xt,
            real_xt=None,
            fake_x0=fake_x0,
            real_x0=None,
            t_disc=t_disc,
            tt=tt,
            x0_noise_sigma=exit_aux.get("opd_x0_noise_sigma"),
        )
        exit_step_value = int(round(float(exit_aux["opd_exit_step"].flatten()[0].item())))
        discriminator_head = self._discriminator_head_for_exit(exit_step_value)
        with self._temporarily_requires_grad(discriminator_model, False):
            pred_fake = self._call_trainable_discriminator(
                discriminator_model=discriminator_model,
                x_t=fake_disc_x,
                t=disc_t,
                c=c,
                tt=disc_tt,
                output="gan",
                discriminator_head=discriminator_head,
            )
        gap_temperature, gap_value, gap_target = self._gen_gap_temperature(
            pred_fake,
            exit_step_value,
        )
        pred_fake_for_loss = pred_fake / gap_temperature
        gan_loss, gan_loss_raw = self._generator_adv_loss(
            pred_fake_for_loss,
            selected_exit_step=exit_step_value,
            return_raw=True,
        )
        flow_matching_loss = torch.zeros_like(gan_loss)
        flow_matching_debug_aux = {}
        if self._generator_flow_matching_enabled():
            fake_x0_clean = fake_x0_clean_live
            if fake_x0_clean is None:
                raise RuntimeError("G phase flow matching requires clean x0 tensor")
            batch_size = int(fake_x0_clean.shape[0])
            fm_t = self._sample_generator_flow_matching_t(
                batch_size,
                fake_x0_clean.device,
            )
            fm_noise = torch.randn_like(fake_x0_clean, dtype=torch.float32)
            fm_sigma = self._broadcast_sigma(fm_t, fake_x0_clean)
            fm_x_t = (1.0 - fm_sigma) * fake_x0_clean.detach().to(
                torch.float32
            ) + fm_sigma * fm_noise
            with torch.no_grad():
                v_pred = self._call_trainable_velocity(
                    discriminator_model=discriminator_model,
                    x_t=fm_x_t.detach(),
                    t=fm_t.detach(),
                    c=c,
                    tt=fm_t.detach(),
                )
            flow_matching_target = fm_x_t.detach().to(
                torch.float32
            ) - fm_sigma.detach() * v_pred.detach().to(torch.float32)
            flow_matching_loss = F.mse_loss(
                fake_x0_clean.to(torch.float32),
                flow_matching_target,
            )
            x0_from_v = flow_matching_target
            flow_matching_debug_aux = {
                "opd_debug_fm_x0_hat": fake_x0_clean.detach(),
                "opd_debug_fm_x_t": fm_x_t.detach(),
                "opd_debug_fm_noise": fm_noise.detach(),
                "opd_debug_fm_v_pred": v_pred.detach(),
                "opd_debug_fm_x0_from_v": x0_from_v.detach(),
                "opd_debug_fm_t": fm_t.detach(),
                "opd_gen_flow_matching_t": fm_t.detach(),
            }
        flow_matching_weighted_loss = (
            float(self.generator_flow_matching_weight) * flow_matching_loss
        )
        dmd_loss = torch.zeros_like(gan_loss)
        dmd_debug_aux = {}
        dmd_stats = {}
        dmd_active = self._dmd_generator_regularizer_active(generator_update_index)
        if dmd_active:
            if fake_x0_clean_live is None:
                raise RuntimeError("G phase DMD requires live clean x0 tensor")
            fake_score_tt = self._dmd_fake_score_tt(exit_aux, fake_x0_clean_live)
            dmd_loss, dmd_stats, dmd_debug_aux = self._dmd_generator_regularizer_loss(
                discriminator_model=discriminator_model,
                fake_x0_clean_live=fake_x0_clean_live,
                c=c,
                fake_score_tt=fake_score_tt,
            )
        dmd_weighted_loss = float(self.generator_dmd_regularizer_weight) * dmd_loss
        refl_loss = torch.zeros_like(gan_loss)
        refl_stats = {}
        refl_active = self._generator_refl_active(step)
        if refl_active:
            if fake_x0_clean_live is None:
                raise RuntimeError("G phase REFL requires live clean x0 tensor")
            refl_reward_train_step = (
                None
                if step is None
                else max(int(step) - int(self.generator_refl_warmup_steps) + 1, 1)
            )
            refl_loss, refl_stats = self._generator_refl_loss(
                reward_adapter=reward_adapter,
                wrapped_model=wrapped_model,
                latents=fake_x0_clean_live,
                text=text,
                prompt_embeds=prompt_embeds,
                prompt_mask=prompt_mask,
                train_step=refl_reward_train_step,
            )
        refl_weighted_loss = float(self.generator_refl_weight) * refl_loss
        reward_gan_gen_loss = torch.zeros_like(gan_loss)
        reward_gan_gen_stats = {}
        reward_gan_gen_active = (
            self._reward_gan_enabled() and float(self.reward_gan_generator_weight) != 0.0
        )
        if reward_gan_gen_active:
            if fake_x0_clean_live is None:
                raise RuntimeError("G phase reward-GAN requires live clean x0 tensor")
            with self._temporarily_requires_grad(reward_adapter, False):
                reward_gan_gen_loss, reward_gan_gen_stats = self._reward_gan_gen_loss(
                    reward_adapter=reward_adapter,
                    wrapped_model=wrapped_model,
                    fake_latents=fake_x0_clean_live,
                    text=text,
                    prompt_embeds=prompt_embeds,
                )
        reward_gan_gen_weighted_loss = float(self.reward_gan_generator_weight) * reward_gan_gen_loss
        loss = (
            gan_loss
            + flow_matching_weighted_loss
            + dmd_weighted_loss
            + refl_weighted_loss
            + reward_gan_gen_weighted_loss
        )
        dmd_update_index_tensor = torch.full(
            (1,),
            float(generator_update_index) if generator_update_index is not None else float("nan"),
            device=fake_xt.device,
            dtype=torch.float32,
        )
        stats = self._pack_loss_stats(
            loss_opd=loss.detach(),
            opd_gen_loss=loss.detach(),
            opd_gen_gan_loss=gan_loss.detach().view(1),
            opd_gen_gan_loss_raw=gan_loss_raw.detach().view(1),
            opd_active_discriminator_exit=torch.full(
                (1,),
                float(exit_step_value if discriminator_head is not None else 0.0),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head=torch.full(
                (1,),
                float(exit_step_value if discriminator_head is not None else 0.0),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head_exit1=torch.full(
                (1,),
                1.0 if discriminator_head == "exit1" else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_active_discriminator_head_exit2=torch.full(
                (1,),
                1.0 if discriminator_head == "exit2" else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_real_source_is_data=torch.full(
                (1,),
                float(real_source_is_data),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_real_source_data_prob=torch.full(
                (1,),
                (
                    float(real_source_data_prob)
                    if real_source_data_prob is not None
                    else float(real_source_is_data)
                ),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_gan_loss_weight=torch.full(
                (1,),
                self._generator_gan_loss_weight_for_exit(exit_step_value),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_flow_matching_loss=flow_matching_loss.detach().view(1),
            opd_gen_flow_matching_weight=torch.full(
                (1,),
                float(self.generator_flow_matching_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_flow_matching_weighted_loss=flow_matching_weighted_loss.detach().view(1),
            opd_gen_flow_matching_t=flow_matching_debug_aux.get(
                "opd_gen_flow_matching_t",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_loss=dmd_loss.detach().view(1),
            opd_gen_dmd_weight=torch.full(
                (1,),
                float(self.generator_dmd_regularizer_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_dmd_weighted_loss=dmd_weighted_loss.detach().view(1),
            opd_gen_refl_loss=refl_loss.detach().view(1),
            opd_gen_refl_weight=torch.full(
                (1,),
                float(self.generator_refl_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_refl_weighted_loss=refl_weighted_loss.detach().view(1),
            opd_reward_gan_disc_loss=torch.zeros(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_disc_weighted_loss=torch.zeros(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_disc_fake_logit=torch.full(
                (1,),
                float("nan"),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_disc_real_logit=torch.full(
                (1,),
                float("nan"),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_gen_loss=reward_gan_gen_loss.detach().view(1),
            opd_reward_gan_gen_weighted_loss=reward_gan_gen_weighted_loss.detach().view(1),
            opd_reward_gan_gen_fake_logit=reward_gan_gen_stats.get(
                "opd_reward_gan_gen_fake_logit",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_reward_gan_discriminator_weight=torch.full(
                (1,),
                float(self.reward_gan_discriminator_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_generator_weight=torch.full(
                (1,),
                float(self.reward_gan_generator_weight),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_reward_gan_active=torch.full(
                (1,),
                1.0 if reward_gan_gen_active else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_refl_active=torch.full(
                (1,),
                1.0 if refl_active else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_refl_score=refl_stats.get(
                "opd_gen_refl_score",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_score_mean=refl_stats.get(
                "opd_gen_refl_score_mean",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_clip_score=refl_stats.get(
                "opd_gen_refl_clip_score",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_clip_keep_ratio=refl_stats.get(
                "opd_gen_refl_clip_keep_ratio",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_iaa=refl_stats.get(
                "opd_gen_refl_iaa",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_iqa=refl_stats.get(
                "opd_gen_refl_iqa",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_refl_ista=refl_stats.get(
                "opd_gen_refl_ista",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_active=torch.full(
                (1,),
                1.0 if dmd_active else 0.0,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_dmd_update_index=dmd_update_index_tensor,
            opd_gen_dmd_update_ratio=torch.full(
                (1,),
                float(self.dmd_generator_update_ratio),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_gen_dmd_sigma=dmd_stats.get(
                "opd_gen_dmd_sigma",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_denom_raw=dmd_stats.get(
                "opd_gen_dmd_denom_raw",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_denom_clamped=dmd_stats.get(
                "opd_gen_dmd_denom_clamped",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_raw_delta_abs=dmd_stats.get(
                "opd_gen_dmd_raw_delta_abs",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_raw_delta_rms=dmd_stats.get(
                "opd_gen_dmd_raw_delta_rms",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_grad_abs=dmd_stats.get(
                "opd_gen_dmd_grad_abs",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_grad_rms=dmd_stats.get(
                "opd_gen_dmd_grad_rms",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_fake_score_tt=dmd_stats.get(
                "opd_gen_dmd_fake_score_tt",
                torch.full(
                    (1,),
                    float("nan"),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_fake_score_has_tt=dmd_stats.get(
                "opd_gen_dmd_fake_score_has_tt",
                torch.full(
                    (1,),
                    0.0,
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_fake_smooth_samples=dmd_stats.get(
                "opd_gen_dmd_fake_smooth_samples",
                torch.full(
                    (1,),
                    float(self.dmd_fake_x0_smooth_num_samples),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_fake_smooth_noise_scale=dmd_stats.get(
                "opd_gen_dmd_fake_smooth_noise_scale",
                torch.full(
                    (1,),
                    float(self.dmd_fake_x0_smooth_noise_scale),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_dmd_real_rollout_steps=dmd_stats.get(
                "opd_gen_dmd_real_rollout_steps",
                torch.full(
                    (1,),
                    float(self.dmd_real_score_rollout_steps),
                    device=fake_xt.device,
                    dtype=torch.float32,
                ),
            ),
            opd_gen_fake_logit=pred_fake.detach(),
            opd_exit_gap_temperature=torch.full(
                (1,),
                float(gap_temperature),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_exit_gap_value=torch.full(
                (1,),
                float(gap_value),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_exit_gap_target=torch.full(
                (1,),
                float(gap_target),
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            opd_disc_t=disc_t.detach(),
            opd_disc_tt=disc_tt.detach(),
            opd_phase_is_generator=torch.ones(
                1,
                device=fake_xt.device,
                dtype=torch.float32,
            ),
            **self._early_exit_stat_tensors(exit_aux),
        )
        aux = {
            "opd_disc_t": disc_t.detach(),
            "opd_disc_tt": disc_tt.detach(),
            "opd_disc_input_t": disc_t.detach(),
            "opd_disc_input_tt": disc_tt.detach(),
            "opd_gen_fake_logit": pred_fake.detach(),
            "opd_gen_base_loss_for_grad": gan_loss
            + dmd_weighted_loss
            + reward_gan_gen_weighted_loss,
            "opd_gen_refl_loss_for_grad": refl_loss,
            "opd_debug_student_xt": fake_xt.detach(),
            **exit_aux,
            **flow_matching_debug_aux,
            **dmd_debug_aux,
        }
        return self._format_return(
            loss,
            stats,
            aux,
            return_loss_stats,
            return_log_tensors,
        )

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
        discriminator_model: Optional[ModelLike] = None,
        frozen_discriminator_model: Optional[ModelLike] = None,
        phase: str = "generator",
        **kwargs,
    ):
        generator_update_index = kwargs.pop("generator_update_index", None)
        reward_adapter = kwargs.pop("reward_adapter", None)
        wrapped_model = kwargs.pop("wrapped_model", None)
        text = kwargs.pop("text", None)
        prompt_embeds = kwargs.pop("prompt_embeds", None)
        prompt_mask = kwargs.pop("prompt_mask", None)
        real_image_latents = kwargs.pop("real_image_latents", None)
        real_source = kwargs.pop("real_source", None)
        real_source_data_prob = kwargs.pop("real_source_data_prob", None)
        del kwargs
        if discriminator_model is None:
            raise ValueError("discriminator_model is required")
        phase = str(phase)
        if phase not in {"generator", "discriminator"}:
            raise ValueError("phase must be 'generator' or 'discriminator'")
        real_source_is_data = self._resolve_real_source_is_data(real_source)
        if real_source_data_prob is not None:
            real_source_data_prob = float(real_source_data_prob)

        if phase == "discriminator":
            return self._discriminator_phase_step(
                student_model=student_model,
                teacher_model=teacher_model,
                latent_shape=latent_shape,
                c=c,
                step=step,
                initial_noise=initial_noise,
                real_image_latents=real_image_latents,
                real_source_is_data=real_source_is_data,
                real_source_data_prob=real_source_data_prob,
                discriminator_model=discriminator_model,
                frozen_discriminator_model=frozen_discriminator_model,
                return_loss_stats=return_loss_stats,
                return_log_tensors=return_log_tensors,
                reward_adapter=reward_adapter,
                wrapped_model=wrapped_model,
                text=text,
                prompt_embeds=prompt_embeds,
            )

        return self._generator_phase_step(
            student_model=student_model,
            teacher_model=teacher_model,
            latent_shape=latent_shape,
            c=c,
            step=step,
            initial_noise=initial_noise,
            discriminator_model=discriminator_model,
            return_loss_stats=return_loss_stats,
            return_log_tensors=return_log_tensors,
            generator_update_index=generator_update_index,
            real_source_is_data=real_source_is_data,
            real_source_data_prob=real_source_data_prob,
            reward_adapter=reward_adapter,
            wrapped_model=wrapped_model,
            text=text,
            prompt_embeds=prompt_embeds,
            prompt_mask=prompt_mask,
        )
