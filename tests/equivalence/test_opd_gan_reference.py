import sys
from pathlib import Path

import pytest
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def test_dual_exit_point_config_normalizes_choices_and_weights():
    from verl_distill.algorithms.opd_gan.dual_config import (
        normalize_exit_point_choices,
        normalize_exit_point_weights,
    )

    points, exit_steps = normalize_exit_point_choices("[4, 3, 1]", 4)
    weights = normalize_exit_point_weights("1.0, 0.5, 2.0", points)

    assert points == (4, 3, 1)
    assert exit_steps == (1, 2, 4)
    assert weights == (1.0, 0.5, 2.0)


@pytest.mark.parametrize(
    "choices, match",
    [
        ([], "must not be empty"),
        ([4, 4], "must not contain duplicates"),
        ([5], r"\[1, 4\]"),
    ],
)
def test_dual_exit_point_config_rejects_invalid_choices(choices, match):
    from verl_distill.algorithms.opd_gan.dual_config import normalize_exit_point_choices

    with pytest.raises(ValueError, match=match):
        normalize_exit_point_choices(choices, 4)


def test_dual_alignment_loss_matches_pearson_plus_weighted_mse():
    from verl_distill.algorithms.opd_gan.losses import dual_alignment_loss

    pred = torch.tensor(
        [
            [[1.0, 2.0], [4.0, 8.0], [2.0, -1.0]],
            [[-1.0, 0.5], [3.0, 1.0], [5.0, 4.0]],
        ],
        dtype=torch.float32,
    )
    target = torch.tensor(
        [
            [[2.0, -1.0], [1.0, 3.0], [0.0, 5.0]],
            [[4.0, 2.0], [-2.0, 1.5], [3.0, -0.5]],
        ],
        dtype=torch.float32,
    )

    total, aux = dual_alignment_loss(
        pred,
        target,
        mse_weight=0.5,
        pearson_eps=1.0e-6,
    )

    pred_centered = pred - pred.mean(dim=1, keepdim=True)
    target_centered = target - target.mean(dim=1, keepdim=True)
    pred_std = pred_centered.square().mean(dim=1, keepdim=True).sqrt()
    target_std = target_centered.square().mean(dim=1, keepdim=True).sqrt()
    expected_pearson = torch.nn.functional.mse_loss(
        pred_centered / (pred_std + 1.0e-6),
        target_centered / (target_std + 1.0e-6),
    )
    expected_mse = (pred - target).square().mean()

    torch.testing.assert_close(total, expected_pearson + 0.5 * expected_mse)
    torch.testing.assert_close(aux["pearson"], expected_pearson)
    torch.testing.assert_close(aux["mse"], expected_mse)


def test_dual_alignment_loss_flatten_mode_uses_all_non_batch_dimensions():
    from verl_distill.algorithms.opd_gan.losses import dual_alignment_loss

    pred = torch.tensor([[[1.0, 2.0], [4.0, 8.0], [2.0, -1.0]]])
    target = torch.tensor([[[2.0, -1.0], [1.0, 3.0], [0.0, 5.0]]])

    total, aux = dual_alignment_loss(
        pred,
        target,
        mse_weight=0.0,
        pearson_eps=1.0e-6,
        pearson_mode="flatten",
    )

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    pred_std = pred_centered.square().mean(dim=1, keepdim=True).sqrt()
    target_std = target_centered.square().mean(dim=1, keepdim=True).sqrt()
    expected = torch.nn.functional.mse_loss(
        pred_centered / (pred_std + 1.0e-6),
        target_centered / (target_std + 1.0e-6),
    )

    torch.testing.assert_close(total, expected)
    torch.testing.assert_close(aux["pearson"], expected)


def test_dual_alignment_loss_supports_channel_and_flatten_pearson_modes():
    from verl_distill.algorithms.opd_gan.losses import dual_alignment_loss

    pred = torch.tensor(
        [
            [
                [[1.0, 2.0], [1.0, 2.0]],
                [[3.0, 1.0], [5.0, 3.0]],
            ]
        ]
    )
    target = torch.tensor(
        [
            [
                [[2.0, 5.0], [3.0, 4.0]],
                [[4.0, 1.0], [1.0, 6.0]],
            ]
        ]
    )

    channel_total, channel_aux = dual_alignment_loss(
        pred,
        target,
        mse_weight=0.0,
        pearson_eps=1.0e-6,
        pearson_mode="channel",
    )
    flatten_total, flatten_aux = dual_alignment_loss(
        pred,
        target,
        mse_weight=0.0,
        pearson_eps=1.0e-6,
        pearson_mode="flatten",
    )

    pred_flat = pred.flatten(1)
    target_flat = target.flatten(1)
    pred_centered = pred_flat - pred_flat.mean(dim=1, keepdim=True)
    target_centered = target_flat - target_flat.mean(dim=1, keepdim=True)
    pred_std = pred_centered.square().mean(dim=1, keepdim=True).sqrt()
    target_std = target_centered.square().mean(dim=1, keepdim=True).sqrt()
    expected_flatten = torch.nn.functional.mse_loss(
        pred_centered / (pred_std + 1.0e-6),
        target_centered / (target_std + 1.0e-6),
    )

    torch.testing.assert_close(flatten_total, expected_flatten)
    torch.testing.assert_close(flatten_aux["pearson"], expected_flatten)
    assert not torch.allclose(channel_total, flatten_total)
    torch.testing.assert_close(channel_total, channel_aux["pearson"])


def test_pair_diff_stats_detaches_reference_tensor():
    from verl_distill.algorithms.opd_gan.losses import pair_diff_stats

    fake = torch.tensor([[1.0, 3.0]], requires_grad=True)
    real = torch.tensor([[0.0, 1.0]], requires_grad=True)

    stats = pair_diff_stats("diff", fake, real)
    loss = stats["diff_mse"].sum()
    loss.backward()

    torch.testing.assert_close(stats["diff_l1"], torch.tensor([1.5]))
    torch.testing.assert_close(stats["diff_rmse"], torch.tensor([1.5811388]))
    assert fake.grad is not None
    assert real.grad is None


def test_rollout_step_from_velocity_uses_ode_update_when_ratio_is_zero():
    from verl_distill.algorithms.opd_gan.rollout import rollout_step_from_velocity

    x_t = torch.tensor([[2.0, 4.0]])
    sigma_cur = torch.tensor([0.75])
    sigma_next = torch.tensor([0.25])
    velocity = torch.tensor([[1.0, -2.0]])

    out = rollout_step_from_velocity(
        x_t=x_t,
        sigma_cur=sigma_cur,
        sigma_next=sigma_next,
        velocity=velocity,
        rollout_stochast_ratio=0.0,
    )

    torch.testing.assert_close(out, torch.tensor([[1.5, 5.0]]))


def test_rollout_step_from_velocity_mixes_terminal_noise_for_stochastic_rollout():
    from verl_distill.algorithms.opd_gan.rollout import rollout_step_from_velocity

    x_t = torch.tensor([[2.0, 4.0]])
    sigma_cur = torch.tensor([0.75])
    sigma_next = torch.tensor([0.25])
    velocity = torch.tensor([[1.0, -2.0]])
    noise = torch.tensor([[0.5, -1.5]])

    out = rollout_step_from_velocity(
        x_t=x_t,
        sigma_cur=sigma_cur,
        sigma_next=sigma_next,
        velocity=velocity,
        noise=noise,
        rollout_stochast_ratio=1.0,
    )

    x0_hat = x_t - sigma_cur.view(1, 1) * velocity
    expected = (1.0 - sigma_next.view(1, 1)) * x0_hat + sigma_next.view(1, 1) * noise
    torch.testing.assert_close(out, expected)


def test_cat_or_empty_preserves_reference_device_and_float_dtype():
    from verl_distill.algorithms.opd_gan.rollout import cat_or_empty

    reference = torch.ones(2, dtype=torch.float16)

    torch.testing.assert_close(
        cat_or_empty([torch.tensor([1.0]), torch.tensor([2.0])], reference),
        torch.tensor([1.0, 2.0]),
    )
    empty = cat_or_empty([], reference)
    assert empty.numel() == 0
    assert empty.device == reference.device
    assert empty.dtype == torch.float32


class _TinyVelocity(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.velocity = torch.nn.Parameter(torch.tensor(float(value)))

    def forward(self, x_t, t, c):
        del t, c
        return torch.ones_like(x_t) * self.velocity


class _TinyDualDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.align_scale = torch.nn.Parameter(torch.tensor(0.75))
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.25))

    def discriminate(
        self,
        x_t,
        t,
        c,
        tt,
        return_raw,
        discriminator_output=None,
        **kwargs,
    ):
        del t, c, tt, return_raw, kwargs
        flat = x_t.flatten(1).mean(dim=1, keepdim=True)
        align = torch.stack(
            [
                flat * self.align_scale,
                (flat + 0.25) * self.align_scale,
            ],
            dim=1,
        )
        logits = (flat.squeeze(1) * self.logit_scale).reshape(-1)
        if discriminator_output == "gan":
            return logits
        if discriminator_output == "align":
            return align
        return {"align": align, "logits": logits}


class _RecordingDualDiscriminator(_TinyDualDiscriminator):
    def __init__(self):
        super().__init__()
        self.calls = []

    def discriminate(
        self,
        x_t,
        t,
        c,
        tt,
        return_raw,
        discriminator_output=None,
        **kwargs,
    ):
        self.calls.append(
            {
                "output": discriminator_output,
                "discriminator_head": kwargs.get("discriminator_head"),
                "t": t.detach().clone(),
                "tt": tt.detach().clone(),
                "x_requires_grad": bool(x_t.requires_grad),
                "param_requires_grad": tuple(
                    bool(param.requires_grad) for param in self.parameters()
                ),
            }
        )
        return super().discriminate(
            x_t,
            t,
            c,
            tt,
            return_raw,
            discriminator_output=discriminator_output,
            **kwargs,
        )

    def forward(self, x_t, t, c, tt=None, **kwargs):
        del c, kwargs
        self.calls.append(
            {
                "output": "velocity",
                "t": t.detach().clone(),
                "tt": tt.detach().clone(),
            }
        )
        return torch.ones_like(x_t) * 0.25


class _TinyFrozenDiscriminator(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.calls = []

    def discriminate(self, x_t, t, c, tt, return_raw, **kwargs):
        del c, return_raw, kwargs
        self.calls.append(
            {
                "t": t.detach().clone(),
                "tt": tt.detach().clone(),
            }
        )
        flat = x_t.flatten(1).mean(dim=1, keepdim=True)
        return torch.stack(
            [
                flat * self.scale,
                (flat + 0.5) * self.scale,
            ],
            dim=1,
        )


class _TinyDMDDualDiscriminator(_TinyDualDiscriminator):
    def __init__(self):
        super().__init__()
        self.score_scale = torch.nn.Parameter(torch.tensor(0.0))
        self.score_tt_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.score_calls = []

    def forward(self, x_t, t, c, tt=None):
        del c
        t_view = t.reshape(-1, *([1] * (x_t.ndim - 1))).to(
            device=x_t.device,
            dtype=x_t.dtype,
        )
        velocity = x_t * self.score_scale + t_view * 0.0
        if tt is not None:
            tt_view = tt.reshape(-1, *([1] * (x_t.ndim - 1))).to(
                device=x_t.device,
                dtype=x_t.dtype,
            )
            velocity = velocity + tt_view * self.score_tt_scale
            self.score_calls.append("fake")
        else:
            self.score_calls.append("real")
        return velocity


def _tiny_dual_method(**overrides):
    from verl_distill.algorithms.opd_gan.gan_method import (
        DualDistilledDiscriminatorOPD,
    )

    kwargs = {
        "num_train_timestep": 1000,
        "num_student_steps": 1,
        "teacher_micro_steps": 1,
        "timestep_shift": 1.0,
        "initial_warmup_step_size": 0.0,
        "exit_step_mode": "fixed",
        "fixed_exit_step": 1,
        "rollout_stochast_ratio": 0.0,
        "discriminator_update_ratio": 1,
        "dual_align_mse_weight": 0.5,
        "dual_align_pearson_eps": 1.0e-6,
    }
    kwargs.update(overrides)
    return DualDistilledDiscriminatorOPD(**kwargs)


def test_separate_exit_discriminator_head_selection():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0_full_teacher_real_separate_discs",
        num_student_steps=2,
        teacher_micro_steps=2,
        exit_point_choices=(2, 1),
    )

    assert method._discriminator_head_for_exit(1) == "exit1"
    assert method._discriminator_head_for_exit(2) == "exit2"
    with pytest.raises(ValueError, match="exit_step"):
        method._discriminator_head_for_exit(3)


@pytest.mark.parametrize(
    "phase, exit_step, expected_outputs",
    [
        ("discriminator", 1, ["both", "both"]),
        ("discriminator", 2, ["both", "both"]),
        ("generator", 1, ["gan"]),
        ("generator", 2, ["gan"]),
    ],
)
def test_separate_exit_discriminator_phase_routes_trainable_head(
    phase,
    exit_step,
    expected_outputs,
):
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0_full_teacher_real_separate_discs",
        num_student_steps=2,
        teacher_micro_steps=2,
        fixed_exit_step=exit_step,
        exit_point_choices=(2, 1),
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _RecordingDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)
    kwargs = {}
    if phase == "discriminator":
        kwargs["frozen_discriminator_model"] = frozen

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase=phase,
        discriminator_model=discriminator,
        return_loss_stats=True,
        return_log_tensors=True,
        **kwargs,
    )

    assert loss.requires_grad
    assert [call["output"] for call in discriminator.calls] == expected_outputs
    assert {call["discriminator_head"] for call in discriminator.calls} == {f"exit{exit_step}"}
    assert stats["opd_active_discriminator_exit"][0].item() == float(exit_step)
    assert stats["opd_active_discriminator_head"][0].item() == float(exit_step)
    assert stats["opd_active_discriminator_head_exit1"][0].item() == float(exit_step == 1)
    assert stats["opd_active_discriminator_head_exit2"][0].item() == float(exit_step == 2)


@pytest.mark.parametrize("exit_step, expected_weight", [(1, 0.5), (2, 1.5)])
def test_discriminator_loss_applies_selected_exit_weight(
    exit_step,
    expected_weight,
):
    common_kwargs = {
        "loss_mode": "dual_distilled_early_exit_x0_full_teacher_real_separate_discs",
        "num_student_steps": 2,
        "teacher_micro_steps": 2,
        "fixed_exit_step": exit_step,
        "exit_point_choices": (2, 1),
    }
    base_method = _tiny_dual_method(**common_kwargs)
    weighted_method = _tiny_dual_method(
        **common_kwargs,
        discriminator_loss_weight_by_exit=(0.5, 1.5),
    )
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    def run(method):
        torch.manual_seed(1234)
        return method.training_step(
            student_model=_TinyVelocity(0.2),
            teacher_model=_TinyVelocity(0.0),
            latent_shape=tuple(initial_noise.shape),
            c=c,
            initial_noise=initial_noise,
            phase="discriminator",
            discriminator_model=_TinyDualDiscriminator(),
            frozen_discriminator_model=_TinyFrozenDiscriminator(),
            return_loss_stats=True,
            return_log_tensors=True,
        )

    base_loss, _base_stats, _ = run(base_method)
    weighted_loss, weighted_stats, _ = run(weighted_method)

    torch.testing.assert_close(weighted_loss, base_loss * expected_weight)
    assert weighted_stats["opd_disc_loss_weight"][0].item() == expected_weight


def _early_exit_x0_noise_residuals(method):
    torch.manual_seed(123)
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    initial_noise = torch.zeros(2, 1, 2, 2)
    (
        _fake_xt,
        _real_xt,
        fake_x0,
        real_x0,
        _t_disc,
        _tt,
        aux,
    ) = method._early_exit_xt_pair(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=[torch.zeros(2, 1)],
        initial_noise=initial_noise,
        step=None,
        grad_enabled=False,
        include_real=True,
    )
    sigma = aux["opd_x0_noise_sigma"].view(-1, 1, 1, 1)
    fake_clean = aux["opd_debug_student_x0_clean"]
    real_clean = aux["opd_debug_teacher_x0_clean"]
    fake_noise = (fake_x0 - (1.0 - sigma) * fake_clean) / sigma
    real_noise = (real_x0 - (1.0 - sigma) * real_clean) / sigma
    return fake_noise, real_noise


def test_dual_x0_noise_default_shares_fake_real_noise():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0_full_teacher_real",
        x0_noise_sigma_by_exit=(0.5,),
    )

    fake_noise, real_noise = _early_exit_x0_noise_residuals(method)

    torch.testing.assert_close(fake_noise, real_noise)


def test_dual_x0_noise_can_disable_fake_real_sharing():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0_full_teacher_real",
        x0_noise_sigma_by_exit=(0.5,),
        x0_noise_share_fake_real=False,
    )

    fake_noise, real_noise = _early_exit_x0_noise_residuals(method)

    assert not torch.allclose(fake_noise, real_noise)


def test_dual_x0_fm_weight_uses_x0_noise_sigma_for_trainable_discriminator_t():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0",
        generator_flow_matching_weight=0.5,
        generator_flow_matching_t_min=0.35,
        generator_flow_matching_t_max=0.35,
        x0_noise_sigma_by_exit=(0.2,),
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _RecordingDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    d_loss, _, d_aux = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    assert d_loss.requires_grad
    for call in discriminator.calls:
        torch.testing.assert_close(call["t"], torch.full((2,), 0.005))
        torch.testing.assert_close(call["tt"], torch.full((2,), 0.005))
    for call in frozen.calls:
        torch.testing.assert_close(call["t"], torch.full((2,), 0.005))
        torch.testing.assert_close(call["tt"], torch.full((2,), 0.005))
    torch.testing.assert_close(d_aux["opd_disc_t"], torch.full((2,), 0.005))

    discriminator.calls.clear()
    g_loss, g_stats, g_aux = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    g_loss.backward()

    assert "opd_gen_flow_matching_loss" in g_stats
    assert "opd_gen_flow_matching_weight" in g_stats
    for key in (
        "opd_debug_fm_x0_hat",
        "opd_debug_fm_x_t",
        "opd_debug_fm_v_pred",
        "opd_debug_fm_x0_from_v",
        "opd_debug_fm_t",
    ):
        assert key in g_aux
    torch.testing.assert_close(
        g_aux["opd_debug_fm_x0_from_v"],
        g_aux["opd_debug_fm_x_t"]
        - g_aux["opd_debug_fm_t"].view(-1, 1, 1, 1) * g_aux["opd_debug_fm_v_pred"],
    )
    expected_fm_loss = torch.nn.functional.mse_loss(
        g_aux["opd_debug_fm_x0_hat"],
        g_aux["opd_debug_fm_x0_from_v"],
    )
    torch.testing.assert_close(
        g_stats["opd_gen_flow_matching_loss"][0],
        expected_fm_loss,
    )
    assert (g_aux["opd_debug_fm_x_t"] - g_aux["opd_debug_fm_x0_hat"]).abs().mean().item() > 0.0
    assert student.velocity.grad is not None
    assert student.velocity.grad.abs().item() > 0.0
    assert discriminator.logit_scale.grad is None
    gan_calls = [call for call in discriminator.calls if call["output"] == "gan"]
    velocity_calls = [call for call in discriminator.calls if call["output"] == "velocity"]
    assert len(gan_calls) == 1
    assert len(velocity_calls) == 1
    for call in gan_calls:
        torch.testing.assert_close(call["t"], torch.full((2,), 0.005))
        torch.testing.assert_close(call["tt"], torch.full((2,), 0.005))
    for call in velocity_calls:
        torch.testing.assert_close(call["t"], torch.full((2,), 0.35))
        torch.testing.assert_close(call["tt"], torch.full((2,), 0.35))
    torch.testing.assert_close(g_aux["opd_disc_t"], torch.full((2,), 0.005))
    torch.testing.assert_close(g_aux["opd_debug_fm_t"], torch.full((2,), 0.35))


def test_dual_x0_noise_uses_same_noise_for_fake_and_real_discriminator_inputs():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0",
        x0_noise_sigma_by_exit=(0.2,),
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(-0.1)
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 2, 2)

    result = method._early_exit_xt_pair(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        step=1,
        grad_enabled=False,
        include_real=True,
    )
    fake_x0 = result[2]
    real_x0 = result[3]
    aux = result[6]

    sigma = aux["opd_x0_noise_sigma"].view(-1, 1, 1, 1)
    fake_clean = aux["opd_debug_student_x0_clean"]
    real_clean = aux["opd_debug_teacher_x0_clean"]
    fake_noise = (fake_x0 - (1.0 - sigma) * fake_clean) / sigma
    real_noise = (real_x0 - (1.0 - sigma) * real_clean) / sigma

    torch.testing.assert_close(fake_noise, real_noise)


def test_dual_x0_noise_can_use_independent_fake_and_real_noise():
    method = _tiny_dual_method(
        loss_mode="dual_distilled_early_exit_x0",
        x0_noise_sigma_by_exit=(0.2,),
        x0_noise_share_fake_real=False,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(-0.1)
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 2, 2)

    result = method._early_exit_xt_pair(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        step=1,
        grad_enabled=False,
        include_real=True,
    )
    fake_x0, real_x0, aux = result[2], result[3], result[6]
    sigma = aux["opd_x0_noise_sigma"].view(-1, 1, 1, 1)
    fake_noise = (fake_x0 - (1.0 - sigma) * aux["opd_debug_student_x0_clean"]) / sigma
    real_noise = (real_x0 - (1.0 - sigma) * aux["opd_debug_teacher_x0_clean"]) / sigma

    assert not torch.equal(fake_noise, real_noise)


def test_dual_training_step_generator_phase_backprops_only_to_student():
    method = _tiny_dual_method()
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, aux = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    assert "opd_gen_gan_loss" in stats
    assert "opd_debug_student_xt" in aux
    assert student.velocity.grad is not None
    assert student.velocity.grad.abs().item() > 0.0
    assert discriminator.logit_scale.grad is None
    assert all(param.requires_grad for param in discriminator.parameters())


def test_dual_training_step_discriminator_phase_backprops_only_to_trainable_head():
    method = _tiny_dual_method()
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, aux = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    assert "opd_disc_loss" in stats
    assert "opd_dual_align_loss" in stats
    assert "opd_debug_teacher_x0" in aux
    assert discriminator.align_scale.grad is not None
    assert discriminator.align_scale.grad.abs().item() > 0.0
    assert discriminator.logit_scale.grad is not None
    assert discriminator.logit_scale.grad.abs().item() > 0.0
    assert student.velocity.grad is None
    assert frozen.scale.grad is None


def test_dual_apt_r1_backprops_to_unperturbed_real_prediction():
    torch.manual_seed(123)
    method = _tiny_dual_method(
        apt_r1_weight=1.0,
        apt_r1_sigma_min=0.25,
        apt_r1_sigma_max=0.25,
    )
    discriminator = _TinyDualDiscriminator()
    real_disc_x = torch.ones(2, 1, 1, 1)
    pred_real = torch.full((2,), 0.75, requires_grad=True)
    c = [torch.zeros(2, 1)]
    disc_t = torch.full((2,), 0.005)
    disc_tt = torch.full((2,), 0.005)

    loss, _, _ = method._apt_r1_real_consistency_loss(
        discriminator_model=discriminator,
        real_disc_x=real_disc_x,
        pred_real=pred_real,
        disc_t=disc_t,
        c=c,
        disc_tt=disc_tt,
        discriminator_head=None,
    )
    loss.backward()

    assert pred_real.grad is not None
    assert pred_real.grad.abs().sum().item() > 0.0
    assert discriminator.logit_scale.grad is not None
    assert discriminator.logit_scale.grad.abs().item() > 0.0


def test_dual_discriminator_phase_adds_apt_r1_real_consistency_loss():
    torch.manual_seed(123)
    method = _tiny_dual_method(
        apt_r1_weight=5.0,
        apt_r1_sigma_min=0.25,
        apt_r1_sigma_max=0.25,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    assert stats["opd_apt_r1_loss"][0].item() > 0.0
    torch.testing.assert_close(stats["opd_apt_r1_weight"][0], torch.tensor(5.0))
    torch.testing.assert_close(stats["opd_apt_r1_sigma"][0], torch.tensor(0.25))
    assert stats["opd_apt_r1_real_perturbed_logit"][0].numel() == 1
    assert discriminator.logit_scale.grad is not None
    assert discriminator.logit_scale.grad.abs().item() > 0.0
    assert student.velocity.grad is None
    assert frozen.scale.grad is None


def test_dual_apt_r1_weight_is_not_scaled_by_exit_weight():
    torch.manual_seed(123)
    method = _tiny_dual_method(
        apt_r1_weight=5.0,
        apt_r1_sigma_min=0.25,
        apt_r1_sigma_max=0.25,
        discriminator_loss_weight_by_exit=(0.125,),
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        return_loss_stats=True,
        return_log_tensors=True,
    )

    disc_gan = stats["opd_disc_gan_loss"][0] / stats["opd_disc_gan_loss"][1]
    align = stats["opd_dual_align_loss"][0] / stats["opd_dual_align_loss"][1]
    r1 = stats["opd_apt_r1_loss"][0] / stats["opd_apt_r1_loss"][1]
    expected = 0.125 * (disc_gan + float(method.dual_align_loss_weight) * align)
    expected = expected + 5.0 * r1

    torch.testing.assert_close(loss.detach(), expected)


def test_dual_dmd_score_loss_backprops_to_fake_score_in_discriminator_phase():
    method = _tiny_dual_method(
        generator_regularizer_type="dmd",
        dmd_fake_score_loss_weight=1.0,
        generator_dmd_regularizer_weight=1.0,
        dmd_score_t_min=0.25,
        dmd_score_t_max=0.25,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDMDDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    assert stats["opd_dmd_score_loss"][0].item() > 0.0
    assert stats["opd_dmd_score_fake_score_tt"][0].item() > 0.0
    assert discriminator.score_tt_scale.grad is not None
    assert discriminator.score_tt_scale.grad.abs().item() > 0.0
    assert discriminator.score_calls == ["fake"]
    assert student.velocity.grad is None


def test_dual_dmd_generator_regularizer_uses_cadence_and_freezes_score():
    method = _tiny_dual_method(
        generator_regularizer_type="dmd",
        generator_dmd_regularizer_weight=1.0,
        dmd_fake_score_loss_weight=1.0,
        dmd_generator_update_ratio=5,
        dmd_score_t_min=0.25,
        dmd_score_t_max=0.25,
        dmd_fake_x0_smooth_num_samples=1,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDMDDualDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    inactive_loss, inactive_stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        generator_update_index=4,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    assert inactive_loss.requires_grad
    assert inactive_stats["opd_gen_dmd_active"][0].item() == 0.0
    assert inactive_stats["opd_gen_dmd_loss"][0].item() == 0.0
    assert discriminator.score_calls == []

    active_loss, active_stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        generator_update_index=5,
        return_loss_stats=True,
        return_log_tensors=True,
    )
    active_loss.backward()

    assert active_stats["opd_gen_dmd_active"][0].item() == 1.0
    assert active_stats["opd_gen_dmd_loss"][0].item() > 0.0
    assert active_stats["opd_gen_dmd_update_index"][0].item() == 5.0
    assert discriminator.score_calls == ["fake", "real"]
    assert student.velocity.grad is not None
    assert discriminator.score_tt_scale.grad is None


def test_dual_dmd_generator_real_score_can_use_multistep_rollout():
    method = _tiny_dual_method(
        generator_regularizer_type="dmd",
        generator_dmd_regularizer_weight=1.0,
        dmd_fake_score_loss_weight=1.0,
        dmd_generator_update_ratio=1,
        dmd_score_t_min=0.25,
        dmd_score_t_max=0.25,
        dmd_fake_x0_smooth_num_samples=1,
        dmd_real_score_rollout_steps=3,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDMDDualDiscriminator()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    _loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        generator_update_index=1,
        return_loss_stats=True,
        return_log_tensors=True,
    )

    assert stats["opd_gen_dmd_real_rollout_steps"][0].item() == 3.0
    assert discriminator.score_calls.count("fake") == 1
    assert discriminator.score_calls.count("real") == 3


class _TinyREFLRewardAdapter:
    def __init__(self):
        self.seen_requires_grad = None
        self.calls = 0
        self.train_steps = []

    def score_from_latents(self, wrapped_model, latents, **kwargs):
        self.train_steps.append(kwargs.get("train_step"))
        self.calls += 1
        self.seen_requires_grad = bool(latents.requires_grad)
        pixels = wrapped_model.latents_to_pixels(latents)
        scores = pixels.flatten(1).mean(dim=1)
        return scores, {
            "reward_score_mean": scores.detach().mean(),
            "reward_iaa": scores.detach() + 1.0,
        }


class _FakeRewardGanAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.calls = []

    def reward_gan_logits_from_latents(
        self,
        wrapped_model,
        latents,
        text=None,
        prompt_embeds=None,
    ):
        del wrapped_model, text, prompt_embeds
        self.calls.append(latents.requires_grad)
        logits = latents.flatten(1).mean(dim=1) * self.weight
        return logits, {"reward_gan_logit_mean": logits.detach().mean()}


class _TinyWrappedModel:
    def latents_to_pixels(self, latents):
        return latents


def test_dual_generator_phase_adds_refl_loss_and_backprops_to_student():
    method = _tiny_dual_method(generator_refl_weight=2.0)
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    reward_adapter = _TinyREFLRewardAdapter()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    assert reward_adapter.seen_requires_grad is True
    assert stats["opd_gen_refl_active"][0].item() == 1.0
    assert stats["opd_gen_refl_weight"][0].item() == 2.0
    assert stats["opd_gen_refl_loss"][0].item() < 0.0
    gen_loss = stats["opd_gen_loss"][0] / stats["opd_gen_loss"][1]
    expected = (
        stats["opd_gen_gan_loss"][0] / stats["opd_gen_gan_loss"][1]
        + stats["opd_gen_dmd_weighted_loss"][0] / stats["opd_gen_dmd_weighted_loss"][1]
        + stats["opd_gen_refl_weighted_loss"][0] / stats["opd_gen_refl_weighted_loss"][1]
    )
    torch.testing.assert_close(gen_loss, expected)
    assert student.velocity.grad is not None
    assert student.velocity.grad.abs().item() > 0.0


def test_dual_generator_refl_waits_for_warmup_step_before_calling_adapter():
    method = _tiny_dual_method(
        generator_refl_weight=2.0,
        generator_refl_warmup_steps=5,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    reward_adapter = _TinyREFLRewardAdapter()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    _, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        step=4,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        return_loss_stats=True,
        return_log_tensors=True,
    )

    assert reward_adapter.calls == 0
    assert stats["opd_gen_refl_active"][0].item() == 0.0
    assert stats["opd_gen_refl_weighted_loss"][0].item() == 0.0

    _, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(initial_noise.shape),
        c=c,
        step=5,
        initial_noise=initial_noise,
        phase="generator",
        discriminator_model=discriminator,
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        return_loss_stats=True,
        return_log_tensors=True,
    )

    assert reward_adapter.calls == 1
    assert stats["opd_gen_refl_active"][0].item() == 1.0


def test_dual_generator_refl_passes_step_relative_to_refl_warmup():
    method = _tiny_dual_method(
        generator_refl_weight=1.0,
        generator_refl_warmup_steps=1200,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    reward_adapter = _TinyREFLRewardAdapter()
    c = [torch.zeros(2, 1)]
    initial_noise = torch.ones(2, 1, 1, 1)

    for step in (1200, 1700):
        method.training_step(
            student_model=student,
            teacher_model=teacher,
            latent_shape=tuple(initial_noise.shape),
            c=c,
            step=step,
            initial_noise=initial_noise,
            phase="generator",
            discriminator_model=discriminator,
            reward_adapter=reward_adapter,
            wrapped_model=_TinyWrappedModel(),
            return_loss_stats=True,
            return_log_tensors=True,
        )

    assert reward_adapter.train_steps == [1, 501]


def test_dual_generator_refl_clip_skips_batch_when_mean_score_is_high():
    method = _tiny_dual_method(
        generator_refl_weight=1.0,
        generator_refl_clip_score=62.0,
    )
    reward_adapter = _TinyREFLRewardAdapter()
    latents = torch.tensor([[[[70.0]]], [[[60.0]]]], requires_grad=True)

    loss, stats = method._generator_refl_loss(
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        latents=latents,
    )
    loss.backward()

    torch.testing.assert_close(loss.detach(), torch.tensor(0.0))
    torch.testing.assert_close(latents.grad[0], torch.zeros_like(latents.grad[0]))
    torch.testing.assert_close(latents.grad[1], torch.zeros_like(latents.grad[1]))
    torch.testing.assert_close(
        stats["opd_gen_refl_clip_score"],
        torch.tensor([62.0]),
    )
    torch.testing.assert_close(
        stats["opd_gen_refl_clip_keep_ratio"],
        torch.tensor([0.0]),
    )


def test_dual_discriminator_invoke_disables_separate_r_modulation():
    from verl_distill.algorithms.opd_gan.gan_method import (
        DualDistilledDiscriminatorOPD,
    )

    class _RecordingDiscriminator:
        def __init__(self):
            self.kwargs = None

        def discriminate(self, x_t, **kwargs):
            self.kwargs = kwargs
            return torch.zeros(x_t.shape[0])

    discriminator = _RecordingDiscriminator()
    x_t = torch.zeros(2, 1, 1, 1)
    t = torch.ones(2)
    c = [torch.zeros(2, 1)]

    DualDistilledDiscriminatorOPD._invoke_discriminator(
        discriminator_model=discriminator,
        x_t=x_t,
        t=t,
        c=c,
        tt=t,
        return_raw=True,
    )

    assert discriminator.kwargs["disable_separate_r_modulation"] is True


def test_dual_method_accepts_generator_adv_loss_min():
    method = _tiny_dual_method(
        generator_adv_loss_min=0.2,
    )

    assert method.generator_adv_loss_min == 0.2

    method = _tiny_dual_method(
        generator_adv_loss_min=0.0,
    )

    assert method.generator_adv_loss_min == 0.0


def test_dual_method_accepts_generator_gan_loss_weight_by_exit():
    method = _tiny_dual_method(
        num_student_steps=2,
        generator_gan_loss_weight_by_exit=(0.25, 2.0),
    )

    assert method.generator_gan_loss_weight_by_exit == (0.25, 2.0)


def test_dual_method_accepts_reward_gan_flags():
    method = _tiny_dual_method(
        generator_refl_mode="reward_gan",
        reward_gan_discriminator_weight=1.25,
        reward_gan_generator_weight=0.5,
    )

    assert method.generator_refl_mode == "reward_gan"
    assert method.reward_gan_discriminator_weight == 1.25
    assert method.reward_gan_generator_weight == 0.5
    assert method._reward_gan_enabled()
    assert not method._generator_refl_enabled()


def test_dual_discriminator_phase_adds_reward_gan_loss():
    method = _tiny_dual_method(
        generator_refl_mode="reward_gan",
        reward_gan_discriminator_weight=1.5,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    frozen = _TinyFrozenDiscriminator()
    reward_adapter = _FakeRewardGanAdapter()
    c = [torch.zeros(2, 1)]
    fake_clean = torch.tensor([[[[0.25]]], [[[0.5]]]], dtype=torch.float32)
    real_clean = torch.tensor([[[[1.0]]], [[[1.5]]]], dtype=torch.float32)
    fake_xt = fake_clean + 0.1
    real_xt = real_clean + 0.2

    def fake_pair(**kwargs):
        del kwargs
        aux = {
            "opd_debug_student_x0_clean": fake_clean,
            "opd_debug_teacher_x0_clean": real_clean,
            "opd_exit_step": torch.ones(1),
            "opd_exit_sigma": torch.full((2,), 0.005),
            "opd_exit_rollout_steps": torch.ones(1),
            "opd_student_next_abs": torch.zeros(2),
            "opd_teacher_next_abs": torch.zeros(2),
            "opd_alpha_roll": torch.zeros(1),
            "opd_alpha_roll_prob": torch.zeros(1),
            "opd_rollout_stochast_ratio": torch.zeros(1),
            "opd_x0_noise_sigma": torch.zeros(2),
        }
        return (
            fake_xt,
            real_xt,
            fake_clean,
            real_clean,
            torch.full((2,), 0.005),
            torch.full((2,), 0.005),
            aux,
        )

    method._early_exit_xt_pair = fake_pair

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(fake_clean.shape),
        c=c,
        initial_noise=torch.zeros_like(fake_clean),
        phase="discriminator",
        discriminator_model=discriminator,
        frozen_discriminator_model=frozen,
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        return_loss_stats=True,
        return_log_tensors=True,
    )

    fake_logits = fake_clean.flatten(1).mean(dim=1) * reward_adapter.weight.detach()
    real_logits = real_clean.flatten(1).mean(dim=1) * reward_adapter.weight.detach()
    expected_reward = (
        torch.nn.functional.softplus(fake_logits).mean()
        + torch.nn.functional.softplus(-real_logits).mean()
    )
    expected_weighted = expected_reward * 1.5
    expected_total = (
        stats["opd_disc_gan_loss"][0] / stats["opd_disc_gan_loss"][1]
        + stats["opd_dual_align_loss"][0] / stats["opd_dual_align_loss"][1]
        + stats["opd_dmd_score_weighted_loss"][0] / stats["opd_dmd_score_weighted_loss"][1]
        + stats["opd_reward_gan_disc_weighted_loss"][0]
        / stats["opd_reward_gan_disc_weighted_loss"][1]
    )

    assert reward_adapter.calls == [False, False]
    torch.testing.assert_close(stats["opd_reward_gan_active"][0], torch.tensor(1.0))
    torch.testing.assert_close(
        stats["opd_reward_gan_discriminator_weight"][0],
        torch.tensor(1.5),
    )
    torch.testing.assert_close(stats["opd_reward_gan_disc_loss"][0], expected_reward)
    torch.testing.assert_close(
        stats["opd_reward_gan_disc_weighted_loss"][0],
        expected_weighted,
    )
    torch.testing.assert_close(stats["opd_reward_gan_disc_fake_logit"][0], fake_logits.sum())
    torch.testing.assert_close(stats["opd_reward_gan_disc_real_logit"][0], real_logits.sum())
    torch.testing.assert_close(loss.detach(), expected_total)


def test_dual_generator_phase_adds_reward_gan_loss_and_freezes_c():
    method = _tiny_dual_method(
        generator_refl_mode="reward_gan",
        reward_gan_generator_weight=0.75,
    )
    student = _TinyVelocity(0.2)
    teacher = _TinyVelocity(0.0)
    discriminator = _TinyDualDiscriminator()
    reward_adapter = _FakeRewardGanAdapter()
    c = [torch.zeros(2, 1)]
    fake_clean = torch.tensor(
        [[[[0.25]]], [[[0.75]]]],
        dtype=torch.float32,
        requires_grad=True,
    )

    def fake_pair(**kwargs):
        del kwargs
        aux = {
            "_opd_live_student_x0_clean": fake_clean,
            "opd_debug_student_x0_clean": fake_clean.detach(),
            "opd_exit_step": torch.ones(1),
            "opd_exit_sigma": torch.full((2,), 0.005),
            "opd_exit_rollout_steps": torch.ones(1),
            "opd_student_next_abs": torch.zeros(2),
            "opd_teacher_next_abs": torch.zeros(2),
            "opd_alpha_roll": torch.zeros(1),
            "opd_alpha_roll_prob": torch.zeros(1),
            "opd_rollout_stochast_ratio": torch.zeros(1),
            "opd_x0_noise_sigma": torch.zeros(2),
        }
        return (
            fake_clean,
            None,
            fake_clean,
            None,
            torch.full((2,), 0.005),
            torch.full((2,), 0.005),
            aux,
        )

    method._early_exit_xt_pair = fake_pair

    loss, stats, _ = method.training_step(
        student_model=student,
        teacher_model=teacher,
        latent_shape=tuple(fake_clean.shape),
        c=c,
        initial_noise=torch.zeros_like(fake_clean.detach()),
        phase="generator",
        discriminator_model=discriminator,
        reward_adapter=reward_adapter,
        wrapped_model=_TinyWrappedModel(),
        return_loss_stats=True,
        return_log_tensors=True,
    )
    loss.backward()

    fake_logits = fake_clean.detach().flatten(1).mean(dim=1) * reward_adapter.weight.detach()
    expected_reward = torch.nn.functional.softplus(-fake_logits).mean()
    expected_weighted = expected_reward * 0.75
    expected_total = (
        stats["opd_gen_gan_loss"][0] / stats["opd_gen_gan_loss"][1]
        + stats["opd_gen_flow_matching_weighted_loss"][0]
        / stats["opd_gen_flow_matching_weighted_loss"][1]
        + stats["opd_gen_dmd_weighted_loss"][0] / stats["opd_gen_dmd_weighted_loss"][1]
        + stats["opd_gen_refl_weighted_loss"][0] / stats["opd_gen_refl_weighted_loss"][1]
        + stats["opd_reward_gan_gen_weighted_loss"][0]
        / stats["opd_reward_gan_gen_weighted_loss"][1]
    )

    assert reward_adapter.calls == [True]
    torch.testing.assert_close(stats["opd_reward_gan_active"][0], torch.tensor(1.0))
    torch.testing.assert_close(
        stats["opd_reward_gan_generator_weight"][0],
        torch.tensor(0.75),
    )
    torch.testing.assert_close(stats["opd_reward_gan_gen_loss"][0], expected_reward)
    torch.testing.assert_close(
        stats["opd_reward_gan_gen_weighted_loss"][0],
        expected_weighted,
    )
    torch.testing.assert_close(stats["opd_reward_gan_gen_fake_logit"][0], fake_logits.sum())
    torch.testing.assert_close(loss.detach(), expected_total)
    assert fake_clean.grad is not None
    assert fake_clean.grad.abs().sum().item() > 0.0
    assert reward_adapter.weight.grad is None
    assert all(param.requires_grad for param in reward_adapter.parameters())


def test_dual_method_accepts_discriminator_loss_weight_by_exit():
    method = _tiny_dual_method(
        num_student_steps=2,
        discriminator_loss_weight_by_exit=(0.5, 1.5),
    )

    assert method.discriminator_loss_weight_by_exit == (0.5, 1.5)
    assert method._discriminator_loss_weight_for_exit(1) == 0.5
    assert method._discriminator_loss_weight_for_exit(2) == 1.5


def test_dual_method_rejects_invalid_generator_gan_loss_weight_by_exit():
    with pytest.raises(ValueError, match="generator_gan_loss_weight_by_exit"):
        _tiny_dual_method(
            num_student_steps=2,
            generator_gan_loss_weight_by_exit=(1.0,),
        )

    with pytest.raises(ValueError, match="generator_gan_loss_weight_by_exit"):
        _tiny_dual_method(
            num_student_steps=2,
            generator_gan_loss_weight_by_exit=(1.0, -0.1),
        )


def test_dual_method_rejects_invalid_discriminator_loss_weight_by_exit():
    with pytest.raises(ValueError, match="discriminator_loss_weight_by_exit"):
        _tiny_dual_method(
            num_student_steps=2,
            discriminator_loss_weight_by_exit=(1.0,),
        )

    with pytest.raises(ValueError, match="discriminator_loss_weight_by_exit"):
        _tiny_dual_method(
            num_student_steps=2,
            discriminator_loss_weight_by_exit=(1.0, -0.1),
        )


def test_dual_method_rejects_invalid_apt_r1_sigma_range():
    with pytest.raises(ValueError, match="apt_r1_weight"):
        _tiny_dual_method(apt_r1_weight=-0.1)

    with pytest.raises(ValueError, match="apt_r1_sigma_max"):
        _tiny_dual_method(apt_r1_sigma_min=0.45, apt_r1_sigma_max=0.05)

    with pytest.raises(ValueError, match="apt_r1_sigma"):
        _tiny_dual_method(apt_r1_sigma=(0.05, 0.25, 0.45))


def test_dual_method_rejects_invalid_generator_adv_loss_min():
    with pytest.raises(ValueError, match="generator_adv_loss_min"):
        _tiny_dual_method(generator_adv_loss_min=-0.01)


def test_generator_adv_loss_min_skips_gradient_when_loss_is_small():
    method = _tiny_dual_method(generator_adv_loss_min=0.2)
    pred_fake = torch.tensor([10.0], requires_grad=True)

    loss, raw_loss = method._generator_adv_loss(pred_fake, return_raw=True)

    expected_raw = torch.nn.functional.softplus(-pred_fake).mean()
    torch.testing.assert_close(raw_loss, expected_raw)
    torch.testing.assert_close(loss, expected_raw * 0.0)
    loss.backward()
    assert pred_fake.grad is not None
    torch.testing.assert_close(pred_fake.grad, torch.zeros_like(pred_fake))


def test_generator_adv_loss_min_keeps_softplus_gradient_when_loss_is_large():
    method = _tiny_dual_method(generator_adv_loss_min=0.2)
    pred_fake = torch.tensor([0.0], requires_grad=True)

    loss, raw_loss = method._generator_adv_loss(pred_fake, return_raw=True)

    expected_raw = torch.nn.functional.softplus(-pred_fake).mean()
    torch.testing.assert_close(raw_loss, expected_raw)
    torch.testing.assert_close(loss, expected_raw)

    loss.backward()
    assert pred_fake.grad is not None
    assert pred_fake.grad.item() < 0.0


def test_generator_adv_loss_applies_selected_exit_weight_after_min_clip():
    method = _tiny_dual_method(
        num_student_steps=2,
        generator_gan_loss_weight_by_exit=(0.5, 2.0),
    )
    pred_fake = torch.tensor([0.0], requires_grad=True)

    loss_exit1, raw_exit1 = method._generator_adv_loss(
        pred_fake,
        selected_exit_step=1,
        return_raw=True,
    )
    loss_exit2, raw_exit2 = method._generator_adv_loss(
        pred_fake,
        selected_exit_step=2,
        return_raw=True,
    )

    expected_raw = torch.nn.functional.softplus(-pred_fake).mean()
    torch.testing.assert_close(raw_exit1, expected_raw)
    torch.testing.assert_close(raw_exit2, expected_raw)
    torch.testing.assert_close(loss_exit1, expected_raw * 0.5)
    torch.testing.assert_close(loss_exit2, expected_raw * 2.0)
