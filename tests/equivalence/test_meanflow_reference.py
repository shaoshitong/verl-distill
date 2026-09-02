import sys
from pathlib import Path

import pytest
import torch
from torch import nn

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class _IntervalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.25))
        self.calls = []

    def forward(self, z, t, c=None, tt=None, **kwargs):
        aux_time_meta = kwargs.get("aux_time_meta", {})
        call_tag = aux_time_meta.get("call_tag")
        del c, kwargs
        self.calls.append(
            {
                "shape": tuple(z.shape),
                "t": t.detach().clone(),
                "tt": None if tt is None else tt.detach().clone(),
                "grad": bool(torch.is_grad_enabled()),
                "call_tag": call_tag,
            }
        )
        t_view = t.reshape(-1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        tt_view = (
            t_view if tt is None else tt.reshape(-1, 1, 1, 1).to(device=z.device, dtype=z.dtype)
        )
        return z * self.scale.to(dtype=z.dtype) + 0.01 * (t_view + tt_view)


class _Teacher(nn.Module):
    def __init__(self, cond_value=2.0, uncond_value=-1.0):
        super().__init__()
        self.cond_value = float(cond_value)
        self.uncond_value = float(uncond_value)
        self.calls = []

    def forward(self, z, t, c=None, tt=None, **kwargs):
        del tt, kwargs
        self.calls.append(
            {
                "shape": tuple(z.shape),
                "t": t.detach().clone(),
                "cond_batch": None if c is None else int(c[0].shape[0]),
                "grad": bool(torch.is_grad_enabled()),
            }
        )
        if c is None:
            values = torch.full(
                (z.shape[0],),
                self.cond_value,
                device=z.device,
                dtype=z.dtype,
            )
        else:
            cond_sign = c[0].reshape(c[0].shape[0], -1).sum(dim=1).to(device=z.device)
            values = torch.where(
                cond_sign < 0,
                torch.full_like(cond_sign, self.uncond_value, dtype=z.dtype),
                torch.full_like(cond_sign, self.cond_value, dtype=z.dtype),
            )
        view_shape = (values.shape[0],) + (1,) * (z.ndim - 1)
        return torch.ones_like(z) * values.reshape(view_shape)


def _cond(batch_size):
    return [
        torch.ones(batch_size, 2, 4, dtype=torch.float32),
        torch.ones(batch_size, 2, dtype=torch.float32),
    ]


def _uncond(batch_size):
    return [
        -torch.ones(batch_size, 2, 4, dtype=torch.float32),
        torch.ones(batch_size, 2, dtype=torch.float32),
    ]


def test_terminal_cfg_scale_schedule_matches_imagenet_run():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    method = ZImageMeanFlow(
        split_fd_terminal_cfg_scale_start=-0.5,
        split_fd_terminal_cfg_scale_end=0.5,
        split_fd_terminal_cfg_scale_steps=15000,
    )

    assert method.scheduled_terminal_cfg_scale(0) == pytest.approx(-0.5)
    assert method.scheduled_terminal_cfg_scale(7500) == pytest.approx(0.0)
    assert method.scheduled_terminal_cfg_scale(15000) == pytest.approx(0.5)
    assert method.scheduled_terminal_cfg_scale(999999) == pytest.approx(0.5)


def test_flow_shift_maps_sigma_grid_with_shift_three():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    method = ZImageMeanFlow(flow_shift=3.0)
    sigma = torch.tensor([0.0, 0.25, 0.5, 1.0])
    shifted = method.apply_flow_shift(sigma)

    expected = 3.0 * sigma / (1.0 + (3.0 - 1.0) * sigma)
    assert torch.allclose(shifted, expected)
    assert shifted[0].item() == pytest.approx(0.0)
    assert shifted[-1].item() == pytest.approx(1.0)


def test_method_registered_in_methodes_registry():
    from verl_distill.algorithms.registry import ALGORITHMS

    assert "meanflow" in ALGORITHMS
    method = ALGORITHMS["meanflow"](teacher_cfg_scale=2.5, flow_shift=3.0)
    assert method.teacher_cfg_scale == pytest.approx(2.5)
    assert method.flow_shift == pytest.approx(3.0)


def test_training_step_uses_tt_for_interval_calls_and_returns_stats():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(1)
    method = ZImageMeanFlow(
        fd_delta=0.005,
        teacher_cfg_scale=2.5,
        flow_shift=3.0,
        split_alpha_min=0.5,
        split_alpha_max=0.5,
        fm_lambda=1.0,
        gap_max_start=0.3,
        gap_max_end=0.3,
        rt_curriculum_steps=1,
    )
    student = _IntervalModel()
    teacher = _Teacher()
    x = torch.randn(2, 4, 8, 8)

    loss, stats = method.training_step(
        student,
        x,
        c=_cond(2),
        e=_uncond(2),
        step=0,
        teacher=teacher,
        return_loss_stats=True,
    )

    assert loss.requires_grad
    assert float(loss.detach()) >= 0.0
    for key in [
        "total",
        "loss_t",
        "loss_split",
        "loss_fm",
        "split_fd_terminal_cfg_scale",
        "anchor_r_frac",
        "split_gap_mean",
    ]:
        assert key in stats
    interval_calls = [call for call in student.calls if call["tt"] is not None]
    assert interval_calls
    assert any(
        call["t"].shape == call["tt"].shape and not torch.allclose(call["t"], call["tt"])
        for call in interval_calls
    )
    assert all(torch.all(call["t"] >= call["tt"]) for call in interval_calls)


def test_training_step_supports_batch_size_four_end_to_end():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(11)
    batch_size = 4
    method = ZImageMeanFlow(
        fd_delta=0.005,
        teacher_cfg_scale=2.5,
        flow_shift=6.0,
        split_alpha_min=0.5,
        split_alpha_max=0.5,
        fm_lambda=1.0,
        gap_max_start=0.3,
        gap_max_end=0.3,
        rt_curriculum_steps=1,
    )
    student = _IntervalModel()
    teacher = _Teacher()
    x = torch.randn(batch_size, 4, 8, 8)

    loss, stats = method.training_step(
        student,
        x,
        c=_cond(batch_size),
        e=_uncond(batch_size),
        step=0,
        teacher=teacher,
        return_loss_stats=True,
    )
    loss.backward()

    assert loss.ndim == 0
    assert torch.isfinite(loss.detach())
    assert student.scale.grad is not None
    assert torch.isfinite(student.scale.grad)
    anchor_r_frac = stats["anchor_r_frac"][0]
    assert anchor_r_frac >= 0.0
    assert anchor_r_frac <= 1.0
    assert student.calls
    assert teacher.calls
    assert all(call["shape"][0] == batch_size for call in student.calls)
    assert all(call["shape"][0] == batch_size for call in teacher.calls)
    assert all(call["t"].numel() == batch_size for call in student.calls)
    assert all(call["t"].numel() == batch_size for call in teacher.calls)
    interval_calls = [call for call in student.calls if call["tt"] is not None]
    assert interval_calls
    assert all(call["tt"].numel() == batch_size for call in interval_calls)
    assert all(call["cond_batch"] == batch_size for call in teacher.calls)


def test_discrete_rt_sampler_uses_raw_grid_before_flow_shift():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(21)
    method = ZImageMeanFlow(
        fd_delta=0.005,
        flow_shift=6.0,
        rt_discrete_grid=True,
        rt_grid_step=0.05,
        gap_min=0.005,
        gap_max_start=0.10,
        gap_max_end=0.10,
        rt_curriculum_steps=0,
    )

    r, t, r_raw, t_raw = method._sample_rt(
        batch_size=128,
        device=torch.device("cpu"),
        dtype=torch.float32,
        step=0,
    )

    raw_values = torch.cat([r_raw, t_raw])
    raw_steps = raw_values / 0.05
    torch.testing.assert_close(raw_steps, raw_steps.round(), atol=1.0e-6, rtol=0.0)
    assert torch.all(t_raw > r_raw)
    assert torch.all((t_raw - r_raw) >= 0.05 - 1.0e-6)
    assert torch.all((t_raw - r_raw) <= 0.10 + 1.0e-6)
    torch.testing.assert_close(r, method.apply_flow_shift(r_raw))
    torch.testing.assert_close(t, method.apply_flow_shift(t_raw))


def test_discrete_split_uses_grid_and_masks_one_step_gaps():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(22)
    method = ZImageMeanFlow(
        rt_discrete_grid=True,
        rt_grid_step=0.05,
        flow_shift=6.0,
    )
    r_raw = torch.tensor([0.20, 0.20, 0.70])
    t_raw = torch.tensor([0.25, 0.30, 0.85])
    r = method.apply_flow_shift(r_raw)
    t = method.apply_flow_shift(t_raw)

    s, s_raw, split_mask = method._sample_split(r, t, r_raw, t_raw)

    torch.testing.assert_close(
        s_raw / 0.05,
        (s_raw / 0.05).round(),
        atol=1.0e-6,
        rtol=0.0,
    )
    assert split_mask.tolist() == [False, True, True]
    assert torch.all(s_raw >= r_raw)
    assert torch.all(s_raw <= t_raw)
    torch.testing.assert_close(s, method.apply_flow_shift(s_raw))


def test_discrete_one_step_gaps_mask_split_loss_without_skipping_forwards():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(24)
    method = ZImageMeanFlow(
        fd_delta=0.05,
        teacher_cfg_scale=1.0,
        flow_shift=6.0,
        rt_discrete_grid=True,
        rt_grid_step=0.05,
        gap_min=0.05,
        gap_max_start=0.05,
        gap_max_end=0.05,
        rt_curriculum_steps=0,
    )
    student = _IntervalModel()
    student.accepts_aux_time_meta = True
    x = torch.randn(4, 4, 8, 8)

    loss, stats = method.training_step(
        student,
        x,
        c=_cond(4),
        e=_uncond(4),
        step=0,
        teacher=_Teacher(),
        return_loss_stats=True,
    )

    assert torch.isfinite(loss.detach())
    assert stats["split_eligible_frac"][0].item() == pytest.approx(0.0)
    assert stats["loss_split"][0].item() == pytest.approx(0.0)
    call_tags = [call["call_tag"] for call in student.calls]
    assert "meanflow_st" in call_tags
    assert "meanflow_rs" in call_tags


def test_loss_reduction_mean_scales_per_sample_losses_by_num_elements():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    x = torch.randn(2, 4, 8, 8)
    kwargs = dict(
        fd_delta=0.005,
        teacher_cfg_scale=2.5,
        flow_shift=3.0,
        split_alpha_min=0.5,
        split_alpha_max=0.5,
        gap_max_start=0.3,
        gap_max_end=0.3,
        rt_curriculum_steps=1,
    )

    torch.manual_seed(23)
    sum_loss, sum_stats = ZImageMeanFlow(
        loss_reduction="sum",
        **kwargs,
    ).training_step(
        _IntervalModel(),
        x,
        c=_cond(2),
        e=_uncond(2),
        step=0,
        teacher=_Teacher(),
        return_loss_stats=True,
    )
    torch.manual_seed(23)
    mean_loss, mean_stats = ZImageMeanFlow(
        loss_reduction="mean",
        **kwargs,
    ).training_step(
        _IntervalModel(),
        x,
        c=_cond(2),
        e=_uncond(2),
        step=0,
        teacher=_Teacher(),
        return_loss_stats=True,
    )

    num_elements = x[0].numel()
    torch.testing.assert_close(mean_loss * num_elements, sum_loss)
    torch.testing.assert_close(
        mean_stats["loss_t"][0] * num_elements,
        sum_stats["loss_t"][0],
    )
    torch.testing.assert_close(
        mean_stats["loss_split"][0] * num_elements,
        sum_stats["loss_split"][0],
    )
    torch.testing.assert_close(
        mean_stats["loss_fm"][0] * num_elements,
        sum_stats["loss_fm"][0],
    )


def test_sampling_loop_passes_current_sigma_as_main_time():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    class _Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, x, t, tt=None, **kwargs):
            del kwargs
            self.calls.append((t.detach().clone(), tt.detach().clone()))
            return torch.zeros_like(x)

    method = ZImageMeanFlow(flow_shift=1.0)
    model = _Recorder()
    initial = torch.randn(2, 4, 8, 8)

    out = method.sampling_loop(initial, model, sampling_steps=2, flow_shift=1.0)

    assert out.shape == (3, *initial.shape)
    assert len(model.calls) == 2
    torch.testing.assert_close(model.calls[0][0], torch.ones(2))
    torch.testing.assert_close(model.calls[0][1], torch.full((2,), 0.5))
    torch.testing.assert_close(model.calls[1][0], torch.full((2,), 0.5))
    torch.testing.assert_close(model.calls[1][1], torch.zeros(2))


def test_sampling_loop_keeps_solver_state_float32_for_low_precision_noise():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    class _Recorder(nn.Module):
        def forward(self, x, t, tt=None, **kwargs):
            del t, tt, kwargs
            return torch.zeros_like(x)

    method = ZImageMeanFlow(flow_shift=1.0)
    initial = torch.randn(2, 4, 8, 8).to(torch.bfloat16)

    out = method.sampling_loop(initial, _Recorder(), sampling_steps=1, flow_shift=1.0)

    assert out.dtype == torch.float32


def test_split_target_does_not_call_teacher():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    torch.manual_seed(3)
    method = ZImageMeanFlow(
        fd_delta=0.005,
        teacher_cfg_scale=2.5,
        flow_shift=3.0,
        split_alpha_min=0.5,
        split_alpha_max=0.5,
        gap_max_start=0.3,
        gap_max_end=0.3,
        rt_curriculum_steps=1,
    )
    student = _IntervalModel()
    teacher = _Teacher()
    x = torch.randn(2, 4, 8, 8)

    method.training_step(
        student,
        x,
        c=_cond(2),
        e=_uncond(2),
        step=0,
        teacher=teacher,
        return_loss_stats=False,
    )

    assert len(teacher.calls) == 4
    assert all(not call["grad"] for call in teacher.calls)


def test_teacher_cfg_scale_one_skips_unconditional_teacher_forward():
    from verl_distill.algorithms.meanflow import ZImageMeanFlow

    method = ZImageMeanFlow(teacher_cfg_scale=1.0)
    teacher = _Teacher(cond_value=2.0, uncond_value=-1.0)
    z = torch.randn(2, 4, 8, 8)
    t = torch.full((2,), 0.5)

    velocity = method._teacher_cfg_velocity(
        teacher,
        z,
        t,
        c=_cond(2),
        e=_uncond(2),
    )

    assert len(teacher.calls) == 1
    assert teacher.calls[0]["cond_batch"] == 2
    torch.testing.assert_close(velocity, torch.full_like(z, 2.0))
