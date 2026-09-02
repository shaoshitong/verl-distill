import sys
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

try:
    import diffusers

    if not hasattr(diffusers, "Flux2KleinPipeline"):
        diffusers.Flux2KleinPipeline = object
except ImportError:
    pass

from verl_distill.algorithms.dmd.dmdr import DMDR  # noqa: E402
from verl_distill.algorithms.dmd.full_model import FullModelDMD  # noqa: E402
from verl_distill.algorithms.dmd.method import StandardDMD  # noqa: E402


class _ConstantFlowModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)

    def forward(self, x_t, t, c):
        del t, c
        return torch.full_like(x_t, self.value)


class _EchoFlowModel(torch.nn.Module):
    def forward(self, x_t, t, c):
        del t, c
        return x_t


class _ConditionValueFlowModel(torch.nn.Module):
    def forward(self, x_t, t, c, **kwargs):
        del t, kwargs
        value = c[0].to(device=x_t.device, dtype=x_t.dtype).view(-1, 1, 1, 1)
        return torch.ones_like(x_t) * value


class _RecordingFlowModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.calls = []

    def forward(self, x_t, t, c, tt=None, **kwargs):
        del c, kwargs
        self.calls.append(
            {
                "t": t.detach().clone(),
                "tt": None if tt is None else tt.detach().clone(),
                "shape": tuple(x_t.shape),
            }
        )
        return torch.full_like(x_t, self.value)


class _RecordingTTOnlyFlowModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.calls = []

    def forward(self, x_t, t, c, tt=None):
        del c
        self.calls.append(
            {
                "t": t.detach().clone(),
                "tt": None if tt is None else tt.detach().clone(),
                "shape": tuple(x_t.shape),
            }
        )
        return torch.full_like(x_t, self.value)


class _RecordingAuxMetaFlowModel(torch.nn.Module):
    accepts_aux_time_meta = True

    def __init__(self, value: float):
        super().__init__()
        self.value = float(value)
        self.calls = []

    def forward(self, x_t, t, c, tt=None, aux_time_meta=None):
        del c
        self.calls.append(
            {
                "t": t.detach().clone(),
                "tt": None if tt is None else tt.detach().clone(),
                "aux_time_meta": aux_time_meta,
            }
        )
        return torch.full_like(x_t, self.value)


def test_predict_x0_from_flow_selects_explicit_fake_and_real_score_branches():
    method = StandardDMD()
    x_t = torch.zeros(1, 1, 2, 2)
    sigma = torch.tensor([0.25], dtype=torch.float32)
    cond = [torch.zeros(1, 1)]
    score_model = {
        "fake": _ConstantFlowModel(2.0),
        "real": _ConstantFlowModel(5.0),
    }

    pred_fake_x0, pred_fake_flow = method._predict_x0_from_flow(
        score_model,
        x_t,
        sigma,
        c=cond,
        use_lora=True,
    )
    pred_real_x0, pred_real_flow = method._predict_x0_from_flow(
        score_model,
        x_t,
        sigma,
        c=cond,
        use_lora=False,
    )

    torch.testing.assert_close(pred_fake_flow, torch.full_like(x_t, 2.0))
    torch.testing.assert_close(pred_real_flow, torch.full_like(x_t, 5.0))
    torch.testing.assert_close(pred_fake_x0, torch.full_like(x_t, -0.5))
    torch.testing.assert_close(pred_real_x0, torch.full_like(x_t, -1.25))


def test_predict_x0_from_flow_uses_reference_dmd_cfg_scale_semantics():
    method = StandardDMD()
    x_t = torch.zeros(1, 1, 2, 2)
    sigma = torch.tensor([0.25], dtype=torch.float32)
    cond = [torch.tensor([[2.0]], dtype=torch.float32)]
    uncond = [torch.tensor([[1.0]], dtype=torch.float32)]

    pred_x0, pred_flow = method._predict_x0_from_flow(
        _ConditionValueFlowModel(),
        x_t,
        sigma,
        c=cond,
        e=uncond,
        guidance_scale=5.5,
    )

    expected_flow = torch.full_like(x_t, 7.5)
    torch.testing.assert_close(pred_flow, expected_flow)
    torch.testing.assert_close(pred_x0, -0.25 * expected_flow)


def test_call_model_forwards_aux_time_meta_only_to_opt_in_model():
    method = StandardDMD()
    method.train_step = 17
    x_t = torch.zeros(2, 1, 2, 2)
    t = torch.tensor([0.25, 0.125], dtype=torch.float32)
    tt = torch.tensor([0.75, 0.5], dtype=torch.float32)
    cond = [torch.zeros(2, 1)]

    opt_in_model = _RecordingAuxMetaFlowModel(0.0)
    method._call_model(
        opt_in_model,
        x_t,
        t,
        cond,
        target_timestep=tt,
        target_timestep_log_tag="score_loss_fake_score",
    )

    assert len(opt_in_model.calls) == 1
    torch.testing.assert_close(opt_in_model.calls[0]["tt"], tt)
    assert opt_in_model.calls[0]["aux_time_meta"] == {
        "step": 17,
        "call_tag": "score_loss_fake_score",
    }

    tt_only_model = _RecordingTTOnlyFlowModel(0.0)
    method._call_model(
        tt_only_model,
        x_t,
        t,
        cond,
        target_timestep=tt,
        target_timestep_log_tag="score_loss_fake_score",
    )

    assert len(tt_only_model.calls) == 1
    torch.testing.assert_close(tt_only_model.calls[0]["tt"], tt)


def test_score_loss_accepts_explicit_fake_and_real_score_models():
    torch.manual_seed(0)
    method = StandardDMD()
    x_real = torch.randn(2, 1, 4, 4)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(1.0)
    score_model = {
        "fake": _ConstantFlowModel(2.0),
        "real": _ConstantFlowModel(3.0),
    }

    loss, stats, aux = method.score_loss(
        generator_model=generator_model,
        score_model=score_model,
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss_score" in stats
    assert "score_sigma" in aux
    assert aux["score_sigma"].shape[0] == x_real.shape[0]


def test_fullmodel_score_loss_passes_generator_sigma_as_fake_score_tt():
    method = FullModelDMD(fake_score_use_generator_timestep=True)
    x_real = torch.zeros(2, 1, 2, 2)
    x_fake = torch.ones_like(x_real)
    gen_sigma = torch.tensor([0.75, 0.5], dtype=torch.float32)
    score_sigma = torch.tensor([0.25, 0.125], dtype=torch.float32)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(0.0)
    fake_score = _RecordingFlowModel(0.0)
    real_score = _RecordingFlowModel(0.0)

    def _fake_generate_one_step_latents(generator_model, x_real, c, **kwargs):
        return (
            x_fake.clone(),
            {
                "gen_input_sigma": gen_sigma.clone(),
                "gen_backward_simulation": torch.zeros(2, dtype=torch.float32),
            },
        )

    method.generate_one_step_latents = _fake_generate_one_step_latents
    method._sample_sigmas = lambda batch_size, device: score_sigma.to(device=device)

    loss, stats, aux = method.score_loss(
        generator_model=generator_model,
        score_model={"fake": fake_score, "real": real_score},
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert torch.isfinite(loss)
    assert len(fake_score.calls) == 1
    torch.testing.assert_close(fake_score.calls[0]["t"], score_sigma)
    torch.testing.assert_close(fake_score.calls[0]["tt"], gen_sigma)
    assert real_score.calls == []
    torch.testing.assert_close(
        stats["score_loss_fake_score_t"][0],
        score_sigma.sum().to(dtype=stats["score_loss_fake_score_t"][0].dtype),
    )
    torch.testing.assert_close(
        stats["score_loss_fake_score_tt"][0],
        gen_sigma.sum().to(dtype=stats["score_loss_fake_score_tt"][0].dtype),
    )
    torch.testing.assert_close(
        stats["score_loss_fake_score_has_tt"][0],
        torch.tensor(2.0, dtype=stats["score_loss_fake_score_has_tt"][0].dtype),
    )
    torch.testing.assert_close(aux["score_fake_score_tt"], gen_sigma)
    torch.testing.assert_close(aux["score_fake_score_t"], score_sigma)
    torch.testing.assert_close(aux["score_fake_score_has_tt"], torch.ones_like(gen_sigma))
    assert "score_loss_fake_score_t" not in aux
    assert "score_loss_fake_score_tt" not in aux
    assert "score_loss_fake_score_has_tt" not in aux


def test_fullmodel_score_loss_passes_tt_without_aux_time_kwargs():
    method = FullModelDMD(fake_score_use_generator_timestep=True)
    x_real = torch.zeros(2, 1, 2, 2)
    x_fake = torch.ones_like(x_real)
    gen_sigma = torch.tensor([0.75, 0.5], dtype=torch.float32)
    score_sigma = torch.tensor([0.25, 0.125], dtype=torch.float32)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    fake_score = _RecordingTTOnlyFlowModel(0.0)

    def _fake_generate_one_step_latents(generator_model, x_real, c, **kwargs):
        return (
            x_fake.clone(),
            {
                "gen_input_sigma": gen_sigma.clone(),
                "gen_backward_simulation": torch.zeros(2, dtype=torch.float32),
            },
        )

    method.generate_one_step_latents = _fake_generate_one_step_latents
    method._sample_sigmas = lambda batch_size, device: score_sigma.to(device=device)

    loss, _, aux = method.score_loss(
        generator_model=_ConstantFlowModel(0.0),
        score_model={"fake": fake_score, "real": _RecordingFlowModel(0.0)},
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert torch.isfinite(loss)
    assert len(fake_score.calls) == 1
    torch.testing.assert_close(fake_score.calls[0]["t"], score_sigma)
    torch.testing.assert_close(fake_score.calls[0]["tt"], gen_sigma)
    torch.testing.assert_close(aux["score_fake_score_tt"], gen_sigma)
    assert "score_loss_fake_score_tt" not in aux


def test_fullmodel_generator_dmd_loss_passes_generator_sigma_only_to_fake_score():
    method = FullModelDMD(fake_score_use_generator_timestep=True)
    x_real = torch.zeros(2, 1, 2, 2)
    x_fake = torch.ones_like(x_real)
    gen_sigma = torch.tensor([0.75, 0.5], dtype=torch.float32)
    dm_sigma = torch.tensor([0.25, 0.125], dtype=torch.float32)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(0.0)
    fake_score = _RecordingFlowModel(0.0)
    real_score = _RecordingFlowModel(0.0)

    def _fake_generate_one_step_latents(generator_model, x_real, c, **kwargs):
        return (
            x_fake.clone(),
            {
                "gen_input_sigma": gen_sigma.clone(),
                "gen_backward_simulation": torch.zeros(2, dtype=torch.float32),
            },
        )

    method.generate_one_step_latents = _fake_generate_one_step_latents
    method._sample_sigmas = lambda batch_size, device: dm_sigma.to(device=device)

    loss, stats, aux = method.generator_dmd_loss(
        generator_model=generator_model,
        score_model={"fake": fake_score, "real": real_score},
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert torch.isfinite(loss)
    assert len(fake_score.calls) == 1
    assert len(real_score.calls) == 2
    torch.testing.assert_close(fake_score.calls[0]["t"], dm_sigma)
    torch.testing.assert_close(fake_score.calls[0]["tt"], gen_sigma)
    for real_call in real_score.calls:
        torch.testing.assert_close(real_call["t"], dm_sigma)
        assert real_call["tt"] is None
    torch.testing.assert_close(
        stats["dmd_fake_score_t"][0],
        dm_sigma.sum().to(dtype=stats["dmd_fake_score_t"][0].dtype),
    )
    torch.testing.assert_close(
        stats["dmd_fake_score_tt"][0],
        gen_sigma.sum().to(dtype=stats["dmd_fake_score_tt"][0].dtype),
    )
    torch.testing.assert_close(
        stats["dmd_fake_score_has_tt"][0],
        torch.tensor(2.0, dtype=stats["dmd_fake_score_has_tt"][0].dtype),
    )
    torch.testing.assert_close(aux["dmd_fake_score_tt"], gen_sigma)


def test_fullmodel_generator_timestep_flag_defaults_to_no_fake_score_tt():
    method = FullModelDMD()
    x_real = torch.zeros(1, 1, 2, 2)
    x_fake = torch.ones_like(x_real)
    gen_sigma = torch.tensor([0.75], dtype=torch.float32)
    score_sigma = torch.tensor([0.25], dtype=torch.float32)
    cond = [torch.zeros(1, 1)]
    uncond = [torch.zeros(1, 1)]
    fake_score = _RecordingFlowModel(0.0)
    real_score = _RecordingFlowModel(0.0)

    def _fake_generate_one_step_latents(generator_model, x_real, c, **kwargs):
        return (
            x_fake.clone(),
            {
                "gen_input_sigma": gen_sigma.clone(),
                "gen_backward_simulation": torch.zeros(1, dtype=torch.float32),
            },
        )

    method.generate_one_step_latents = _fake_generate_one_step_latents
    method._sample_sigmas = lambda batch_size, device: score_sigma.to(device=device)

    method.score_loss(
        generator_model=_ConstantFlowModel(0.0),
        score_model={"fake": fake_score, "real": real_score},
        x_real=x_real,
        c=cond,
        e=uncond,
    )

    assert len(fake_score.calls) == 1
    assert fake_score.calls[0]["tt"] is None

    fake_score.calls.clear()
    real_score.calls.clear()

    method.generator_dmd_loss(
        generator_model=_ConstantFlowModel(0.0),
        score_model={"fake": fake_score, "real": real_score},
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert len(fake_score.calls) == 1
    assert fake_score.calls[-1]["tt"] is None


def test_score_loss_can_use_flow_target_mse():
    method = StandardDMD(score_loss_target="flow")
    x_real = torch.zeros(2, 1, 2, 2)
    x_fake = torch.ones_like(x_real)
    sigma = torch.tensor([0.5, 0.25], dtype=torch.float32)
    noise = torch.full_like(x_real, 3.0)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(0.0)
    score_model = {
        "fake": _ConstantFlowModel(0.0),
        "real": _ConstantFlowModel(0.0),
    }

    method.generate_one_step_latents = lambda generator_model, x_real, c: (
        x_fake.clone(),
        {"gen_backward_simulation": torch.zeros(x_fake.shape[0], dtype=torch.float32)},
    )
    method._sample_sigmas = lambda batch_size, device: sigma.to(device=device)

    with patch("verl_distill.algorithms.dmd.method.torch.randn_like", return_value=noise):
        loss, stats, aux = method.score_loss(
            generator_model=generator_model,
            score_model=score_model,
            x_real=x_real,
            c=cond,
            e=uncond,
            return_log_tensors=True,
        )

    expected_target = noise - x_fake
    expected_loss = expected_target.square().flatten(1).mean(dim=1).mean()
    torch.testing.assert_close(loss, expected_loss)
    assert torch.isfinite(stats["score_pred_err"][0])
    assert "score_sigma" in aux


def test_ode_warmup_generator_sigmas_use_configured_biased_distribution():
    method = StandardDMD(
        num_denoising_step=4,
        timestep_shift=1.0,
        ode_warmup_generator_sigma_probs=[0.8, 0.16, 0.032, 0.008],
    )
    captured = {}

    def fake_multinomial(probs, num_samples, replacement):
        captured["probs"] = probs.detach().cpu()
        captured["num_samples"] = num_samples
        captured["replacement"] = replacement
        return torch.tensor([0, 1, 2, 3], device=probs.device)

    with patch("verl_distill.algorithms.dmd.method.torch.multinomial", fake_multinomial):
        sigmas = method._sample_ode_warmup_generator_sigmas(4, torch.device("cpu"))

    torch.testing.assert_close(
        sigmas,
        torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float32),
    )
    torch.testing.assert_close(
        captured["probs"],
        torch.tensor([0.8, 0.16, 0.032, 0.008], dtype=torch.float32),
    )
    assert captured["num_samples"] == 4
    assert captured["replacement"] is True


def test_regular_generator_sigmas_stay_uniform_when_ode_warmup_probs_are_set():
    method = StandardDMD(
        num_denoising_step=4,
        timestep_shift=1.0,
        ode_warmup_generator_sigma_probs=[0.8, 0.16, 0.032, 0.008],
    )

    with patch(
        "verl_distill.algorithms.dmd.method.torch.randint",
        return_value=torch.tensor([0, 1, 2, 3]),
    ):
        sigmas = method._sample_generator_sigmas(4, torch.device("cpu"))

    torch.testing.assert_close(
        sigmas,
        torch.tensor([1.0, 0.75, 0.5, 0.25], dtype=torch.float32),
    )


def test_ode_warmup_rank_weight_strength_is_configurable_and_validated():
    method = StandardDMD(ode_warmup_rank_weight_strength=2.0)

    assert method.ode_warmup_rank_weight_strength == 2.0
    with pytest.raises(ValueError, match="ode_warmup_rank_weight_strength"):
        StandardDMD(ode_warmup_rank_weight_strength=-1.0)


def test_data_free_requires_backward_simulation_and_zero_real_image_losses():
    with pytest.raises(ValueError, match="backward_simulation=True"):
        StandardDMD(data_free=True, backward_simulation=False)

    with pytest.raises(ValueError, match="gt_grad_loss_weight == 0"):
        StandardDMD(
            data_free=True,
            backward_simulation=True,
            gt_grad_loss_weight=0.1,
        )

    with pytest.raises(ValueError, match="generator_recon_weight == 0"):
        StandardDMD(
            data_free=True,
            backward_simulation=True,
            gt_grad_loss_weight=0.0,
            generator_recon_weight=0.1,
        )


def test_score_loss_data_free_accepts_latent_shape_without_x_real():
    torch.manual_seed(0)
    method = StandardDMD(
        data_free=True,
        backward_simulation=True,
        gt_grad_loss_weight=0.0,
        generator_recon_weight=0.0,
    )
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(1.0)
    score_model = {
        "fake": _ConstantFlowModel(2.0),
        "real": _ConstantFlowModel(3.0),
    }

    loss, stats, aux = method.score_loss(
        generator_model=generator_model,
        score_model=score_model,
        x_real=None,
        latent_shape=(2, 1, 4, 4),
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss_score" in stats
    assert aux["score_sigma"].shape[0] == 2


def test_generator_loss_data_free_accepts_latent_shape_without_x_real():
    torch.manual_seed(0)
    method = StandardDMD(
        data_free=True,
        backward_simulation=True,
        gt_grad_loss_weight=0.0,
        generator_recon_weight=0.0,
    )
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(1.0)
    score_model = {
        "fake": _ConstantFlowModel(2.0),
        "real": _ConstantFlowModel(3.0),
    }

    loss, stats, aux = method.generator_loss(
        generator_model=generator_model,
        score_model=score_model,
        x_real=None,
        latent_shape=(2, 1, 4, 4),
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert "loss_gen" in stats
    assert aux["x_fake_live"].shape == (2, 1, 4, 4)


def test_generate_one_step_latents_uses_backward_simulation_inputs_when_enabled():
    method = StandardDMD(
        num_denoising_step=4,
        timestep_shift=1.0,
        backward_simulation=True,
    )
    x_real = torch.zeros(2, 1, 2, 2)
    cond = [torch.zeros(2, 1)]
    generator_model = _EchoFlowModel()

    method._sample_generator_step_indices = lambda batch_size, device: torch.tensor(
        [0, 2],
        device=device,
        dtype=torch.long,
    )
    method._simulate_backward_generator_inputs = lambda **kwargs: torch.full(
        kwargs["x_shape"],
        7.0,
        device=kwargs["step_indices"].device,
        dtype=torch.float32,
    )

    x_fake, meta = method.generate_one_step_latents(
        generator_model=generator_model,
        x_real=x_real,
        c=cond,
    )

    expected = torch.tensor([0.0, 3.5], dtype=torch.float32).view(2, 1, 1, 1).expand_as(x_fake)
    torch.testing.assert_close(x_fake, expected)
    torch.testing.assert_close(
        meta["gen_input_sigma"], torch.tensor([1.0, 0.5], dtype=torch.float32)
    )
    torch.testing.assert_close(meta["gen_backward_simulation"], torch.ones(2, dtype=torch.float32))
    torch.testing.assert_close(meta["gen_step_index"], torch.tensor([0, 2], dtype=torch.long))


def test_generate_one_step_latents_falls_back_to_override_path_with_backward_simulation_enabled():
    method = StandardDMD(
        num_denoising_step=4,
        timestep_shift=1.0,
        backward_simulation=True,
    )
    x_real = torch.zeros(1, 1, 2, 2)
    cond = [torch.zeros(1, 1)]
    generator_model = _EchoFlowModel()

    def _unexpected_simulation(**kwargs):
        raise AssertionError("backward simulation should not run when overrides are provided")

    method._simulate_backward_generator_inputs = _unexpected_simulation

    x_fake, meta = method.generate_one_step_latents(
        generator_model=generator_model,
        x_real=x_real,
        c=cond,
        sigma_override=torch.tensor([0.5], dtype=torch.float32),
        noise_override=torch.ones_like(x_real),
    )

    torch.testing.assert_close(x_fake, torch.full_like(x_fake, 0.25))
    torch.testing.assert_close(meta["gen_backward_simulation"], torch.zeros(1, dtype=torch.float32))


def test_backward_simulation_samples_one_shared_step_per_batch():
    torch.manual_seed(0)
    method = StandardDMD(
        num_denoising_step=4,
        backward_simulation=True,
    )

    step_indices = method._sample_generator_step_indices(batch_size=8, device=torch.device("cpu"))

    assert step_indices.shape == (8,)
    assert torch.unique(step_indices).numel() == 1
    assert int(step_indices[0].item()) in {0, 1, 2, 3}


def test_standard_dmd_accepts_warmup_and_legacy_rcgm_fields():
    method = StandardDMD(
        warmup_type="rcgm",
        warmup_iterations=3,
        rcgm_warmup_loss_weight=1.0,
        rcgm_warmup_teacher_steps=4,
        rcgm_loss_weight=0.25,
        rcgm_sigma_eps=0.02,
        rcgm_teacher_steps=2,
    )

    assert method.is_rcgm_warmup_step(1)
    assert method.is_rcgm_warmup_step(3)
    assert not method.is_rcgm_warmup_step(4)
    assert method.rcgm_loss_weight == 0.25
    assert method.rcgm_sigma_eps == 0.02
    assert method.rcgm_teacher_steps == 2


def test_standard_dmd_ode_warmup_updates_generator_every_step():
    method = StandardDMD(
        warmup_type="ode_pair",
        warmup_iterations=2,
        ode_warmup_loss_weight=1.0,
        dfake_gen_update_ratio=5,
    )

    assert method.should_update_generator(1)
    assert method.should_update_generator(2)
    assert not method.should_update_generator(3)
    assert method.should_update_generator(5)


def test_standard_dmd_warmup_generator_loss_aux_includes_dm_sigma():
    x_real = torch.zeros(2, 1, 2, 2)
    initial_noise = torch.ones_like(x_real)
    cond = [torch.zeros(2, 1)]
    uncond = [torch.zeros(2, 1)]
    generator_model = _ConstantFlowModel(1.0)
    score_model = {
        "fake": _ConstantFlowModel(2.0),
        "real": _ConstantFlowModel(3.0),
    }

    ode_method = StandardDMD(
        warmup_type="ode_pair",
        warmup_iterations=1,
        ode_warmup_loss_weight=1.0,
        timestep_shift=1.0,
    )
    ode_method.set_train_step(1)
    _, _, ode_aux = ode_method.generator_loss(
        generator_model=generator_model,
        score_model=score_model,
        x_real=x_real,
        initial_noise=initial_noise,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert "dm_sigma" in ode_aux
    torch.testing.assert_close(ode_aux["dm_sigma"], ode_aux["gen_input_sigma"])

    rcgm_method = StandardDMD(
        warmup_type="rcgm",
        warmup_iterations=1,
        rcgm_warmup_loss_weight=1.0,
        timestep_shift=1.0,
    )
    rcgm_method.set_train_step(1)
    _, _, rcgm_aux = rcgm_method.generator_loss(
        generator_model=generator_model,
        score_model=score_model,
        x_real=x_real,
        c=cond,
        e=uncond,
        return_log_tensors=True,
    )

    assert "dm_sigma" in rcgm_aux
    torch.testing.assert_close(rcgm_aux["dm_sigma"], rcgm_aux["rcgm_sigma_t"])


def test_dmdr_set_train_step_updates_standard_dmd_warmup_state():
    method = DMDR(
        warmup_type="ode_pair",
        warmup_iterations=1,
        ode_warmup_loss_weight=1.0,
    )

    method.set_train_step(1)

    assert method.is_ode_pair_warmup_step()
