import pytest
import torch

from verl_distill.algorithms.dmd.method import StandardDMD


def build_method(**overrides):
    params = dict(
        generator_objective="teacher_feature_ste",
        score_objective="teacher_feature_mse",
        teacher_feature_layers=[5, 15, 25],
        teacher_feature_include_latent=True,
        teacher_feature_include_last=False,
        teacher_feature_normalize=True,
    )
    params.update(overrides)
    return StandardDMD(**params)


def test_representation_keys_follow_teacher_order():
    method = build_method()
    assert method.teacher_feature_representation_keys() == (
        "latent",
        "layer_5",
        "layer_15",
        "layer_25",
    )
    assert build_method(
        teacher_feature_include_latent=False, teacher_feature_include_last=True
    ).teacher_feature_representation_keys() == ("layer_5", "layer_15", "layer_25", "pre_projector")


def test_balance_weights_enforce_target_gradient_ratio():
    method = build_method(teacher_feature_grad_balance={"ratio": 2.0})
    squared = {"latent": 1.0, "layer_5": 4.0, "layer_15": 64.0, "layer_25": 256.0}
    weights, norms = method.teacher_feature_balance_weights(squared, role="generator")
    assert norms == {"latent": 1.0, "layer_5": 2.0, "layer_15": 8.0, "layer_25": 16.0}
    assert weights["latent"] == 1.0
    contributions = {key: weights[key] * norms[key] for key in norms}
    assert contributions["latent"] == pytest.approx(1.0)
    assert contributions["layer_5"] == pytest.approx(0.5)
    assert contributions["layer_15"] == pytest.approx(0.25)
    assert contributions["layer_25"] == pytest.approx(0.125)


def test_balance_weights_clamp_tiny_norms_and_respect_anchor():
    method = build_method(
        teacher_feature_layers=[5, 15],
        teacher_feature_grad_balance={"ratio": 4.0, "anchor": "layer_5", "max_weight": 8.0},
    )
    squared = {"latent": 100.0, "layer_5": 4.0, "layer_15": 0.0}
    weights, norms = method.teacher_feature_balance_weights(squared, role="generator")
    assert weights["layer_5"] == 1.0
    # latent precedes the anchor, so its target norm is ratio ** 1 of the anchor norm.
    assert contributions(weights, norms)["latent"] == pytest.approx(8.0)
    # Zero-variance representations would otherwise explode; max_weight caps them.
    assert weights["layer_15"] == 8.0


def contributions(weights, norms):
    return {key: weights[key] * norms[key] for key in norms}


def test_balance_disabled_by_default_and_by_flag():
    assert build_method().teacher_feature_grad_balance is None
    assert (
        build_method(teacher_feature_grad_balance={"enabled": False}).teacher_feature_grad_balance
        is None
    )
    method = build_method(teacher_feature_grad_balance={"score": False})
    assert method.teacher_feature_balance_for("generator") is not None
    assert method.teacher_feature_balance_for("score") is None


def test_balance_config_validation_rejects_bad_configs():
    with pytest.raises(ValueError, match="Unknown teacher_feature_grad_balance keys"):
        build_method(teacher_feature_grad_balance={"ration": 2.0})
    with pytest.raises(ValueError, match="ratio must be finite and positive"):
        build_method(teacher_feature_grad_balance={"ratio": 0.0})
    with pytest.raises(ValueError, match="anchor must be 'first'"):
        build_method(teacher_feature_grad_balance={"anchor": "layer_7"})
    with pytest.raises(ValueError, match="measure_every_n_updates"):
        build_method(teacher_feature_grad_balance={"measure_every_n_updates": 0})
    with pytest.raises(ValueError, match="probe_accumulation"):
        build_method(teacher_feature_grad_balance={"probe_accumulation": "middle"})
    with pytest.raises(ValueError, match="at least two teacher representations"):
        build_method(
            teacher_feature_layers=[5],
            teacher_feature_include_latent=False,
            teacher_feature_grad_balance={},
        )
    with pytest.raises(ValueError, match="generator_objective=teacher_feature_ste"):
        build_method(generator_objective="dmd_mse", teacher_feature_grad_balance={})
    with pytest.raises(ValueError, match="score_objective=teacher_feature_mse"):
        build_method(score_objective="mse", teacher_feature_grad_balance={})


def test_balance_requires_norms_for_every_representation():
    method = build_method(teacher_feature_grad_balance={})
    with pytest.raises(ValueError, match="missing norms"):
        method.teacher_feature_balance_weights({"latent": 1.0}, role="generator")
    with pytest.raises(RuntimeError, match="not enabled"):
        build_method().teacher_feature_balance_weights(
            {"latent": 1.0, "layer_5": 1.0}, role="generator"
        )


def test_use_teacher_feature_weights_restores_original_mapping():
    method = build_method(teacher_feature_grad_balance={})
    original_generator = dict(method.generator_teacher_feature_weights)
    original_score = dict(method.score_teacher_feature_weights)
    with method.use_teacher_feature_weights("generator", {"latent": 1.0}):
        assert method.generator_teacher_feature_weights == {"latent": 1.0}
        assert method.score_teacher_feature_weights == original_score
    assert method.generator_teacher_feature_weights == original_generator
    method.set_teacher_feature_weights("score", {"layer_5": 0.25})
    assert method.score_teacher_feature_weights == {"layer_5": 0.25}
    with pytest.raises(ValueError, match="role"):
        method.set_teacher_feature_weights("discriminator", {"latent": 1.0})


def test_generator_feature_loss_types_validate_and_default_to_ste():
    method = build_method()
    assert method.generator_teacher_feature_loss_type == {
        "latent": "ste",
        "layer_5": "ste",
        "layer_15": "ste",
        "layer_25": "ste",
    }
    method = build_method(
        generator_teacher_feature_loss_type={
            "latent": "ste",
            "layer_5": "dmd_mse",
            "layer_15": "dmd_mse",
            "layer_25": "mse",
        }
    )
    assert method.generator_teacher_feature_loss_type["layer_25"] == "mse"
    assert method.generator_teacher_feature_loss_type["latent"] == "ste"
    assert build_method(
        generator_teacher_feature_loss_type="dmd_mse"
    ).generator_teacher_feature_loss_type == {
        k: "dmd_mse" for k in ("latent", "layer_5", "layer_15", "layer_25")
    }
    with pytest.raises(ValueError, match="must be one of"):
        build_method(generator_teacher_feature_loss_type="full")
    with pytest.raises(ValueError, match="Unknown generator_teacher_feature_loss_type"):
        build_method(generator_teacher_feature_loss_type={"layer_9": "mse"})
    with pytest.raises(ValueError, match="generator_objective=teacher_feature_ste"):
        build_method(generator_objective="dmd_mse", generator_teacher_feature_loss_type="mse")


def test_dmd_mse_without_normalize_matches_ste_gradient_direction():
    torch.manual_seed(0)
    live = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    real = torch.randn(2, 3, 4, dtype=torch.float64)
    fake = torch.randn(2, 3, 4, dtype=torch.float64)
    keys = {"layer_5": live}
    ste = StandardDMD._teacher_feature_ste_losses(
        keys, {"layer_5": real}, {"layer_5": fake}, normalize=False
    )["layer_5"]
    ste_grad = torch.autograd.grad(ste, live, retain_graph=True)[0]
    live.grad = None
    dmd = StandardDMD._teacher_feature_dmd_mse_losses(
        keys, {"layer_5": real}, {"layer_5": fake}, normalize=False
    )["layer_5"]
    dmd_grad = torch.autograd.grad(dmd, live, retain_graph=True)[0]
    # Both are exactly 2*(f-r)/N: the STE surrogate and the honest differentiable DMD loss agree.
    torch.testing.assert_close(ste_grad, dmd_grad)
    torch.testing.assert_close(ste_grad, 2 * (fake - real) / real.numel())
    # The honest MSE arm is a genuinely different objective (pull to real, no repulsion).
    live.grad = None
    mse = StandardDMD._teacher_feature_mse_losses(keys, {"layer_5": real})["layer_5"]
    mse_grad = torch.autograd.grad(mse, live)[0]
    assert not torch.allclose(mse_grad, ste_grad)
    torch.testing.assert_close(mse_grad, 2 * (live.detach() - real) / real.numel())


def test_dmd_mse_normalize_matches_ste_scaled_gradient():
    torch.manual_seed(1)
    live = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    real = torch.randn(2, 3, 4, dtype=torch.float64)
    fake = torch.randn(2, 3, 4, dtype=torch.float64)
    keys = {"layer_15": live}
    ste = StandardDMD._teacher_feature_ste_losses(
        keys, {"layer_15": real}, {"layer_15": fake}, normalize=True, eps=1e-6
    )["layer_15"]
    ste_grad = torch.autograd.grad(ste, live, retain_graph=True)[0]
    dmd = StandardDMD._teacher_feature_dmd_mse_losses(
        keys, {"layer_15": real}, {"layer_15": fake}, normalize=True, eps=1e-6
    )["layer_15"]
    dmd_grad = torch.autograd.grad(dmd, live)[0]
    torch.testing.assert_close(ste_grad, dmd_grad)


def test_mixed_loss_mapping_keeps_latent_ste_and_switches_features_to_mse():
    method = build_method(
        generator_teacher_feature_loss_type={
            "latent": "ste",
            "layer_5": "mse",
            "layer_15": "mse",
            "layer_25": "mse",
        }
    )
    live = {
        "latent": torch.randn(2, 4, 3, 3, dtype=torch.float64, requires_grad=True),
        "layer_5": torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True),
        "layer_15": torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True),
        "layer_25": torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True),
    }
    real = {k: v.detach().clone() + 0.1 for k, v in live.items()}
    fake = {k: v.detach().clone() - 0.2 for k, v in live.items()}
    losses, diagnostics = method._generator_teacher_feature_losses(live, real, fake)
    assert set(losses) == set(live)
    # feature layers now carry a genuine MSE value, the latent keeps the STE surrogate value
    torch.testing.assert_close(
        losses["layer_15"],
        StandardDMD._teacher_feature_mse_losses(
            {"layer_15": live["layer_15"]}, {"layer_15": real["layer_15"]}
        )["layer_15"],
    )
    for key in ("layer_5", "layer_15", "layer_25"):
        grad = torch.autograd.grad(losses[key], live[key], retain_graph=True)[0]
        torch.testing.assert_close(grad, 2 * (live[key].detach() - real[key]) / real[key].numel())
    # the STE layer still reports the DMD normalization diagnostics
    assert "teacher_feature_direction_rms_latent" in diagnostics
    assert "teacher_feature_direction_rms_layer_15" not in diagnostics


def test_full_bwd_pair_loss_differentiates_both_sides_and_skips_h_live():
    theta = torch.randn(4, dtype=torch.float64, requires_grad=True)
    base = torch.randn(2, 3, 4, dtype=torch.float64)
    real = base + 0.3 * theta.sum()
    fake = base - 0.7 * theta.sum()
    losses, diagnostics = StandardDMD._teacher_feature_pair_losses(
        {"layer_15": real}, {"layer_15": fake}, return_stats=True
    )
    grad = torch.autograd.grad(losses["layer_15"], theta, retain_graph=True)[0]
    expected = ((real - fake).square()).mean()
    expected_grad = torch.autograd.grad(expected, theta, retain_graph=True)[0]
    torch.testing.assert_close(losses["layer_15"], expected)
    torch.testing.assert_close(grad, expected_grad)
    # both sides are differentiable, so the gradient is genuinely non-zero
    assert grad.abs().sum() > 0
    # and the expression is symmetric in real/fake
    swapped, _ = StandardDMD._teacher_feature_pair_losses(
        {"layer_15": fake}, {"layer_15": real}, return_stats=True
    )
    torch.testing.assert_close(losses["layer_15"], swapped["layer_15"])
    assert "teacher_feature_pair_rms_layer_15" in diagnostics


def test_full_bwd_config_skips_the_live_representation():
    method = build_method(generator_teacher_feature_loss_type="full_bwd")
    assert method.generator_teacher_feature_needs_live_query() is True
    assert method.generator_teacher_feature_needs_live() is False
    live = {
        "latent": torch.randn(2, 4, 3, 3, dtype=torch.float64),
        "layer_5": torch.randn(2, 3, 4, dtype=torch.float64),
        "layer_15": torch.randn(2, 3, 4, dtype=torch.float64),
        "layer_25": torch.randn(2, 3, 4, dtype=torch.float64),
    }
    real = {k: v + 0.1 for k, v in live.items()}
    fake = {k: v - 0.2 for k, v in live.items()}
    losses, _ = method._generator_teacher_feature_losses({}, real, fake)
    assert set(losses) == set(live)
    mixed = build_method(
        generator_teacher_feature_loss_type={"latent": "ste", "layer_5": "full_bwd"}
    )
    assert mixed.generator_teacher_feature_needs_live() is True
    assert mixed.generator_teacher_feature_needs_live_query() is True
    with pytest.raises(ValueError, match="requires the live representation"):
        mixed._generator_teacher_feature_losses({}, real, fake)


def test_target_pair_ste_value_is_the_target_distance_and_gradient_rides_both_points():
    torch.manual_seed(3)
    x_real = torch.randn(2, 3, 4, dtype=torch.float64)
    x_fake = torch.randn(2, 3, 4, dtype=torch.float64)
    x_hat = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
    proxy = x_hat - x_hat.detach()
    real_proxy = x_real + proxy
    fake_proxy = x_fake + proxy

    # value: identical to the detached target distance because the live term cancels
    losses, diagnostics = StandardDMD._teacher_feature_target_pair_losses(
        {"layer_5": real_proxy}, {"layer_5": fake_proxy}, return_stats=True
    )
    torch.testing.assert_close(losses["layer_5"], (x_fake - x_real).square().mean())
    assert "teacher_feature_target_pair_rms_layer_5" in diagnostics

    # linear "teacher" (same Jacobian at both points) -> the two branches cancel exactly
    linear_grad = torch.autograd.grad(losses["layer_5"], x_hat, retain_graph=True)[0]
    torch.testing.assert_close(linear_grad, torch.zeros_like(linear_grad))

    # nonlinear teacher -> the straight-through term injects a genuine gradient, and it equals the
    # gradient of the same expression written out directly
    squared = StandardDMD._teacher_feature_target_pair_losses(
        {"layer_5": real_proxy**3}, {"layer_5": fake_proxy**3}
    )["layer_5"]
    grad = torch.autograd.grad(squared, x_hat, retain_graph=True)[0]
    assert grad.abs().sum() > 0
    reference = ((fake_proxy**3) - (real_proxy**3)).square().mean()
    reference_grad = torch.autograd.grad(reference, x_hat)[0]
    torch.testing.assert_close(grad, reference_grad)
