import torch
from torch import nn

from verl_distill.algorithms.dmd import StandardDMD


class RecordingScore(nn.Module):
    def __init__(self, slope):
        super().__init__()
        self.slope = nn.Parameter(torch.tensor(slope))
        self.calls = []

    def forward(self, x, t, c=None):
        self.calls.append((x.detach().clone(), t.detach().clone(), torch.is_grad_enabled()))
        return self.slope * x


class RecordingDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(2.0))
        self.inputs = []

    def forward(self, x, return_features=False):
        self.inputs.append(x.detach().clone())
        features = self.weight * x.flatten(2).transpose(1, 2)
        logits = features.flatten(1).mean(1)
        return {"logits": logits, "features": features} if return_features else logits


def setup_pair(monkeypatch):
    method = StandardDMD(
        generator_objective="gan",
        gan_discriminator_noise_sigma=0,
        min_step_percent=0.5,
        max_step_percent=0.5,
    )
    sample = nn.Parameter(torch.full((2, 1, 2, 2), 0.3))
    meta = dict(gen_input_sigma=torch.ones(2), gen_backward_simulation=torch.zeros(2))
    monkeypatch.setattr(method, "generate_one_step_latents", lambda *a, **kw: (sample, meta))
    score = {"fake": RecordingScore(0.7), "real": RecordingScore(-0.4)}
    disc = RecordingDiscriminator()
    kwargs = dict(
        generator_model=nn.Identity(),
        score_model=score,
        discriminator_model=disc,
        x_real=torch.full_like(sample, 99),
        c=[],
        e=None,
    )
    return method, sample, score, disc, kwargs


def test_discriminator_uses_paired_score_x0_and_only_updates_d(monkeypatch):
    method, sample, score, disc, kwargs = setup_pair(monkeypatch)
    loss, _, aux = method.discriminator_loss(**kwargs, return_debug_tensors=True)
    fake_query, fake_t, fake_grad_enabled = score["fake"].calls[0]
    real_query, real_t, real_grad_enabled = score["real"].calls[0]
    assert torch.equal(fake_query, real_query)
    assert torch.equal(fake_t, real_t)
    assert not fake_grad_enabled and not real_grad_enabled
    torch.testing.assert_close(disc.inputs[0], fake_query * (1 - 0.5 * 0.7))
    torch.testing.assert_close(disc.inputs[1], real_query * (1 + 0.5 * 0.4))
    torch.testing.assert_close(aux["gan_disc_x_fake"], disc.inputs[0])
    loss.backward()
    assert disc.weight.grad is not None and disc.weight.grad.abs() > 0
    assert sample.grad is None
    assert all(branch.slope.grad is None for branch in score.values())


def test_generator_uses_identity_ste_without_score_jacobian(monkeypatch):
    method, sample, score, disc, kwargs = setup_pair(monkeypatch)
    disc.requires_grad_(False)
    loss, _, aux = method.generator_loss(**kwargs, return_debug_tensors=True)
    expected = aux["pred_fake_x0"].clone().requires_grad_()
    expected_loss = torch.nn.functional.softplus(-2 * expected.flatten(1).mean(1)).mean()
    (expected_grad,) = torch.autograd.grad(expected_loss, expected)
    loss.backward()
    torch.testing.assert_close(disc.inputs[0], expected.detach())
    torch.testing.assert_close(sample.grad, expected_grad)
    assert score["real"].calls == []
    assert not score["fake"].calls[0][2]
    assert all(branch.slope.grad is None for branch in score.values())
    assert disc.weight.grad is None


def test_feature_r1_value_and_both_sides_receive_gradients():
    method = StandardDMD(gan_r1_noise_std=0.1)
    values = [torch.tensor([v], requires_grad=True) for v in (1.0, 1.2, 2.0, 2.3)]
    loss = method._gan_feature_r1(*values)
    torch.testing.assert_close(loss, torch.tensor(13.0))
    loss.backward()
    for value, expected in zip(values, (-40.0, 40.0, -60.0, 60.0)):
        torch.testing.assert_close(value.grad, torch.tensor([expected]))


def test_r1_discriminator_loss_and_gradient_ownership(monkeypatch):
    method, sample, score, disc, kwargs = setup_pair(monkeypatch)
    method.gan_r1_weight = 1.0
    loss, stats, _ = method.discriminator_loss(**kwargs, return_debug_tensors=True)
    assert len(disc.inputs) == 4
    fake, real, perturbed_real, perturbed_fake = disc.inputs
    expected_r1 = method._gan_feature_r1(2 * real, 2 * perturbed_real, 2 * fake, 2 * perturbed_fake)
    torch.testing.assert_close(stats["gan_r1_loss"][0].float(), expected_r1)
    torch.testing.assert_close(
        loss.detach(), stats["gan_discriminator_raw_loss"][0].float() + expected_r1
    )
    assert expected_r1 > 0
    loss.backward()
    assert disc.weight.grad is not None and torch.isfinite(disc.weight.grad)
    assert sample.grad is None
    assert all(branch.slope.grad is None for branch in score.values())


def test_discriminator_noise_preserves_reference_input_gradient():
    method = StandardDMD(generator_objective="gan", gan_discriminator_noise_sigma=0.2)
    x = torch.ones(2, 1, 2, 2, requires_grad=True)
    method._gan_discriminator_input(x).sum().backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 0.8))


def test_feature_ste_exact_direction_and_detached_target():
    live = torch.randn(2, 3, 8, dtype=torch.float64, requires_grad=True)
    real = torch.randn_like(live, requires_grad=True)
    fake = torch.randn_like(live, requires_grad=True)
    loss = StandardDMD._gan_feature_ste_loss(live, real, fake)
    delta = real.detach() - fake.detach()
    assert loss.dtype == torch.float64
    torch.testing.assert_close(loss, delta.square().sum(dim=1).mean())
    loss.backward()
    torch.testing.assert_close(live.grad, -2 * delta / (live.shape[0] * live.shape[2]))
    assert real.grad is None and fake.grad is None


def test_feature_ste_sgd_moves_toward_real_in_identity_case():
    live = nn.Parameter(torch.zeros(1, 1, 1, dtype=torch.float64))
    real = torch.ones_like(live)
    fake = live.detach().clone()
    optimizer = torch.optim.SGD([live], lr=0.1)
    before = (live.detach() - real).square().sum()
    StandardDMD._gan_feature_ste_loss(live, real, fake).backward()
    optimizer.step()
    torch.testing.assert_close(live.detach(), torch.full_like(live, 0.2))
    assert (live.detach() - real).square().sum() < before


def test_feature_ste_sums_tokens_but_averages_batch_and_channels():
    live = torch.zeros(2, 3, 5, dtype=torch.float64)
    real = torch.ones_like(live)
    fake = torch.zeros_like(live)
    loss = StandardDMD._gan_feature_ste_loss(live, real, fake)
    torch.testing.assert_close(loss, torch.tensor(3.0, dtype=torch.float64))
    for repeats, factor in [((1, 2, 1), 2), ((2, 1, 1), 1), ((1, 1, 2), 1)]:
        repeated = StandardDMD._gan_feature_ste_loss(
            live.repeat(*repeats),
            real.repeat(*repeats),
            fake.repeat(*repeats),
        )
        torch.testing.assert_close(repeated, factor * loss)


def test_feature_ste_uses_shared_noisy_query_and_only_updates_g(monkeypatch):
    method, sample, score, disc, kwargs = setup_pair(monkeypatch)
    method.gan_generator_loss_type = "feature_ste"
    disc.requires_grad_(False)

    def unexpected_renoise(*args):
        raise AssertionError("Feature STE must not add discriminator re-noising")

    monkeypatch.setattr(method, "_gan_discriminator_input", unexpected_renoise)
    loss, _, aux = method.generator_loss(**kwargs, return_debug_tensors=True)
    assert len(disc.inputs) == 3
    torch.testing.assert_close(disc.inputs[0], aux["pred_fake_x0"])
    torch.testing.assert_close(disc.inputs[1], aux["pred_real_x0"])
    torch.testing.assert_close(disc.inputs[2], aux["dmd_noisy"], rtol=0, atol=0)
    delta = 2 * (aux["pred_real_x0"].double() - aux["pred_fake_x0"].double())
    features_delta = delta.flatten(2).transpose(1, 2)
    torch.testing.assert_close(loss, features_delta.square().sum(dim=1).mean())
    loss.backward()
    # D's feature Jacobian is 2; the query interpolation contributes (1-t)=0.5.
    torch.testing.assert_close(
        sample.grad, (-2 * delta / (sample.shape[0] * sample.shape[1])).float()
    )
    assert disc.weight.grad is None
    assert all(branch.slope.grad is None for branch in score.values())
    assert not score["real"].calls[0][2] and not score["fake"].calls[0][2]


def test_phase_metrics_keep_header_column_order(tmp_path):
    import csv

    from verl_distill.trainers.dmd import _append_stats_row

    path = tmp_path / "stats.tsv"
    _append_stats_row(path, {"step": 1, "fake_logit": 0.0, "real_logit": 0.0})
    _append_stats_row(path, {"step": 2, "real_logit": 2.0, "fake_logit": -1.0})
    with path.open() as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert rows[1]["real_logit"] == "2.0"
    assert rows[1]["fake_logit"] == "-1.0"
