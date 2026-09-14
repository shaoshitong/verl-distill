import pytest
import torch

from verl_distill.algorithms.dmd import StandardDMD
from verl_distill.algorithms.dmd.full_model import FullModelDMD
from verl_distill.models.zimage.feature_discriminator import TeacherFeatureDiscriminator
from verl_distill.models.zimage.modeling import GenTransformer
from verl_distill.models.zimage.transformer import ZImageTransformer2DModelWrapper


@pytest.mark.parametrize("method_cls", [StandardDMD, FullModelDMD])
def test_latent_only_baseline_losses_and_gradients(monkeypatch, method_cls):
    class NoFeatures(torch.nn.Module):
        def forward(self, *args, **kwargs):
            raise AssertionError("Latent-only objective must not extract teacher features")

    method = method_cls(
        generator_objective="teacher_feature_ste",
        score_objective="teacher_feature_mse",
        teacher_feature_layers=[],
        teacher_feature_include_latent=True,
        teacher_feature_include_last=False,
        teacher_feature_normalize=True,
    )
    assert method.generator_teacher_feature_weights == {"latent": 1.0}
    assert method.score_teacher_feature_weights == {"latent": 1.0}
    sample = torch.randn(2, 4, 3, 3, requires_grad=True)
    real, fake = torch.randn_like(sample), torch.randn_like(sample)
    meta = dict(gen_input_sigma=torch.ones(2), gen_backward_simulation=torch.zeros(2))
    monkeypatch.setattr(method, "generate_one_step_latents", lambda *a, **k: (sample, meta))
    monkeypatch.setattr(
        method,
        "_gan_score_pair",
        lambda *a, **k: {"pred_real_x0": real, "pred_fake_x0": fake, "dm_sigma": torch.ones(2)},
    )
    score = {"real": NoFeatures(), "fake": NoFeatures()}
    loss, stats = method.generator_loss(None, score, sample, None, None)
    denominator = (sample.detach().double() - real.double()).abs().mean((1, 2, 3), keepdim=True)
    direction = (real.double() - fake.double()) / denominator.clamp_min(method.grad_norm_eps)
    torch.testing.assert_close(loss, direction.square().mean())
    torch.testing.assert_close(
        torch.autograd.grad(loss, sample)[0], (-2 * direction / sample.numel()).float()
    )
    assert "teacher_feature_loss_latent" in stats
    prediction = torch.randn_like(sample, requires_grad=True)
    fake_loss, fake_stats = method._score_teacher_feature_loss(score, prediction, sample, None)
    expected = (prediction.double() - sample.detach().double()).square().mean()
    torch.testing.assert_close(fake_loss, expected)
    torch.testing.assert_close(
        torch.autograd.grad(fake_loss, prediction)[0],
        (2 * (prediction.detach() - sample.detach()) / sample.numel()),
    )
    assert "teacher_feature_loss_latent" in fake_stats


def tiny_teacher(n_layers=4):
    backbone = ZImageTransformer2DModelWrapper(
        all_patch_size=(2,),
        all_f_patch_size=(1,),
        in_channels=4,
        dim=64,
        n_layers=n_layers,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        norm_eps=1e-5,
        qk_norm=True,
        cap_feat_dim=32,
        siglip_feat_dim=None,
        rope_theta=256.0,
        t_scale=1000.0,
        axes_dims=[8, 4, 4],
        axes_lens=[64, 64, 64],
    )
    with torch.no_grad():
        backbone.x_pad_token.zero_()
        backbone.cap_pad_token.zero_()
    return GenTransformer(backbone, 8, aux_time_embed=False).eval().requires_grad_(False)


def test_teacher_features_match_opd_head_path():
    teacher = tiny_teacher()
    disc = TeacherFeatureDiscriminator(64, (1, 2, 3), transformer_layers=2, transformer_heads=4)
    x = torch.randn(1, 4, 8, 8)
    c = [torch.randn(1, 4, 32), torch.ones(1, 4)]
    t = torch.tensor([0.2])
    features = teacher(x, t, c=c, feature_layers=(1, 2, 3))
    assert set(features) == {"layer_1", "layer_2", "layer_3", "pre_projector"}
    assert all(value.shape[-1] == 64 for value in features.values())
    expected = disc(features)
    with_features = disc(features, return_features=True)
    torch.testing.assert_close(with_features["logits"], expected)
    torch.testing.assert_close(disc.head.out_mlp[-1](with_features["features"]), expected)
    teacher.transformer.multi_feature_discriminator_head = disc.head
    teacher.transformer.multi_feature_discriminator_layer_numbers = (1, 2, 3)
    teacher.aux_time_embed = True
    actual = teacher(x, t, c=c, tt=t, skip_aux_time=True, discriminator_mode=True, return_raw=True)
    torch.testing.assert_close(actual, expected)


def test_teacher_feature_loss_four_layer_sum_and_element_mean():
    live = {str(i): torch.zeros(2, 3, 5, dtype=torch.float64, requires_grad=True) for i in range(4)}
    real = {k: torch.full_like(v, int(k) + 1, requires_grad=True) for k, v in live.items()}
    fake = {k: torch.zeros_like(v, requires_grad=True) for k, v in live.items()}
    losses = StandardDMD._teacher_feature_ste_losses(live, real, fake)
    total = sum(losses.values())
    torch.testing.assert_close(total, torch.tensor(30.0, dtype=torch.float64))
    total.backward()
    for k, h in live.items():
        torch.testing.assert_close(h.grad, -2 * real[k].detach() / h.numel())
        assert real[k].grad is None and fake[k].grad is None

    def repeated(tensors):
        return {k: v.detach().repeat(2, 3, 4) for k, v in tensors.items()}

    repeated_losses = StandardDMD._teacher_feature_ste_losses(
        repeated(live), repeated(real), repeated(fake)
    )
    torch.testing.assert_close(sum(repeated_losses.values()), total.detach())


@pytest.mark.parametrize(
    "single_normalized,multirep", [(False, False), (True, False), (False, True)]
)
def test_teacher_feature_g_anchors_clean_x0_and_matches_four_layer_vjp(
    monkeypatch, single_normalized, multirep
):
    from torch import nn

    teacher = tiny_teacher(n_layers=30)
    teacher.enable_gradient_checkpointing()

    class FakeScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.slope = nn.Parameter(torch.tensor(0.7))

        def forward(self, x, t, c=None):
            return self.slope * x

    fake_score = FakeScore()
    sample = nn.Parameter(torch.randn(1, 4, 8, 8))
    layer_outputs = {}

    def capture_layer(number):
        def capture(module, args, output):
            layer_outputs[number] = output.detach().clone()

        return capture

    layer_handles = [
        teacher.transformer.layers[number - 1].register_forward_hook(capture_layer(number))
        for number in (5, 15, 25, 30)
    ]
    method = StandardDMD(
        generator_objective="teacher_feature_ste",
        real_guidance_scale=0.0,
        min_step_percent=0.5,
        max_step_percent=0.5,
        teacher_feature_layers=[5] if single_normalized else [5, 15, 25],
        teacher_feature_include_last=not (single_normalized or multirep),
        teacher_feature_include_latent=multirep,
        teacher_feature_normalize=single_normalized or multirep,
    )
    meta = dict(gen_input_sigma=torch.ones(1), gen_backward_simulation=torch.zeros(1))
    monkeypatch.setattr(method, "generate_one_step_latents", lambda *a, **kw: (sample, meta))
    records = []

    def record(module, args, kwargs, output):
        if "feature_layers" in kwargs:
            records.append((args[0].detach().clone(), args[1].detach().clone(), output))

    handle = teacher.register_forward_hook(record, with_kwargs=True)
    c = [torch.randn(1, 4, 32), torch.ones(1, 4)]
    loss, _, aux = method.generator_loss(
        nn.Identity(),
        {"real": teacher, "fake": fake_score},
        torch.zeros_like(sample),
        c,
        None,
        return_debug_tensors=True,
    )
    handle.remove()
    for layer_handle in layer_handles:
        layer_handle.remove()
    assert len(records) == 3
    for (x, t, _), expected in zip(
        records, (aux["pred_real_x0"], aux["pred_fake_x0"], sample.detach())
    ):
        torch.testing.assert_close(x, expected, rtol=0, atol=0)
        torch.testing.assert_close(t, torch.tensor([0.2]))
    hr, hf, h = [item[2] for item in records]
    assert set(h) == (
        {"layer_1", "pre_projector"}
        if single_normalized
        else {"layer_1", "layer_2", "layer_3", "pre_projector"}
    )
    for key, number in zip(("layer_1", "layer_2", "layer_3", "pre_projector"), (5, 15, 25, 30)):
        if single_normalized and key != "layer_1":
            continue
        torch.testing.assert_close(
            h[key], layer_outputs[number][:, : h[key].shape[1]], rtol=0, atol=0
        )
    assert all(v.requires_grad for v in h.values())
    assert not any(v.requires_grad for d in (hr, hf) for v in d.values())
    (actual,) = torch.autograd.grad(loss, sample, retain_graph=True)
    keys = ["layer_1"] if single_normalized else list(h)
    if multirep:
        keys = ["latent", "layer_1", "layer_2", "layer_3"]
        hr = {**hr, "latent": aux["pred_real_x0"]}
        hf = {**hf, "latent": aux["pred_fake_x0"]}
        h = {**h, "latent": sample}
    vectors = []
    for k in keys:
        direction = hr[k].double() - hf[k].double()
        if single_normalized or multirep:
            denom = (
                (h[k].detach().double() - hr[k].double())
                .abs()
                .mean(tuple(range(1, h[k].ndim)), keepdim=True)
            )
            direction = direction / denom.clamp_min(method.grad_norm_eps)
        vectors.append((-2 * direction / h[k].numel()).to(h[k].dtype))
    (expected,) = torch.autograd.grad(
        tuple(h[k] for k in keys), sample, grad_outputs=tuple(vectors)
    )
    torch.testing.assert_close(actual, expected)
    assert torch.isfinite(actual).all() and actual.abs().sum() > 0
    assert fake_score.slope.grad is None and all(p.grad is None for p in teacher.parameters())
    assert not method.uses_gan_objective()


@pytest.mark.parametrize("method_cls", [StandardDMD, FullModelDMD])
def test_score_multirep_regression_matches_direct_loss_and_only_updates_fake(
    monkeypatch, method_cls
):
    from torch import nn

    teacher = tiny_teacher(n_layers=30)
    teacher.enable_gradient_checkpointing()
    generator = nn.Conv2d(4, 4, 1)

    class FakeScore(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 1)

        def forward(self, x, t, c=None):
            return self.conv(x)

    fake = FakeScore()
    weights = {"latent": 1.0, "layer_5": 2.0, "layer_15": 3.0, "layer_25": 4.0}
    method = method_cls(
        score_objective="teacher_feature_mse",
        teacher_feature_layers=[5, 15, 25],
        teacher_feature_include_latent=True,
        teacher_feature_include_last=False,
        score_teacher_feature_weights=weights,
    )
    meta = dict(gen_input_sigma=torch.ones(1), gen_backward_simulation=torch.zeros(1))
    monkeypatch.setattr(method, "generate_one_step_latents", lambda g, x, c: (g(x), meta))
    records = []

    def record(module, args, kwargs, output):
        if "feature_layers" in kwargs:
            records.append((args[0], args[1], output))

    handle = teacher.register_forward_hook(record, with_kwargs=True)
    c = [torch.randn(1, 4, 32), torch.ones(1, 4)]
    loss, stats, aux = method.score_loss(
        generator,
        {"real": teacher, "fake": fake},
        torch.randn(1, 4, 8, 8),
        c,
        None,
        return_debug_tensors=True,
    )
    handle.remove()
    assert len(records) == 2
    target_x, t, target = records[0]
    pred_x, pt, pred = records[1]
    torch.testing.assert_close(target_x, aux["x_fake"])
    torch.testing.assert_close(pred_x, aux["pred_x0"])
    torch.testing.assert_close(t, torch.tensor([0.2]))
    torch.testing.assert_close(pt, t)
    assert not target_x.requires_grad and pred_x.requires_grad
    assert all(not h.requires_grad for h in target.values())
    expected = (pred_x.double() - target_x.double()).square().mean()
    for slot, layer in enumerate((5, 15, 25), 1):
        expected += (
            weights[f"layer_{layer}"]
            * (pred[f"layer_{slot}"].double() - target[f"layer_{slot}"].double()).square().mean()
        )
    torch.testing.assert_close(loss, expected)
    for key in weights:
        assert f"teacher_feature_loss_{key}" in stats
    params = tuple(fake.parameters())
    actual_grad = torch.autograd.grad(loss, params, retain_graph=True)
    expected_grad = torch.autograd.grad(expected, params, retain_graph=True)
    for actual, reference in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual, reference)
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in params)
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in generator.parameters())


def test_multirep_reductions_normalize_per_sample_and_keep_identity():
    shapes = {
        "latent": (2, 4, 3, 3),
        "layer_5": (2, 3, 4),
        "layer_15": (2, 6, 8),
        "layer_25": (2, 4, 5),
    }
    live = {k: torch.zeros(s, dtype=torch.float64, requires_grad=True) for k, s in shapes.items()}
    real = {
        k: torch.stack([torch.ones(s[1:]), torch.full(s[1:], 2.0)]).double().requires_grad_()
        for k, s in shapes.items()
    }
    fake = {k: torch.zeros_like(v, requires_grad=True) for k, v in real.items()}
    losses, stats = StandardDMD._teacher_feature_ste_losses(
        live, real, fake, normalize=True, return_stats=True
    )
    total = sum(losses.values())
    torch.testing.assert_close(total, torch.tensor(4.0, dtype=torch.float64))
    total.backward()
    for k, h in live.items():
        torch.testing.assert_close(h.grad, torch.full_like(h, -2 / h.numel()))
        torch.testing.assert_close(
            stats[f"teacher_feature_denom_{k}"], torch.tensor([1.0, 2.0], dtype=torch.float64)
        )
        assert real[k].grad is None and fake[k].grad is None
    regression = StandardDMD._teacher_feature_mse_losses(live, real)
    torch.testing.assert_close(sum(regression.values()), torch.tensor(10.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"score_objective": "teacher_feature_mse", "score_loss_target": "flow"},
        {"score_objective": "teacher_feature_mse", "score_use_weighting": True},
        {"generator_teacher_feature_weights": {"latent": 1.0}},
        {"score_teacher_feature_weights": {"layer_5": -1.0}},
    ],
)
def test_invalid_teacher_representation_configuration(kwargs):
    with pytest.raises(ValueError):
        StandardDMD(**kwargs)


def test_teacher_feature_normalization_is_detached_per_sample_direction():
    live = torch.zeros(2, 3, 4, dtype=torch.float64, requires_grad=True)
    real = torch.stack([torch.ones(3, 4), torch.full((3, 4), 2.0)]).double().requires_grad_()
    fake = torch.zeros_like(real, requires_grad=True)
    loss = StandardDMD._teacher_feature_ste_losses(
        {"layer_5": live},
        {"layer_5": real},
        {"layer_5": fake},
        normalize=True,
    )["layer_5"]
    torch.testing.assert_close(loss, torch.tensor(1.0, dtype=torch.float64))
    loss.backward()
    torch.testing.assert_close(live.grad, torch.full_like(live, -2 / live.numel()))
    assert real.grad is None and fake.grad is None
    zero = torch.zeros_like(live, requires_grad=True)
    finite = StandardDMD._teacher_feature_ste_losses(
        {"layer_5": zero},
        {"layer_5": zero.detach()},
        {"layer_5": fake.detach()},
        normalize=True,
    )["layer_5"]
    finite.backward()
    assert torch.isfinite(finite) and torch.isfinite(zero.grad).all()


def test_frozen_teacher_backpropagates_only_to_g_input():
    teacher = tiny_teacher()
    teacher.enable_gradient_checkpointing()
    disc = TeacherFeatureDiscriminator(64, (1, 2, 3), transformer_layers=2, transformer_heads=4)
    method = StandardDMD(generator_objective="gan")
    c = [torch.randn(1, 4, 32), torch.ones(1, 4)]
    x = torch.randn(1, 4, 8, 8)
    score = {"real": teacher}
    loss = method._call_discriminator(disc, x, score_model=score, c=c).square().mean()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in disc.parameters())
    assert all(p.grad is None for p in teacher.parameters())
    disc.zero_grad(set_to_none=True)
    disc.requires_grad_(False)
    x.requires_grad_(True)
    loss = torch.nn.functional.softplus(
        -method._call_discriminator(disc, x, score_model=score, c=c)
    ).mean()
    loss.backward()
    assert x.grad is not None and torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    assert all(p.grad is None for p in teacher.parameters())
    assert all(p.grad is None for p in disc.parameters())


def test_feature_teacher_preserves_native_score_forward():
    from diffusers import ZImageTransformer2DModel

    teacher = tiny_teacher()
    native = ZImageTransformer2DModel(
        in_channels=4,
        dim=64,
        n_layers=4,
        n_refiner_layers=1,
        n_heads=4,
        n_kv_heads=4,
        cap_feat_dim=32,
        axes_dims=[8, 4, 4],
        axes_lens=[64, 64, 64],
    )
    source = teacher.transformer.state_dict()
    native.load_state_dict({k: source[k] for k in native.state_dict()}, strict=True)
    native = GenTransformer(native, 8, aux_time_embed=False).eval()
    x = torch.randn(1, 4, 8, 8)
    c = [torch.randn(1, 4, 32), torch.ones(1, 4)]
    with torch.no_grad():
        expected = native(x, torch.tensor([0.37]), c=c)
        actual = teacher(x, torch.tensor([0.37]), c=c)
    torch.testing.assert_close(actual, expected)


def test_dino_checkpoint_loads_trunk_and_preserves_fresh_gan_projection(tmp_path):
    import torch.distributed.checkpoint as dcp

    from verl_distill.models.zimage.discriminator import ZImageMultiFeatureDiscriminatorHead

    source = ZImageMultiFeatureDiscriminatorHead(
        hidden_dim=64,
        num_features=4,
        fusion="channel",
        norm="new",
        transformer_layers=2,
        transformer_heads=4,
        output_dim=7,
        output_mode="tokens",
        use_time_embedding=False,
    )
    state = source.state_dict()
    checkpoint = tmp_path / "fsdp_state"
    dcp.save({"teacher_discriminator_model": state}, checkpoint_id=checkpoint, no_dist=True)
    disc = TeacherFeatureDiscriminator(64, (1, 2, 3), transformer_layers=2, transformer_heads=4)
    fresh = {k: v.clone() for k, v in disc.head.state_dict().items() if k.startswith("out_mlp.2.")}
    report = disc.load_pretrained(tmp_path)
    for key, value in disc.head.state_dict().items():
        torch.testing.assert_close(value, fresh[key] if key in fresh else state[key])
    assert report["loaded_tensors"] == len(state) - 2
    del state["fusion_blocks.0.linear1.weight"]
    broken = tmp_path / "broken"
    dcp.save({"teacher_discriminator_model": state}, checkpoint_id=broken, no_dist=True)
    with pytest.raises(ValueError, match="fusion_blocks.0.linear1.weight"):
        disc.load_pretrained(broken)
