import pytest
import torch

from verl_distill.algorithms.dmd.qwen_image21 import dmd_surrogate, fake_score_loss, renoise
from verl_distill.models.qwen_image21.guidance import combine_cfg, geometry_metrics
from verl_distill.trainers.qwen_image21 import snapshot_tensors
from verl_distill.trainers.qwen_image21_debug import tensor_stats
from verl_distill.trainers.qwen_image21_metrics import score_bin_report, score_bin_sums


def test_fp64_snapshot_and_statistics_preserve_small_differences(tmp_path):
    original = torch.tensor([2**24 + .25, 2**24 + .5], dtype=torch.float64)
    saved = snapshot_tensors({"diff_x0": original})["diff_x0"]
    assert saved.dtype == torch.float64 and saved.data_ptr() != original.data_ptr()
    torch.save(saved, tmp_path / "diff.pt")
    torch.testing.assert_close(torch.load(tmp_path / "diff.pt", weights_only=True), original, atol=0, rtol=0)
    stats = tensor_stats(original)
    assert stats["dtype"] == "torch.float64"
    assert stats["std"] == .125
    assert tensor_stats(original.float())["dtype"] == "torch.float32"


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_surrogate_dtype_controls_math_without_changing_gradient_direction(dtype):
    h = torch.randn(2, 5, 3, requires_grad=True)
    noisy = torch.randn_like(h)
    fake = torch.randn_like(h, requires_grad=True)
    real = torch.randn_like(h, requires_grad=True)
    loss, aux = dmd_surrogate(h, noisy, fake, real, torch.tensor([.2, .8]), dtype=dtype)
    assert loss.dtype == getattr(torch, dtype)
    assert aux["diff_x0"].dtype == getattr(torch, dtype)
    loss.backward()
    torch.testing.assert_close(h.grad, (4 * aux["normalized_direction"] / h.numel()).float())
    assert fake.grad is None and real.grad is None
    with pytest.raises(ValueError):
        dmd_surrogate(h, noisy, fake, real, torch.tensor([.2, .8]), dtype="bfloat16")


def test_score_bins_are_independent_of_generator_sigma_and_reduce_correctly():
    rank_a = [{"generator_sigma": .4, "score_sigma": .95, "velocity_mse": 2., "loss_weighted": .005},
              {"generator_sigma": 1., "score_sigma": .95, "velocity_mse": 4., "loss_weighted": .01}]
    rank_b = [{"generator_sigma": .4, "score_sigma": .05, "velocity_mse": 10.},
              {"generator_sigma": .8, "score_sigma": .95, "velocity_mse": 6.}]
    report = score_bin_report(score_bin_sums(rank_a) + score_bin_sums(rank_b))
    assert report[-1]["count"] == 3
    assert report[-1]["means"]["velocity_mse"] == 4
    assert report[-1]["means"]["loss_weighted"] == pytest.approx(.0075)
    assert report[0]["means"]["velocity_mse"] == 10
    assert report[-1]["means"]["direction_rms"] is None


def test_fake_velocity_diagnostic_is_unweighted():
    clean, noise = torch.ones(1, 2, 3), torch.zeros(1, 2, 3)
    s = torch.tensor([.95])
    noisy = renoise(clean, noise, s)
    loss, aux = fake_score_loss(noisy, torch.zeros_like(clean), clean, noise, s)
    assert aux["velocity_mse"].item() == 1
    assert aux["velocity_weight"].item() == pytest.approx(.0025)
    assert loss.item() == pytest.approx(.0025)


def test_cfg_arithmetic_and_geometry():
    cond, neg = torch.ones(1, 3, 4), torch.zeros(1, 3, 4)
    assert combine_cfg(cond, None, 1) is cond
    torch.testing.assert_close(combine_cfg(cond, neg, 4), torch.full_like(cond, 4.).double())
    with pytest.raises(ValueError):
        combine_cfg(cond, None, 4)
    with pytest.raises(ValueError):
        combine_cfg(cond, neg, float("nan"))
    geom = geometry_metrics(torch.zeros(2), torch.tensor([0., 1.]), torch.tensor([1., 0.]))
    assert geom["angle_H_degrees"] == 90
    assert geom["mse_RF"] == 1


def test_negative_encoding_preserves_reference_order_and_latents(tmp_path):
    from types import SimpleNamespace
    from PIL import Image
    from verl_distill.data.qwen_image21 import sha256
    from verl_distill.models.qwen_image21.guidance import encode_negative_condition

    names = ["red.png", "blue.png"]
    for name, color in zip(names, ["red", "blue"]):
        Image.new("RGBA", (32, 32), color).save(tmp_path / name)
    row = {"height": 32, "width": 32, "reference_images": names,
           "reference_sha256": [sha256(tmp_path / name) for name in names]}
    observed = {}

    def encode_prompt(**kwargs):
        observed["prompt"] = kwargs["prompt"]
        observed["colors"] = [im.getpixel((0, 0)) for im in kwargs["image"]]
        return torch.zeros(1, 4, 8), None, torch.tensor([[False, True, True, False]])

    pipe = SimpleNamespace(encode_prompt=encode_prompt, image_processor=SimpleNamespace(
        resize=lambda im, **kwargs: im.copy()))
    positive = {"reference_latents": torch.randn(1, 8, 64), "img_shapes": [[(1, 2, 2)] * 3]}
    negative = encode_negative_condition(pipe, row, positive, tmp_path, "cpu")
    assert observed["prompt"] == ""
    assert observed["colors"] == [(255, 0, 0, 255), (0, 0, 255, 255)]
    assert negative["reference_latents"] is positive["reference_latents"]
    assert negative["img_shapes"] == positive["img_shapes"]
    assert negative["img_mask"].tolist() == [[False, True, True, False, True]]
