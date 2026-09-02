import torch

from verl_distill.algorithms.opd_gan.losses import dual_alignment_loss


def test_alignment_loss_is_zero_for_equal_inputs():
    value = torch.randn(2, 4, 3, 3)
    loss, stats = dual_alignment_loss(value, value, mse_weight=1.0, pearson_eps=1.0e-6)
    assert torch.allclose(loss, torch.zeros_like(loss), atol=1.0e-10)
    assert torch.allclose(stats["mse"], torch.zeros_like(stats["mse"]))
