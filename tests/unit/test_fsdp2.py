import torch

from verl_distill.engine.fsdp2 import clip_grad_norm


def test_clip_grad_norm_uses_l2_norm_and_shared_coefficient():
    first = torch.nn.Parameter(torch.tensor([3.0]))
    second = torch.nn.Parameter(torch.tensor([4.0]))
    first.grad = torch.tensor([3.0])
    second.grad = torch.tensor([4.0])

    norm = clip_grad_norm([first, second], 2.5)

    assert norm == 5.0
    torch.testing.assert_close(first.grad, torch.tensor([1.5]))
    torch.testing.assert_close(second.grad, torch.tensor([2.0]))
