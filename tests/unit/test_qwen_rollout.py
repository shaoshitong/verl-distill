import pytest
import torch

from verl_distill.algorithms.dmd.qwen_rollout import detached_prefix_rollout


@pytest.mark.parametrize("exit_index", range(6))
@pytest.mark.parametrize("train_exit", [False, True])
def test_rollout_exact_prefix_and_last_call_only_gradient(exit_index, train_exit):
    levels = torch.tensor([1., .93, .84, .72, .57, .4, 0.])
    initial = torch.tensor([[[2.]]], requires_grad=True)
    weight = torch.tensor(.3, requires_grad=True)
    calls = []
    def predict(x, t):
        calls.append((x.detach().clone(), t.item(), torch.is_grad_enabled(), x.requires_grad))
        return weight * x + t
    generated, exit_input = detached_prefix_rollout(initial, levels, exit_index, predict, train_exit=train_exit)
    expected = initial.detach().clone()
    for i in range(exit_index):
        expected += (levels[i + 1] - levels[i]) * (weight.detach() * expected + levels[i])
    torch.testing.assert_close(exit_input, expected)
    torch.testing.assert_close(generated.detach(), expected - levels[exit_index] * (weight.detach() * expected + levels[exit_index]))
    assert len(calls) == exit_index + 1
    assert [c[1] for c in calls] == levels[:exit_index + 1].tolist()
    assert all(not c[2] and not c[3] for c in calls[:-1])
    assert calls[-1][2] == train_exit and not calls[-1][3]
    assert not exit_input.requires_grad
    if train_exit:
        generated.sum().backward()
        torch.testing.assert_close(weight.grad, (-levels[exit_index] * expected).sum())
        assert initial.grad is None
    else:
        assert not generated.requires_grad and weight.grad is None


def test_invalid_exit_rejected_before_model_call():
    with pytest.raises(ValueError):
        detached_prefix_rollout(torch.ones(1, 1, 1), torch.tensor([1., 0.]), 1,
                                lambda *args: pytest.fail("must not call model"), train_exit=True)
