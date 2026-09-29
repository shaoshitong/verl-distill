"""CPU numerical checks against independent detached-prefix exits."""
from pathlib import Path
import copy
import pytest
import torch
from verl_distill.algorithms.dmd.qwen_rollout import detached_prefix_rollout, detached_rollout_step
from verl_distill.algorithms.dmd.qwen_image21 import dmd_surrogate, renoise
from verl_distill.config import load_config
from verl_distill.models.qwen_image21.configuration import validate_qwen_config

@pytest.mark.parametrize('phase', ['generator', 'fake'])
def test_six_exit_gradient_matches_independent_prefixes(phase):
    torch.manual_seed(37)
    levels = torch.tensor([1., .94, .86, .73, .59, .4, 0.])
    initial = torch.randn(1, 4, 3)
    g = torch.nn.Parameter(torch.tensor(.17))
    f = torch.nn.Parameter(torch.tensor(.23))
    sigmas = [.02 + .96 * torch.rand(1) for _ in range(6)]
    noises = [torch.randn_like(initial) for _ in range(6)]
    calls = []
    def predict(x, t):
        calls.append(t.item())
        return g * x + t
    def objective(y, index):
        sigma, noise = sigmas[index], noises[index]
        q = renoise(y.detach(), noise, sigma)
        if phase == 'fake':
            return ((f*q - (noise-y.detach()))**2).mean()
        with torch.no_grad():
            fv, rv = f*q, .31*q
        return dmd_surrogate(y, q, fv, rv, sigma, loss_weight=1.)[0]
    x = initial
    outputs = []
    for index in range(6):
        y, exit_input, x = detached_rollout_step(x, levels, index, predict, train_exit=phase=='generator')
        assert not x.requires_grad and x.grad_fn is None
        assert not exit_input.requires_grad
        assert y.requires_grad == (phase=='generator')
        outputs.append(y.detach())
        (objective(y, index)/6).backward()
    assert len(calls) == 6
    active, inactive = (g, f) if phase == 'generator' else (f, g)
    actual = active.grad.clone()
    assert inactive.grad is None
    active.grad = None
    calls.clear()
    for index in range(6):
        y, _ = detached_prefix_rollout(initial, levels, index, predict, train_exit=phase=='generator')
        torch.testing.assert_close(y.detach(), outputs[index])
        (objective(y, index)/6).backward()
    assert len(calls) == 21
    torch.testing.assert_close(active.grad, actual)
    assert inactive.grad is None


def test_config_default_and_opt_in():
    root=Path(__file__).resolve().parents[1]
    config=load_config(root/'configs/recipes/qwen_image21/reflow300_dmd2_clip5_debug6nfe.yaml')
    original=copy.deepcopy(config)
    validate_qwen_config(config)
    assert config == original  # No default injection changing old checkpoint contracts.
    config['method']['params']['dmd_rollout_loss_mode']='all_exits'
    validate_qwen_config(config)
    config['method']['params']['dmd_rollout_loss_mode']='typo'
    with pytest.raises(ValueError, match='dmd_rollout_loss_mode'):
        validate_qwen_config(config)
    config['method']['params']['dmd_rollout_loss_mode']='all_exits'
    config['method']['params']['generator_input']='renoised_data'
    with pytest.raises(ValueError, match='all_exits requires'):
        validate_qwen_config(config)
