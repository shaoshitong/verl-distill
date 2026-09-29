from unittest.mock import patch

import pytest
import torch

from verl_distill.algorithms.dmd.full_model import FullModelDMD


@pytest.mark.parametrize('initial,expected', [
    (0.8, [0.8, 0.4, 0.2, 0.3, 0.25]),
    (0.1, [0.1, 0.55, 0.325, 0.2125, 0.26875, 0.240625]),
    (0.24, [0.24]),
])
def test_search_and_fixed_real_query(initial, expected):
    method = FullModelDMD(adaptive_fake_timestep=True, adaptive_fake_timestep_max_iters=8)
    fake_calls, real_calls, queries = [], [], []
    def predict(model, noisy, sigma, **kwargs):
        (fake_calls if kwargs['use_lora'] else real_calls).append(sigma.item())
        queries.append((sigma.clone(), noisy.clone()))
        pred = torch.ones_like(noisy) * (sigma.view(-1, 1, 1, 1) if kwargs['use_lora'] else 1.0)
        return pred, torch.zeros_like(noisy)
    with patch.object(method, '_sample_sigmas', return_value=torch.tensor([initial])), \
         patch.object(method, '_predict_x0_from_flow', side_effect=predict):
        _, stats = method._compute_dmd_grad({}, torch.zeros(1, 1, 2, 2), [], None)
    assert fake_calls == pytest.approx(expected)
    assert real_calls == pytest.approx([initial])
    assert 0.22 <= stats['dm_fake_distance_final'].item() <= 0.26
    for sigma, noisy in queries:
        torch.testing.assert_close(noisy / sigma, queries[0][1] / initial)


@pytest.mark.parametrize("mode", ["mae_range", "match_real_mse"])
def test_score_phase_unchanged(mode):
    class Flow(torch.nn.Module):
        def forward(self, x, t, c):
            return x * 0.1
    model = Flow()
    x = torch.ones(1, 1, 2, 2)
    results = []
    for enabled in (False, True):
        method = FullModelDMD(adaptive_fake_timestep=enabled, score_flow_shift=4, adaptive_fake_distance_mode=mode)
        torch.manual_seed(123)
        results.append(method.score_loss(model, {'fake': model, 'real': model}, x, [], None))
    torch.testing.assert_close(results[0][0], results[1][0])
    assert results[0][1].keys() == results[1][1].keys()
    for key in results[0][1]:
        torch.testing.assert_close(results[0][1][key], results[1][1][key])


@pytest.mark.parametrize('mode', ['match_real_mse', 'match_real_l2_squared'])
@pytest.mark.parametrize('initial', [0.1, 0.6, 0.9])
def test_match_real_squared_distance(mode, initial):
    method = FullModelDMD(adaptive_fake_timestep=True,
                          adaptive_fake_distance_mode=mode,
                          adaptive_fake_distance_tolerance=0.05,
                          adaptive_fake_timestep_max_iters=8)
    real_calls = []
    def predict(model, noisy, sigma, **kwargs):
        if not kwargs['use_lora']:
            real_calls.append(sigma.clone())
        value = sigma if kwargs['use_lora'] else torch.full_like(sigma, 0.6)
        return torch.ones_like(noisy) * value[:, None, None, None], torch.zeros_like(noisy)
    with patch.object(method, '_sample_sigmas', return_value=torch.tensor([initial])), \
         patch.object(method, '_predict_x0_from_flow', side_effect=predict):
        _, stats, extra = method._compute_dmd_grad({}, torch.zeros(1, 1, 2, 2), [], None, return_extra=True)
    assert len(real_calls) == 1
    assert real_calls[0].item() == pytest.approx(initial)
    assert stats['dm_distance_gap_abs'].item() <= 0.05
    assert stats['dm_real_mse'].item() == pytest.approx(0.36)
    assert stats['dm_real_l2_squared'].item() == pytest.approx(1.44)
    assert stats['dm_fake_distance_target_hit'].item() == 1
    assert extra['dm_cos_real_fake'].item() == pytest.approx(1)
    if initial < 0.6:
        assert stats['dm_fake_sigma'].item() > initial
    elif initial > 0.6:
        assert stats['dm_fake_sigma'].item() < initial


def test_triangle_cosine_orientation():
    method = FullModelDMD(adaptive_fake_distance_mode='match_real_mse')
    def predict(model, noisy, sigma, **kwargs):
        pred = torch.tensor([1., 0.] if kwargs['use_lora'] else [0., 1.]).reshape(1, 1, 1, 2)
        return pred, torch.zeros_like(pred)
    with patch.object(method, '_sample_sigmas', return_value=torch.tensor([0.5])), \
         patch.object(method, '_predict_x0_from_flow', side_effect=predict):
        _, stats = method._compute_dmd_grad({}, torch.zeros(1, 1, 1, 2), [], None)
    assert stats['dm_cos_real_fake'].item() == pytest.approx(0)
    assert stats['dm_cos_real_real_minus_fake'].item() == pytest.approx(2 ** -0.5)
    assert stats['dm_cos_fake_real_minus_fake'].item() == pytest.approx(-2 ** -0.5)
