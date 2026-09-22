"""Pinned Diffusers CPU tests: six-point scheduling and small real Qwen blocks."""
import json

import numpy as np
import pytest
import torch

from verl_distill.models.qwen_image21.modeling import QwenSchedule, predict_velocity


@pytest.mark.parametrize("height,width", [(2048, 2048), (1696, 2400), (1440, 2880)])
def test_official_grid_stretch_and_independent_score(tmp_path, height, width):
    pytest.importorskip("diffusers.pipelines.qwenimage21.pipeline_qwenimage21")
    from diffusers import FlowMatchEulerDiscreteScheduler
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

    config = {"base_image_seq_len": 256, "max_image_seq_len": 8192,
              "base_shift": .5, "max_shift": .9, "shift_terminal": .02,
              "use_dynamic_shifting": True, "time_shift_type": "exponential",
              "num_train_timesteps": 1000}
    (tmp_path / "scheduler").mkdir()
    (tmp_path / "scheduler/scheduler_config.json").write_text(json.dumps(config))
    adapter = QwenSchedule(tmp_path)
    actual = adapter.levels(height, width)
    official = FlowMatchEulerDiscreteScheduler.from_config({**config, "shift_terminal": .4})
    mu = calculate_shift((height // 16) * (width // 16), 256, 8192, .5, .9)
    official.set_timesteps(6, sigmas=np.linspace(1, 1 / 6, 6), mu=mu)
    torch.testing.assert_close(actual, official.sigmas, rtol=0, atol=0)
    assert len(actual) == 7 and actual[-1].item() == 0
    assert actual[-2].item() == pytest.approx(.4, abs=1e-7)
    assert (actual[:-1] > actual[1:]).all()
    assert adapter.levels(height, width, 1000, generator=False)[-2].item() == pytest.approx(.02, abs=1e-7)
    original = FlowMatchEulerDiscreteScheduler.from_config(config)
    original.set_timesteps(25, sigmas=np.linspace(1, 1 / 25, 25), mu=mu)
    torch.testing.assert_close(adapter.levels(height, width, 25, generator=False),
                               original.sigmas, rtol=0, atol=0)
    assert adapter.config["shift_terminal"] == .02
    if height == width == 2048:
        torch.testing.assert_close(actual, torch.tensor(
            [1., .94658935, .87597263, .77823925, .63405871, .39999998, 0.]),
            rtol=0, atol=2e-7)
    for _ in range(10):
        assert .02 - 1e-7 <= adapter.score_sigma(height, width, .02, .98, "cpu").item() <= .98 + 1e-7


@pytest.mark.parametrize("references", [0, 2])
def test_target_adapter_matches_official_forward_and_backward(references):
    pytest.importorskip("diffusers.models.transformers.transformer_qwenimage21")
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel(
        num_layers=2, attention_head_dim=16, num_attention_heads=2,
        context_in_dim=16, axes_dims_rope=(4, 6, 6)).to(torch.bfloat16)
    refs = torch.randn(1, 4 * references, 64, dtype=torch.bfloat16) if references else None
    # Two text positions followed by one image_pad slot per reference and target.
    mask = torch.tensor([[False, False] + [True] * (references + 1)])
    condition = {"encoder_hidden_states": torch.randn(1, 2 + references, 16, dtype=torch.bfloat16),
                 "encoder_hidden_states_mask": torch.ones(1, 2 + references, dtype=torch.bool),
                 "img_mask": mask, "img_shapes": [[(1, 2, 2)] * (references + 1)],
                 "reference_latents": refs}
    target, sigma = torch.randn(1, 4, 64), torch.tensor([.4])
    direct = model(hidden_states=torch.cat([refs, target.to(torch.bfloat16)], 1)
                   if references else target.to(torch.bfloat16), timestep=sigma,
                   **{k: v for k, v in condition.items() if k != "reference_latents"},
                   return_dict=False)[0][:, -4:].float()
    adapted = predict_velocity(model, target, sigma, condition)
    torch.testing.assert_close(adapted, direct)
    assert adapted.shape == target.shape
    adapted.square().mean().backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters())
