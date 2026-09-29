from types import SimpleNamespace

import torch

from verl_distill.trainers.dmd import _save_dmd_training_debug


def test_pair_dump_preserves_signed_diff_and_uses_only_three_decodes(tmp_path):
    calls = []
    def decode(latent):
        calls.append(latent)
        return torch.zeros(1, 3, 4, 4)
    student = SimpleNamespace(device=torch.device('cpu'), latents_to_pixels=decode)
    fake = torch.ones(1, 4, 2, 2)
    real = fake * 2
    aux = {'x_fake': fake * 0.5, 'pred_fake_x0': fake, 'pred_real_x0': real,
           'dmd_raw_diff': fake-real, 'dmd_normalized_grad': fake-real,
           'dm_real_mse': torch.tensor([2.25]), 'dm_cos_fake_real_minus_fake': torch.tensor([1.]),
           'dm_sigma': torch.tensor([0.8]), 'dm_fake_sigma': torch.tensor([0.6])}
    _save_dmd_training_debug(tmp_path, student, 1005, ['test'], fake, aux, pair_only=True)
    directory = tmp_path / 'debug_dmd_tensors' / 'step-001005'
    saved = torch.load(directory / 'tensors.pt', weights_only=True)
    torch.testing.assert_close(saved['dmd_raw_diff'], fake-real)
    assert saved["dm_real_mse"].item() == 2.25
    assert saved["dm_cos_fake_real_minus_fake"].item() == 1.0
    assert "dm_real_mse=[2.25]" in (directory / "metadata.txt").read_text()
    assert len(calls) == 3
    for name in ('fake_score_pred_x0.jpg', 'real_score_pred_x0.jpg',
                 'generator_denoised_latent.jpg', 'diff_rms_fixed_scale.png'):
        assert (directory / name).is_file()
