import torch

from verl_distill.models.zimage.checkpoints import load_component_checkpoint


def test_load_component_checkpoint_accepts_prefixed_keys(tmp_path):
    source = torch.nn.Linear(3, 2)
    target = torch.nn.Linear(3, 2)
    path = tmp_path / "head.pt"
    torch.save(
        {
            f"teacher_discriminator_model.head.{key}": value
            for key, value in source.state_dict().items()
        },
        path,
    )
    result = load_component_checkpoint(path, target, source_markers=("head",))
    assert result == {"matched": 2, "missing": []}
    for key, value in source.state_dict().items():
        torch.testing.assert_close(target.state_dict()[key], value)


def test_load_component_checkpoint_supports_dual_warm_start_aliases(tmp_path):
    class Target(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.align_norm = torch.nn.LayerNorm(3)
            self.gan_projector = torch.nn.Linear(3, 1)

    source_norm = torch.nn.LayerNorm(3)
    source_norm.weight.data.fill_(2.0)
    path = tmp_path / "legacy.pt"
    torch.save(
        {
            "out_norm.weight": source_norm.weight,
            "out_norm.bias": source_norm.bias,
        },
        path,
    )
    target = Target()
    original_gan = target.gan_projector.weight.detach().clone()
    result = load_component_checkpoint(
        path,
        target,
        key_aliases={"align_norm": "out_norm"},
        strict=False,
    )
    assert result["matched"] == 2
    torch.testing.assert_close(target.align_norm.weight, source_norm.weight)
    torch.testing.assert_close(target.gan_projector.weight, original_gan)
