import importlib.util
from pathlib import Path

import torch

spec = importlib.util.spec_from_file_location(
    "grad_diagnostic", Path(__file__).parents[2] / "scripts/diagnose_dmd_layer_gradients.py"
)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_sharded_gram_matches_full_vectors_and_projection_identity():
    gradients = torch.tensor(
        [[1.0, 0.0, 2.0, 0.0], [0.0, 1.0, 0.0, 2.0], [3.0, 4.0, 1.0, 1.0]], dtype=torch.float64
    )
    gradients = torch.cat([gradients, gradients.sum(0, keepdim=True)])
    rank0 = [[v[:1], v[1:2]] for v in gradients]
    rank1 = [[v[2:3], v[3:]] for v in gradients]
    gram = diagnostic.gradient_gram(rank0, chunk_size=1) + diagnostic.gradient_gram(
        rank1, chunk_size=1
    )
    torch.testing.assert_close(gram, gradients @ gradients.T)
    report = diagnostic.describe_gram(gram, ["latent", "layer_5", "layer_15", "sum"], 2.83)
    assert abs(sum(report["projection_fraction_of_total"].values()) - 1) < 1e-12
    assert report["sum_vs_joint_backward_relative_residual"] < 1e-12
    assert report["combined_clip_scale"] < report["latent_plus_layer5_clip_scale"]
    assert report["cosines"]["latent:layer_5"] == 0


def test_gram_preserves_cancellation():
    snapshots = [
        [torch.tensor([1.0, 0.0])],
        [torch.tensor([-1.0, 0.0])],
        [torch.tensor([0.0, 2.0])],
        [torch.tensor([0.0, 2.0])],
    ]
    report = diagnostic.describe_gram(
        diagnostic.gradient_gram(snapshots), ["latent", "layer_5", "layer_15", "sum"], 2.83
    )
    assert report["latent_plus_layer5_norm"] == 0
    assert report["cosines"]["latent:layer_5"] == -1
    assert report["projection_fraction_of_total"]["layer_15"] == 1
    assert report["combined_clip_scale"] == 1


def test_sf_training_weights_restore_z_not_eval_weights(tmp_path):
    import torch.distributed.checkpoint as dcp

    model = torch.nn.ModuleDict(
        {"generator": torch.nn.Linear(2, 2), "fake_score": torch.nn.Linear(2, 2)}
    )
    saved = {key: value.detach().clone() for key, value in model.state_dict().items()}
    z = {key: value + 7 for key, value in saved.items() if key.startswith("generator.")}
    dcp.save(
        {
            "model": saved,
            "step": torch.tensor(1500),
            "optimizer": {
                "state": {key: {"z": value} for key, value in z.items()},
                "param_groups": [
                    {"betas": (0.0, 0.999), "train_mode": False, "k": 83},
                    {"lr": 1e-5},
                ],
            },
        },
        checkpoint_id=tmp_path,
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    step, group = diagnostic.load_sf_training_weights(str(tmp_path), model)
    assert step == 1500 and group == {"betas": (0.0, 0.999), "train_mode": False, "k": 83}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, z[key] if key in z else saved[key])
