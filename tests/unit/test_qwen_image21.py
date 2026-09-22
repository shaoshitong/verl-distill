"""CPU regression tests. Added with the implementation; not executed during delivery."""
import copy
from pathlib import Path

import pytest
import torch

from verl_distill.algorithms.dmd.qwen_image21 import (
    UpdateState, dmd_surrogate, epsilon_from_velocity, fake_score_loss,
    reflow_loss, renoise, x0_from_velocity,
)
from verl_distill.config import load_config, validate_config
from verl_distill.data.qwen_image21 import (
    QwenPairDataset, RankCursor, atomic_json, canonical_hash, sha256, validate_complete, within,
)
from verl_distill.models.qwen_image21.configuration import MODEL_REVISION


@pytest.mark.parametrize("sigma", [.02, .4, .5, .98, 1.])
def test_flow_endpoints(sigma):
    clean, noise = torch.randn(2, 7, 64), torch.randn(2, 7, 64)
    s = torch.full((2,), sigma)
    noisy, velocity = renoise(clean, noise, s), noise - clean
    torch.testing.assert_close(x0_from_velocity(noisy, velocity, s), clean)
    torch.testing.assert_close(epsilon_from_velocity(noisy, velocity, s), noise)
    assert reflow_loss(velocity, clean, noise).item() == 0


@pytest.mark.parametrize("sigma", [.02, .5, .98])
def test_epsilon_equivalent_loss_and_gradient(sigma):
    target, noise = torch.randn(1, 8, 64), torch.randn(1, 8, 64)
    s = torch.tensor([sigma])
    noisy = renoise(target, noise, s)
    velocity = torch.randn_like(noisy, requires_grad=True)
    uncapped, _ = fake_score_loss(noisy, velocity, target, noise, s, max_weight=1e9)
    direct = (epsilon_from_velocity(noisy, velocity, s) - noise).square().mean()
    torch.testing.assert_close(uncapped, direct, atol=2e-5, rtol=2e-5)
    g1 = torch.autograd.grad(uncapped, velocity, retain_graph=True)[0]
    g2 = torch.autograd.grad(direct, velocity)[0]
    torch.testing.assert_close(g1, g2, atol=2e-6, rtol=2e-5)


def test_cap_is_explicit_and_target_is_detached():
    target = torch.ones(1, 2, 4, requires_grad=True)
    noise = torch.zeros_like(target)
    s = torch.tensor([.02])
    noisy = renoise(target, noise, s)
    velocity = torch.zeros_like(noisy, requires_grad=True)
    loss, stats = fake_score_loss(noisy, velocity, target, noise, s)
    assert stats["weight_raw"].item() == pytest.approx(2401.)
    assert stats["weight"].item() == 50
    assert stats["cap_fraction"].item() == 1
    torch.testing.assert_close(loss, 50 * (stats["fake_x0"] - target.detach()).square().mean())
    loss.backward()
    assert velocity.grad is not None
    assert target.grad is None


def test_dmd_gradient_sign_scale_and_score_freeze():
    generated = torch.tensor([[[1., 2.]], [[4., 7.]]], requires_grad=True)
    fake_v = torch.zeros_like(generated, requires_grad=True)
    real_v = torch.ones_like(generated, requires_grad=True)
    s = torch.tensor([.4, .8])
    noisy = generated.detach() + 1
    loss, aux = dmd_surrogate(generated, noisy, fake_v, real_v, s)
    loss.backward()
    expected = 4 * aux["normalized_direction"] / generated.numel()
    torch.testing.assert_close(generated.grad, expected)
    assert (generated.grad > 0).all()  # descent moves toward real_x0
    assert fake_v.grad is None and real_v.grad is None
    torch.testing.assert_close(aux["diff_x0"], aux["fake_x0"] - aux["real_x0"])


def test_schedule_count_resume_and_final_generator():
    state, phases = UpdateState(), []
    while (phase := state.next_phase(1000, 3000)) != "complete":
        if phase != "reflow":
            state.dmd_initialized = True
        phases.append(phase)
        state.advance(phase)
        # Simulate counter serialization at every update boundary.
        state = UpdateState(**state.state_dict())
        state.validate(1000, 3000)
    assert phases[:1000] == ["reflow"] * 1000
    assert phases[1000:] == (["fake_score"] * 5 + ["generator"]) * 600
    assert state.fake_updates == 3000 and state.generator_updates == 600
    assert phases[-1] == "generator"
    with pytest.raises(ValueError):
        UpdateState(1000, 5, 2, True).validate(1000, 3000)


def test_cursor_same_world_resume_and_rank_partition():
    cursors = [RankCursor(17, rank, 4, 42) for rank in range(4)]
    samples = [[cursor.next_index() for _ in range(4)] for cursor in cursors]
    assert len(set(sum(samples, []))) == 16
    saved = cursors[2].state_dict()
    expected = [cursors[2].next_index() for _ in range(20)]
    restored = RankCursor(17, 2, 4, 42)
    restored.load_state_dict(saved)
    assert [restored.next_index() for _ in range(20)] == expected
    with pytest.raises(ValueError, match="world_size"):
        RankCursor(17, 2, 3, 42).load_state_dict(saved)


def test_recipe_counts_ga_and_rejected_overrides(monkeypatch):
    for name in ("QWEN21_TRAIN_MANIFEST", "QWEN21_EVAL_MANIFEST", "QWEN21_CONDITION_CACHE", "OUTPUT_DIR"):
        monkeypatch.setenv(name, "/unused/" + name)
    recipe = Path(__file__).resolve().parents[2] / "configs/recipes/qwen_image21/reflow_dmd_fsdp1.yaml"
    config = load_config(recipe)
    assert config["method"]["params"]["fake_updates"] == 3000
    assert config["runtime"]["reflow_gradient_accumulation_steps"] == 1
    assert config["runtime"]["gradient_accumulation_steps"] == 4
    for section, key, bad in (("runtime", "gradient_accumulation_steps", 1),
                              ("distributed", "fsdp_backend", "fsdp2")):
        edited = copy.deepcopy(config)
        edited[section][key] = bad
        with pytest.raises(ValueError):
            validate_config(edited)


def test_complete_manifest_pack_hash_and_missing_reference(tmp_path):
    from PIL import Image

    root, refs = tmp_path / "outputs", tmp_path / "refs"
    directory = root / "edit" / "ab" / "abcd"
    directory.mkdir(parents=True)
    refs.mkdir()
    Image.new("RGBA", (32, 32), "red").save(refs / "first.png")
    Image.new("RGBA", (32, 32), "blue").save(refs / "second.png")
    Image.new("RGBA", (32, 32)).save(directory / "image.png")
    clean = torch.arange(256).reshape(1, 64, 1, 2, 2).float()
    torch.save(clean, directory / "x0_latent.pt")
    torch.save(-clean, directory / "initial_noise.pt")
    payloads = {name: sha256(directory / name) for name in
                ("image.png", "initial_noise.pt", "x0_latent.pt")}
    meta = {"id": "abcd", "kind": "edit", "prompt": "Use image 2 then image 1", "seed": 7,
            "width": 32, "height": 32, "reference_images": ["first.png", "second.png"],
            "model_revision": MODEL_REVISION, "num_inference_steps": 40,
            "true_cfg_scale": 1., "vae_tiling": False, "model_cpu_offload": True,
            "use_kv_cache": True, "sha256": payloads}
    atomic_json(directory / "complete.json", meta)
    row = validate_complete(directory / "complete.json", root, refs)
    assert row["reference_images"] == ["first.png", "second.png"]
    assert row["reference_sha256"] == [sha256(refs / name) for name in row["reference_images"]]
    manifest = tmp_path / "train.json"
    atomic_json(manifest, {"schema": 1, "purpose": "train", "output_root": str(root),
                          "records": [row], "records_sha256": canonical_hash([row])})
    sample = QwenPairDataset(manifest)[0]
    torch.testing.assert_close(sample["clean"].transpose(1, 2).reshape_as(clean), clean)
    torch.testing.assert_close(sample["noise"], -sample["clean"])
    (refs / "second.png").unlink()
    with pytest.raises(FileNotFoundError):
        validate_complete(directory / "complete.json", root, refs)
    torch.save(clean + 1, directory / "x0_latent.pt")
    with pytest.raises(ValueError, match="payload changed"):
        QwenPairDataset(manifest)[0]
    with pytest.raises(ValueError, match="escapes"):
        within(root, "../outside")
