import importlib.util
import json
from pathlib import Path

import pytest
import yaml

from verl_distill.algorithms import build_algorithm
from verl_distill.config import load_config
from verl_distill.trainers.dmd import _tracked_stats_keys

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_cross", REPO / "scripts/prepare_dmd_cross_ablation.py"
)
prepare_module = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare_module)


@pytest.fixture
def resource_env(monkeypatch):
    for key in (
        "ZIMAGE_MODEL_PATH",
        "ZIMAGE_TEACHER_MODEL_PATH",
        "ZIMAGE_FAKE_SCORE_MODEL_PATH",
        "TRAIN_LANCE_DATA_DIR",
        "ZIMAGE_ODE_PAIR_DIR",
        "RESUME_FROM",
    ):
        monkeypatch.setenv(key, "/example/" + key.lower())


def test_cross_recipes_select_requested_objectives(resource_env):
    for recipe, feature_score in [
        ("dmd_cross_fake0_gen0to5", False),
        ("dmd_cross_fake0to5_gen0", True),
    ]:
        config = load_config(recipe)
        method = build_algorithm(config["method"]["name"], config["method"]["params"])
        assert method.score_objective == ("teacher_feature_mse" if feature_score else "mse")
        assert method.generator_teacher_feature_anchor == "live"
        assert method.teacher_feature_representation_keys() == (
            "latent",
            "layer_1",
            "layer_2",
            "layer_3",
            "layer_4",
            "layer_5",
        )
        assert method.generator_teacher_feature_weights["latent"] == 1
        for i in range(1, 6):
            assert method.generator_teacher_feature_weights[f"layer_{i}"] == (
                0 if feature_score else 1
            )
            assert f"gen/teacher_feature_loss_layer_{i}" in _tracked_stats_keys(method)
        assert config["optimizer"]["generator"]["lr"] == 5e-5
        assert config["optimizer"]["generator"]["betas"] == [0, 0.999]
        assert config["runtime"]["resume_optimizer_state"] is False


def test_prepare_resolves_new_paths_and_refuses_overwrite(resource_env, tmp_path):
    output = tmp_path / "new run"
    prepare_module.prepare(output)
    assert not (output / "queue_state.json").exists()  # Preparation did not launch training.
    assert json.loads((output / "queue_settings.json").read_text())["gpu_occupier_script"] is None
    for name in prepare_module.JOBS:
        config = yaml.safe_load((output / name / "resolved_config.yaml").read_text())
        assert config["runtime"]["output_dir"] == str(output / name)
        assert config["runtime"]["resume_from"] == "/example/resume_from"
        assert (output / name / "train.sh").is_file()
    with pytest.raises(FileExistsError):
        prepare_module.prepare(output)
