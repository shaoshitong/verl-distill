import pytest

from verl_distill.config import load_config, validate_config


def test_recipe_composition(monkeypatch):
    monkeypatch.setenv("ZIMAGE_MODEL_PATH", "/public/model")
    monkeypatch.setenv("TRAIN_MANIFEST", "/public/prompts.jsonl")
    monkeypatch.setenv("DISCRIMINATOR_CHECKPOINT", "/public/discriminator")
    monkeypatch.setenv("FROZEN_DISCRIMINATOR_CHECKPOINT", "/public/frozen")
    config = load_config("configs/recipes/zimage/opd_gan.yaml")
    assert config["model"]["pretrained_model"] == "/public/model"
    assert config["method"]["name"] == "opd_gan"
    assert config["method"]["params"]["num_student_steps"] == 4
    assert config["method"]["params"]["rollout_mode"] == "trajectory_bernoulli"
    assert config["optimizer"]["generator"]["lr"] == 5.0e-6
    assert config["runtime"]["gradient_accumulation_steps"] == 4
    assert config["ema"]["decay"] == 0.99
    assert config["discriminator"]["multifeature_layers"] == [4, 12, 20]


def test_builtin_recipe_alias(monkeypatch):
    monkeypatch.setenv("ZIMAGE_MODEL_PATH", "/public/model")
    monkeypatch.setenv("ZIMAGE_TEACHER_MODEL_PATH", "/public/teacher")
    monkeypatch.setenv("ZIMAGE_FAKE_SCORE_MODEL_PATH", "/public/fake-score")
    monkeypatch.setenv("TRAIN_MANIFEST", "/public/train.jsonl")
    monkeypatch.setenv("TRAIN_IMAGE_ROOT", "/public/images")
    config = load_config("dmd")
    assert config["method"]["name"] == "dmd_full"
    assert config["model"]["pretrained_model"] == "/public/model"
    assert config["model"]["teacher_model"] == "/public/teacher"
    assert config["model"]["fake_score_model"] == "/public/fake-score"


def test_dmd_full_requires_explicit_teacher_and_fake_score():
    config = {
        "method": {"name": "dmd_full", "params": {}},
        "model": {"pretrained_model": "/public/model"},
        "data": {
            "format": "image_jsonl",
            "manifest": "/public/train.jsonl",
            "image_root": "/public/images",
        },
        "runtime": {},
        "distributed": {},
        "optimizer": {},
    }
    with pytest.raises(ValueError, match="model.teacher_model"):
        validate_config(config)


def test_config_validation_rejects_unknown_sections():
    with pytest.raises(ValueError, match="Unknown top-level"):
        validate_config({"method": {}, "typo": {}})
