from verl_distill.algorithms.dmd import FullModelDMD
from verl_distill.algorithms.meanflow import ZImageMeanFlow
from verl_distill.algorithms.opd_gan import DualDistilledDiscriminatorOPD
from verl_distill.config import load_config


def test_dmd_recipe_preserves_reference_defaults(monkeypatch):
    monkeypatch.setenv("ZIMAGE_MODEL_PATH", "unused")
    monkeypatch.setenv("ZIMAGE_TEACHER_MODEL_PATH", "unused")
    monkeypatch.setenv("ZIMAGE_FAKE_SCORE_MODEL_PATH", "unused")
    monkeypatch.setenv("TRAIN_MANIFEST", "unused")
    monkeypatch.setenv("TRAIN_IMAGE_ROOT", "unused")
    params = load_config("configs/recipes/zimage/dmd.yaml")["method"]["params"]
    method = FullModelDMD(**params)
    assert method.num_denoising_step == 4
    assert method.timestep_shift == 5.0
    assert method.dfake_gen_update_ratio == 5
    assert method.fake_score_use_generator_timestep is True


def test_meanflow_recipe_preserves_reference_defaults(monkeypatch):
    monkeypatch.setenv("ZIMAGE_MODEL_PATH", "unused")
    monkeypatch.setenv("TRAIN_MANIFEST", "unused")
    monkeypatch.setenv("TRAIN_IMAGE_ROOT", "unused")
    params = load_config("configs/recipes/zimage/meanflow.yaml")["method"]["params"]
    method = ZImageMeanFlow(**params)
    assert method.fd_delta == 0.005
    assert method.flow_shift == 3.0
    assert method.rt_curriculum_steps == 50000


def test_opd_recipe_preserves_reference_defaults(monkeypatch):
    monkeypatch.setenv("ZIMAGE_MODEL_PATH", "unused")
    monkeypatch.setenv("TRAIN_MANIFEST", "unused")
    monkeypatch.setenv("DISCRIMINATOR_CHECKPOINT", "unused")
    monkeypatch.setenv("FROZEN_DISCRIMINATOR_CHECKPOINT", "unused")
    params = load_config("configs/recipes/zimage/opd_gan.yaml")["method"]["params"]
    method = DualDistilledDiscriminatorOPD(**params)
    assert method.num_student_steps == 4
    assert method.teacher_micro_steps == 2
    assert method.timestep_shift == 3.0
