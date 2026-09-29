import pytest

from verl_distill.config import load_config


@pytest.fixture
def qwen_hf_config(monkeypatch):
    for name in (
        "QWEN21_MODEL_PATH",
        "QWEN21_TRAIN_MANIFEST",
        "QWEN21_EVAL_MANIFEST",
        "QWEN21_CONDITION_CACHE",
        "QWEN21_NEGATIVE_CONDITION_CACHE",
        "QWEN21_DEBUG_CSV",
        "OUTPUT_DIR",
    ):
        monkeypatch.setenv(name, "/unused/" + name)
    monkeypatch.delenv("QWEN21_RESUME_FROM", raising=False)
    return load_config("qwen_image21/dmd_hf_cfg2_fsdp1")
