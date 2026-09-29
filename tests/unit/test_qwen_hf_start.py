import pytest

from verl_distill.algorithms.dmd.qwen_image21 import UpdateState
from verl_distill.models.qwen_image21.configuration import validate_qwen_config


def test_hf_only_config_and_five_to_one_schedule(qwen_hf_config):
    config = qwen_hf_config
    validate_qwen_config(config)
    assert config["method"]["params"]["reflow_updates"] == 0
    assert config["method"]["params"]["teacher_cfg_scale"] == 2.0
    assert not config["runtime"].get("resume_from")
    assert not config["runtime"].get("init_reflow_from")
    state = UpdateState()
    state.validate(0, 3000)
    assert state.next_phase(0, 3000) == "fake_score"
    state.dmd_initialized = True
    for _ in range(5):
        assert state.next_phase(0, 3000) == "fake_score"
        state.advance("fake_score")
        state.validate(0, 3000)
    assert state.next_phase(0, 3000) == "generator"
    state.advance("generator")
    state.validate(0, 3000)
    assert state.state_dict() == dict(
        reflow_updates=0, fake_updates=5, generator_updates=1, dmd_initialized=True
    )
    assert state.next_phase(0, 3000) == "fake_score"


def test_negative_reflow_budget_is_invalid(qwen_hf_config):
    config = qwen_hf_config
    config["method"]["params"]["reflow_updates"] = -1
    with pytest.raises(ValueError, match="nonnegative"):
        validate_qwen_config(config)
