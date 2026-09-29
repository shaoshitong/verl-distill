import copy

import pytest

from verl_distill.engine.qwen_checkpoint import (
    condition_cache_rebuild_contract_compatible as compatible,
)


@pytest.fixture
def contract():
    return {
        "condition_index": "old",
        "manifest_hashes": {"train": "train", "eval": "eval"},
        "model_identity": "model",
        "teacher_guidance": {
            "scale": 4.0,
            "negative_index": "old_neg",
            "reference_images_preserved": True,
        },
        "runtime": {
            "offload_inactive_for_fake": True,
            "offload_scores_for_generator_backward": True,
            "gradient_accumulation_steps": 1,
        },
        "optimizer": {"generator": {"lr": 5e-7}},
        "params": {"nfe": 6},
    }


def changed(c):
    n = copy.deepcopy(c)
    n["condition_index"] = "new"
    n["teacher_guidance"]["negative_index"] = "new_neg"
    return n


def test_requires_explicit_cache_authorization(contract):
    assert not compatible(contract, changed(contract))
    assert compatible(contract, changed(contract), enabled=True)


def test_offload_change_requires_infra_authorization(contract):
    n = changed(contract)
    n["runtime"]["offload_inactive_for_fake"] = False
    assert not compatible(contract, n, enabled=True)
    assert compatible(contract, n, enabled=True, allow_infra_change=True)


@pytest.mark.parametrize(
    "section,key,value",
    [
        ("manifest_hashes", "train", "different"),
        ("teacher_guidance", "scale", 8.0),
        ("teacher_guidance", "reference_images_preserved", False),
        ("runtime", "gradient_accumulation_steps", 4),
        ("params", "nfe", 4),
    ],
)
def test_other_training_changes_still_rejected(contract, section, key, value):
    n = changed(contract)
    n[section][key] = value
    assert not compatible(contract, n, enabled=True, allow_infra_change=True)


def test_model_and_optimizer_changes_rejected(contract):
    n = changed(contract)
    n["model_identity"] = "different"
    assert not compatible(contract, n, enabled=True, allow_infra_change=True)
    n = changed(contract)
    n["optimizer"]["generator"]["lr"] = 1e-6
    assert not compatible(contract, n, enabled=True, allow_infra_change=True)
