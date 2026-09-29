import copy

import pytest
import torch

from verl_distill.data.qwen_image21 import atomic_json, sha256
from verl_distill.engine.qwen_checkpoint import inspect_reflow_initialization
from verl_distill.models.qwen_image21.configuration import validate_qwen_config
from verl_distill.models.qwen_image21.negative_cache import NegativeConditionStore, negative_key


def test_negative_cache_reuses_text_but_retains_reference_layout_and_target_mask(tmp_path):
    identity = {"model": "pinned"}
    hashes = {"train": "a", "eval": "b"}
    row = {
        "id": "x",
        "prompt": "positive",
        "height": 64,
        "width": 32,
        "reference_images": ["a.png", "b.png"],
        "reference_sha256": ["a", "b"],
    }
    key = negative_key(row, identity)
    assert negative_key({**row, "prompt": "different", "height": 32}, identity) == key
    assert negative_key({**row, "reference_images": ["b.png", "a.png"]}, identity) != key
    payload = {
        "key": key,
        "condition": {
            "encoder_hidden_states": torch.zeros(1, 3, 8, dtype=torch.bfloat16),
            "encoder_hidden_states_mask": None,
            "img_mask": torch.tensor([[False, True, True]]),
        },
    }
    path = tmp_path / "negative.pt"
    torch.save(payload, path)
    atomic_json(
        tmp_path / "index.json",
        {
            "schema": 1,
            "model_identity": identity,
            "manifest_hashes": hashes,
            "negative_prompt": "",
            "entries": {key: {"file": "negative.pt", "sha256": sha256(path)}},
        },
    )
    store = NegativeConditionStore(tmp_path, identity, hashes)
    positive = {"reference_latents": torch.randn(1, 4, 64), "img_shapes": [[(1, 2, 2)] * 3]}
    value = store.get(row, positive, "cpu")
    assert value["reference_latents"] is positive["reference_latents"]
    assert value["img_shapes"] is positive["img_shapes"]
    assert value["img_mask"].tolist() == [[False, True, True, True, True]]
    payload["key"] = "wrong"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="identity"):
        store.get(row, positive, "cpu")


def test_reshard_initialization_accepts_only_complete_reflow_weights(tmp_path):
    old = {
        "params": {
            "reflow_updates": 100,
            "nfe": 6,
            "generator_terminal": 0.4,
            "generator_input": "renoised_data",
        },
        "model_identity": "a",
        "scheduler": "b",
        "manifest_hashes": {"train": "x"},
        "fsdp_use_orig_params": True,
    }
    saved = {
        "world_size": 8,
        "contract": old,
        "models": ["generator"],
        "files": {"generator/.metadata": 1},
        "updates": {
            "reflow_updates": 100,
            "fake_updates": 0,
            "generator_updates": 0,
            "dmd_initialized": False,
        },
    }
    (tmp_path / "generator").mkdir()
    (tmp_path / "generator/.metadata").write_bytes(b"x")

    def publish():
        atomic_json(tmp_path / "state.json", saved)
        atomic_json(tmp_path / "COMPLETE", {"state_sha256": sha256(tmp_path / "state.json")})

    publish()
    new = copy.deepcopy(old)
    new["params"].update(teacher_cfg_scale=4.0, fake_loss="velocity_mse")
    assert inspect_reflow_initialization(tmp_path, new)["world_size"] == 8
    rollout = copy.deepcopy(new)
    rollout["params"]["generator_input"] = "rollout_dataset_noise"
    assert inspect_reflow_initialization(tmp_path, rollout)["world_size"] == 8
    rollout["params"]["generator_terminal"] = .375
    with pytest.raises(ValueError, match="generator_terminal"):
        inspect_reflow_initialization(tmp_path, rollout)
    saved["contract"]["params"]["reflow_updates"] = 500
    publish()
    assert inspect_reflow_initialization(tmp_path, new)["updates"]["reflow_updates"] == 100
    wrong_boundary = copy.deepcopy(new)
    wrong_boundary["params"]["reflow_updates"] = 99
    with pytest.raises(ValueError, match="boundary"):
        inspect_reflow_initialization(tmp_path, wrong_boundary)
    saved["updates"]["fake_updates"] = 5
    publish()
    with pytest.raises(ValueError, match="REFLOW-only"):
        inspect_reflow_initialization(tmp_path, new)
    saved["updates"]["fake_updates"] = 0
    publish()
    new["model_identity"] = "other"
    with pytest.raises(ValueError, match="model_identity"):
        inspect_reflow_initialization(tmp_path, new)


@pytest.mark.parametrize("teacher_cfg", [2.0, 4.0])
def test_ga1_teacher_cfg_requires_cache_and_cannot_mix_resume(teacher_cfg, qwen_hf_config):
    config = copy.deepcopy(qwen_hf_config)
    config["method"]["params"]["teacher_cfg_scale"] = teacher_cfg
    validate_qwen_config(config)
    assert config["runtime"]["gradient_accumulation_steps"] == 1
    assert config["method"]["params"]["teacher_cfg_scale"] == teacher_cfg
    changed = copy.deepcopy(config)
    changed["data"].pop("negative_condition_cache")
    with pytest.raises(ValueError, match="negative cache"):
        validate_qwen_config(changed)
    changed = copy.deepcopy(config)
    changed["runtime"]["resume_from"] = "other"
    changed["runtime"]["init_reflow_from"] = "reflow"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_qwen_config(changed)


def test_nested_cache_reuse_preserves_pinned_payload(tmp_path):
    from verl_distill.models.qwen_image21.modeling import ConditionStore, save_condition

    identity = {"model": "same"}
    row = {"id": "same"}
    root = tmp_path / "base"
    key, entry = save_condition(root, row, identity, {"value": torch.tensor([3.0])})
    atomic_json(
        root / "index.json",
        {
            "schema": 1,
            "model_identity": identity,
            "manifest_hashes": {"train": "base"},
            "entries": {key: entry},
        },
    )
    for n in range(2):
        new = tmp_path / f"level{n}"
        new.mkdir()
        atomic_json(
            new / "index.json",
            {
                "schema": 1,
                "model_identity": identity,
                "manifest_hashes": {"train": str(n)},
                "entries": {key: {**entry, "source": "base"}},
                "reused_cache": {"root": str(root), "index_sha256": sha256(root / "index.json")},
            },
        )
        root = new
    assert ConditionStore(root, identity, {"train": "1"}).get(row, "cpu")["value"].item() == 3.0


def test_initialization_data_extension_requires_unchanged_original_records(tmp_path):
    from verl_distill.data.qwen_image21 import canonical_hash

    row = {"id": "old", "kind": "t2i", "prompt": "original"}
    train = tmp_path / "train.json"
    atomic_json(
        train,
        {
            "schema": 1,
            "purpose": "train",
            "records": [row],
            "records_sha256": canonical_hash([row]),
        },
    )
    old = {
        "params": {
            "reflow_updates": 100,
            "nfe": 6,
            "generator_terminal": 0.4,
            "generator_input": "renoised_data",
        },
        "model_identity": "m",
        "scheduler": "s",
        "manifest_hashes": {"train": sha256(train), "eval": "unchanged"},
        "fsdp_use_orig_params": True,
    }
    cp = tmp_path / "cp"
    (cp / "generator").mkdir(parents=True)
    (cp / "generator/.metadata").write_bytes(b"x")
    saved = {
        "world_size": 8,
        "contract": old,
        "models": ["generator"],
        "files": {"generator/.metadata": 1},
        "config": {"data": {"manifest": str(train)}},
        "updates": {
            "reflow_updates": 100,
            "fake_updates": 0,
            "generator_updates": 0,
            "dmd_initialized": False,
        },
    }
    atomic_json(cp / "state.json", saved)
    atomic_json(cp / "COMPLETE", {"state_sha256": sha256(cp / "state.json")})
    current = copy.deepcopy(old)
    current["manifest_hashes"]["train"] = "extended"
    rows = [row, {"id": "new", "kind": "t2i", "prompt": "new"}]
    with pytest.raises(ValueError, match="manifest_hashes"):
        inspect_reflow_initialization(cp, current, current_records=rows)
    inspect_reflow_initialization(cp, current, current_records=rows, allow_data_extension=True)
    with pytest.raises(ValueError, match="unchanged"):
        inspect_reflow_initialization(
            cp, current, current_records=[{**row, "prompt": "changed"}], allow_data_extension=True
        )
