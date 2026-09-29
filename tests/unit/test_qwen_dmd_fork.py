import copy

from verl_distill.engine.qwen_checkpoint import dmd_fork_contract_compatible


def test_dmd_fork_allows_only_requested_changes_at_reflow_boundary():
    contract = {"params": {"reflow_updates": 100, "fake_initialization": "reflow_generator"},
                "optimizer": {"generator": {"lr": 5e-7}, "fake_score": {"lr": 5e-6}},
                "runtime": {"gradient_accumulation_steps": 4}, "manifest_hashes": {"train": "abc"}}
    saved = {"contract": contract, "updates": {"reflow_updates": 100,
             "fake_updates": 0, "generator_updates": 0, "dmd_initialized": False}}
    current = copy.deepcopy(contract)
    current["params"]["fake_initialization"] = "hf"
    current["optimizer"]["generator"]["lr"] = 2e-6
    current["optimizer"]["fake_score"]["lr"] = 8e-6
    assert dmd_fork_contract_compatible(saved, current, True)
    blocks = copy.deepcopy(current)
    blocks["runtime"]["train_transformer_blocks_only"] = True
    blocks["trainable_policy"] = "dit_transformer_blocks_only_v1"
    blocks["params"]["dmd_surrogate_dtype"] = "float64"
    blocks["optimizer"]["generator"]["betas"] = [0., .999]
    assert dmd_fork_contract_compatible(saved, blocks, True)
    blocks["params"]["generator_loss_weight"] = 1.0
    assert dmd_fork_contract_compatible(saved, blocks, True)
    blocks["params"]["fake_loss"] = "velocity_mse"
    assert dmd_fork_contract_compatible(saved, blocks, True)
    assert not dmd_fork_contract_compatible(saved, current, False)
    changed = copy.deepcopy(current)
    changed["runtime"]["gradient_accumulation_steps"] = 8
    assert not dmd_fork_contract_compatible(saved, changed, True)
    changed = copy.deepcopy(current)
    changed["manifest_hashes"]["train"] = "other"
    assert not dmd_fork_contract_compatible(saved, changed, True)
    for key, value in [("reflow_updates", 99), ("fake_updates", 5), ("dmd_initialized", True)]:
        changed = copy.deepcopy(saved)
        changed["updates"][key] = value
        assert not dmd_fork_contract_compatible(changed, current, True)
