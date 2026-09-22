"""Atomic, same-world-size FSDP1 phase checkpoints using the framework's DCP helpers."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist

from verl_distill.data.qwen_image21 import atomic_json, sha256, within
from verl_distill.engine.checkpoint import (
    load_distributed_training_state, restore_rng_state as restore_legacy_rng_state,
    save_distributed_training_state,
)


def capture_rng_state():
    """One GPU per rank: do not initialize CUDA contexts on the other seven GPUs."""
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(), "cuda_local": torch.cuda.get_rng_state()}


def restore_rng_state(state):
    if "cuda_local" not in state:
        restore_legacy_rng_state(state)
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    torch.cuda.set_rng_state(state["cuda_local"])


def collective_call(label, function):
    """For local IO/validation only: all ranks decide before entering the next collective."""
    error, value = None, None
    try:
        value = function()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = [None] * dist.get_world_size()
    dist.all_gather_object(errors, error)
    if any(errors):
        raise RuntimeError(f"{label}: " + "; ".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e))
    return value


def save_checkpoint(path, models, optimizers, state, cursor, contract, config, *, rank):
    path = Path(path)
    staging = path.with_name(path.name + ".incomplete")

    def reserve():
        if rank == 0:
            if path.exists() or staging.exists():
                raise FileExistsError(f"Checkpoint path already exists: {path}")
            staging.mkdir(parents=True)
    collective_call("reserve checkpoint", reserve)
    for name in models:
        # All ranks call DCP in exactly this model order.
        save_distributed_training_state(
            staging / name, models[name], optimizers[name],
            step=state.reflow_updates + state.fake_updates + state.generator_updates,
            save_rng_sidecar=False)
    rng = capture_rng_state()

    def sidecar():
        # Local IO failures must be gathered before any rank advances to commit.
        with (staging / f"rank-{rank:05d}.pt").open("wb") as handle:
            torch.save({"cursor": cursor.state_dict(), "rng": rng}, handle)
            handle.flush()
            os.fsync(handle.fileno())
    collective_call("checkpoint sidecars", sidecar)

    def commit():
        if rank == 0:
            for name in models:
                if not (staging / name / ".metadata").is_file():
                    raise FileNotFoundError(f"Missing DCP metadata for {name}")
            for worker in range(dist.get_world_size()):
                if not (staging / f"rank-{worker:05d}.pt").is_file():
                    raise FileNotFoundError(f"Missing rank {worker} cursor/RNG")
            files = {str(p.relative_to(staging)): p.stat().st_size
                     for p in staging.rglob("*") if p.is_file()}
            if any(size <= 0 for size in files.values()):
                raise ValueError("Empty checkpoint file")
            atomic_json(staging / "state.json", {"updates": state.state_dict(),
                        "world_size": dist.get_world_size(), "contract": contract,
                        "models": list(models), "config": config, "files": files})
            atomic_json(staging / "COMPLETE", {"updates": state.state_dict(),
                        "state_sha256": sha256(staging / "state.json")})
            os.rename(staging, path)
    collective_call("commit checkpoint", commit)
    return path


def infrastructure_contract_compatible(saved, current):
    # Explicitly allow only the new sampler and attention implementation. Every
    # optimizer/data/model/GA/phase setting remains protected by exact equality.
    import copy
    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    for value in (a, b):
        for key in ("data_ordering", "bucket_batches", "attention_backend", "offload_scores_for_generator_backward", "offload_inactive_for_fake", "early_dmd_checkpoint_steps"):
            value.get("runtime", {}).pop(key, None)
    return a == b


def refinement_contract_compatible(saved, current, plan, *, allow_infra_change=False):
    """Explicit REFLOW refinement fork; all unrequested training fields stay locked."""
    import copy
    if not plan or saved["updates"] != {"reflow_updates": plan["source_step"],
            "fake_updates": 0, "generator_updates": 0, "dmd_initialized": False}:
        return False
    old = copy.deepcopy(saved["contract"])
    old["params"]["reflow_updates"] = plan["source_step"] + plan["updates"]
    old["runtime"]["reflow_gradient_accumulation_steps"] = plan["ga"]
    old["runtime"]["reflow_checkpoint_steps"] = [plan["source_step"] + plan["updates"]]
    old["optimizer"]["reflow"]["lr"] = float(old["optimizer"]["reflow"]["lr"]) / plan["lr_divisor"]
    if plan.get("replace_training_data", False):
        old["manifest_hashes"]["train"] = current["manifest_hashes"]["train"]
        old["condition_index"] = current["condition_index"]
    return (infrastructure_contract_compatible(old, current) if allow_infra_change else old == current)


def dmd_fork_contract_compatible(saved, current, enabled):
    """Fork only a completed REFLOW checkpoint; permit DMD initialization/LR changes."""
    import copy
    if not enabled:
        return False
    old = copy.deepcopy(saved["contract"])
    if saved["updates"] != {"reflow_updates": old["params"]["reflow_updates"],
            "fake_updates": 0, "generator_updates": 0, "dmd_initialized": False}:
        return False
    if current["params"].get("fake_initialization") not in ("hf", "reflow_generator"):
        return False
    old["params"]["fake_initialization"] = current["params"]["fake_initialization"]
    for name in ("generator", "fake_score"):
        old["optimizer"][name]["lr"] = current["optimizer"][name]["lr"]
    return old == current


def inspect_checkpoint(path, contract, *, allow_infra_change=False, refinement=None, dmd_fork=False):
    path = Path(path)
    if not (path / "COMPLETE").is_file() or path.name.endswith(".incomplete"):
        raise ValueError("Refusing incomplete checkpoint")
    saved = json.loads((path / "state.json").read_text())
    complete = json.loads((path / "COMPLETE").read_text())
    if complete.get("state_sha256") and sha256(path / "state.json") != complete["state_sha256"]:
        raise ValueError("Checkpoint state digest mismatch")
    for relative, size in saved.get("files", {}).items():
        file = within(path, relative)
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"Missing or truncated checkpoint file: {file}")
    if saved["world_size"] != dist.get_world_size():
        raise ValueError("Qwen resume currently supports the same world size only")
    if saved["contract"] != contract and not (allow_infra_change and
            infrastructure_contract_compatible(saved["contract"], contract)) and not refinement_contract_compatible(
                saved, contract, refinement, allow_infra_change=allow_infra_change) and not dmd_fork_contract_compatible(saved, contract, dmd_fork):
        raise ValueError("Checkpoint runtime/data/model/config contract changed")
    return saved


def restore_checkpoint(path, models, optimizers, cursor, *, rank, reset_data_cursor=False):
    for name in models:
        # The top-level sidecar is authoritative; restore RNG only after all models.
        load_distributed_training_state(Path(path) / name, models[name], optimizers[name],
                                        restore_rng_sidecar=False)
    extra = collective_call("load resume cursor/RNG", lambda: torch.load(
        Path(path) / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False))
    if not reset_data_cursor:
        cursor.load_state_dict(extra["cursor"])
    # Restore once after all model/optimizer loading (which may consume RNG).
    restore_rng_state(extra["rng"])
