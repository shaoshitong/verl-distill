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
    load_distributed_model_state,
    load_distributed_training_state,
    save_distributed_training_state,
)
from verl_distill.engine.checkpoint import (
    restore_rng_state as restore_legacy_rng_state,
)


def capture_rng_state():
    """One GPU per rank: do not initialize CUDA contexts on the other seven GPUs."""
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda_local": torch.cuda.get_rng_state(),
    }


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
        raise RuntimeError(
            f"{label}: " + "; ".join(f"rank {i}: {e}" for i, e in enumerate(errors) if e)
        )
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
            staging / name,
            models[name],
            optimizers[name],
            step=state.reflow_updates + state.fake_updates + state.generator_updates,
            save_rng_sidecar=False,
        )
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
            files = {
                str(p.relative_to(staging)): p.stat().st_size
                for p in staging.rglob("*")
                if p.is_file()
            }
            if any(size <= 0 for size in files.values()):
                raise ValueError("Empty checkpoint file")
            atomic_json(
                staging / "state.json",
                {
                    "updates": state.state_dict(),
                    "world_size": dist.get_world_size(),
                    "contract": contract,
                    "models": list(models),
                    "config": config,
                    "files": files,
                },
            )
            atomic_json(
                staging / "COMPLETE",
                {"updates": state.state_dict(), "state_sha256": sha256(staging / "state.json")},
            )
            os.rename(staging, path)

    collective_call("commit checkpoint", commit)
    return path


def infrastructure_contract_compatible(saved, current):
    # Explicitly allow only the new sampler and attention implementation. Every
    # optimizer/data/model/GA/phase setting remains protected by exact equality.
    import copy

    a, b = copy.deepcopy(saved), copy.deepcopy(current)
    for value in (a, b):
        for key in (
            "data_ordering",
            "bucket_batches",
            "attention_backend",
            "offload_scores_for_generator_backward",
            "offload_inactive_for_fake",
            "early_dmd_checkpoint_steps",
            "reflow_checkpoint_steps",
        ):
            value.get("runtime", {}).pop(key, None)
    return a == b


def refinement_contract_compatible(saved, current, plan, *, allow_infra_change=False):
    """Explicit REFLOW refinement fork; all unrequested training fields stay locked."""
    import copy

    if not plan or saved["updates"] != {
        "reflow_updates": plan["source_step"],
        "fake_updates": 0,
        "generator_updates": 0,
        "dmd_initialized": False,
    }:
        return False
    old = copy.deepcopy(saved["contract"])
    old["params"]["reflow_updates"] = plan["source_step"] + plan["updates"]
    old["runtime"]["reflow_gradient_accumulation_steps"] = plan["ga"]
    old["runtime"]["reflow_checkpoint_steps"] = [plan["source_step"] + plan["updates"]]
    old["optimizer"]["reflow"]["lr"] = float(old["optimizer"]["reflow"]["lr"]) / plan["lr_divisor"]
    if plan.get("replace_training_data", False):
        old["manifest_hashes"]["train"] = current["manifest_hashes"]["train"]
        old["condition_index"] = current["condition_index"]
    return (
        infrastructure_contract_compatible(old, current) if allow_infra_change else old == current
    )


def dmd_fork_contract_compatible(saved, current, enabled):
    """Fork only a completed REFLOW checkpoint; permit DMD initialization/LR changes."""
    import copy

    if not enabled:
        return False
    old = copy.deepcopy(saved["contract"])
    if saved["updates"] != {
        "reflow_updates": old["params"]["reflow_updates"],
        "fake_updates": 0,
        "generator_updates": 0,
        "dmd_initialized": False,
    }:
        return False
    if current["params"].get("fake_initialization") not in ("hf", "reflow_generator"):
        return False
    old["params"]["fake_initialization"] = current["params"]["fake_initialization"]
    # Explicit DMD fork from REFLOW: remove the legacy 4x Generator loss gain.
    # This exception never applies to continuation of an existing DMD phase.
    if current["params"].get("generator_loss_weight") == 1.0:
        old["params"]["generator_loss_weight"] = 1.0
    if current["params"].get("fake_loss") == "velocity_mse":
        old["params"]["fake_loss"] = "velocity_mse"
    for name in ("generator", "fake_score"):
        old["optimizer"][name]["lr"] = current["optimizer"][name]["lr"]
    old["optimizer"]["generator"]["betas"] = current["optimizer"]["generator"].get("betas")
    if "betas" not in current["optimizer"]["generator"]:
        old["optimizer"]["generator"].pop("betas", None)
    if current.get("runtime", {}).get("train_transformer_blocks_only") is True:
        old["runtime"]["train_transformer_blocks_only"] = True
        old["trainable_policy"] = "dit_transformer_blocks_only_v1"
    if current["params"].get("dmd_surrogate_dtype") == "float64":
        old["params"]["dmd_surrogate_dtype"] = "float64"
    return old == current


def inspect_reflow_initialization(
    path, contract, *, allow_data_extension=False, current_records=None
):
    """Explicit weights-only fork; allow DCP resharding, never restore old rank RNG/cursor."""
    path = Path(path)
    if path.name.endswith(".incomplete") or not (path / "COMPLETE").is_file():
        raise ValueError("Refusing incomplete REFLOW initialization")
    saved = json.loads((path / "state.json").read_text())
    complete = json.loads((path / "COMPLETE").read_text())
    if complete.get("state_sha256") != sha256(path / "state.json"):
        raise ValueError("REFLOW initialization state digest mismatch")
    for relative, size in saved.get("files", {}).items():
        file = within(path, relative)
        if not file.is_file() or file.stat().st_size != size:
            raise ValueError(f"Missing or truncated initialization file: {file}")
    old = saved["contract"]
    completed = saved["updates"]["reflow_updates"]
    if not (0 < completed <= old["params"]["reflow_updates"]):
        raise ValueError("Invalid source REFLOW progress")
    if completed != contract["params"]["reflow_updates"]:
        raise ValueError("Target REFLOW boundary must match source completed updates")
    expected = {
        "reflow_updates": completed,
        "fake_updates": 0,
        "generator_updates": 0,
        "dmd_initialized": False,
    }
    if saved["updates"] != expected or saved["models"] != ["generator"]:
        raise ValueError("Weights-only initialization requires a completed REFLOW-only checkpoint")
    for key in ("model_identity", "scheduler", "fsdp_use_orig_params"):
        if old[key] != contract[key]:
            raise ValueError(f"REFLOW initialization contract mismatch: {key}")
    if old["manifest_hashes"] != contract["manifest_hashes"]:
        if not allow_data_extension or current_records is None:
            raise ValueError("REFLOW initialization contract mismatch: manifest_hashes")
        if old["manifest_hashes"]["eval"] != contract["manifest_hashes"]["eval"]:
            raise ValueError("Data extension may not change evaluation")
        from verl_distill.data.qwen_image21 import read_manifest

        previous_path = saved["config"]["data"]["manifest"]
        if sha256(previous_path) != old["manifest_hashes"]["train"]:
            raise ValueError("Original training manifest changed")
        previous = read_manifest(previous_path)["records"]
        current = {(r["kind"], r["id"]): r for r in current_records}
        if any(current.get((r["kind"], r["id"])) != r for r in previous):
            raise ValueError("Data extension must preserve all original records unchanged")
    for key in ("nfe", "generator_terminal", "generator_input"):
        # Weights-only DMD fork: the REFLOW source remains trained on paired
        # data, while the explicitly configured new DMD input may be rollout.
        if key == "generator_input" and old["params"][key] == "renoised_data" and contract["params"][key] == "rollout_dataset_noise":
            continue
        if old["params"][key] != contract["params"][key]:
            raise ValueError(f"REFLOW initialization recipe mismatch: {key}")
    if not (path / "generator" / ".metadata").is_file():
        raise ValueError("Missing Generator DCP metadata")
    return saved


def initialize_reflow_weights(path, generator):
    # DCP maps the saved tensor shards onto the current FSDP world size.
    rng = capture_rng_state()
    load_distributed_model_state(Path(path) / "generator", generator)
    restore_rng_state(rng)


def condition_cache_rebuild_contract_compatible(
    saved, current, *, enabled=False, allow_infra_change=False
):
    """Explicit cache regeneration: keep data/model/optimization identities locked."""
    if not enabled:
        return False
    import copy

    old = copy.deepcopy(saved)
    if "condition_index" not in old or "condition_index" not in current:
        return False
    old["condition_index"] = current["condition_index"]
    if "teacher_guidance" in old and "teacher_guidance" in current:
        if (
            "negative_index" in old["teacher_guidance"]
            and "negative_index" in current["teacher_guidance"]
        ):
            old["teacher_guidance"]["negative_index"] = current["teacher_guidance"][
                "negative_index"
            ]
    return (
        infrastructure_contract_compatible(old, current) if allow_infra_change else old == current
    )


def inspect_checkpoint(
    path,
    contract,
    *,
    allow_infra_change=False,
    refinement=None,
    dmd_fork=False,
    allow_condition_cache_rebuild=False,
):
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
    if (
        saved["contract"] != contract
        and not (
            allow_infra_change and infrastructure_contract_compatible(saved["contract"], contract)
        )
        and not condition_cache_rebuild_contract_compatible(
            saved["contract"],
            contract,
            enabled=allow_condition_cache_rebuild,
            allow_infra_change=allow_infra_change,
        )
        and not refinement_contract_compatible(
            saved, contract, refinement, allow_infra_change=allow_infra_change
        )
        and not dmd_fork_contract_compatible(saved, contract, dmd_fork)
    ):
        raise ValueError("Checkpoint runtime/data/model/config contract changed")
    return saved


def restore_checkpoint(
    path, models, optimizers, cursor, *, rank, reset_data_cursor=False, model_only=False
):
    for name in models:
        # The top-level sidecar is authoritative; restore RNG only after all models.
        if model_only:
            load_distributed_model_state(Path(path) / name, models[name])
        else:
            load_distributed_training_state(
                Path(path) / name, models[name], optimizers[name], restore_rng_sidecar=False
            )
    extra = collective_call(
        "load resume cursor/RNG",
        lambda: torch.load(
            Path(path) / f"rank-{rank:05d}.pt", map_location="cpu", weights_only=False
        ),
    )
    if not reset_data_cursor:
        cursor.load_state_dict(extra["cursor"])
    # Restore once after all model/optimizer loading (which may consume RNG).
    restore_rng_state(extra["rng"])
