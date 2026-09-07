from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_state_dict,
    set_model_state_dict,
    set_state_dict,
)


def capture_rng_state() -> dict[str, Any]:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


def save_training_state(path: str | Path, state: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({**state, "rng_state": capture_rng_state()}, destination)


def load_training_state(path: str | Path, *, map_location="cpu") -> dict[str, Any]:
    state = torch.load(Path(path), map_location=map_location, weights_only=False)
    restore_rng_state(state.pop("rng_state"))
    return state


def _initialize_lazy_optimizer_state(optimizers) -> None:
    if optimizers is None:
        return
    if isinstance(optimizers, torch.optim.Optimizer):
        optimizer_iter = (optimizers,)
    else:
        optimizer_iter = optimizers
    for optimizer in optimizer_iter:
        initialize = getattr(optimizer, "initialize_missing_state", None)
        if callable(initialize):
            initialize()


def save_distributed_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizers,
    *,
    step: int,
    extra_state: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    _initialize_lazy_optimizer_state(optimizers)
    model_state, optimizer_state = get_state_dict(model, optimizers, options=options)
    dcp.save(
        {
            "model": model_state,
            "optimizer": optimizer_state,
            "step": torch.tensor(int(step), dtype=torch.int64),
        },
        checkpoint_id=str(path),
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    torch.save(capture_rng_state(), path / f"rng-rank{rank}.pt")
    if extra_state is not None:
        torch.save(extra_state, path / f"extra-rank{rank}.pt")


def save_distributed_model_state(
    path: str | Path,
    model: torch.nn.Module,
    *,
    step: int,
) -> None:
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    dcp.save(
        {
            "model": get_model_state_dict(model, options=options),
            "step": torch.tensor(int(step), dtype=torch.int64),
        },
        checkpoint_id=str(Path(path)),
    )


def load_distributed_model_state(path: str | Path, model: torch.nn.Module) -> int:
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    state = {"model": model_state, "step": torch.zeros((), dtype=torch.int64)}
    dcp.load(state, checkpoint_id=str(Path(path)))
    set_model_state_dict(model, model_state_dict=state["model"], options=options)
    return int(state["step"].item())


def load_distributed_training_state(
    path: str | Path,
    model: torch.nn.Module,
    optimizers,
    *,
    extra_state: dict[str, Any] | None = None,
    allow_partial_optimizer_state: bool = False,
) -> int:
    path = Path(path)
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    _initialize_lazy_optimizer_state(optimizers)
    model_state, optimizer_state = get_state_dict(model, optimizers, options=options)
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "step": torch.zeros((), dtype=torch.int64),
    }
    planner = DefaultLoadPlanner(allow_partial_load=True) if allow_partial_optimizer_state else None
    dcp.load(state, checkpoint_id=str(path), planner=planner)
    set_state_dict(
        model,
        optimizers,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    rng_path = path / f"rng-rank{rank}.pt"
    if not rng_path.is_file():
        raise FileNotFoundError(f"Distributed RNG sidecar is missing: {rng_path}")
    restore_rng_state(torch.load(rng_path, map_location="cpu", weights_only=False))
    if extra_state is not None:
        extra_path = path / f"extra-rank{rank}.pt"
        if not extra_path.is_file():
            raise FileNotFoundError(f"Distributed extra-state sidecar is missing: {extra_path}")
        loaded_extra = torch.load(extra_path, map_location="cpu", weights_only=False)
        extra_state.clear()
        extra_state.update(loaded_extra)
    return int(state["step"].item())
