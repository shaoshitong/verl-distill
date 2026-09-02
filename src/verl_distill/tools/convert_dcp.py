from __future__ import annotations

from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp


def _checkpoint_dir(path: Path) -> Path:
    if (path / ".metadata").is_file():
        return path
    if (path / "fsdp_state" / ".metadata").is_file():
        return path / "fsdp_state"
    raise FileNotFoundError(f"Distributed checkpoint metadata not found under {path}")


def convert_dcp_component(
    checkpoint_path,
    output_path,
    *,
    state_name="teacher_discriminator_model",
):
    checkpoint_dir = _checkpoint_dir(Path(checkpoint_path))
    metadata = dcp.FileSystemReader(str(checkpoint_dir)).read_metadata()
    prefix = f"{state_name}."
    tensors = {}
    for key, tensor_metadata in metadata.state_dict_metadata.items():
        key = str(key)
        if not key.startswith(prefix):
            continue
        properties = getattr(tensor_metadata, "properties", None)
        size = getattr(tensor_metadata, "size", None)
        if properties is None or size is None:
            continue
        tensors[key[len(prefix) :]] = torch.empty(
            tuple(int(value) for value in size),
            dtype=properties.dtype,
            device="cpu",
        )
    if not tensors:
        raise KeyError(f"No tensor metadata found for state '{state_name}'")
    dcp.load(
        state_dict={state_name: tensors},
        checkpoint_id=str(checkpoint_dir),
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".safetensors":
        from safetensors.torch import save_file

        save_file(tensors, str(output_path))
    else:
        torch.save(tensors, output_path)
    return {"state_name": state_name, "tensor_count": len(tensors), "output": output_path}
