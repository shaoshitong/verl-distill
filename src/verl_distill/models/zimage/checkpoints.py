from __future__ import annotations

from pathlib import Path

import torch


def _read_state(path: Path):
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file

        return load_file(str(path), device="cpu")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model", "module"):
        if isinstance(payload, dict) and isinstance(payload.get(key), dict):
            payload = payload[key]
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint must contain a state dictionary: {path}")
    return payload


def load_component_checkpoint(
    path,
    module,
    *,
    source_markers=(),
    key_aliases=None,
    strict=True,
):
    path = Path(path)
    if path.is_dir():
        candidates = [
            path / "model.safetensors",
            path / "pytorch_model.bin",
            path / "model.pt",
        ]
        path = next((candidate for candidate in candidates if candidate.is_file()), path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Consolidated checkpoint not found: {path}. Convert legacy distributed "
            "checkpoints before running the public trainer."
        )
    source = _read_state(path)
    target = module.state_dict()
    matched = {}
    key_aliases = dict(key_aliases or {})
    for target_key, target_value in target.items():
        source_target_key = target_key
        for target_prefix, source_prefix in key_aliases.items():
            if target_key == target_prefix or target_key.startswith(f"{target_prefix}."):
                source_target_key = source_prefix + target_key[len(target_prefix) :]
                break
        candidates = [source_target_key]
        candidates.extend(f"{marker}.{source_target_key}" for marker in source_markers)
        candidates.extend(
            source_key for source_key in source if source_key.endswith(f".{source_target_key}")
        )
        source_key = next(
            (
                candidate
                for candidate in candidates
                if candidate in source and source[candidate].shape == target_value.shape
            ),
            None,
        )
        if source_key is not None:
            matched[target_key] = source[source_key]
    missing = sorted(set(target) - set(matched))
    if strict and missing:
        preview = ", ".join(missing[:8])
        raise KeyError(f"Checkpoint is missing {len(missing)} component keys: {preview}")
    if not matched:
        raise KeyError(f"Checkpoint has no weights matching {type(module).__name__}")
    module.load_state_dict(matched, strict=False)
    return {"matched": len(matched), "missing": missing}
