from __future__ import annotations

import os
import re
from importlib.resources import files
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)(?:,([^}]*))?\}")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        name, default = match.groups()
        if name in os.environ:
            return os.environ[name]
        if default is not None:
            return default
        raise ValueError(f"Required environment variable is not set: {name}")

    return _ENV_PATTERN.sub(replace, value)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        recipe = str(path).removesuffix(".yaml")
        if "/" not in recipe:
            recipe = f"zimage/{recipe}"
        checkout_candidate = (
            Path(__file__).resolve().parents[2] / "configs" / "recipes" / f"{recipe}.yaml"
        )
        candidate = (
            checkout_candidate
            if checkout_candidate.is_file()
            else files("verl_distill").joinpath("configs", "recipes", f"{recipe}.yaml")
        )
        if not candidate.is_file():
            raise FileNotFoundError(f"Configuration file or built-in recipe not found: {path}")
        path = Path(str(candidate))
    path = path.resolve()
    with path.open(encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    includes = current.pop("defaults", [])
    merged: dict[str, Any] = {}
    for relative in includes:
        include_path = (path.parent / relative).resolve()
        merged = _deep_merge(merged, _load_config(include_path))
    return _resolve_env(_deep_merge(merged, current))


def validate_config(config: dict[str, Any]) -> None:
    allowed_sections = {
        "adapter",
        "data",
        "discriminator",
        "distributed",
        "ema",
        "method",
        "model",
        "optimizer",
        "runtime",
    }
    unknown = sorted(set(config) - allowed_sections)
    if unknown:
        raise ValueError(f"Unknown top-level configuration sections: {unknown}")
    method = config.get("method", {})
    name = method.get("name")
    if name not in {"dmd", "dmd_full", "meanflow", "opd_gan"}:
        raise ValueError("method.name must be one of: dmd, dmd_full, meanflow, opd_gan")
    if not isinstance(method.get("params"), dict):
        raise ValueError("method.params must be a mapping")
    for section in ("model", "data", "runtime", "distributed", "optimizer"):
        if not isinstance(config.get(section), dict):
            raise ValueError(f"{section} must be a mapping")
    if not config["model"].get("pretrained_model"):
        raise ValueError("model.pretrained_model is required")
    data_format = config["data"].get("format")
    if data_format not in {"image_jsonl", "prompt_jsonl", "image_lance", "lance"}:
        raise ValueError("data.format must be image_jsonl, prompt_jsonl, image_lance, or lance")
    if data_format in {"image_jsonl", "prompt_jsonl"} and not config["data"].get("manifest"):
        raise ValueError("data.manifest is required")
    runtime = config["runtime"]
    for key in ("micro_batch_size", "gradient_accumulation_steps"):
        if int(runtime.get(key, 1)) < 1:
            raise ValueError(f"runtime.{key} must be at least 1")
    if int(runtime.get("max_train_steps", 0)) < 0:
        raise ValueError("runtime.max_train_steps must be non-negative")
    if name in {"dmd", "dmd_full", "meanflow"}:
        if data_format in {"image_lance", "lance"}:
            if not config["data"].get("lance_data_dir"):
                raise ValueError(f"{name} requires data.lance_data_dir for Lance data")
        elif not config["data"].get("image_root"):
            raise ValueError(f"{name} requires data.image_root")
    if name == "dmd_full":
        if not config["model"].get("teacher_model"):
            raise ValueError("dmd_full requires model.teacher_model")
        if not config["model"].get("fake_score_model"):
            raise ValueError("dmd_full requires model.fake_score_model")
    if name == "opd_gan":
        discriminator = config.get("discriminator", {})
        for key in ("pretrained_checkpoint", "frozen_checkpoint"):
            if not discriminator.get(key):
                raise ValueError(f"opd_gan requires discriminator.{key}")


def load_config(path: str | Path) -> dict[str, Any]:
    config = _load_config(path)
    validate_config(config)
    return config
