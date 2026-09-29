#!/usr/bin/env python3
"""Evaluate two verl-distill DMD checkpoints on qwen-image-bench at 1024px.

This reuses the older Z-Image DCP sampling utilities from
model_comparison_result_complete, but adds support for the verl-distill DCP
checkpoint key prefix: model.generator.transformer.*.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.api import CheckpointException


ZROOT = Path(os.environ.get("ZIMAGE_EVAL_ROOT", "/path/to/zimage_eval_root"))
MCRC = ZROOT / "model_comparison_result_complete"
sys.path.insert(0, str(MCRC))

import run_lunara_aesthetic_8gpu_benchmark as bench  # noqa: E402


QWEN_BENCH_CASES = (
    MCRC
    / "qwen-image-bench-diversity"
    / "shift5-k128-spatial85-siglip-dino-edgew2-active_per_exit_000050000-50cases-8seeds"
    / "qwen_image_bench_cases_50.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    ZROOT
    / "verl-distill-runs"
    / "qwen_image_bench_1024_step5000_20260906"
)
FM1_MODEL_PATH = ZROOT / "Z-Image-FM1"


DCP_MODELS = [
    {
        "name": "genSF_fakeAdamW_step5000_4step_cfg0_shift6",
        "checkpoint": (
            ZROOT
            / "verl-distill-runs"
            / "dmd_refaligned_sf_gen_lr5e5_fakeadamw_lr1e5_resume1000_5000_20260906"
            / "checkpoints"
            / "step-5000"
        ),
        "sampling_steps": 4,
        "cfg_scale": 0.0,
        "flow_shift": 6.0,
    },
    {
        "name": "adamw_ref_step5000_4step_cfg0_shift6",
        "checkpoint": (
            ZROOT
            / "verl-distill-runs"
            / "dmd_refaligned_fsdp1_ode_warmup_50000_20260902_v2"
            / "checkpoints"
            / "step-5000"
        ),
        "sampling_steps": 4,
        "cfg_scale": 0.0,
        "flow_shift": 6.0,
    },
]


def _checkpoint_metadata_keys(checkpoint_id: str | Path) -> set[str] | None:
    metadata_path = Path(checkpoint_id) / ".metadata"
    if not metadata_path.exists():
        return None
    with metadata_path.open("rb") as f:
        metadata = pickle.load(f)
    state_dict_metadata = getattr(metadata, "state_dict_metadata", None)
    if not state_dict_metadata:
        return None
    return set(state_dict_metadata.keys())


def _resolve_checkpoint_dir(path: str | Path) -> str:
    path = Path(path)
    if (path / ".metadata").exists():
        return str(path)
    if path.name == "fsdp_state":
        return str(path)
    fsdp_state = path / "fsdp_state"
    if (fsdp_state / ".metadata").exists():
        return str(fsdp_state)
    return str(path)


def _load_dcp_weights(transformer, checkpoint_id: str | Path, rank: int):
    checkpoint_id = _resolve_checkpoint_dir(checkpoint_id)
    attempts = [
        ("verl", ["model", "generator", "transformer"], "model.generator.transformer.xxx"),
        ("flat", ["generator_model"], "generator_model.xxx"),
        ("nested", ["generator_model", "transformer"], "generator_model.transformer.xxx"),
        ("student", ["student_model"], "student_model.xxx"),
    ]
    metadata_keys = _checkpoint_metadata_keys(checkpoint_id)
    if metadata_keys:
        def matches(attempt):
            prefix = ".".join(attempt[1])
            return any(key == prefix or key.startswith(prefix + ".") for key in metadata_keys)

        attempts = sorted(attempts, key=lambda attempt: not matches(attempt))

    last_error = None
    for wrapper_kind, path, label in attempts:
        ref_state = {k: v.contiguous() for k, v in transformer.state_dict().items()}
        if wrapper_kind == "verl":
            wrapper = {"model": {"generator": {"transformer": ref_state}}}
        elif wrapper_kind == "flat":
            wrapper = {"generator_model": ref_state}
        elif wrapper_kind == "nested":
            wrapper = {"generator_model": {"transformer": ref_state}}
        else:
            wrapper = {"student_model": ref_state}
        try:
            dcp.load(state_dict=wrapper, checkpoint_id=checkpoint_id)
            loaded_sd = wrapper
            for part in path:
                loaded_sd = loaded_sd[part]
            missing, unexpected = transformer.load_state_dict(loaded_sd, strict=False)
            bench.log(
                rank,
                f"DCP loaded from {checkpoint_id} using {label}; "
                f"missing={len(missing)}, unexpected={len(unexpected)}",
            )
            return {
                "key_format": label,
                "loaded_keys": len(loaded_sd),
                "missing_count": len(missing),
                "unexpected_count": len(unexpected),
                "missing_sample": missing[:10],
                "unexpected_sample": unexpected[:10],
            }
        except (Exception, CheckpointException) as exc:
            last_error = exc
            bench.log(rank, f"DCP load attempt {label} failed: {exc}")
            if dist.is_initialized():
                dist.barrier()
    raise RuntimeError(f"failed to load DCP checkpoint {checkpoint_id}: {last_error}")


def _load_qwen_cases(path: str | Path, *, limit: int, seed_slots: int, prompt_lang: str):
    items = []
    with Path(path).open(encoding="utf-8") as f:
        for line_index, line in enumerate(f):
            if line_index >= limit:
                break
            raw = json.loads(line)
            qwen_id = int(raw["ID"])
            prompt_key = "prompt_cn" if prompt_lang == "cn" else "prompt_en"
            for seed_slot in range(seed_slots):
                items.append(
                    {
                        "case_index": len(items),
                        "qwen_id": qwen_id,
                        "seed_slot": seed_slot,
                        "prompt_index": qwen_id,
                        "dataset": "qwen-image-bench",
                        "split": "diversity-50",
                        "height": 1024,
                        "width": 1024,
                        "prompt": raw[prompt_key],
                        "prompt_cn": raw.get("prompt_cn"),
                        "prompt_en": raw.get("prompt_en"),
                        "dims_cn": raw.get("dims_cn"),
                        "dims_en": raw.get("dims_en"),
                        "seed": 42_000_000 + qwen_id + 1_000_000 * seed_slot,
                    }
                )
    expected = limit * seed_slots
    if len(items) != expected:
        raise ValueError(f"loaded {len(items)} items from {path}, expected {expected}")
    return items


def _numeric_output_path(output_dir: Path, item: dict) -> Path:
    return output_dir / f"case_{int(item['qwen_id']):06d}_seed{int(item['seed_slot'])}.png"


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _selected_models(names: list[str], models=None):
    models = DCP_MODELS if models is None else models
    if names == ["all"]:
        return models
    lookup = {cfg["name"]: cfg for cfg in models}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise ValueError(f"unknown model names: {missing}; valid={list(lookup)}")
    return [lookup[name] for name in names]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases-jsonl", default=str(QWEN_BENCH_CASES))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--seed-slots", type=int, default=8)
    parser.add_argument("--prompt-lang", choices=["en", "cn"], default="en")
    parser.add_argument("--models", nargs="+", default=["all"])
    parser.add_argument("--model-config-json", help="JSON list of named checkpoint/sampling configurations")
    args = parser.parse_args()
    models = None
    if args.model_config_json:
        models = json.loads(Path(args.model_config_json).read_text())
        if not isinstance(models, list) or not models:
            raise ValueError("model-config-json must contain a nonempty list")
        for model in models:
            if not all(key in model for key in ("name", "checkpoint", "sampling_steps", "cfg_scale", "flow_shift")):
                raise ValueError("Each model needs name, checkpoint, sampling_steps, cfg_scale, flow_shift")
            if Path(model["name"]).name != model["name"] or model["name"] in {".", ".."}:
                raise ValueError("Model name must be a single directory name")
        if len({model["name"] for model in models}) != len(models):
            raise ValueError("Model names must be unique")

    bench.BASE_MODEL_PATH = str(FM1_MODEL_PATH)
    bench.load_dcp_weights = _load_dcp_weights
    bench.resolve_fsdp_dir = _resolve_checkpoint_dir
    bench.numeric_output_path = _numeric_output_path

    rank, world_size, local_rank = bench.init_dist()
    all_items = _load_qwen_cases(
        args.cases_jsonl,
        limit=args.limit,
        seed_slots=args.seed_slots,
        prompt_lang=args.prompt_lang,
    )
    rank_items = [item for item in all_items if item["case_index"] % world_size == rank]
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = _selected_models(args.models, models)
    if rank == 0:
        _write_json(
            output_root / "benchmark_settings.json",
            {
                "benchmark": "qwen-image-bench",
                "cases_jsonl": args.cases_jsonl,
                "limit": args.limit,
                "seed_slots": args.seed_slots,
                "num_images_per_model": len(all_items),
                "resolution": "1024x1024",
                "prompt_lang": args.prompt_lang,
                "seed_formula": "42000000 + qwen_id + 1000000 * seed_slot",
                "world_size": world_size,
                "base_model_for_runtime": str(FM1_MODEL_PATH),
                "model_names": [cfg["name"] for cfg in selected],
            },
        )
    bench.log(rank, f"local_rank={local_rank}; assigned {len(rank_items)} / {len(all_items)} images")

    try:
        for model_cfg in selected:
            cfg = dict(model_cfg)
            cfg["checkpoint"] = str(cfg["checkpoint"])
            bench.run_dcp_model(cfg, rank, rank_items, output_root, args.cases_jsonl, len(all_items))
            if dist.is_initialized():
                dist.barrier()
    finally:
        bench.cleanup_dist()


if __name__ == "__main__":
    main()
