#!/usr/bin/env python3
"""Measure per-representation generator gradient norms under each teacher-feature loss formulation.

Read-only: no optimizer steps and no checkpoint writes. Run with
``torchrun --standalone --nproc-per-node=8`` and ``PYTHONPATH=src``.

Every formulation replays the SAME gradient-accumulation window and the same RNG state, so the
per-layer gradient norms and losses are directly comparable. This decides whether the layer-15/25
gradient blow-up comes from the straight-through surrogate or from the teacher representations.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import logging
import traceback
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist

from verl_distill.algorithms import build_algorithm
from verl_distill.config import load_config
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.checkpoint import capture_rng_state, restore_rng_state
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.fsdp1 import apply_zimage_fsdp1
from verl_distill.trainers.common import extract_image_batch, set_sampler_epoch, set_seed
from verl_distill.trainers.dmd import (
    _build_models,
    _dtype_from_config,
    _global_grad_squared_norm,
    _gradient_sync_context,
)

MODES = (
    ("ste_norm", "ste", True),
    ("ste_raw", "ste", False),
    ("dmd_mse_norm", "dmd_mse", True),
    ("dmd_mse_raw", "dmd_mse", False),
    ("mse", "mse", False),
    ("full_bwd_norm", "full_bwd", True),
    ("full_bwd_raw", "full_bwd", False),
    ("pair_ste", "pair_ste", False),
    ("mixed_ste_pair_ste", "mixed_pair", False),
    ("mixed_ste_pair_ste_norm", "mixed_pair", True),
    ("anchor_real", "ste:real", True),
    ("anchor_fake", "ste:fake", True),
    ("anchor_mid", "ste:mid", True),
)
MODES_BY_NAME = {name: (name, kind, normalize) for name, kind, normalize in MODES}


def load_sf_training_weights(checkpoint, model):
    """Reuse the verified schedule-free z restore from the layer-gradient diagnostic."""
    spec = importlib.util.spec_from_file_location(
        "layer_grad_diagnostic", Path(__file__).with_name("diagnose_dmd_layer_gradients.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_sf_training_weights(checkpoint, model)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument(
        "--modes",
        default="ste_norm,ste_raw,dmd_mse_norm,dmd_mse_raw,mse",
        help=f"comma separated subset of: {', '.join(MODES_BY_NAME)}",
    )
    args = parser.parse_args()
    selected = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = [name for name in selected if name not in MODES_BY_NAME]
    if unknown or not selected:
        raise ValueError(f"unknown or empty --modes: {unknown or args.modes}")
    modes = [MODES_BY_NAME[name] for name in selected]
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    context = initialize_distributed(cfg["distributed"].get("backend", "nccl"))
    output = Path(args.output_dir)
    try:
        fsdp = cfg["distributed"]
        if str(fsdp.get("fsdp_backend", "")).strip().lower() != "fsdp1":
            raise ValueError("This diagnostic must run with the training FSDP1 setup")
        if args.windows <= 0:
            raise ValueError("windows must be positive")
        set_seed(cfg["runtime"].get("seed", 42), context.rank)
        method = build_algorithm(cfg["method"]["name"], cfg["method"]["params"])
        method.teacher_feature_grad_balance = None
        keys = method.teacher_feature_representation_keys()
        if len(keys) < 2:
            raise ValueError("Expected a multi-representation teacher feature configuration")
        student, score = _build_models(cfg, context.device)
        options = dict(
            local_rank=context.local_rank,
            param_dtype=_dtype_from_config(fsdp.get("param_dtype"), torch.bfloat16),
            reduce_dtype=_dtype_from_config(fsdp.get("reduce_dtype"), torch.float32),
            buffer_dtype=_dtype_from_config(fsdp.get("buffer_dtype"), torch.float32),
        )
        no_split = list(getattr(score["fake"].transformer, "_no_split_modules", []))
        student.transformer = apply_zimage_fsdp1(
            student.transformer, no_split_modules=student.get_no_split_modules(), **options
        )
        for key in score:
            score[key] = apply_zimage_fsdp1(score[key], no_split_modules=no_split, **options)
        parameters = [
            parameter for parameter in student.transformer.parameters() if parameter.requires_grad
        ]
        checkpoint_model = torch.nn.ModuleDict(
            {"generator": student.transformer, "fake_score": score["fake"]}
        )
        step, _ = load_sf_training_weights(args.checkpoint, checkpoint_model)
        gc.collect()
        torch.cuda.empty_cache()
        student.transformer.train()
        score["fake"].train()
        score["real"].eval()
        before = [p.detach().to("cpu", copy=True) for p in parameters]
        accumulation = int(cfg["runtime"]["gradient_accumulation_steps"])
        dataset = build_dataset(cfg["data"])
        loader = build_dataloader(
            dataset,
            rank=context.rank,
            world_size=context.world_size,
            batch_size=cfg["runtime"]["micro_batch_size"],
            num_workers=int(cfg["runtime"].get("num_workers", 4)),
        )
        set_sampler_epoch(loader, 0)
        iterator = iter(loader)
        report = dict(
            checkpoint=args.checkpoint,
            checkpoint_step=step,
            config=args.config,
            seed=args.seed,
            world_size=context.world_size,
            accumulation=accumulation,
            representations=list(keys),
            modes=[{"name": n, "type": t, "normalize": bool(norm)} for n, t, norm in modes],
            measurement="per-representation generator parameter gradient norm, global L2, after GA",
            windows=[],
        )
        output.mkdir(parents=True, exist_ok=True)
        for window in range(args.windows):
            batches = []
            for _ in range(accumulation):
                text, image = extract_image_batch(next(iterator), "loss mode diagnostic")
                with torch.no_grad():
                    latents = student.pixels_to_latents(image.to(context.device)).float()
                    prompt, mask, uncond, uncond_mask = student.encode_prompt(text, do_cfg=True)
                batches.append(
                    (
                        latents,
                        [prompt.float(), mask.float()],
                        [uncond.float(), uncond_mask.float()],
                    )
                )
            rng = capture_rng_state()
            window_result = {"window": window, "modes": {}}
            reference_grads: dict[str, list] = {}
            for name, kind, normalize in modes:
                anchor = "live"
                if ":" in kind:
                    kind, anchor = kind.split(":", 1)
                method.generator_teacher_feature_anchor = anchor
                if kind == "mixed_pair":
                    # latent (the identity representation) keeps the STE surrogate; the real
                    # teacher-feature layers use the target-point straight-through form.
                    method.generator_teacher_feature_loss_type = {
                        key: ("ste" if key == keys[0] else "pair_ste") for key in keys
                    }
                else:
                    method.generator_teacher_feature_loss_type = {key: kind for key in keys}
                method.teacher_feature_normalize = bool(normalize)
                per_key = {}
                for active in keys:
                    with method.use_teacher_feature_weights(
                        "generator", {key: 1.0 if key == active else 0.0 for key in keys}
                    ):
                        restore_rng_state(rng)
                        student.transformer.zero_grad(set_to_none=True)
                        loss_total = 0.0
                        for index, (latents, c, e) in enumerate(batches):
                            frozen = (
                                method._frozen_parameters(score["fake"])
                                if kind == "full_bwd"
                                else nullcontext()
                            )
                            with frozen:
                                with _gradient_sync_context(
                                    student.transformer, index == accumulation - 1
                                ):
                                    with torch.autocast(
                                        device_type=context.device.type,
                                        dtype=torch.bfloat16,
                                        enabled=context.device.type == "cuda",
                                        cache_enabled=False,
                                    ):
                                        loss, stats = method.generator_loss(
                                            student.transformer, score, latents, c, e
                                        )
                                    (loss / accumulation).backward()
                            loss_total += float(loss.detach()) / accumulation
                    squared = _global_grad_squared_norm(parameters).item()
                    per_key[active] = dict(grad_norm=squared**0.5, loss=loss_total)
                    if name == "ste_norm":
                        reference_grads[active] = [
                            None if p.grad is None else p.grad.detach().to("cpu", copy=True)
                            for p in parameters
                        ]
                    elif active in reference_grads:
                        # cosine of this formulation's per-representation gradient against the
                        # current STE one, over the whole parameter vector and all ranks
                        reference = reference_grads[active]
                        local = torch.zeros(3, dtype=torch.float64, device=context.device)
                        for ref, parameter in zip(reference, parameters, strict=True):
                            if ref is None or parameter.grad is None:
                                continue
                            a = ref.float().flatten().double().to(context.device)
                            b = parameter.grad.detach().float().flatten().double()
                            local[0] += a.dot(b)
                            local[1] += a.dot(a)
                            local[2] += b.dot(b)
                        if dist.is_available() and dist.is_initialized():
                            dist.all_reduce(local, op=dist.ReduceOp.SUM)
                        denominator = float((local[1] * local[2]).sqrt())
                        per_key[active]["cos_vs_ste"] = (
                            float(local[0]) / denominator if denominator else 0.0
                        )
                    student.transformer.zero_grad(set_to_none=True)
                window_result["modes"][name] = per_key
                if context.is_main_process:
                    print(
                        f"window={window} mode={name} "
                        + json.dumps({k: round(v["grad_norm"], 4) for k, v in per_key.items()}),
                        flush=True,
                    )
            report["windows"].append(window_result)
            if context.is_main_process:
                (output / "report.json").write_text(json.dumps(report, indent=2))
            del batches
            gc.collect()
        for original, parameter in zip(before, parameters, strict=True):
            if not torch.equal(original, parameter.detach().cpu()):
                raise RuntimeError("Generator weights changed during the diagnostic")
        report["complete"] = True
        if context.is_main_process:
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print("COMPLETE", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
