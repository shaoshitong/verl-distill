#!/usr/bin/env python3
"""Compare the three parameter-space branch Jacobians per representation.

For a shared upstream feature-space vector u_k = (h_real - h_fake), measure

  v_live = (d h_live / d theta)^T u_k        <- what the STE surrogate back-propagates
  v_real = (d h_real / d theta)^T u_k        <- the real-branch term
  v_fake = (d h_fake / d theta)^T u_k        <- the fake-branch term

The STE gradient is -v_live/denom, the fully differentiable objective gradient is v_real - v_fake.
So cos(v_live, v_real), cos(v_live, v_fake), cos(v_live, v_real - v_fake) state directly whether the
live branch still represents the other two as depth grows. Read-only, no optimizer steps.
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
    _gradient_sync_context,
)


def load_sf_training_weights(checkpoint, model):
    spec = importlib.util.spec_from_file_location(
        "layer_grad_diagnostic", Path(__file__).with_name("diagnose_dmd_layer_gradients.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.load_sf_training_weights(checkpoint, model)


def _global_inner(products: dict[str, float]) -> dict[str, float]:
    if not (dist.is_available() and dist.is_initialized()):
        return products
    keys = sorted(products)
    tensor = torch.tensor([products[key] for key in keys], dtype=torch.float64, device="cuda")
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return {key: float(value) for key, value in zip(keys, tensor.tolist(), strict=True)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--accumulation", type=int, default=0, help="0 keeps the config value")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    context = initialize_distributed(cfg["distributed"].get("backend", "nccl"))
    output = Path(args.output_dir)
    try:
        fsdp = cfg["distributed"]
        if str(fsdp.get("fsdp_backend", "")).strip().lower() != "fsdp1":
            raise ValueError("This diagnostic must run with the training FSDP1 setup")
        set_seed(cfg["runtime"].get("seed", 42), context.rank)
        method = build_algorithm(cfg["method"]["name"], cfg["method"]["params"])
        method.teacher_feature_grad_balance = None
        keys = method.teacher_feature_representation_keys()
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
        accumulation = args.accumulation or int(cfg["runtime"]["gradient_accumulation_steps"])
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
            representations=list(keys),
            accumulation=accumulation,
            definition=(
                "v_branch = (d h_branch / d theta)^T u with u = h_real - h_fake; "
                "STE grad = -v_live/denom, differentiable-objective grad = v_real - v_fake"
            ),
            windows=[],
        )
        output.mkdir(parents=True, exist_ok=True)
        for window in range(args.windows):
            batches = []
            for _ in range(accumulation):
                text, image = extract_image_batch(next(iterator), "branch jacobian diagnostic")
                with torch.no_grad():
                    latents = student.pixels_to_latents(image.to(context.device)).float()
                    prompt, mask, uncond, uncond_mask = student.encode_prompt(text, do_cfg=True)
                batches.append(
                    (latents, [prompt.float(), mask.float()], [uncond.float(), uncond_mask.float()])
                )
            rng = capture_rng_state()

            # ---- 1. detached forward to fix the shared upstream vector u_k per batch ----
            upstream = [{key: None for key in keys} for _ in batches]
            x0_batches = []
            for index, (latents, c, e) in enumerate(batches):
                restore_rng_state(rng)
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x_fake, gen_meta = method.generate_one_step_latents(
                        student.transformer, latents, c, latent_shape=latents.shape
                    )
                    pair = method._gan_score_pair(
                        score, x_fake, c, e, gen_meta["gen_input_sigma"], include_real=True
                    )
                    teacher = score["real"]
                    real = method._call_teacher_representations(teacher, pair["pred_real_x0"], c)
                    fake = method._call_teacher_representations(teacher, pair["pred_fake_x0"], c)
                x0_batches.append((latents, c, e, gen_meta["gen_input_sigma"]))
                for key in keys:
                    upstream[index][key] = (real[key].double() - fake[key].double()).detach()
                del real, fake, pair, x_fake

            # ---- 2. one backward per (branch, representation) with the shared u_k ----
            snapshots: dict[str, dict[str, list]] = {}
            for branch in ("live", "real", "fake"):
                for key in keys:
                    restore_rng_state(rng)
                    student.transformer.zero_grad(set_to_none=True)
                    for index, (latents, c, e, gen_sigma) in enumerate(x0_batches):
                        # The fake-score parameters must stay frozen around the backward too, otherwise
                        # gradient checkpointing recomputes the forward with different metadata.
                        frozen = (
                            method._frozen_parameters(score["fake"])
                            if branch != "live"
                            else nullcontext()
                        )
                        with frozen:
                            with _gradient_sync_context(
                                student.transformer, index == accumulation - 1
                            ):
                                with torch.autocast("cuda", dtype=torch.bfloat16):
                                    x_fake, gen_meta = method.generate_one_step_latents(
                                        student.transformer, latents, c, latent_shape=latents.shape
                                    )
                                    if branch == "live":
                                        features = method._call_teacher_representations(
                                            score["real"], x_fake, c
                                        )
                                    else:
                                        pair = method._teacher_pair(
                                            score,
                                            x_fake,
                                            c,
                                            e,
                                            gen_sigma,
                                            include_real=True,
                                            detach_query=False,
                                        )
                                        source = pair[
                                            "pred_real_x0" if branch == "real" else "pred_fake_x0"
                                        ]
                                        features = method._call_teacher_representations(
                                            score["real"], source, c
                                        )
                                    surrogate = (
                                        features[key].double() * upstream[index][key]
                                    ).sum() / accumulation
                                surrogate.backward()
                    snapshots[f"{branch}:{key}"] = [
                        None if p.grad is None else p.grad.detach().to("cpu", copy=True)
                        for p in parameters
                    ]
                    student.transformer.zero_grad(set_to_none=True)
                    gc.collect()

            # ---- 3. pairwise inner products, all-reduced across ranks ----
            names = list(snapshots)
            totals = {name: {"dot": 0.0, "norm": 0.0} for name in names}
            pairs = {f"{a}|{b}": 0.0 for i, a in enumerate(names) for b in names[i + 1 :]}
            sums: dict[str, float] = {}
            for index in range(len(parameters)):
                vectors = {
                    name: None if snapshots[name][index] is None else snapshots[name][index].float()
                    for name in names
                }
                for name, vector in vectors.items():
                    if vector is None:
                        continue
                    squared = float(vector.double().dot(vector.double()))
                    totals[name]["norm"] += squared
                for i, a in enumerate(names):
                    if vectors[a] is None:
                        continue
                    for b in names[i + 1 :]:
                        if vectors[b] is None:
                            continue
                        pairs[f"{a}|{b}"] += float(vectors[a].double().dot(vectors[b].double()))
            # v_real - v_fake needs a dedicated combination term
            for index in range(len(parameters)):
                for key in keys:
                    real = snapshots[f"real:{key}"][index]
                    fake = snapshots[f"fake:{key}"][index]
                    live = snapshots[f"live:{key}"][index]
                    if real is None or fake is None:
                        continue
                    combined = real.float() - fake.float()
                    squared = float(combined.double().dot(combined.double()))
                    sums[f"real_minus_fake:{key}"] = (
                        sums.get(f"real_minus_fake:{key}", 0.0) + squared
                    )
                    sums[f"live_dot_real_minus_fake:{key}"] = sums.get(
                        f"live_dot_real_minus_fake:{key}", 0.0
                    ) + float(live.float().double().dot(combined.double()))
            reduced_totals = _global_inner({f"{name}|norm": totals[name]["norm"] for name in names})
            reduced_pairs = _global_inner(pairs)
            reduced_sums = _global_inner(sums)
            result = {}
            for key in keys:

                def norm(branch):
                    return max(reduced_totals[f"{branch}:{key}|norm"], 0.0) ** 0.5

                def cos(a, b):
                    denominator = norm(a) * norm(b)
                    return (
                        reduced_pairs.get(f"{a}:{key}|{b}:{key}", 0.0) / denominator
                        if denominator
                        else 0.0
                    )

                live_norm = norm("live")
                real_norm = norm("real")
                fake_norm = norm("fake")
                combined_norm = max(reduced_sums.get(f"real_minus_fake:{key}", 0.0), 0.0) ** 0.5
                result[key] = dict(
                    norm_live=live_norm,
                    norm_real=real_norm,
                    norm_fake=fake_norm,
                    norm_real_minus_fake=combined_norm,
                    cos_live_real=cos("live", "real"),
                    cos_live_fake=cos("live", "fake"),
                    cos_real_fake=cos("real", "fake"),
                    cos_live_vs_real_minus_fake=(
                        reduced_sums.get(f"live_dot_real_minus_fake:{key}", 0.0)
                        / (live_norm * combined_norm)
                        if live_norm * combined_norm
                        else 0.0
                    ),
                    real_minus_fake_over_live=(combined_norm / live_norm if live_norm else 0.0),
                )
            report["windows"].append(dict(window=window, branch_jacobians=result))
            if context.is_main_process:
                print(f"window={window} " + json.dumps(result, indent=2), flush=True)
                (output / "branch_jacobians.json").write_text(json.dumps(report, indent=2))
            del snapshots, batches, x0_batches, upstream
            gc.collect()
        report["complete"] = True
        if context.is_main_process:
            (output / "branch_jacobians.json").write_text(json.dumps(report, indent=2))
            print("COMPLETE", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
