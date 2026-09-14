#!/usr/bin/env python3
"""Measure globally averaged FSDP1 G gradients without optimizer updates."""

from __future__ import annotations

import argparse
import gc
import json
import logging
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    set_model_state_dict,
)

from verl_distill.algorithms import build_algorithm
from verl_distill.config import load_config
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.checkpoint import capture_rng_state, restore_rng_state
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.fsdp1 import apply_zimage_fsdp1
from verl_distill.trainers.common import extract_image_batch, set_sampler_epoch, set_seed
from verl_distill.trainers.dmd import _build_models, _dtype_from_config, _gradient_sync_context


def load_sf_training_weights(checkpoint, model):
    # For beta1=0, ScheduleFreeAdamW.train() copies z into model parameters.
    # Read z directly with the model's sharding; moments are irrelevant without updates.
    options = StateDictOptions(full_state_dict=False, cpu_offload=True)
    model_state = get_model_state_dict(model, options=options)
    state = {"model": model_state, "step": torch.zeros((), dtype=torch.int64)}
    dcp.load(state, checkpoint_id=checkpoint)
    group = {"betas": (0.0, 0.999), "train_mode": False, "k": 0}
    z_state = {key: value for key, value in model_state.items() if key.startswith("generator.")}
    if not z_state:
        raise RuntimeError("No generator parameters selected for SF z restore")
    optimizer = {f"optimizer.state.{key}.z": value for key, value in z_state.items()}
    optimizer.update({f"optimizer.param_groups.0.{key}": value for key, value in group.items()})
    # Explicit flat keys avoid DCP's legacy nested-list fallback on partial reads.
    planner = dcp.DefaultLoadPlanner(flatten_state_dict=False)
    dcp.load(optimizer, checkpoint_id=checkpoint, planner=planner)
    group = {key: planner.state_dict[f"optimizer.param_groups.0.{key}"] for key in group}
    if group["betas"][0] != 0.0:
        raise ValueError("Direct z restore is valid only for beta1=0")
    for key in z_state:
        model_state[key] = optimizer[f"optimizer.state.{key}.z"]
    set_model_state_dict(model, model_state_dict=model_state, options=options)
    return int(state["step"].item()), group


def gradient_gram(snapshots, chunk_size=1_048_576):
    count = len(snapshots)
    gram = torch.zeros(count, count, dtype=torch.float64)
    if any(len(snapshot) != len(snapshots[0]) for snapshot in snapshots):
        raise ValueError("Gradient snapshots have different parameter counts")
    for tensors in zip(*snapshots, strict=True):
        if any(t.shape != tensors[0].shape for t in tensors):
            raise ValueError("Gradient shard shapes changed")
        for start in range(0, tensors[0].numel(), chunk_size):
            chunks = [t.reshape(-1)[start : start + chunk_size].double() for t in tensors]
            for i in range(count):
                for j in range(i, count):
                    gram[i, j] += torch.dot(chunks[i], chunks[j])
    return gram + gram.triu(1).T


def describe_gram(gram, names, clip):
    norms = gram.diag().clamp_min(0).sqrt()
    cosines = {}
    for i, left in enumerate(names):
        for j in range(i + 1, len(names)):
            denominator = float(norms[i] * norms[j])
            cosines[f"{left}:{names[j]}"] = float(gram[i, j]) / denominator if denominator else None
    total_sq = float(gram[-1, -1])
    low_sq = float(gram[0, 0] + gram[1, 1] + 2 * gram[0, 1])
    low_norm = max(0.0, low_sq) ** 0.5
    total_norm = float(norms[-1])
    weights = torch.tensor([1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
    residual = float((weights @ gram @ weights).clamp_min(0).sqrt())
    return {
        "grad_norms": dict(zip(names, norms.tolist(), strict=True)),
        "cosines": cosines,
        "latent_plus_layer5_norm": low_norm,
        "layer15_to_latent_plus_layer5_norm_ratio": float(norms[2]) / max(low_norm, 1e-30),
        "projection_fraction_of_total": {
            name: float(gram[i, -1]) / max(total_sq, 1e-30) for i, name in enumerate(names[:-1])
        },
        "clip_threshold": clip,
        "combined_clip_scale": min(1.0, clip / (total_norm + 1e-6)),
        "latent_plus_layer5_clip_scale": min(1.0, clip / (low_norm + 1e-6)),
        "sum_vs_joint_backward_relative_residual": residual / max(total_norm, 1e-30),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260910)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    cfg = load_config(args.config)
    context = initialize_distributed(cfg["distributed"].get("backend", "nccl"))
    output = Path(args.output_dir)
    try:
        if context.world_size != 8 or cfg["distributed"]["fsdp_backend"] != "fsdp1":
            raise ValueError("Diagnostic must use the training setup: 8 GPUs and FSDP1")
        if args.windows <= 0:
            raise ValueError("windows must be positive")
        set_seed(cfg["runtime"].get("seed", 42), context.rank)
        method = build_algorithm(cfg["method"]["name"], cfg["method"]["params"])
        names = ["latent", "layer_5", "layer_15"]
        if method.generator_teacher_feature_weights != dict.fromkeys(names, 1.0):
            raise ValueError("Expected the B experiment: latent/5/15 with unit weights")
        if not method.teacher_feature_normalize:
            raise ValueError("Expected independently normalized feature STE")
        student, score = _build_models(cfg, context.device)
        fsdp = cfg["distributed"]
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
        parameters = list(student.transformer.parameters())
        checkpoint_model = torch.nn.ModuleDict(
            {"generator": student.transformer, "fake_score": score["fake"]}
        )
        step, group = load_sf_training_weights(args.checkpoint, checkpoint_model)
        mode_before = [group["train_mode"]]
        mode_after = [True]
        optimizer_steps = [group["k"]]
        gc.collect()
        torch.cuda.empty_cache()
        student.transformer.train()
        score["fake"].train()
        score["real"].eval()
        before = [p.detach().to("cpu", copy=True) for p in parameters]
        set_seed(args.seed, context.rank)
        dataset = build_dataset(cfg["data"])
        loader = build_dataloader(
            dataset,
            rank=context.rank,
            world_size=context.world_size,
            batch_size=cfg["runtime"]["micro_batch_size"],
            num_workers=cfg["runtime"].get("num_workers", 4),
        )
        set_sampler_epoch(loader, 0)
        iterator = iter(loader)
        accumulation = cfg["runtime"]["gradient_accumulation_steps"]
        if accumulation != 4:
            raise ValueError("Expected GA4")
        report = dict(
            checkpoint=args.checkpoint,
            checkpoint_step=step,
            config=args.config,
            seed=args.seed,
            world_size=context.world_size,
            accumulation=accumulation,
            precision=fsdp,
            sf_train_mode_before=mode_before,
            sf_train_mode_after=mode_after,
            sf_optimizer_steps=optimizer_steps,
            optimizer_updates=0,
            sf_restore="Direct checkpoint z restore, equivalent to train() for verified beta1=0; no optimizer created",
            measurement="G parameter gradients after GA4 and distributed averaging, before clipping",
            batch_source="Fresh fixed training batches, not an exact replay of checkpoint step",
            windows=[],
        )
        output.mkdir(parents=True, exist_ok=True)
        if context.is_main_process:
            print(
                f"Loaded step={step}; SF mode {mode_before}->{mode_after}, k={optimizer_steps}",
                flush=True,
            )
        all_grams = []
        for window in range(args.windows):
            batches = []
            prompt_rows = []
            for _ in range(accumulation):
                text, image = extract_image_batch(next(iterator), "gradient diagnostic")
                with torch.no_grad():
                    latents = student.pixels_to_latents(image.to(context.device)).float()
                    prompt, mask, uncond, uncond_mask = student.encode_prompt(text, do_cfg=True)
                batches.append(
                    (latents, [prompt.float(), mask.float()], [uncond.float(), uncond_mask.float()])
                )
                prompt_rows.append(list(text))
            (output / f"window-{window:02d}-rank-{context.rank}-prompts.json").write_text(
                json.dumps(prompt_rows, ensure_ascii=False)
            )
            rng = capture_rng_state()
            snapshots = []
            losses = []
            reference_stats = None
            for component in names + ["sum"]:
                restore_rng_state(rng)
                method.generator_teacher_feature_weights = {
                    key: float(component == "sum" or component == key) for key in names
                }
                student.transformer.zero_grad(set_to_none=True)
                loss_sum = torch.zeros((), device=context.device, dtype=torch.float64)
                component_stats = []
                for i, (latents, c, e) in enumerate(batches):
                    with _gradient_sync_context(student.transformer, i == accumulation - 1):
                        with torch.autocast("cuda", dtype=torch.bfloat16, cache_enabled=False):
                            loss, stats = method.generator_loss(
                                student.transformer, score, latents, c, e
                            )
                        (loss / accumulation).backward()
                    loss_sum += loss.detach() / accumulation
                    component_stats.append(
                        [
                            float(
                                stats[f"teacher_feature_loss_{key}"][0]
                                / stats[f"teacher_feature_loss_{key}"][1]
                            )
                            for key in names
                        ]
                    )
                if reference_stats is None:
                    reference_stats = component_stats
                elif component_stats != reference_stats:
                    raise RuntimeError("Layer passes did not reproduce identical inputs/losses")
                snapshot = []
                for parameter in parameters:
                    grad = parameter.grad
                    if grad is not None and not torch.isfinite(grad).all():
                        raise RuntimeError("Nonfinite gradient; refusing to sanitize measurements")
                    snapshot.append(
                        torch.zeros_like(parameter, device="cpu")
                        if grad is None
                        else grad.detach().to("cpu", copy=True)
                    )
                snapshots.append(snapshot)
                student.transformer.zero_grad(set_to_none=True)
                dist.all_reduce(loss_sum)
                losses.append(float(loss_sum / context.world_size))
                if context.is_main_process:
                    print(
                        f"window={window} component={component} loss={losses[-1]:.6g} gradient captured",
                        flush=True,
                    )
            gram = gradient_gram(snapshots).to(context.device)
            dist.all_reduce(gram)
            gram = gram.cpu()
            all_grams.append(gram)
            result = describe_gram(
                gram, names + ["sum"], cfg["optimizer"]["generator"]["max_grad_norm"]
            )
            result.update(
                window=window, losses=dict(zip(names + ["sum"], losses)), gram=gram.tolist()
            )
            report["windows"].append(result)
            if context.is_main_process:
                print(json.dumps(result), flush=True)
                (output / "report.json").write_text(json.dumps(report, indent=2))
            del snapshots, batches
            gc.collect()
        for original, parameter in zip(before, parameters, strict=True):
            if not torch.equal(original, parameter.detach().cpu()):
                raise RuntimeError("Generator weights changed during diagnostic")
        if any(p.grad is not None for model in score.values() for p in model.parameters()):
            raise RuntimeError("Unexpected fake/teacher parameter gradient")
        report["aggregate"] = describe_gram(
            torch.stack(all_grams).mean(0),
            names + ["sum"],
            cfg["optimizer"]["generator"]["max_grad_norm"],
        )
        report["aggregate_definition"] = (
            "Gram matrices averaged over windows; norms are RMS over windows"
        )
        report["generator_weights_unchanged"] = True
        report["complete"] = True
        if context.is_main_process:
            (output / "report.json").write_text(json.dumps(report, indent=2))
            print("COMPLETE " + json.dumps(report["aggregate"]), flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
