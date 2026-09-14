#!/usr/bin/env python3
"""Measure per-representation teacher-feature geometry and STE-vs-MSE gradient agreement.

Forward pass answers "are h_live / h_real / h_fake close?" (the STE premise); the extra
backward pairs answer "does the STE gradient point where an honest MSE regression would?".
Read-only: no optimizer steps, no checkpoint writes.
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


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    a = left.flatten().double()
    b = right.flatten().double()
    denominator = float(a.norm() * b.norm())
    return float(a.dot(b)) / denominator if denominator else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--windows", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--skip-grad-cosine", action="store_true")
    parser.add_argument(
        "--precision",
        default="bf16",
        choices=["bf16", "fp32"],
        help="compute precision for the teacher feature geometry (fp32 isolates bf16 rounding)",
    )
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
        if args.precision == "fp32":
            options.update(
                param_dtype=torch.float32, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            )

        def autocast():
            if args.precision == "bf16":
                return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
            return nullcontext()

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
            representations=list(keys),
            precision=args.precision,
            norm_definition="RMS over all elements of the batch tensor; cosines over the flattened batch",
            windows=[],
        )
        output.mkdir(parents=True, exist_ok=True)
        for window in range(args.windows):
            batches = []
            for _ in range(accumulation):
                text, image = extract_image_batch(next(iterator), "geometry diagnostic")
                with torch.no_grad():
                    latents = student.pixels_to_latents(image.to(context.device)).float()
                    prompt, mask, uncond, uncond_mask = student.encode_prompt(text, do_cfg=True)
                batches.append(
                    (latents, [prompt.float(), mask.float()], [uncond.float(), uncond_mask.float()])
                )
            rng = capture_rng_state()
            restore_rng_state(rng)
            per_key = {key: [] for key in keys}
            for latents, c, e in batches:
                with torch.no_grad():
                    with autocast():
                        x_fake, gen_meta = method.generate_one_step_latents(
                            student.transformer, latents, c, latent_shape=latents.shape
                        )
                        pair = method._gan_score_pair(
                            score, x_fake, c, e, gen_meta["gen_input_sigma"], include_real=True
                        )
                        teacher = score["real"]
                        live = method._call_teacher_representations(teacher, x_fake, c)
                        real = method._call_teacher_representations(
                            teacher, pair["pred_real_x0"], c
                        )
                        fake = method._call_teacher_representations(
                            teacher, pair["pred_fake_x0"], c
                        )
                for key in keys:
                    h_live = live[key].double()
                    h_real = real[key].double()
                    h_fake = fake[key].double()
                    norm_live = float(h_live.square().mean().sqrt())
                    diff_lr = h_live - h_real
                    diff_lf = h_live - h_fake
                    diff_rf = h_real - h_fake

                    def rms(value):
                        return float(value.square().mean().sqrt())

                    denom = float(diff_lr.abs().mean())
                    # STE surrogate direction is -(h_real - h_fake)/denom; the honest
                    # "pull h_live to h_real" direction is -(h_live - h_real).
                    per_key[key].append(
                        dict(
                            norm_live=norm_live,
                            norm_real=rms(h_real),
                            norm_fake=rms(h_fake),
                            rms_live_real=rms(diff_lr),
                            rms_live_fake=rms(diff_lf),
                            rms_real_fake=rms(diff_rf),
                            relative_live_real=rms(diff_lr) / max(norm_live, 1e-30),
                            relative_live_fake=rms(diff_lf) / max(norm_live, 1e-30),
                            relative_real_fake=rms(diff_rf) / max(norm_live, 1e-30),
                            cosine_live_real=_cosine(h_live, h_real),
                            cosine_live_fake=_cosine(h_live, h_fake),
                            cosine_real_fake=_cosine(h_real, h_fake),
                            # feature-space gradient of the STE surrogate is -(h_r - h_f)/denom,
                            # that of the honest ||h_live - h_real||^2 is (h_live - h_r):
                            # cos = cos(h_f - h_r, h_live - h_r) = -cos(h_r - h_f, h_live - h_r)
                            cosine_ste_vs_mse_direction=-_cosine(diff_rf, diff_lr),
                            denom=denom,
                            real_fake_over_denom=rms(diff_rf) / max(denom, 1e-30),
                        )
                    )
            averaged = {
                key: {name: sum(item[name] for item in items) / len(items) for name in items[0]}
                for key, items in per_key.items()
            }
            window_result = dict(window=window, geometry=averaged)
            if not args.skip_grad_cosine:
                grad_cosine = {}
                for key in keys:
                    live_key = {k: 1.0 if k == key else 0.0 for k in keys}

                    def grads_for(kind, batches=batches):
                        method.generator_teacher_feature_loss_type = {k: kind for k in keys}
                        with method.use_teacher_feature_weights("generator", live_key):
                            restore_rng_state(rng)
                            student.transformer.zero_grad(set_to_none=True)
                            for index, (latents, c, e) in enumerate(batches):
                                with _gradient_sync_context(
                                    student.transformer, index == accumulation - 1
                                ):
                                    with autocast():
                                        loss, _ = method.generator_loss(
                                            student.transformer, score, latents, c, e
                                        )
                                    (loss / accumulation).backward()
                        snapshot = [
                            None if p.grad is None else p.grad.detach().to("cpu", copy=True)
                            for p in parameters
                        ]
                        student.transformer.zero_grad(set_to_none=True)
                        return snapshot

                    ste = grads_for("ste")
                    mse = grads_for("mse")
                    dot = 0.0
                    left = 0.0
                    right = 0.0
                    for a, b in zip(ste, mse, strict=True):
                        if a is None or b is None:
                            continue
                        ad = a.float().flatten()
                        bd = b.detach().to("cpu").float().flatten()
                        dot += float(ad.double().dot(bd.double()))
                        left += float(ad.double().dot(ad.double()))
                        right += float(bd.double().dot(bd.double()))
                    denominator = (left**0.5) * (right**0.5)
                    grad_cosine[key] = dict(
                        cosine=dot / denominator if denominator else 0.0,
                        ste_norm=left**0.5,
                        mse_norm=right**0.5,
                    )
                    del ste, mse
                    gc.collect()
                window_result["grad_cosine_ste_vs_mse"] = grad_cosine
            report["windows"].append(window_result)
            if context.is_main_process:
                print(f"window={window} " + json.dumps(window_result, indent=2), flush=True)
                (output / "geometry.json").write_text(json.dumps(report, indent=2))
            del batches
            gc.collect()
        report["complete"] = True
        if context.is_main_process:
            (output / "geometry.json").write_text(json.dumps(report, indent=2))
            print("COMPLETE", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
