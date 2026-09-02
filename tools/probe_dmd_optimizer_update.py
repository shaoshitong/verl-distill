from __future__ import annotations

import torch
import torch.distributed as dist

from verl_distill.algorithms import build_algorithm
from verl_distill.config import load_config
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2, clip_grad_norm
from verl_distill.trainers.common import extract_image_batch, set_sampler_epoch, set_seed
from verl_distill.trainers.dmd import _build_models, _build_optimizer


def _stats(parameters, device: torch.device) -> torch.Tensor:
    out = torch.zeros(4, device=device, dtype=torch.float64)
    for parameter in parameters:
        data = parameter.detach()
        if hasattr(data, "to_local"):
            data = data.to_local()
        data = data.float()
        out[0] += data.numel()
        out[1] += data.double().sum()
        out[2] += data.double().square().sum()
        out[3] += data.double().abs().sum()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    return out


def _delta(after: torch.Tensor, before: torch.Tensor) -> str:
    return (
        f"sum_delta={(after[1] - before[1]).item():.9e} "
        f"sqsum_delta={(after[2] - before[2]).item():.9e} "
        f"abssum_delta={(after[3] - before[3]).item():.9e}"
    )


def main() -> None:
    context = initialize_distributed("nccl")
    set_seed(42, context.rank)
    config = load_config("dmd_1000_debug")
    config["runtime"]["max_train_steps"] = 5
    method = build_algorithm(config["method"]["name"], config["method"]["params"])
    student, score = _build_models(config, context.device)
    student.transformer.transformer.float()
    score["real"].transformer.float()
    score["fake"].transformer.float()
    apply_zimage_fsdp2(student.transformer.transformer, param_dtype=torch.bfloat16)
    apply_zimage_fsdp2(score["real"].transformer, param_dtype=torch.bfloat16)
    apply_zimage_fsdp2(score["fake"].transformer, param_dtype=torch.bfloat16)

    generator_parameters = [
        parameter for parameter in student.transformer.parameters() if parameter.requires_grad
    ]
    fake_parameters = [
        parameter for parameter in score["fake"].parameters() if parameter.requires_grad
    ]
    real_parameters = list(score["real"].parameters())
    generator_optimizer = _build_optimizer(generator_parameters, config["optimizer"]["generator"])
    fake_optimizer = _build_optimizer(fake_parameters, config["optimizer"]["fake_score"])

    dataset = build_dataset(config["data"])
    loader = build_dataloader(
        dataset,
        rank=context.rank,
        world_size=context.world_size,
        batch_size=int(config["runtime"].get("micro_batch_size", 1)),
        num_workers=int(config["runtime"].get("num_workers", 4)),
    )
    accumulation = int(config["runtime"].get("gradient_accumulation_steps", 1))
    set_sampler_epoch(loader, 0)
    data_iterator = iter(loader)
    max_grad_norm = float(config["optimizer"]["generator"].get("max_grad_norm", 1.0))
    fake_max_grad_norm = float(
        config["optimizer"]["fake_score"].get("max_grad_norm", max_grad_norm)
    )

    if context.is_main_process:
        print(
            "config",
            f"generator_lr={generator_optimizer.param_groups[0]['lr']}",
            f"fake_score_lr={fake_optimizer.param_groups[0]['lr']}",
            f"generator_clip={max_grad_norm}",
            f"fake_score_clip={fake_max_grad_norm}",
            f"dfake_gen_update_ratio={method.dfake_gen_update_ratio}",
            f"accumulation={accumulation}",
        )
        print(
            "dtypes",
            f"generator_first={generator_parameters[0].dtype}",
            f"fake_first={fake_parameters[0].dtype}",
            f"real_first={real_parameters[0].dtype}",
        )

    real_before_all = _stats(real_parameters, context.device)
    for global_step in range(1, 6):
        method.set_train_step(global_step)
        update_generator = method.should_update_generator(global_step)
        fake_optimizer.zero_grad(set_to_none=True)
        if update_generator:
            generator_optimizer.zero_grad(set_to_none=True)
        accumulated_score = torch.zeros((), device=context.device)
        accumulated_generator = torch.zeros((), device=context.device)
        fake_before = _stats(fake_parameters, context.device)
        generator_before = (
            _stats(generator_parameters, context.device) if update_generator else None
        )
        for _ in range(accumulation):
            try:
                batch = next(data_iterator)
            except StopIteration:
                data_iterator = iter(loader)
                batch = next(data_iterator)
            text, image = extract_image_batch(batch, "DMD")
            image = image.to(context.device, non_blocking=True)
            with torch.no_grad():
                prompt, prompt_mask, uncond, uncond_mask = student.encode_prompt(text, do_cfg=True)
                latents = student.pixels_to_latents(image).float()
            c = [prompt.float(), prompt_mask.float()]
            e = [uncond.float(), uncond_mask.float()]
            with torch.autocast(
                device_type=context.device.type,
                dtype=torch.bfloat16,
                enabled=context.device.type == "cuda",
            ):
                score_loss, _ = method.score_loss(
                    generator_model=student.transformer,
                    score_model=score,
                    x_real=latents,
                    c=c,
                    e=e,
                    latent_shape=latents.shape,
                )
            (score_loss / accumulation).backward()
            accumulated_score += score_loss.detach()
            if update_generator:
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.bfloat16,
                    enabled=context.device.type == "cuda",
                ):
                    generator_loss, _ = method.generator_loss(
                        generator_model=student.transformer,
                        score_model=score,
                        x_real=latents,
                        c=c,
                        e=e,
                        latent_shape=latents.shape,
                    )
                (generator_loss / accumulation).backward()
                accumulated_generator += generator_loss.detach()
        fake_grad_norm = clip_grad_norm(fake_parameters, fake_max_grad_norm)
        fake_optimizer.step()
        fake_after = _stats(fake_parameters, context.device)
        generator_grad_norm = 0.0
        generator_delta = "not_updated"
        if update_generator:
            generator_grad_norm = clip_grad_norm(generator_parameters, max_grad_norm)
            generator_optimizer.step()
            generator_after = _stats(generator_parameters, context.device)
            generator_delta = _delta(generator_after, generator_before)
        if context.is_main_process:
            print(
                f"step={global_step}",
                f"score_loss={(accumulated_score / accumulation).item():.9e}",
                f"generator_loss={(accumulated_generator / accumulation).item():.9e}",
                f"fake_grad_norm={fake_grad_norm:.9e}",
                f"generator_grad_norm={generator_grad_norm:.9e}",
                f"fake_delta={_delta(fake_after, fake_before)}",
                f"generator_delta={generator_delta}",
            )
    real_after_all = _stats(real_parameters, context.device)
    if context.is_main_process:
        print(f"real_score_total_delta={_delta(real_after_all, real_before_all)}")
    cleanup_distributed()


if __name__ == "__main__":
    main()
