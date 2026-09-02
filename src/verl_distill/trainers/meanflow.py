from __future__ import annotations

import logging
from pathlib import Path

import torch

from verl_distill.algorithms import build_algorithm
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.checkpoint import (
    load_distributed_training_state,
    load_training_state,
    save_distributed_training_state,
    save_training_state,
)
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.ema import ShardedEMA, use_ema_weights
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2, clip_grad_norm
from verl_distill.models.zimage import load_zimage
from verl_distill.models.zimage.compatibility import require_zimage_diffusers
from verl_distill.trainers.common import (
    extract_image_batch,
    save_debug_samples,
    set_sampler_epoch,
    set_seed,
)

logger = logging.getLogger(__name__)


def _build_models(config, device):
    require_zimage_diffusers()
    from diffusers import ZImageTransformer2DModel

    from verl_distill.models.zimage.modeling import GenTransformer

    model_config = config["model"]
    ZImage = load_zimage()
    student = ZImage(
        model_id=model_config["pretrained_model"],
        aux_time_embed=True,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        device=str(device),
    )
    teacher_path = model_config.get("teacher_model") or model_config["pretrained_model"]
    teacher_transformer = ZImageTransformer2DModel.from_pretrained(
        teacher_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(device)
    teacher = GenTransformer(
        teacher_transformer,
        student.model.vae_scale_factor,
        aux_time_embed=False,
    ).to(device)
    teacher.requires_grad_(False).eval()
    student.transformer.requires_grad_(True)
    if config["runtime"].get("gradient_checkpointing", True):
        student.transformer.transformer.enable_gradient_checkpointing()
    return student, teacher


def train_meanflow(config):
    context = initialize_distributed(config["distributed"].get("backend", "nccl"))
    set_seed(config["runtime"].get("seed", 42), context.rank)
    method = build_algorithm("meanflow", config["method"]["params"])
    student, teacher = _build_models(config, context.device)
    student.transformer.transformer.float()
    teacher.transformer.float()
    time_rotary = getattr(student.transformer.transformer, "time_rotary", None)
    if time_rotary is not None and hasattr(time_rotary, "reset_parameters"):
        time_rotary.reset_parameters()
    if context.world_size > 1:
        apply_zimage_fsdp2(student.transformer.transformer, param_dtype=torch.bfloat16)
        apply_zimage_fsdp2(teacher.transformer, param_dtype=torch.bfloat16)

    parameters = [
        parameter for parameter in student.transformer.parameters() if parameter.requires_grad
    ]
    optimizer_config = config["optimizer"]["generator"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config["lr"]),
        betas=tuple(optimizer_config.get("betas", [0.9, 0.999])),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        foreach=False,
    )
    dataset = build_dataset(config["data"])
    loader = build_dataloader(
        dataset,
        rank=context.rank,
        world_size=context.world_size,
        batch_size=int(config["runtime"].get("micro_batch_size", 1)),
        num_workers=int(config["runtime"].get("num_workers", 4)),
    )
    if len(loader) == 0:
        raise ValueError("Dataloader is empty; reduce micro_batch_size or add training samples")
    accumulation = int(config["runtime"].get("gradient_accumulation_steps", 1))
    max_steps = int(config["runtime"].get("max_train_steps", 0))
    max_grad_norm = float(optimizer_config.get("max_grad_norm", 1.0))
    output_dir = Path(config["runtime"].get("output_dir", "outputs"))
    save_every = int(config["runtime"].get("save_every_n_steps", 1000))
    global_step = 0
    ema = None
    if config.get("ema", {}).get("enabled", True):
        ema = ShardedEMA(
            student.transformer.transformer,
            decay=float(config.get("ema", {}).get("decay", 0.9999)),
        )
    resume_from = str(config["runtime"].get("resume_from", "") or "")
    if resume_from:
        if context.world_size > 1:
            extra_state = {}
            global_step = load_distributed_training_state(
                resume_from, student.transformer, optimizer, extra_state=extra_state
            )
            if ema is not None:
                ema.load_state_dict(extra_state["ema"])
        else:
            state = load_training_state(resume_from, map_location=context.device)
            student.transformer.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            global_step = int(state["step"])
            if ema is not None:
                ema.load_state_dict(state["ema"])
    micro_step = 0
    epoch = 0
    optimizer.zero_grad(set_to_none=True)
    try:
        while max_steps <= 0 or global_step < max_steps:
            set_sampler_epoch(loader, epoch)
            for batch in loader:
                text, image = extract_image_batch(batch, "MeanFlow")
                image = image.to(context.device, non_blocking=True)
                with torch.no_grad():
                    prompt, prompt_mask, uncond, uncond_mask = student.encode_prompt(
                        text, do_cfg=True
                    )
                    latents = student.pixels_to_latents(image).float()
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.bfloat16,
                    enabled=context.device.type == "cuda",
                ):
                    loss, _ = method.training_step(
                        student.transformer,
                        latents,
                        c=[prompt.float(), prompt_mask.float()],
                        e=[uncond.float(), uncond_mask.float()],
                        step=global_step + 1,
                        teacher=teacher,
                        return_loss_stats=True,
                    )
                (loss / accumulation).backward()
                micro_step += 1
                if micro_step % accumulation:
                    continue
                if max_grad_norm > 0:
                    clip_grad_norm(parameters, max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(student.transformer.transformer)
                global_step += 1
                if context.is_main_process:
                    logger.info("step=%d loss=%.6g", global_step, float(loss.detach()))
                debug_every = int(config["runtime"].get("debug_every_n_steps", 0) or 0)
                if debug_every > 0 and global_step % debug_every == 0:
                    with use_ema_weights(ema, student.transformer.transformer):
                        save_debug_samples(student, method, config, context, global_step)
                if save_every > 0 and global_step % save_every == 0:
                    if context.world_size > 1:
                        save_distributed_training_state(
                            output_dir / "checkpoints" / f"step-{global_step}",
                            student.transformer,
                            optimizer,
                            step=global_step,
                            extra_state={"ema": ema.state_dict() if ema is not None else None},
                        )
                    elif context.is_main_process:
                        save_training_state(
                            output_dir / "checkpoints" / f"step-{global_step}.pt",
                            {
                                "step": global_step,
                                "model": student.transformer.state_dict(),
                                "optimizer": optimizer.state_dict(),
                                "ema": ema.state_dict() if ema is not None else None,
                            },
                        )
                if max_steps > 0 and global_step >= max_steps:
                    break
            epoch += 1
    finally:
        cleanup_distributed()
    return global_step
