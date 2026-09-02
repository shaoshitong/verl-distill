from __future__ import annotations

import logging
from pathlib import Path

import torch

from verl_distill.algorithms import build_algorithm
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.checkpoint import (
    load_distributed_training_state,
    load_training_state,
    save_distributed_model_state,
    save_distributed_training_state,
    save_training_state,
)
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.ema import ShardedEMA, use_ema_weights
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2, clip_grad_norm
from verl_distill.engine.steps import opd_phase_for_step
from verl_distill.models.zimage import load_zimage
from verl_distill.models.zimage.checkpoints import load_component_checkpoint
from verl_distill.models.zimage.compatibility import require_zimage_diffusers
from verl_distill.trainers.common import (
    extract_batch,
    save_debug_samples,
    set_sampler_epoch,
    set_seed,
)

logger = logging.getLogger(__name__)


def _head_config(config):
    head = config["discriminator"]
    return {
        "layer_numbers": tuple(head["multifeature_layers"]),
        "fusion": head.get("multifeature_fusion", "channel"),
        "norm": head.get("multifeature_norm", "new"),
        "transformer_layers": int(head.get("transformer_layers", 5)),
        "transformer_heads": int(head.get("transformer_heads", 8)),
        "mlp_ratio": float(head.get("mlp_ratio", 4.0)),
    }


def _build_models(config, device):
    require_zimage_diffusers()
    from verl_distill.models.zimage.modeling import GenTransformer
    from verl_distill.models.zimage.transformer import ZImageTransformer2DModelWrapper

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
    backbone = ZImageTransformer2DModelWrapper.from_pretrained(
        teacher_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(device)
    teacher = GenTransformer(backbone, student.model.vae_scale_factor, aux_time_embed=True).to(
        device
    )
    teacher.requires_grad_(False)
    head_config = _head_config(config)
    teacher.init_dual_projector_multi_feature_discriminator_head(
        **head_config,
        align_output_dim=int(config["discriminator"].get("output_dim", 4096)),
        use_time_embedding=bool(config["discriminator"].get("use_time_embedding", True)),
    )
    dual_head = teacher.transformer.dual_projector_multi_feature_discriminator_head
    load_component_checkpoint(
        config["discriminator"]["pretrained_checkpoint"],
        dual_head,
        source_markers=("dual_projector_multi_feature_discriminator_head", "head"),
        key_aliases={
            "align_norm": "out_norm",
            "align_projector": "out_mlp",
        },
        strict=False,
    )
    teacher.init_multi_feature_discriminator_head(
        **head_config,
        output_dim=int(config["discriminator"].get("output_dim", 4096)),
        output_mode="tokens",
        use_time_embedding=False,
    )
    frozen_head = teacher.transformer.multi_feature_discriminator_head
    load_component_checkpoint(
        config["discriminator"]["frozen_checkpoint"],
        frozen_head,
        source_markers=("multi_feature_discriminator_head", "head"),
    )
    teacher.requires_grad_(False)
    for parameter in dual_head.parameters():
        parameter.requires_grad_(True)
    teacher.eval()
    dual_head.train()
    student.transformer.requires_grad_(True)
    if config["runtime"].get("gradient_checkpointing", True):

        def checkpoint_dynamic_forward(module, *args):
            return torch.utils.checkpoint.checkpoint(
                module.__call__,
                *args,
                use_reentrant=True,
            )

        student.transformer.transformer.enable_gradient_checkpointing(checkpoint_dynamic_forward)
        teacher.transformer.enable_gradient_checkpointing(checkpoint_dynamic_forward)
    return student, teacher, teacher, teacher, list(dual_head.parameters())


def _optimizer(parameters, config):
    return torch.optim.AdamW(
        list(parameters),
        lr=float(config["lr"]),
        betas=tuple(config.get("betas", [0.9, 0.999])),
        weight_decay=float(config.get("weight_decay", 0.0)),
        foreach=False,
    )


def _scheduler(optimizer, config):
    scheduler_type = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_type in {"", "none", "constant"}:
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={scheduler_type!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config.get("lr_decay_steps", 3000))),
        eta_min=float(config.get("lr_min", 0.0)),
    )


def _latent_shape(model, config, batch_size):
    resolution = int(config["data"].get("resolution", 1024))
    height = int(config["data"].get("height", resolution))
    width = int(config["data"].get("width", resolution))
    scale = int(model.model.vae_scale_factor)
    if height % scale or width % scale:
        raise ValueError(f"Image size must be divisible by VAE scale factor {scale}")
    return (batch_size, int(model.transformer.in_channels), height // scale, width // scale)


def train_opd_gan(config):
    context = initialize_distributed(config["distributed"].get("backend", "nccl"))
    set_seed(config["runtime"].get("seed", 42), context.rank)
    method = build_algorithm("opd_gan", config["method"]["params"])
    student, teacher, discriminator, frozen_discriminator, _ = _build_models(config, context.device)
    student.transformer.transformer.float()
    teacher.transformer.float()
    if context.world_size > 1:
        apply_zimage_fsdp2(student.transformer.transformer, param_dtype=None)
        apply_zimage_fsdp2(teacher.transformer, param_dtype=None)
    generator_parameters = [
        parameter for parameter in student.transformer.parameters() if parameter.requires_grad
    ]
    discriminator_parameters = [
        parameter
        for parameter in discriminator.transformer.dual_projector_multi_feature_discriminator_head.parameters()
        if parameter.requires_grad
    ]
    generator_optimizer = _optimizer(generator_parameters, config["optimizer"]["generator"])
    discriminator_optimizer = _optimizer(
        discriminator_parameters, config["optimizer"]["discriminator"]
    )
    generator_scheduler = _scheduler(generator_optimizer, config["optimizer"]["generator"])
    checkpoint_model = torch.nn.ModuleDict(
        {
            "generator": student.transformer,
            "discriminator": discriminator.transformer.dual_projector_multi_feature_discriminator_head,
        }
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
        raise ValueError("Dataloader is empty; reduce micro_batch_size or add samples")
    accumulation = int(config["runtime"].get("gradient_accumulation_steps", 1))
    if len(loader) < accumulation:
        raise ValueError("Dataloader must provide at least gradient_accumulation_steps batches")
    max_steps = int(config["runtime"].get("max_train_steps", 0))
    save_every = int(config["runtime"].get("save_every_n_steps", 1000))
    output_dir = Path(config["runtime"].get("output_dir", "outputs"))
    global_step = 0
    generator_update_index = 0
    ema = None
    if config.get("ema", {}).get("enabled", True):
        ema = ShardedEMA(
            student.transformer.transformer,
            decay=float(config.get("ema", {}).get("decay", 0.99)),
        )
    resume_from = str(config["runtime"].get("resume_from", "") or "")
    if resume_from:
        if context.world_size > 1:
            extra_state = {}
            global_step = load_distributed_training_state(
                resume_from,
                checkpoint_model,
                [generator_optimizer, discriminator_optimizer],
                extra_state=extra_state,
            )
            if generator_scheduler is not None:
                generator_scheduler.load_state_dict(extra_state["generator_scheduler"])
            if ema is not None:
                ema.load_state_dict(extra_state["ema"])
            cycle = int(method.discriminator_update_ratio) + 1
            generator_update_index = int(
                extra_state.get("generator_update_index", global_step // cycle)
            )
        else:
            state = load_training_state(resume_from, map_location=context.device)
            student.transformer.load_state_dict(state["generator"])
            discriminator.transformer.dual_projector_multi_feature_discriminator_head.load_state_dict(
                state["discriminator"]
            )
            generator_optimizer.load_state_dict(state["generator_optimizer"])
            discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
            if generator_scheduler is not None:
                generator_scheduler.load_state_dict(state["generator_scheduler"])
            if ema is not None:
                ema.load_state_dict(state["ema"])
            global_step = int(state["step"])
            generator_update_index = int(state.get("generator_update_index", 0))
    epoch = 0
    set_sampler_epoch(loader, epoch)
    data_iterator = iter(loader)
    try:
        while max_steps <= 0 or global_step < max_steps:
            phase = opd_phase_for_step(method, global_step)
            optimizer = generator_optimizer if phase == "generator" else discriminator_optimizer
            parameters = generator_parameters if phase == "generator" else discriminator_parameters
            optimizer.zero_grad(set_to_none=True)
            accumulated_loss = torch.zeros((), device=context.device)
            discriminator_grad_flags = None
            if phase == "generator":
                discriminator_grad_flags = [
                    parameter.requires_grad for parameter in discriminator_parameters
                ]
                for parameter in discriminator_parameters:
                    parameter.requires_grad_(False)
            for micro_step in range(1, accumulation + 1):
                try:
                    batch = next(data_iterator)
                except StopIteration:
                    epoch += 1
                    set_sampler_epoch(loader, epoch)
                    data_iterator = iter(loader)
                    batch = next(data_iterator)
                text, image = extract_batch(batch)
                with torch.no_grad():
                    prompt, prompt_mask, _, _ = student.encode_prompt(text, do_cfg=False)
                batch_size = int(prompt.shape[0])
                real_latents = None
                if torch.is_tensor(image) and image.numel() > 0:
                    real_latents = student.pixels_to_latents(
                        image.to(context.device, non_blocking=True)
                    ).float()
                latent_shape = (
                    tuple(real_latents.shape)
                    if real_latents is not None
                    else _latent_shape(student, config, batch_size)
                )
                kwargs = {}
                if phase == "generator":
                    kwargs["generator_update_index"] = generator_update_index + 1
                with torch.autocast(
                    device_type=context.device.type,
                    dtype=torch.bfloat16,
                    enabled=context.device.type == "cuda",
                ):
                    loss, _ = method.training_step(
                        student_model=student.transformer,
                        teacher_model=teacher,
                        discriminator_model=discriminator,
                        frozen_discriminator_model=frozen_discriminator,
                        latent_shape=latent_shape,
                        c=[prompt.float(), prompt_mask.float()],
                        step=global_step + 1,
                        real_image_latents=real_latents,
                        phase=phase,
                        return_loss_stats=True,
                        **kwargs,
                    )
                (loss / accumulation).backward()
                accumulated_loss += loss.detach()
            if discriminator_grad_flags is not None:
                for parameter, requires_grad in zip(
                    discriminator_parameters, discriminator_grad_flags, strict=True
                ):
                    parameter.requires_grad_(requires_grad)
            optimizer_name = "generator" if phase == "generator" else "discriminator"
            max_grad_norm = float(config["optimizer"][optimizer_name].get("max_grad_norm", 5.0))
            clip_grad_norm(parameters, max_grad_norm)
            optimizer.step()
            global_step += 1
            if phase == "generator":
                generator_update_index += 1
                if generator_scheduler is not None:
                    generator_scheduler.step()
                if ema is not None:
                    ema.update(student.transformer.transformer)
            if context.is_main_process:
                logger.info(
                    "step=%d phase=%s loss=%.6g",
                    global_step,
                    phase,
                    float(accumulated_loss / accumulation),
                )
            debug_every = int(config["runtime"].get("debug_every_n_steps", 0) or 0)
            if debug_every > 0 and global_step % debug_every == 0:
                with use_ema_weights(ema, student.transformer.transformer):
                    save_debug_samples(student, method, config, context, global_step)
            if save_every > 0 and global_step % save_every == 0:
                if context.world_size > 1:
                    checkpoint_path = output_dir / "checkpoints" / f"step-{global_step}"
                    if config["runtime"].get("save_optimizer_state", True):
                        save_distributed_training_state(
                            checkpoint_path,
                            checkpoint_model,
                            [generator_optimizer, discriminator_optimizer],
                            step=global_step,
                            extra_state={
                                "generator_update_index": generator_update_index,
                                "generator_scheduler": (
                                    generator_scheduler.state_dict()
                                    if generator_scheduler is not None
                                    else None
                                ),
                                "ema": ema.state_dict() if ema is not None else None,
                            },
                        )
                    else:
                        save_distributed_model_state(
                            checkpoint_path, student.transformer, step=global_step
                        )
                elif context.is_main_process:
                    save_training_state(
                        output_dir / "checkpoints" / f"step-{global_step}.pt",
                        {
                            "step": global_step,
                            "generator_update_index": generator_update_index,
                            "generator": student.transformer.state_dict(),
                            "discriminator": discriminator.transformer.dual_projector_multi_feature_discriminator_head.state_dict(),
                            "generator_optimizer": generator_optimizer.state_dict(),
                            "discriminator_optimizer": discriminator_optimizer.state_dict(),
                            "generator_scheduler": (
                                generator_scheduler.state_dict()
                                if generator_scheduler is not None
                                else None
                            ),
                            "ema": ema.state_dict() if ema is not None else None,
                        },
                    )
            if max_steps > 0 and global_step >= max_steps:
                break
    finally:
        cleanup_distributed()
    return global_step
