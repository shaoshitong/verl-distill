from __future__ import annotations

import csv
import logging
import math
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Optional, Sequence

import torch
import torch.distributed as dist
from PIL import Image

from verl_distill.algorithms import build_algorithm
from verl_distill.data import build_dataloader, build_dataset
from verl_distill.engine.checkpoint import (
    capture_rng_state,
    load_distributed_model_state,
    load_distributed_training_state,
    load_training_state,
    restore_rng_state,
    save_distributed_training_state,
    save_training_state,
)
from verl_distill.engine.distributed import cleanup_distributed, initialize_distributed
from verl_distill.engine.fsdp1 import apply_zimage_fsdp1
from verl_distill.engine.fsdp2 import apply_zimage_fsdp2, clip_grad_norm
from verl_distill.models.zimage import load_zimage
from verl_distill.models.zimage.compatibility import require_zimage_diffusers
from verl_distill.models.zimage.feature_discriminator import TeacherFeatureDiscriminator
from verl_distill.optim import AdamWScheduleFree
from verl_distill.trainers.common import (
    extract_image_batch,
    save_debug_samples,
    set_sampler_epoch,
    set_seed,
)

logger = logging.getLogger(__name__)


TRACKED_STATS_KEYS = (
    "score/loss_score",
    "score/score_loss_fake_score_has_tt",
    "score/score_loss_fake_score_t",
    "score/score_loss_fake_score_tt",
    "score/score_pred_err",
    "score/score_sigma",
    "gen/dmd_fake_score_has_tt",
    "gen/dmd_fake_score_t",
    "gen/dmd_fake_score_tt",
    "gen/dm_grad_abs",
    "gen/dm_sigma",
    "gen/gen_input_sigma",
    "gen/dmd_pearson4_pearson_loss",
    "gen/dmd_pearson4_raw_mse_loss",
    "gen/dmd_pearson4_pearson_weight",
    "gen/dmd_pearson4_mse_weight",
    "gen/loss_gen_dm",
    "gen/loss_gen_gan",
    "gen/loss_gen_teacher_features",
    "gen/teacher_feature_loss_layer_5",
    "gen/teacher_feature_loss_layer_15",
    "gen/teacher_feature_loss_layer_25",
    "gen/teacher_feature_loss_pre_projector",
    "gen/x_fake_rms",
    "gan/gan_discriminator_loss",
    "gan/gan_discriminator_raw_loss",
    "gan/gan_r1_loss",
    "gan/gan_r1_weighted_loss",
    "gan/gan_disc_fake_logit",
    "gan/gan_disc_real_logit",
    "gan/gan_disc_fake_prob",
    "gan/gan_disc_real_prob",
    "gan/dm_sigma",
    "gan/gan_disc_x_fake_abs",
    "gan/gan_disc_x_real_abs",
    "gan/gan_disc_gen_input_sigma",
    "gen/gan_generator_raw_loss",
    "gen/gan_feature_ste_loss",
    "gen/gan_feature_delta_abs",
    "gen/gan_feature_live_abs",
    "gen/gan_gen_fake_logit",
    "gen/gan_gen_fake_prob",
    "gen/loss_gen_ode_warmup",
    "gen/loss_gen_ode_warmup_unweighted",
    "gen/ode_reward",
    "gen/ode_reward_score",
    "gen/ode_reward_aesthetic",
    "gen/ode_reward_instruction_following",
    "gen/ode_reward_color_harmony",
    "gen/ode_loss_weight_mean",
    "gen/x_fake_abs",
    "probe/fake_score_grad_abs",
    "probe/fake_score_delta_abs",
    "probe/fake_score_delta_max",
    "probe/generator_grad_abs",
    "probe/generator_delta_abs",
    "probe/generator_delta_max",
)


TRACKED_STATS_KEYS += tuple(
    f"{phase}/teacher_feature_{metric}_{representation}"
    for phase, metrics in (
        ("gen", ("loss", "denom", "direction_rms", "live_rms", "real_rms", "fake_rms")),
        ("score", ("loss", "pred_rms", "target_rms")),
    )
    for metric in metrics
    for representation in ("latent", "layer_5", "layer_15", "layer_25", "pre_projector")
    if f"{phase}/teacher_feature_{metric}_{representation}" not in TRACKED_STATS_KEYS
)


def _tracked_stats_keys(method) -> tuple[str, ...]:
    """Reserve both phases' selected feature columns before the first TSV row."""
    representation_keys = getattr(method, "teacher_feature_representation_keys", None)
    if representation_keys is None:
        return TRACKED_STATS_KEYS
    representations = representation_keys()
    feature_keys = (
        f"{phase}/teacher_feature_{metric}_{representation}"
        for phase, metrics in (
            ("gen", ("loss", "denom", "direction_rms", "live_rms", "real_rms", "fake_rms")),
            ("score", ("loss", "pred_rms", "target_rms")),
        )
        for metric in metrics
        for representation in representations
    )
    return tuple(dict.fromkeys((*TRACKED_STATS_KEYS, *feature_keys)))


@torch.no_grad()
def _module_signature(module: torch.nn.Module, limit: int = 3) -> str:
    chunks = []
    for index, (name, parameter) in enumerate(module.named_parameters()):
        if index >= limit:
            break
        local = parameter.detach()
        if hasattr(local, "to_local"):
            local = local.to_local()
        local = local.float()
        chunks.append(
            f"{name}:shape={tuple(parameter.shape)} mean={local.mean().item():.6g} "
            f"std={local.std(unbiased=False).item():.6g}"
        )
    return " | ".join(chunks) if chunks else "<no parameters>"


@torch.no_grad()
def _dtype_signature(module: torch.nn.Module) -> str:
    counts: dict[str, int] = {}
    for parameter in module.parameters():
        counts[str(parameter.dtype)] = counts.get(str(parameter.dtype), 0) + parameter.numel()
    return ", ".join(f"{dtype}:{count}" for dtype, count in sorted(counts.items()))


@torch.no_grad()
def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    local = tensor.detach()
    if hasattr(local, "to_local"):
        local = local.to_local()
    return local


@torch.no_grad()
def _capture_param_probe(
    module: torch.nn.Module,
    *,
    max_tensors: int = 4,
    max_elements: int = 2048,
) -> dict[str, torch.Tensor]:
    captured = {}
    for name, parameter in module.named_parameters():
        if not parameter.requires_grad:
            continue
        local = _local_tensor(parameter).float().flatten()
        if local.numel() == 0:
            continue
        captured[name] = local[:max_elements].cpu().clone()
        if len(captured) >= max_tensors:
            break
    return captured


@torch.no_grad()
def _param_delta_stats(
    module: torch.nn.Module,
    before: dict[str, torch.Tensor],
    *,
    max_elements: int = 2048,
) -> tuple[float, float]:
    if not before:
        return 0.0, 0.0
    total = 0.0
    count = 0
    max_value = 0.0
    for name, parameter in module.named_parameters():
        previous = before.get(name)
        if previous is None:
            continue
        current = _local_tensor(parameter).float().flatten()[:max_elements].cpu()
        if current.numel() != previous.numel():
            continue
        delta = (current - previous).abs()
        total += float(delta.sum().item())
        count += int(delta.numel())
        max_value = max(max_value, float(delta.max().item()))
    return total / max(count, 1), max_value


@torch.no_grad()
def _grad_abs(parameters: list[torch.nn.Parameter]) -> float:
    total = 0.0
    count = 0
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local = _local_tensor(grad).float()
        total += float(local.abs().sum().item())
        count += int(local.numel())
    return total / max(count, 1)


def _reset_fake_score_time_rotary(
    module: torch.nn.Module,
    *,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> bool:
    transformer = getattr(module, "transformer", module)
    time_rotary = getattr(transformer, "time_rotary", None)
    if time_rotary is None:
        return False
    time_rotary.to(device=device, dtype=dtype)
    if hasattr(time_rotary, "reset_parameters"):
        time_rotary.reset_parameters()
    return True


def _dtype_from_config(value, default: Optional[torch.dtype] = None) -> Optional[torch.dtype]:
    if value is None:
        return default
    if isinstance(value, torch.dtype):
        return value
    text = str(value).strip().lower()
    if text in {"", "none", "null"}:
        return None
    mapping = {
        "fp32": torch.float32,
        "float32": torch.float32,
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported dtype value: {value!r}")
    return mapping[text]


def _build_models(config, device):
    require_zimage_diffusers()
    from diffusers import ZImageTransformer2DModel

    from verl_distill.models.zimage.modeling import GenTransformer
    from verl_distill.models.zimage.transformer import ZImageTransformer2DModelWrapper

    model_path = config["model"]["pretrained_model"]
    teacher_model_path = config["model"].get("teacher_model")
    fake_score_model_path = config["model"].get("fake_score_model")
    if not teacher_model_path:
        raise ValueError("dmd_full requires model.teacher_model; refusing to fallback to generator")
    if not fake_score_model_path:
        raise ValueError(
            "dmd_full requires model.fake_score_model; refusing to fallback to generator"
        )
    logger.info(
        "DMD model paths: generator=%s real_score=%s fake_score=%s",
        model_path,
        teacher_model_path,
        fake_score_model_path,
    )
    ZImage = load_zimage()
    student = ZImage(
        model_id=model_path,
        aux_time_embed=False,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        device=str(device),
    )
    student.transformer.requires_grad_(True)
    real_score_cls = (
        ZImageTransformer2DModelWrapper
        if (
            config["method"]["params"].get("generator_objective") in {"gan", "teacher_feature_ste"}
            or config["method"]["params"].get("score_objective") == "teacher_feature_mse"
        )
        else ZImageTransformer2DModel
    )
    real_score_transformer = real_score_cls.from_pretrained(
        teacher_model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(device)
    real_score = GenTransformer(
        real_score_transformer,
        student.model.vae_scale_factor,
        aux_time_embed=False,
    ).to(device)
    fake_score_aux_time_embed = bool(config["model"].get("fake_score_aux_time_embed", True))
    fake_score_cls = (
        ZImageTransformer2DModelWrapper if fake_score_aux_time_embed else ZImageTransformer2DModel
    )
    fake_score_transformer = fake_score_cls.from_pretrained(
        fake_score_model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=False,
    ).to(device)
    fake_score = GenTransformer(
        fake_score_transformer,
        student.model.vae_scale_factor,
        aux_time_embed=fake_score_aux_time_embed,
    ).to(device)
    fake_score_time_rotary_reset = False
    if fake_score_aux_time_embed:
        fake_score_time_rotary_reset = _reset_fake_score_time_rotary(
            fake_score,
            device=device,
            dtype=torch.float32,
        )
    real_score.requires_grad_(False)
    fake_score.requires_grad_(True)
    real_score.eval()
    fake_score.train()
    logger.info(
        "DMD fake_score aux_time_embed=%s time_rotary_reset=%s",
        fake_score_aux_time_embed,
        fake_score_time_rotary_reset,
    )
    logger.info("DMD generator signature: %s", _module_signature(student.transformer))
    logger.info("DMD real_score signature: %s", _module_signature(real_score))
    logger.info("DMD fake_score signature: %s", _module_signature(fake_score))
    if config["runtime"].get("gradient_checkpointing", True):
        student.transformer.enable_gradient_checkpointing()
        real_score.enable_gradient_checkpointing()
        fake_score.enable_gradient_checkpointing()
    return student, {"real": real_score, "fake": fake_score}


def _optimizer_type(config: dict) -> str:
    return str(config.get("type", config.get("name", "adamw"))).strip().lower().replace("-", "_")


def _build_optimizer(parameters, config, *, role: str = "optimizer", fsdp_backend: str = ""):
    optimizer_type = _optimizer_type(config)
    if optimizer_type in {"adamw", "torch_adamw"}:
        return torch.optim.AdamW(
            list(parameters),
            lr=float(config["lr"]),
            betas=tuple(config.get("betas", [0.9, 0.99])),
            weight_decay=float(config.get("weight_decay", 0.0)),
            foreach=bool(config.get("foreach", True)),
        )
    if optimizer_type in {"adamw_schedule_free", "schedule_free_adamw"}:
        if fsdp_backend != "fsdp1":
            raise ValueError(f"optimizer.{role}.type=adamw_schedule_free requires FSDP1")
        return AdamWScheduleFree(
            list(parameters),
            lr=float(config["lr"]),
            betas=tuple(config.get("betas", [0.9, 0.999])),
            eps=float(config.get("eps", 1.0e-8)),
            weight_decay=float(config.get("weight_decay", 0.0)),
            warmup_steps=int(config.get("warmup_steps", 0)),
            r=float(config.get("r", 0.0)),
            weight_lr_power=float(config.get("weight_lr_power", 2.0)),
            inner_momentum=float(config.get("inner_momentum", 0.0)),
            foreach=bool(config.get("foreach", True)),
        )
    raise ValueError(f"Unsupported {role} optimizer type: {optimizer_type!r}")


def _optimizer_is_schedule_free(optimizer: torch.optim.Optimizer) -> bool:
    return (
        hasattr(optimizer, "train")
        and hasattr(optimizer, "eval")
        and any("train_mode" in group for group in optimizer.param_groups)
    )


def _optimizer_train(optimizer: torch.optim.Optimizer) -> None:
    if _optimizer_is_schedule_free(optimizer):
        optimizer.train()


@contextmanager
def _optimizer_eval_context(optimizer: torch.optim.Optimizer):
    if not _optimizer_is_schedule_free(optimizer):
        yield
        return
    optimizer.eval()
    try:
        yield
    finally:
        optimizer.train()


@contextmanager
def _optimizers_eval_context(optimizers: Sequence[torch.optim.Optimizer]):
    schedule_free_optimizers = [
        optimizer for optimizer in optimizers if _optimizer_is_schedule_free(optimizer)
    ]
    for optimizer in schedule_free_optimizers:
        optimizer.eval()
    try:
        yield
    finally:
        for optimizer in schedule_free_optimizers:
            optimizer.train()


def _optimizer_logged_type(optimizer: torch.optim.Optimizer) -> str:
    return optimizer.__class__.__name__


def _build_scheduler(optimizer, config):
    scheduler_type = str(config.get("lr_scheduler", "none")).lower()
    if scheduler_type in {"", "none", "constant"}:
        return None
    if _optimizer_is_schedule_free(optimizer):
        raise ValueError("Schedule-free optimizer does not support an external lr_scheduler")
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported lr_scheduler={scheduler_type!r}")
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, int(config.get("lr_decay_steps", 3000))),
        eta_min=float(config.get("lr_min", 0.0)),
    )


def _build_feature_discriminator(config: dict, student, device: torch.device) -> torch.nn.Module:
    discriminator_config = config.get("discriminator", {})
    if "latent" in discriminator_config:
        raise ValueError("DMD GAN requires a teacher multi-feature discriminator, not a latent CNN")
    discriminator = TeacherFeatureDiscriminator(
        hidden_dim=int(student.transformer.transformer.config.dim),
        layer_numbers=discriminator_config.get("multifeature_layers", (4, 12, 20)),
        transformer_layers=int(discriminator_config.get("transformer_layers", 5)),
        transformer_heads=int(discriminator_config.get("transformer_heads", 8)),
        mlp_ratio=float(discriminator_config.get("mlp_ratio", 4.0)),
    )
    pretrained = discriminator_config.get("pretrained_checkpoint")
    if not pretrained:
        raise ValueError("Teacher-feature GAN requires discriminator.pretrained_checkpoint")
    report = discriminator.load_pretrained(pretrained)
    logger.info("DINO-distilled discriminator initialization: %s", report)
    return discriminator.to(device)


def _count_named_parameters(module: torch.nn.Module, needle: str) -> tuple[int, int]:
    params = [parameter for name, parameter in module.named_parameters() if needle in name]
    return len(params), sum(parameter.numel() for parameter in params)


def _merge_stats(
    target: dict[str, list[torch.Tensor]],
    stats: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    for key, value in stats.items():
        if not isinstance(value, tuple) or len(value) != 2:
            continue
        total, count = value
        if key not in target:
            target[key] = [
                torch.zeros((), device=total.device, dtype=torch.float64),
                torch.zeros((), device=count.device, dtype=torch.float64),
            ]
        target[key][0] = target[key][0] + total.detach().to(torch.float64)
        target[key][1] = target[key][1] + count.detach().to(torch.float64)


def _stats_to_means(stats: dict[str, list[torch.Tensor]]) -> dict[str, float]:
    means = {}
    for key, (total, count) in stats.items():
        count_value = float(count.item())
        means[key] = float((total / max(count_value, 1.0)).item())
    return means


def _append_stats_row(path: Path, row: dict[str, float | int | bool]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    keys = list(row.keys())
    if not write_header:
        with path.open(newline="", encoding="utf-8") as handle:
            keys = next(csv.reader(handle, delimiter="\t"))
    with path.open("a", encoding="utf-8") as handle:
        if write_header:
            handle.write("\t".join(keys) + "\n")
        handle.write("\t".join(str(row.get(key, 0.0)) for key in keys) + "\n")


def _set_optimizer_hparams(
    optimizer: torch.optim.Optimizer,
    *,
    lr: Optional[float] = None,
    betas: Optional[tuple[float, float]] = None,
) -> None:
    for group in optimizer.param_groups:
        if lr is not None:
            group["lr"] = float(lr)
        if betas is not None:
            group["betas"] = tuple(float(value) for value in betas)


def _clear_optimizer_state(
    optimizer: torch.optim.Optimizer,
    *,
    log_prefix: str,
) -> tuple[int, int]:
    before = len(optimizer.state)
    optimizer.state.clear()
    after = len(optimizer.state)
    logger.info(
        "%s optimizer_state_clear: entries_before=%d entries_after=%d", log_prefix, before, after
    )
    return before, after


def _optimizer_betas(config: dict) -> tuple[float, float]:
    return tuple(float(value) for value in config.get("betas", [0.9, 0.999]))


def _set_requires_gradient_sync(module: torch.nn.Module, enabled: bool) -> None:
    setter = getattr(module, "set_requires_gradient_sync", None)
    if setter is not None:
        setter(bool(enabled), recurse=True)


class _FSDP2GradientSyncContext:
    def __init__(self, module: torch.nn.Module, enabled: bool):
        self.module = module
        self.enabled = bool(enabled)

    def __enter__(self):
        _set_requires_gradient_sync(self.module, self.enabled)
        return None

    def __exit__(self, exc_type, exc, tb):
        _set_requires_gradient_sync(self.module, True)
        return False


def _gradient_sync_context(module: torch.nn.Module, enabled: bool):
    if enabled:
        return nullcontext()
    no_sync = getattr(module, "no_sync", None)
    if no_sync is not None:
        return no_sync()
    return _FSDP2GradientSyncContext(module, False)


@contextmanager
def _temporarily_requires_grad(parameters: list[torch.nn.Parameter], enabled: bool):
    original = [parameter.requires_grad for parameter in parameters]
    for parameter in parameters:
        parameter.requires_grad_(enabled)
    try:
        yield
    finally:
        for parameter, requires_grad in zip(parameters, original, strict=True):
            parameter.requires_grad_(requires_grad)


@torch.no_grad()
def _clip_grad_norm_for(
    module: torch.nn.Module,
    parameters: list[torch.nn.Parameter],
    max_norm: float,
) -> float:
    clipper = getattr(module, "clip_grad_norm_", None)
    if clipper is not None:
        norm = clipper(float(max_norm))
        if isinstance(norm, torch.Tensor):
            return float(norm.detach().float().item())
        return float(norm)
    return clip_grad_norm(parameters, max_norm)


@torch.no_grad()
def _sanitize_nonfinite_grads(parameters: list[torch.nn.Parameter]) -> int:
    replacements = 0
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local = grad
        if hasattr(local, "to_local"):
            local = local.to_local()
        finite = torch.isfinite(local)
        bad = int((~finite).sum().item())
        if bad:
            replacements += bad
            local.nan_to_num_(nan=0.0, posinf=0.0, neginf=0.0)
    if dist.is_available() and dist.is_initialized():
        count = torch.tensor(replacements, device=parameters[0].device if parameters else "cpu")
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        replacements = int(count.item())
    return replacements


@torch.no_grad()
def _global_grad_squared_norm(parameters: list[torch.nn.Parameter]) -> torch.Tensor:
    """Squared L2 norm of the current gradients across all ranks, without clipping."""
    if not parameters:
        raise ValueError("Gradient norm requires at least one parameter")
    total = torch.zeros((), device=parameters[0].device, dtype=torch.float64)
    found = False
    for parameter in parameters:
        grad = parameter.grad
        if grad is None:
            continue
        local = grad
        if hasattr(local, "to_local"):
            local = local.to_local()
        total += local.detach().double().square().sum()
        found = True
    if found and dist.is_available() and dist.is_initialized():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
    return total


@torch.no_grad()
def _reduce_stats_to_global(stats: dict[str, list[torch.Tensor]]) -> None:
    if not (dist.is_available() and dist.is_initialized()):
        return
    for total, count in stats.values():
        dist.all_reduce(total, op=dist.ReduceOp.SUM)
        dist.all_reduce(count, op=dist.ReduceOp.SUM)


@torch.no_grad()
def _reduce_mean_scalar(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    reduced = value.detach().to(torch.float64)
    dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
    reduced = reduced / float(dist.get_world_size())
    return reduced.to(device=value.device, dtype=value.dtype)


@torch.no_grad()
def _save_latent_image(student, latent: torch.Tensor, path: Path) -> None:
    pixels = student.latents_to_pixels(latent[:1].to(student.device)).detach().float().cpu()
    pixels = pixels.clamp(-1, 1).add(1).mul(127.5).byte()
    array = pixels[0].permute(1, 2, 0).numpy()
    Image.fromarray(array).save(path, quality=95)


@torch.no_grad()
def _save_dmd_training_debug(
    output_dir: Path,
    student,
    step: int,
    text,
    latents: torch.Tensor,
    aux: dict[str, torch.Tensor],
) -> None:
    step_dir = output_dir / "debug_dmd_tensors" / f"step-{int(step):06d}"
    step_dir.mkdir(parents=True, exist_ok=True)
    gen_x_t = aux.get("gen_x_t", aux.get("ode_x_t"))
    gen_flow = aux.get("gen_flow", aux.get("ode_pred_flow"))
    gen_sigma = aux.get("gen_input_sigma")
    generator_formula_tensors = {}
    if gen_x_t is not None and gen_flow is not None and gen_sigma is not None:
        sigma_b = gen_sigma.view(-1, *([1] * (gen_x_t.ndim - 1))).to(
            device=gen_x_t.device, dtype=gen_x_t.dtype
        )
        generator_formula_tensors = {
            "generator_x0_minus_sigma_flow": gen_x_t - sigma_b * gen_flow,
            "generator_x0_minus_flow": gen_x_t - gen_flow,
            "generator_x0_plus_sigma_flow": gen_x_t + sigma_b * gen_flow,
            "generator_x0_minus_one_minus_sigma_flow": gen_x_t - (1.0 - sigma_b) * gen_flow,
        }
    tensors = {
        "real_latent": latents.detach(),
        "generator_noisy_latent": gen_x_t,
        "generator_noise": aux.get("gen_noise"),
        "generator_flow": gen_flow,
        "generator_denoised_latent": aux.get("x_fake"),
        "ode_x_t": aux.get("ode_x_t"),
        "ode_target_flow": aux.get("ode_target_flow"),
        "ode_pred_flow": aux.get("ode_pred_flow"),
        "ode_pred_x0": aux.get("pred_x0"),
        "dmd_noisy_latent": aux.get("dmd_noisy"),
        "dmd_noise": aux.get("dmd_noise"),
        "fake_score_pred_x0": aux.get("pred_fake_x0", aux.get("pred_x0")),
        "real_score_pred_x0": aux.get("pred_real_x0"),
        "fake_score_flow": aux.get("fake_score_flow"),
        "real_score_flow": aux.get("real_score_flow"),
        "gan_disc_noisy_fake": aux.get("gan_disc_noisy_fake"),
        "gen_input_sigma": aux.get("gen_input_sigma"),
        "dm_sigma": aux.get("dm_sigma"),
        "dmd_fake_score_t": aux.get("dmd_fake_score_t"),
        "dmd_fake_score_tt": aux.get("dmd_fake_score_tt"),
    }
    tensors.update(generator_formula_tensors)
    tensors = {
        key: value.detach().float().cpu() for key, value in tensors.items() if value is not None
    }
    torch.save(tensors, step_dir / "tensors.pt")
    prompts = list(text) if isinstance(text, (list, tuple)) else [str(text)]
    (step_dir / "prompts.txt").write_text(
        "\n".join(f"{index:02d}\t{prompt}" for index, prompt in enumerate(prompts)) + "\n",
        encoding="utf-8",
    )
    metadata_lines = [
        f"step={int(step)}",
        "generator_noisy_latent = gen_input_sigma * generator_noise + (1 - gen_input_sigma) * real_latent",
        "generator_denoised_latent = generator_noisy_latent - gen_input_sigma * generator_flow",
        "ode_x_t = clean_latent * (1 - gen_input_sigma) + noise_latent * gen_input_sigma during ODE warmup",
        "ode_pred_x0 = ode_x_t - gen_input_sigma * ode_pred_flow during ODE warmup",
        "dmd_noisy_latent = dm_sigma * dmd_noise + (1 - dm_sigma) * generator_denoised_latent",
        "fake_score_pred_x0/real_score_pred_x0 are predictions from dmd_noisy_latent, not from generator_noisy_latent",
    ]
    for key in ("gen_input_sigma", "dm_sigma", "dmd_fake_score_t", "dmd_fake_score_tt"):
        value = tensors.get(key)
        if value is not None:
            metadata_lines.append(f"{key}={value.flatten().tolist()}")
    (step_dir / "metadata.txt").write_text("\n".join(metadata_lines) + "\n", encoding="utf-8")
    image_fields = {
        "real_latent": latents,
        "generator_noisy_latent": gen_x_t,
        "generator_denoised_latent": aux.get("x_fake"),
        "ode_x_t": aux.get("ode_x_t"),
        "ode_pred_x0": aux.get("pred_x0"),
        "generator_x0_minus_flow": generator_formula_tensors.get("generator_x0_minus_flow"),
        "generator_x0_plus_sigma_flow": generator_formula_tensors.get(
            "generator_x0_plus_sigma_flow"
        ),
        "generator_x0_minus_one_minus_sigma_flow": generator_formula_tensors.get(
            "generator_x0_minus_one_minus_sigma_flow"
        ),
        "dmd_noisy_latent": aux.get("dmd_noisy"),
        "fake_score_pred_x0": aux.get("pred_fake_x0", aux.get("pred_x0")),
        "real_score_pred_x0": aux.get("pred_real_x0"),
    }
    for name, tensor in image_fields.items():
        if tensor is not None:
            _save_latent_image(student, tensor.detach(), step_dir / f"{name}.jpg")


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def train_dmd(config):
    context = initialize_distributed(config["distributed"].get("backend", "nccl"))
    set_seed(config["runtime"].get("seed", 42), context.rank)
    method = build_algorithm(config["method"]["name"], config["method"]["params"])
    student, score = _build_models(config, context.device)
    fsdp_config = config.get("distributed", {})
    fsdp_backend = str(fsdp_config.get("fsdp_backend", "fsdp2")).strip().lower()
    if context.world_size <= 1 or fsdp_backend != "fsdp1":
        student.transformer.transformer.float()
        score["real"].transformer.float()
        score["fake"].transformer.float()
    generator_sync_module = student.transformer.transformer
    score_sync_module = score["fake"].transformer
    discriminator = None
    discriminator_sync_module = None
    discriminator_clip_module = None
    discriminator_parameters: list[torch.nn.Parameter] = []
    gan_objective_enabled = bool(getattr(method, "uses_gan_objective", lambda: False)())
    teacher_feature_objective_enabled = bool(
        getattr(method, "uses_teacher_feature_objective", lambda: False)()
    )
    if gan_objective_enabled:
        discriminator = _build_feature_discriminator(config, student, context.device)
        if context.is_main_process:
            logger.info(
                "DMD discriminator: frozen real teacher features=%s; OPD token head params=%d",
                discriminator.feature_layers,
                sum(p.numel() for p in discriminator.parameters()),
            )
        discriminator.train()
        discriminator_sync_module = discriminator
        discriminator_clip_module = discriminator
    generator_clip_module = student.transformer
    score_clip_module = score["fake"]
    if context.world_size > 1:
        fsdp_param_dtype_default = torch.bfloat16 if fsdp_backend == "fsdp1" else torch.float32
        fsdp_reduce_dtype_default = torch.float32 if fsdp_backend == "fsdp1" else torch.bfloat16
        fsdp_param_dtype = _dtype_from_config(
            fsdp_config.get("param_dtype"), fsdp_param_dtype_default
        )
        fsdp_reduce_dtype = _dtype_from_config(
            fsdp_config.get("reduce_dtype"), fsdp_reduce_dtype_default
        )
        fsdp_buffer_dtype = _dtype_from_config(fsdp_config.get("buffer_dtype"), torch.float32)
        if fsdp_backend == "fsdp1":
            if context.is_main_process:
                logger.info(
                    "DMD FSDP1 precision: param_dtype=%s forward_autocast=bfloat16 "
                    "reduce_dtype=%s buffer_dtype=%s",
                    fsdp_param_dtype,
                    fsdp_reduce_dtype,
                    fsdp_buffer_dtype,
                )
            no_split_gen = student.get_no_split_modules()
            no_split_score = list(getattr(score["fake"].transformer, "_no_split_modules", []))
            student.transformer = apply_zimage_fsdp1(
                student.transformer,
                no_split_modules=no_split_gen,
                local_rank=context.local_rank,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
                buffer_dtype=fsdp_buffer_dtype,
            )
            score["real"] = apply_zimage_fsdp1(
                score["real"],
                no_split_modules=no_split_score,
                local_rank=context.local_rank,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
                buffer_dtype=fsdp_buffer_dtype,
            )
            score["fake"] = apply_zimage_fsdp1(
                score["fake"],
                no_split_modules=no_split_score,
                local_rank=context.local_rank,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
                buffer_dtype=fsdp_buffer_dtype,
            )
            if discriminator is not None:
                discriminator = apply_zimage_fsdp1(
                    discriminator,
                    no_split_modules=("TransformerEncoderLayer",),
                    local_rank=context.local_rank,
                    param_dtype=fsdp_param_dtype,
                    reduce_dtype=fsdp_reduce_dtype,
                    buffer_dtype=fsdp_buffer_dtype,
                )
            generator_sync_module = student.transformer
            score_sync_module = score["fake"]
            discriminator_sync_module = discriminator
            generator_clip_module = student.transformer
            score_clip_module = score["fake"]
            discriminator_clip_module = discriminator
        elif fsdp_backend == "fsdp2":
            if discriminator is not None:
                raise ValueError("DMD GAN generator_objective currently requires FSDP1")
            if context.is_main_process:
                logger.info(
                    "DMD FSDP2 precision: param_dtype=%s forward_autocast=bfloat16 reduce_dtype=%s",
                    fsdp_param_dtype,
                    fsdp_reduce_dtype,
                )
            apply_zimage_fsdp2(
                student.transformer.transformer,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
            )
            apply_zimage_fsdp2(
                score["real"].transformer,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
            )
            apply_zimage_fsdp2(
                score["fake"].transformer,
                param_dtype=fsdp_param_dtype,
                reduce_dtype=fsdp_reduce_dtype,
            )
        else:
            raise ValueError(f"Unsupported distributed.fsdp_backend={fsdp_backend!r}")
    student.transformer.train()
    score["fake"].train()
    score["real"].eval()
    if discriminator is not None:
        discriminator.train()
    if context.is_main_process:
        logger.info(
            "DMD generator dtype signature after FSDP/float: %s",
            _dtype_signature(student.transformer),
        )
        logger.info(
            "DMD fake_score dtype signature after FSDP/float: %s", _dtype_signature(score["fake"])
        )

    generator_parameters = [
        parameter for parameter in student.transformer.parameters() if parameter.requires_grad
    ]
    score_parameters = [
        parameter for parameter in score["fake"].parameters() if parameter.requires_grad
    ]
    if not score_parameters:
        raise RuntimeError("No trainable fake-score parameters were created")
    if discriminator is not None:
        discriminator_parameters = [
            parameter for parameter in discriminator.parameters() if parameter.requires_grad
        ]
        if not discriminator_parameters:
            raise RuntimeError("No trainable GAN discriminator parameters were created")
    generator_optimizer = _build_optimizer(
        generator_parameters,
        config["optimizer"]["generator"],
        role="generator",
        fsdp_backend=fsdp_backend,
    )
    score_optimizer = _build_optimizer(
        score_parameters,
        config["optimizer"]["fake_score"],
        role="fake_score",
        fsdp_backend=fsdp_backend,
    )
    discriminator_optimizer = None
    if discriminator is not None:
        discriminator_optimizer = _build_optimizer(
            discriminator_parameters,
            config["optimizer"].get("discriminator", {"lr": 1.0e-5}),
            role="discriminator",
            fsdp_backend=fsdp_backend,
        )
    generator_scheduler = _build_scheduler(generator_optimizer, config["optimizer"]["generator"])
    score_scheduler = _build_scheduler(score_optimizer, config["optimizer"]["fake_score"])
    discriminator_scheduler = None
    if discriminator_optimizer is not None:
        discriminator_scheduler = _build_scheduler(
            discriminator_optimizer,
            config["optimizer"].get("discriminator", {}),
        )
    _optimizer_train(generator_optimizer)
    _optimizer_train(score_optimizer)
    if discriminator_optimizer is not None:
        _optimizer_train(discriminator_optimizer)
    generator_base_lr = float(config["optimizer"]["generator"]["lr"])
    score_base_lr = float(config["optimizer"]["fake_score"]["lr"])
    discriminator_base_lr = float(
        config["optimizer"].get("discriminator", {"lr": 0.0}).get("lr", 0.0)
    )
    generator_base_betas = _optimizer_betas(config["optimizer"]["generator"])
    score_base_betas = _optimizer_betas(config["optimizer"]["fake_score"])
    checkpoint_modules = {
        "generator": student.transformer,
        "fake_score": score["fake"],
    }
    if discriminator is not None:
        checkpoint_modules["discriminator"] = discriminator
    checkpoint_model = torch.nn.ModuleDict(checkpoint_modules)
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
    ode_loader = None
    ode_iterator = None
    ode_epoch = 0
    ode_pair_dir = str(config["method"]["params"].get("ode_warmup_pair_dir", "") or "")
    ode_warmup_enabled = bool(
        getattr(method, "warmup_type", "none") == "ode_pair"
        and int(getattr(method, "warmup_iterations", 0)) > 0
        and float(getattr(method, "ode_warmup_loss_weight", 0.0)) != 0.0
    )
    if ode_warmup_enabled:
        if not ode_pair_dir:
            raise ValueError("method.params.ode_warmup_pair_dir is required for ODE warmup")
        ode_dataset = build_dataset(
            {
                "format": "ode_pair",
                "pair_dir": ode_pair_dir,
                "reward_weighting": config["method"]["params"].get(
                    "ode_warmup_reward_weighting", "source_rank"
                ),
                "reward_weights": config["method"]["params"].get("ode_warmup_reward_weights"),
                "rank_weight_strength": config["method"]["params"].get(
                    "ode_warmup_rank_weight_strength", 1.0
                ),
            }
        )
        ode_loader = build_dataloader(
            ode_dataset,
            rank=context.rank,
            world_size=context.world_size,
            batch_size=1,
            num_workers=int(config["runtime"].get("num_workers", 4)),
        )
        if len(ode_loader) == 0:
            raise ValueError("ODE warmup dataloader is empty")
        set_sampler_epoch(ode_loader, ode_epoch)
        ode_iterator = iter(ode_loader)
        logger.info("DMD ODE warmup pairs: dir=%s samples=%d", ode_pair_dir, len(ode_dataset))
    generator_warmup_lr = config["optimizer"]["generator"].get("warmup_lr", generator_base_lr)
    score_warmup_lr = config["optimizer"]["fake_score"].get("warmup_lr", score_base_lr)
    generator_warmup_betas = tuple(
        float(value)
        for value in config["optimizer"]["generator"].get(
            "warmup_betas",
            config["optimizer"]["generator"].get("generator_warmup_betas", generator_base_betas),
        )
    )
    score_warmup_betas = tuple(
        float(value)
        for value in config["optimizer"]["fake_score"].get("warmup_betas", score_base_betas)
    )
    if ode_warmup_enabled:
        _set_optimizer_hparams(
            generator_optimizer,
            lr=float(generator_warmup_lr),
            betas=generator_warmup_betas,
        )
        _set_optimizer_hparams(
            score_optimizer,
            lr=float(score_warmup_lr),
            betas=score_warmup_betas,
        )
    accumulation = int(config["runtime"].get("gradient_accumulation_steps", 1))
    if accumulation < 1:
        raise ValueError("gradient_accumulation_steps must be at least 1")
    max_steps = int(config["runtime"].get("max_train_steps", 0))
    max_grad_norm = float(config["optimizer"]["generator"].get("max_grad_norm", 1.0))
    score_max_grad_norm = float(
        config["optimizer"]["fake_score"].get("max_grad_norm", max_grad_norm)
    )
    discriminator_max_grad_norm = float(
        config["optimizer"].get("discriminator", {}).get("max_grad_norm", max_grad_norm)
    )
    save_every = int(config["runtime"].get("save_every_n_steps", 1000))
    param_probe_every = int(config["runtime"].get("param_probe_every_n_steps", 50) or 0)
    output_dir = Path(config["runtime"].get("output_dir", "outputs"))
    if hasattr(score["fake"], "set_aux_time_log_path"):
        score["fake"].set_aux_time_log_path(output_dir / "fake_score_aux_time.tsv")
    if context.is_main_process:
        fake_time_rotary_tensors, fake_time_rotary_params = _count_named_parameters(
            score["fake"], "time_rotary"
        )
        logger.info(
            "DMD hyperparameters: generator_lr=%s generator_warmup_lr=%s "
            "fake_score_lr=%s fake_score_warmup_lr=%s generator_clip=%s "
            "fake_score_clip=%s discriminator_lr=%s discriminator_clip=%s "
            "grad_accum=%s dfake_gen_update_ratio=%s cfg_real=%s "
            "cfg_fake=%s nfe=%s timestep_shift=%s warmup_type=%s warmup_iterations=%s "
            "debug_cfg=%s debug_timestep_shift=%s generator_betas=%s "
            "generator_warmup_betas=%s fake_score_betas=%s fake_score_warmup_betas=%s "
            "generator_optimizer=%s fake_score_optimizer=%s discriminator_optimizer=%s "
            "generator_objective=%s",
            config["optimizer"]["generator"].get("lr"),
            config["optimizer"]["generator"].get("warmup_lr"),
            config["optimizer"]["fake_score"].get("lr"),
            config["optimizer"]["fake_score"].get("warmup_lr"),
            max_grad_norm,
            score_max_grad_norm,
            discriminator_base_lr,
            discriminator_max_grad_norm,
            accumulation,
            getattr(method, "dfake_gen_update_ratio", None),
            getattr(method, "real_guidance_scale", None),
            getattr(method, "fake_guidance_scale", None),
            getattr(method, "num_denoising_step", None),
            getattr(method, "timestep_shift", None),
            getattr(method, "warmup_type", None),
            getattr(method, "warmup_iterations", None),
            config["runtime"].get("debug_cfg_scale"),
            config["runtime"].get("debug_timestep_shift"),
            generator_base_betas,
            generator_warmup_betas,
            score_base_betas,
            score_warmup_betas,
            _optimizer_logged_type(generator_optimizer),
            _optimizer_logged_type(score_optimizer),
            (
                _optimizer_logged_type(discriminator_optimizer)
                if discriminator_optimizer is not None
                else "None"
            ),
            getattr(method, "generator_objective", None),
        )
        logger.info(
            "DMD fake_score aux-time: aux_time_embed=%s time_rotary_tensors=%d "
            "time_rotary_params=%d fake_score_use_generator_timestep=%s",
            getattr(score["fake"], "aux_time_embed", None),
            fake_time_rotary_tensors,
            fake_time_rotary_params,
            getattr(method, "fake_score_use_generator_timestep", None),
        )
        if (
            teacher_feature_objective_enabled
            or getattr(method, "score_objective", "mse") == "teacher_feature_mse"
        ):
            logger.info(
                "Teacher representations: layers=%s latent=%s last=%s timestep=%s "
                "generator_normalize=%s generator_weights=%s score_objective=%s score_weights=%s",
                getattr(method, "teacher_feature_layers", None),
                getattr(method, "teacher_feature_include_latent", None),
                getattr(method, "teacher_feature_include_last", None),
                getattr(method, "teacher_feature_timestep", None),
                getattr(method, "teacher_feature_normalize", None),
                getattr(method, "generator_teacher_feature_weights", None),
                getattr(method, "score_objective", None),
                getattr(method, "score_teacher_feature_weights", None),
            )
    global_step = 0
    resume_from = str(config["runtime"].get("resume_from", "") or "")
    resume_optimizer_state = bool(config["runtime"].get("resume_optimizer_state", True))
    allow_partial_optimizer_state = bool(
        config["runtime"].get("allow_partial_optimizer_state", False)
    )
    allow_partial_model_state = bool(config["runtime"].get("allow_partial_model_state", False))
    if resume_from:
        optimizers = [generator_optimizer, score_optimizer]
        if discriminator_optimizer is not None:
            optimizers.append(discriminator_optimizer)
        if context.world_size > 1:
            if resume_optimizer_state:
                global_step = load_distributed_training_state(
                    resume_from,
                    checkpoint_model,
                    optimizers,
                    allow_partial_optimizer_state=allow_partial_optimizer_state,
                )
            else:
                global_step = load_distributed_model_state(
                    resume_from,
                    checkpoint_model,
                    allow_partial_model_state=allow_partial_model_state,
                )
        else:
            state = load_training_state(resume_from, map_location=context.device)
            student.transformer.load_state_dict(state["generator"])
            score["fake"].load_state_dict(state["fake_score"])
            if discriminator is not None:
                if "discriminator" in state:
                    discriminator.load_state_dict(state["discriminator"])
                elif context.is_main_process:
                    logger.info(
                        "Resume checkpoint has no GAN discriminator state; using fresh init"
                    )
            if resume_optimizer_state:
                generator_optimizer.load_state_dict(state["generator_optimizer"])
                score_optimizer.load_state_dict(state["score_optimizer"])
                if discriminator_optimizer is not None and "discriminator_optimizer" in state:
                    discriminator_optimizer.load_state_dict(state["discriminator_optimizer"])
                elif discriminator_optimizer is not None and context.is_main_process:
                    logger.info(
                        "Resume checkpoint has no GAN discriminator optimizer state; "
                        "using fresh optimizer"
                    )
            global_step = int(state["step"])
    _optimizer_train(generator_optimizer)
    _optimizer_train(score_optimizer)
    if discriminator_optimizer is not None:
        _optimizer_train(discriminator_optimizer)
    epoch = 0
    set_sampler_epoch(loader, epoch)
    data_iterator = iter(loader)
    score_balance_counter = 0
    generator_balance_counter = 0
    warmup_iterations = int(getattr(method, "warmup_iterations", 0))
    was_in_ode_warmup = bool(
        ode_warmup_enabled and global_step > 0 and global_step <= warmup_iterations
    )
    generator_optimizer_reset_after_warmup = bool(
        ode_warmup_enabled and global_step > warmup_iterations
    )
    try:
        while max_steps <= 0 or global_step < max_steps:
            global_step += 1
            method.set_train_step(global_step)
            in_ode_warmup = bool(getattr(method, "is_ode_pair_warmup_step", lambda: False)())
            current_accumulation = 1 if in_ode_warmup else accumulation
            if in_ode_warmup:
                _set_optimizer_hparams(
                    generator_optimizer,
                    lr=float(generator_warmup_lr),
                    betas=generator_warmup_betas,
                )
                _set_optimizer_hparams(
                    score_optimizer,
                    lr=float(score_warmup_lr),
                    betas=score_warmup_betas,
                )
            if (
                was_in_ode_warmup
                and not in_ode_warmup
                and not generator_optimizer_reset_after_warmup
            ):
                if context.is_main_process:
                    _clear_optimizer_state(
                        generator_optimizer,
                        log_prefix=f"[optimizer-reset][gen][step={global_step}]",
                    )
                else:
                    generator_optimizer.state.clear()
                _set_optimizer_hparams(
                    generator_optimizer,
                    lr=generator_base_lr,
                    betas=generator_base_betas,
                )
                _set_optimizer_hparams(
                    score_optimizer,
                    lr=score_base_lr,
                    betas=score_base_betas,
                )
                generator_optimizer_reset_after_warmup = True
                if context.is_main_process:
                    logger.info(
                        "[optimizer-reset] step=%d reset generator optimizer after ODE warmup; "
                        "generator lr=%s betas=%s; fake_score lr=%s betas=%s",
                        global_step,
                        generator_base_lr,
                        generator_base_betas,
                        score_base_lr,
                        score_base_betas,
                    )
            update_generator = method.should_update_generator(global_step)
            update_score = True
            update_discriminator = False
            if teacher_feature_objective_enabled and not in_ode_warmup:
                fake_updates = int(method.dfake_gen_update_ratio)
                phase = (global_step - int(method.warmup_iterations) - 1) % (fake_updates + 1)
                update_score = phase < fake_updates
                update_generator = phase == fake_updates
            if discriminator is not None and not in_ode_warmup:
                # Each F/D/G is a complete GA window and one optimizer update.
                fake_updates = int(getattr(method, "dfake_gen_update_ratio", 5))
                phase = (global_step - int(getattr(method, "warmup_iterations", 0)) - 1) % (
                    fake_updates + 2
                )
                update_score = phase < fake_updates
                update_discriminator = phase == fake_updates
                update_generator = phase == fake_updates + 1
            score_optimizer.zero_grad(set_to_none=True)
            if update_discriminator:
                discriminator_optimizer.zero_grad(set_to_none=True)
            if update_generator:
                generator_optimizer.zero_grad(set_to_none=True)
            accumulated_score = torch.zeros((), device=context.device)
            accumulated_discriminator = torch.zeros((), device=context.device)
            accumulated_generator = torch.zeros((), device=context.device)
            score_stats_accum: dict[str, list[torch.Tensor]] = {}
            discriminator_stats_accum: dict[str, list[torch.Tensor]] = {}
            generator_stats_accum: dict[str, list[torch.Tensor]] = {}
            generator_debug_payload = None
            debug_every = int(config["runtime"].get("debug_every_n_steps", 0) or 0)
            save_dmd_debug = bool(
                update_generator
                and debug_every > 0
                and global_step % debug_every == 0
                and context.is_main_process
            )

            def fetch_training_batch():
                nonlocal epoch, ode_epoch, data_iterator, ode_iterator
                if in_ode_warmup:
                    if ode_iterator is None or ode_loader is None:
                        raise ValueError(
                            "ODE warmup requested but ODE dataloader is not initialized"
                        )
                    try:
                        batch = next(ode_iterator)
                    except StopIteration:
                        ode_epoch += 1
                        set_sampler_epoch(ode_loader, ode_epoch)
                        ode_iterator = iter(ode_loader)
                        batch = next(ode_iterator)
                    text = batch["text"]
                    latents = batch["clean_latent"].to(
                        context.device, dtype=torch.float32, non_blocking=True
                    )
                    initial_noise = batch["noise_latent"].to(
                        context.device, dtype=torch.float32, non_blocking=True
                    )
                    ode_weight = batch.get("ode_weight")
                    if ode_weight is not None:
                        ode_weight = ode_weight.to(
                            context.device, dtype=torch.float32, non_blocking=True
                        )
                else:
                    try:
                        batch = next(data_iterator)
                    except StopIteration:
                        epoch += 1
                        set_sampler_epoch(loader, epoch)
                        data_iterator = iter(loader)
                        batch = next(data_iterator)
                    text, image = extract_image_batch(batch, "DMD")
                    image = image.to(context.device, non_blocking=True)
                    with torch.no_grad():
                        latents = student.pixels_to_latents(image).float()
                    initial_noise = None
                    ode_weight = None
                with torch.no_grad():
                    prompt, prompt_mask, uncond, uncond_mask = student.encode_prompt(
                        text, do_cfg=True
                    )
                return {
                    "batch": batch,
                    "text": text,
                    "latents": latents,
                    "initial_noise": initial_noise,
                    "ode_weight": ode_weight,
                    "c": [prompt.float(), prompt_mask.float()],
                    "e": [uncond.float(), uncond_mask.float()],
                }

            def score_forward(item):
                return method.score_loss(
                    generator_model=student.transformer,
                    score_model=score,
                    x_real=item["latents"],
                    c=item["c"],
                    e=item["e"],
                    latent_shape=item["latents"].shape,
                )

            def generator_forward(item, return_debug_tensors=False):
                return method.generator_loss(
                    generator_model=student.transformer,
                    score_model=score,
                    discriminator_model=discriminator,
                    x_real=item["latents"],
                    c=item["c"],
                    e=item["e"],
                    latent_shape=item["latents"].shape,
                    initial_noise=item["initial_noise"],
                    ode_weight=item["ode_weight"],
                    return_debug_tensors=return_debug_tensors,
                )

            balance_lookup = getattr(method, "teacher_feature_balance_for", None)
            score_balance = (
                balance_lookup("score") if update_score and balance_lookup is not None else None
            )
            generator_balance = (
                balance_lookup("generator")
                if update_generator and balance_lookup is not None
                else None
            )
            if in_ode_warmup:
                score_balance = None
                generator_balance = None
            active_balance_role = (
                "score"
                if score_balance is not None
                else "generator"
                if generator_balance is not None
                else None
            )
            prefetched_items = None
            if active_balance_role is not None:
                balance = score_balance if active_balance_role == "score" else generator_balance
                balance_parameters = (
                    score_parameters if active_balance_role == "score" else generator_parameters
                )
                balance_module = (
                    score_clip_module if active_balance_role == "score" else generator_clip_module
                )
                balance_sync_module = (
                    score_sync_module if active_balance_role == "score" else generator_sync_module
                )
                balance_forward = (
                    score_forward if active_balance_role == "score" else generator_forward
                )
                balance_counter = (
                    score_balance_counter
                    if active_balance_role == "score"
                    else generator_balance_counter
                )
                prefetched_items = [fetch_training_batch() for _ in range(current_accumulation)]
                if balance_counter % balance["measure_every_n_updates"] == 0:
                    probe_items = (
                        prefetched_items
                        if balance["probe_accumulation"] == "full"
                        else prefetched_items[-1:]
                    )
                    balance_keys = method.teacher_feature_representation_keys()
                    probe_rng = capture_rng_state()
                    squared_norms: dict[str, float] = {}
                    for balance_key in balance_keys:
                        with method.use_teacher_feature_weights(
                            active_balance_role,
                            {key: 1.0 if key == balance_key else 0.0 for key in balance_keys},
                        ):
                            restore_rng_state(probe_rng)
                            balance_module.zero_grad(set_to_none=True)
                            for probe_index, probe_item in enumerate(probe_items):
                                with _gradient_sync_context(
                                    balance_sync_module,
                                    probe_index == len(probe_items) - 1,
                                ):
                                    with torch.autocast(
                                        device_type=context.device.type,
                                        dtype=torch.bfloat16,
                                        enabled=context.device.type == "cuda",
                                        cache_enabled=False,
                                    ):
                                        probe_loss, _ = balance_forward(probe_item)
                                    (probe_loss / len(probe_items)).backward()
                        squared_norms[balance_key] = float(
                            _global_grad_squared_norm(balance_parameters).item()
                        )
                        balance_module.zero_grad(set_to_none=True)
                    restore_rng_state(probe_rng)
                    if not all(math.isfinite(value) for value in squared_norms.values()):
                        if context.is_main_process:
                            logger.warning(
                                "[grad-balance][%s][step=%d] non-finite probe norms %s; "
                                "keeping the previous weights",
                                active_balance_role,
                                global_step,
                                squared_norms,
                            )
                        measured_balance = False
                    else:
                        balance_weights, balance_norms = method.teacher_feature_balance_weights(
                            squared_norms, role=active_balance_role
                        )
                        method.set_teacher_feature_weights(active_balance_role, balance_weights)
                        measured_balance = True
                    if measured_balance and context.is_main_process:
                        logger.info(
                            "[grad-balance][%s][step=%d] probe=%s norms=%s weights=%s",
                            active_balance_role,
                            global_step,
                            balance["probe_accumulation"],
                            {key: round(value, 6) for key, value in balance_norms.items()},
                            {key: round(value, 6) for key, value in balance_weights.items()},
                        )
                        _append_stats_row(
                            output_dir / "grad_balance.tsv",
                            {
                                "step": global_step,
                                "role": active_balance_role,
                                "probe": balance["probe_accumulation"],
                                **{f"norm_{key}": balance_norms[key] for key in balance_keys},
                                **{f"weight_{key}": balance_weights[key] for key in balance_keys},
                                **{
                                    f"contribution_{key}": balance_weights[key] * balance_norms[key]
                                    for key in balance_keys
                                },
                            },
                        )
                if active_balance_role == "score":
                    score_balance_counter += 1
                else:
                    generator_balance_counter += 1

            for accumulation_index in range(current_accumulation):
                is_last_accumulation = accumulation_index == current_accumulation - 1
                if prefetched_items is not None:
                    item = prefetched_items[accumulation_index]
                else:
                    item = fetch_training_batch()
                batch = item["batch"]
                text = item["text"]
                latents = item["latents"]
                initial_noise = item["initial_noise"]
                c = item["c"]
                e = item["e"]
                if update_score:
                    with _gradient_sync_context(score_sync_module, is_last_accumulation):
                        with torch.autocast(
                            device_type=context.device.type,
                            dtype=torch.bfloat16,
                            enabled=context.device.type == "cuda",
                            cache_enabled=False,
                        ):
                            score_loss, score_stats = score_forward(item)
                        (score_loss / current_accumulation).backward()
                    accumulated_score += score_loss.detach()
                    _merge_stats(score_stats_accum, score_stats)

                if update_discriminator:
                    with _gradient_sync_context(
                        discriminator_sync_module,
                        is_last_accumulation,
                    ):
                        with torch.autocast(
                            device_type=context.device.type,
                            dtype=torch.bfloat16,
                            enabled=context.device.type == "cuda",
                            cache_enabled=False,
                        ):
                            discriminator_loss, discriminator_stats = method.discriminator_loss(
                                generator_model=student.transformer,
                                score_model=score,
                                discriminator_model=discriminator,
                                x_real=latents,
                                c=c,
                                e=e,
                                latent_shape=latents.shape,
                                initial_noise=initial_noise,
                            )
                        (discriminator_loss / current_accumulation).backward()
                    accumulated_discriminator += discriminator_loss.detach()
                    _merge_stats(discriminator_stats_accum, discriminator_stats)

                if update_generator:
                    return_generator_debug = bool(save_dmd_debug and accumulation_index == 0)
                    # Frozen through the backward as well: gradient checkpointing recomputes the
                    # forward during backward, and a requires_grad change in between breaks it.
                    generator_frozen_parameters = list(discriminator_parameters)
                    if getattr(
                        method, "generator_teacher_feature_needs_live_query", lambda: False
                    )():
                        generator_frozen_parameters += list(score_parameters)
                    with (
                        _temporarily_requires_grad(
                            generator_frozen_parameters,
                            False,
                        )
                        if generator_frozen_parameters
                        else nullcontext()
                    ):
                        with _gradient_sync_context(generator_sync_module, is_last_accumulation):
                            with torch.autocast(
                                device_type=context.device.type,
                                dtype=torch.bfloat16,
                                enabled=context.device.type == "cuda",
                                cache_enabled=False,
                            ):
                                generator_out = generator_forward(
                                    item, return_debug_tensors=return_generator_debug
                                )
                                if return_generator_debug:
                                    generator_loss, generator_stats, generator_debug = generator_out
                                else:
                                    generator_loss, generator_stats = generator_out
                            (generator_loss / current_accumulation).backward()
                    accumulated_generator += generator_loss.detach()
                    _merge_stats(generator_stats_accum, generator_stats)
                    if in_ode_warmup:
                        _merge_stats(
                            generator_stats_accum,
                            method._pack_loss_stats(
                                ode_reward=batch["ode_reward"].to(
                                    device=context.device, dtype=torch.float32
                                ),
                                ode_reward_score=batch["ode_reward_score"].to(
                                    device=context.device, dtype=torch.float32
                                ),
                                ode_reward_aesthetic=batch["ode_reward_aesthetic"].to(
                                    device=context.device, dtype=torch.float32
                                ),
                                ode_reward_instruction_following=batch[
                                    "ode_reward_instruction_following"
                                ].to(device=context.device, dtype=torch.float32),
                                ode_reward_color_harmony=batch["ode_reward_color_harmony"].to(
                                    device=context.device, dtype=torch.float32
                                ),
                            ),
                        )
                    if return_generator_debug:
                        generator_debug_payload = (
                            list(text) if isinstance(text, (list, tuple)) else text,
                            latents.detach(),
                            {key: value.detach() for key, value in generator_debug.items()},
                        )
            score_nonfinite_grads = 0
            score_grad_norm = 0.0
            if update_score:
                score_nonfinite_grads = _sanitize_nonfinite_grads(score_parameters)
                score_grad_norm = _clip_grad_norm_for(
                    score_clip_module,
                    score_parameters,
                    score_max_grad_norm,
                )
            discriminator_nonfinite_grads = 0
            discriminator_grad_norm = 0.0
            if update_discriminator:
                discriminator_nonfinite_grads = _sanitize_nonfinite_grads(discriminator_parameters)
                discriminator_grad_norm = _clip_grad_norm_for(
                    discriminator_clip_module,
                    discriminator_parameters,
                    discriminator_max_grad_norm,
                )
            do_param_probe = bool(
                context.is_main_process
                and param_probe_every > 0
                and global_step % param_probe_every == 0
            )
            score_grad_abs = _grad_abs(score_parameters) if do_param_probe else 0.0
            score_probe_before = _capture_param_probe(score["fake"]) if do_param_probe else {}
            if update_score:
                score_optimizer.step()
                if score_scheduler is not None and not in_ode_warmup:
                    score_scheduler.step()
            if update_discriminator:
                discriminator_optimizer.step()
                if discriminator_scheduler is not None:
                    discriminator_scheduler.step()
            generator_grad_norm = 0.0
            generator_grad_abs = 0.0
            generator_delta_abs = 0.0
            generator_delta_max = 0.0
            generator_nonfinite_grads = 0
            if update_generator:
                generator_nonfinite_grads = _sanitize_nonfinite_grads(generator_parameters)
                generator_grad_norm = _clip_grad_norm_for(
                    generator_clip_module,
                    generator_parameters,
                    max_grad_norm,
                )
                generator_grad_abs = _grad_abs(generator_parameters) if do_param_probe else 0.0
                generator_probe_before = (
                    _capture_param_probe(student.transformer) if do_param_probe else {}
                )
                generator_optimizer.step()
                if generator_scheduler is not None and not in_ode_warmup:
                    generator_scheduler.step()
                if do_param_probe:
                    generator_delta_abs, generator_delta_max = _param_delta_stats(
                        student.transformer,
                        generator_probe_before,
                    )
            score_delta_abs = 0.0
            score_delta_max = 0.0
            if do_param_probe:
                score_delta_abs, score_delta_max = _param_delta_stats(
                    score["fake"],
                    score_probe_before,
                )
            if generator_debug_payload is not None and context.is_main_process:
                debug_text, debug_latents, debug_aux = generator_debug_payload
                _save_dmd_training_debug(
                    output_dir,
                    student,
                    global_step,
                    debug_text,
                    debug_latents,
                    debug_aux,
                )
            _reduce_stats_to_global(score_stats_accum)
            _reduce_stats_to_global(discriminator_stats_accum)
            _reduce_stats_to_global(generator_stats_accum)
            accumulated_score_global = _reduce_mean_scalar(accumulated_score)
            accumulated_discriminator_global = _reduce_mean_scalar(accumulated_discriminator)
            accumulated_generator_global = _reduce_mean_scalar(accumulated_generator)
            if context.is_main_process:
                score_means = _stats_to_means(score_stats_accum)
                discriminator_means = _stats_to_means(discriminator_stats_accum)
                generator_means = _stats_to_means(generator_stats_accum)
                stats_row = {
                    "step": global_step,
                    "score_loss": float(accumulated_score_global / current_accumulation),
                    "discriminator_loss": float(
                        accumulated_discriminator_global / current_accumulation
                    ),
                    "generator_loss": float(accumulated_generator_global / current_accumulation),
                    "generator_updated": int(update_generator),
                    "gan_discriminator": int(discriminator is not None),
                    "discriminator_updated": int(update_discriminator),
                    "score_updated": int(update_score),
                    "ode_warmup": int(in_ode_warmup),
                    "accumulation": current_accumulation,
                    "score_grad_norm": score_grad_norm,
                    "discriminator_grad_norm": discriminator_grad_norm,
                    "generator_grad_norm": generator_grad_norm,
                    "score_nonfinite_grads": score_nonfinite_grads,
                    "discriminator_nonfinite_grads": discriminator_nonfinite_grads,
                    "generator_nonfinite_grads": generator_nonfinite_grads
                    if update_generator
                    else 0,
                    "score_lr": score_optimizer.param_groups[0]["lr"],
                    "discriminator_lr": (
                        discriminator_optimizer.param_groups[0]["lr"]
                        if discriminator_optimizer is not None
                        else 0.0
                    ),
                    "generator_lr": generator_optimizer.param_groups[0]["lr"],
                    "score_beta1": score_optimizer.param_groups[0].get("betas", (0.0, 0.0))[0],
                    "generator_beta1": generator_optimizer.param_groups[0].get("betas", (0.0, 0.0))[
                        0
                    ],
                    "probe/fake_score_grad_abs": score_grad_abs,
                    "probe/fake_score_delta_abs": score_delta_abs,
                    "probe/fake_score_delta_max": score_delta_max,
                    "probe/generator_grad_abs": generator_grad_abs,
                    "probe/generator_delta_abs": generator_delta_abs,
                    "probe/generator_delta_max": generator_delta_max,
                }
                for key in sorted(score_means):
                    stats_row[f"score/{key}"] = score_means[key]
                for key in sorted(discriminator_means):
                    stats_row[f"gan/{key}"] = discriminator_means[key]
                for key in sorted(generator_means):
                    stats_row[f"gen/{key}"] = generator_means[key]
                for key in _tracked_stats_keys(method):
                    stats_row.setdefault(key, 0.0)
                _append_stats_row(output_dir / "dmd_stats.tsv", stats_row)
                logger.info(
                    "step=%d score_loss=%.6g discriminator_loss=%.6g "
                    "generator_loss=%.6g ode_warmup=%d score_grad_norm=%.6g "
                    "discriminator_grad_norm=%.6g generator_grad_norm=%.6g "
                    "score_lr=%.6g discriminator_lr=%.6g generator_lr=%.6g "
                    "score_beta1=%.3g generator_beta1=%.3g "
                    "score_bad_grads=%d discriminator_bad_grads=%d generator_bad_grads=%d "
                    "dm_grad_abs=%.6g gan_fake_logit=%.6g gen_sigma=%.6g score_sigma=%.6g",
                    global_step,
                    float(accumulated_score_global / current_accumulation),
                    float(accumulated_discriminator_global / current_accumulation),
                    float(accumulated_generator_global / current_accumulation),
                    int(in_ode_warmup),
                    score_grad_norm,
                    discriminator_grad_norm,
                    generator_grad_norm,
                    score_optimizer.param_groups[0]["lr"],
                    (
                        discriminator_optimizer.param_groups[0]["lr"]
                        if discriminator_optimizer is not None
                        else 0.0
                    ),
                    generator_optimizer.param_groups[0]["lr"],
                    score_optimizer.param_groups[0].get("betas", (0.0, 0.0))[0],
                    generator_optimizer.param_groups[0].get("betas", (0.0, 0.0))[0],
                    score_nonfinite_grads,
                    discriminator_nonfinite_grads,
                    generator_nonfinite_grads if update_generator else 0,
                    generator_means.get("dm_grad_abs", 0.0),
                    generator_means.get(
                        "gan_gen_fake_logit",
                        discriminator_means.get("gan_disc_fake_logit", 0.0),
                    ),
                    generator_means.get("gen_input_sigma", 0.0),
                    score_means.get("score_sigma", 0.0),
                )
                if do_param_probe:
                    logger.info(
                        "param_probe step=%d fake_grad_abs=%.6g fake_delta_abs=%.6g "
                        "fake_delta_max=%.6g gen_grad_abs=%.6g gen_delta_abs=%.6g "
                        "gen_delta_max=%.6g",
                        global_step,
                        score_grad_abs,
                        score_delta_abs,
                        score_delta_max,
                        generator_grad_abs,
                        generator_delta_abs,
                        generator_delta_max,
                    )
            eval_optimizers = [generator_optimizer, score_optimizer]
            if discriminator_optimizer is not None:
                eval_optimizers.append(discriminator_optimizer)
            with _optimizers_eval_context(eval_optimizers):
                save_debug_samples(student, method, config, context, global_step)
            if save_every > 0 and global_step % save_every == 0:
                with _optimizers_eval_context(eval_optimizers):
                    if context.world_size > 1:
                        if context.is_main_process:
                            logger.info("Saving distributed DMD checkpoint at step=%d", global_step)
                        save_optimizers = [generator_optimizer, score_optimizer]
                        if discriminator_optimizer is not None:
                            save_optimizers.append(discriminator_optimizer)
                        save_distributed_training_state(
                            output_dir / "checkpoints" / f"step-{global_step}",
                            checkpoint_model,
                            save_optimizers,
                            step=global_step,
                        )
                        if context.is_main_process:
                            logger.info("Saved distributed DMD checkpoint at step=%d", global_step)
                    elif context.is_main_process:
                        save_training_state(
                            output_dir / "checkpoints" / f"step-{global_step}.pt",
                            {
                                "step": global_step,
                                "generator": student.transformer.state_dict(),
                                "fake_score": score["fake"].state_dict(),
                                **(
                                    {"discriminator": discriminator.state_dict()}
                                    if discriminator is not None
                                    else {}
                                ),
                                "generator_optimizer": generator_optimizer.state_dict(),
                                "score_optimizer": score_optimizer.state_dict(),
                                **(
                                    {
                                        "discriminator_optimizer": (
                                            discriminator_optimizer.state_dict()
                                        )
                                    }
                                    if discriminator_optimizer is not None
                                    else {}
                                ),
                            },
                        )
            was_in_ode_warmup = in_ode_warmup
    finally:
        cleanup_distributed()
    return global_step
