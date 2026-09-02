from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torchvision.utils import save_image


def set_seed(seed: int, rank: int) -> None:
    value = int(seed) + int(rank)
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def extract_image_batch(batch, method_name: str):
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch mapping, got {type(batch)!r}")
    image = batch.get("image")
    if not torch.is_tensor(image) or image.numel() == 0:
        raise ValueError(f"{method_name} requires image tensors")
    return batch["text"], image


def extract_batch(batch):
    if not isinstance(batch, dict):
        raise TypeError(f"Expected batch mapping, got {type(batch)!r}")
    return batch["text"], batch.get("image")


def set_sampler_epoch(loader, epoch: int) -> None:
    set_epoch = getattr(getattr(loader, "sampler", None), "set_epoch", None)
    if set_epoch is not None:
        set_epoch(int(epoch))


@torch.no_grad()
def save_debug_samples(student, method, config, context, step: int) -> None:
    runtime = config["runtime"]
    every = int(runtime.get("debug_every_n_steps", 0) or 0)
    if every <= 0 or int(step) % every:
        return
    prompts = list(
        runtime.get(
            "debug_prompts",
            ["a photo of a bench", "a photo of a cow in a green field"],
        )
    )
    if not prompts:
        return
    output_dir = Path(runtime.get("output_dir", "outputs")) / "debug_samples"
    was_training = student.training
    student.eval()
    try:
        with torch.autocast(
            device_type=context.device.type,
            dtype=torch.bfloat16,
            enabled=context.device.type == "cuda",
        ):
            images = student.sample(
                prompts,
                cfg_scale=float(runtime.get("debug_cfg_scale", 0.0)),
                seed=int(runtime.get("debug_seed", 42)),
                height=int(runtime.get("debug_height", config["data"].get("resolution", 1024))),
                width=int(runtime.get("debug_width", config["data"].get("resolution", 1024))),
                return_traj=bool(runtime.get("debug_return_traj", False)),
                sampler=method.sampling_loop,
                sampler_kwargs={
                    "sampling_steps": int(config["method"]["params"].get("sampling_steps", 4)),
                    **(
                        {
                            "timestep_shift": float(
                                runtime.get(
                                    "debug_timestep_shift",
                                    config["method"]["params"].get(
                                        "debug_timestep_shift",
                                        config["method"]["params"].get("timestep_shift", 5.0),
                                    ),
                                )
                            )
                        }
                        if config["method"]["name"] in {"dmd", "dmd_full"}
                        else {}
                    ),
                    **(
                        {"flow_shift": float(config["method"]["params"]["flow_shift"])}
                        if config["method"]["name"] == "meanflow"
                        else {}
                    ),
                },
            )
        if context.is_main_process:
            step_dir = output_dir / f"step-{int(step):06d}"
            step_dir.mkdir(parents=True, exist_ok=True)
            images = images.detach().float().cpu().clamp(-1, 1)
            if bool(runtime.get("debug_return_traj", False)):
                if images.shape[0] % len(prompts) != 0:
                    raise ValueError(
                        f"debug trajectory image count {images.shape[0]} is not divisible "
                        f"by prompt count {len(prompts)}"
                    )
                columns = images.shape[0] // len(prompts)
                grid = (
                    images.view(columns, len(prompts), *images.shape[-3:])
                    .permute(1, 0, 2, 3, 4)
                    .reshape(-1, *images.shape[-3:])
                )
                save_image((grid + 1) / 2, step_dir / "trajectory.png", nrow=columns)
            else:
                pixels = images.add(1).mul(127.5).byte()
                for index, (prompt, pixel) in enumerate(zip(prompts, pixels, strict=True)):
                    array = pixel.permute(1, 2, 0).numpy()
                    Image.fromarray(array).save(step_dir / f"{index:02d}.jpg", quality=95)
            (step_dir / "prompts.txt").write_text(
                "\n".join(f"{index:02d}\t{prompt}" for index, prompt in enumerate(prompts)) + "\n",
                encoding="utf-8",
            )
    finally:
        student.train(was_training)
    if dist.is_available() and dist.is_initialized():
        barrier_kwargs = (
            {"device_ids": [context.local_rank]} if context.device.type == "cuda" else {}
        )
        dist.barrier(**barrier_kwargs)
