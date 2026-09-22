"""Detailed production-independent DMD diagnostics and 64-prompt six-call evaluation."""
from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image, ImageDraw

from verl_distill.data.qwen_image21 import atomic_json, sha256, within
from verl_distill.engine.qwen_checkpoint import capture_rng_state, restore_rng_state, collective_call
from verl_distill.models.qwen_image21.modeling import predict_velocity

logger = logging.getLogger(__name__)


def debug_event_name(state):
    if not state.dmd_initialized and state.fake_updates == 0:
        return f"reflow_step_{state.reflow_updates:06d}"
    return f"fake_step_{state.fake_updates:06d}"


def tensor_stats(value):
    value = value.detach().float()
    finite = torch.isfinite(value)
    if not bool(finite.all()):
        raise FloatingPointError("Nonfinite diagnostic tensor")
    return {"shape": list(value.shape), "dtype": str(value.dtype),
            "mean": value.mean().item(), "std": value.std(unbiased=False).item(),
            "min": value.min().item(), "max": value.max().item(),
            "l2": value.norm().item(), "finite_fraction": finite.float().mean().item()}


def cosine(a, b):
    a, b = a.detach().float().flatten(), b.detach().float().flatten()
    denom = a.norm() * b.norm()
    return (a.dot(b) / denom).item() if denom > 0 else None


def labeled_grid(items, path, *, cell=384):
    columns = min(4, len(items))
    rows = (len(items) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * cell, rows * (cell + 40)), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (image, label) in enumerate(items):
        thumb = image.convert("RGB")
        thumb.thumbnail((cell, cell))
        x, y = (i % columns) * cell, (i // columns) * (cell + 40)
        canvas.paste(thumb, (x, y + 40))
        draw.text((x + 4, y + 4), label, fill="black")
    canvas.save(path)
    canvas.close()


def heatmap(value, path, scale):
    # Fixed scale across all events. Actual extrema and clipping are recorded separately.
    x = value.detach().float().abs().clamp(0, scale).div(scale).cpu().numpy()
    colors = np.stack([x, np.zeros_like(x), 1 - x], axis=-1)
    Image.fromarray((colors * 255).astype(np.uint8)).save(path)


def save_train_sample(directory, snapshot, decoder, reference_root, *, diff_scale=1.):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    record, tensors = snapshot["record"], snapshot["tensors"]
    stats, files = {}, {}
    for name, value in tensors.items():
        if isinstance(value, torch.Tensor):
            stats[name] = tensor_stats(value)
            torch.save(value, directory / f"{name}.pt")
    images, pixels = {}, {}
    for name in ("generator_x0", "fake_x0", "real_x0"):
        if name in tensors:
            image, pixel = decoder.decode(tensors[name], record["height"], record["width"])
            image.save(directory / f"{name}.png")
            images[name], pixels[name] = image, pixel
    if "diff_x0" in tensors:
        diff = tensors["diff_x0"].float()
        absolute = diff.abs()
        torch.save(absolute, directory / "abs_diff_x0.pt")
        spatial = absolute.mean(-1).reshape(record["height"] // 16, record["width"] // 16)
        heatmap(spatial, directory / "latent_diff_heatmap.png", diff_scale)
        pixel_diff = pixels["fake_x0"] - pixels["real_x0"]
        torch.save(pixel_diff, directory / "pixel_diff.pt")
        pixel_heat = pixel_diff.abs().mean(1)[0]
        heatmap(pixel_heat, directory / "pixel_diff_heatmap.png", 2.)
        stats["difference"] = {
            "mae": absolute.mean().item(), "rmse": diff.square().mean().sqrt().item(),
            "l2": diff.norm().item(), "max": absolute.max().item(),
            "cosine_fake_real": cosine(tensors["fake_x0"], tensors["real_x0"]),
            "fake_target_mse": (tensors["fake_x0"] - tensors["generator_x0"]).square().mean().item(),
            "real_target_mse": (tensors["real_x0"] - tensors["generator_x0"]).square().mean().item(),
            "latent_heatmap_scale": diff_scale,
            "latent_heatmap_clipped_fraction": (spatial > diff_scale).float().mean().item(),
            "pixel_heatmap_scale": 2., "pixel_difference_space": "raw VAE pixels (normally [-1,1])"}
    refs = []
    for index, name in enumerate(record["reference_images"]):
        reference_path = within(reference_root, name)
        if sha256(reference_path) != record["reference_sha256"][index]:
            raise ValueError(f"Debug reference image changed: {reference_path}")
        with Image.open(reference_path) as image:
            refs.append((image.convert("RGB"), f"reference {index+1}"))
    labeled_grid(refs + [(image, name) for name, image in images.items()], directory / "comparison.png")
    for image, _ in refs:
        image.close()
    for image in images.values():
        image.close()
    for path in directory.iterdir():
        if path.is_file():
            files[path.name] = sha256(path)
    atomic_json(directory / "metadata.json", {**snapshot["metadata"], "record": record,
                "tensor_stats": stats, "sha256": files,
                "latent_layout": "B,target_tokens,64 (diffusion-normalized)",
                "diff_definition": "fake_x0-real_x0; images subtracted AFTER separate VAE decoding"})


@torch.no_grad()
def rollout_sample(model, condition, record, schedule, device, *, steps=6,
                   official_schedule=False, capture_trajectory=True):
    levels = (schedule.levels(record["height"], record["width"], device=device)
              if steps == 6 and not official_schedule else
              schedule.levels(record["height"], record["width"], steps,
                              generator=not official_schedule, device=device))
    rng = torch.Generator(device).manual_seed(record["seed"])
    # Generate in BF16 as in the validated production pipeline; integrate in FP32.
    spatial = torch.randn((1, 1, 64, record["height"] // 16, record["width"] // 16),
                          generator=rng, device=device, dtype=torch.bfloat16)
    x = spatial.reshape(1, 64, -1).transpose(1, 2).float()
    initial = x.detach().cpu().clone()
    trajectory = []
    for index, (sigma, next_sigma) in enumerate(zip(levels[:-1], levels[1:], strict=True)):
        velocity = predict_velocity(model, x, sigma.reshape(1), condition)
        x0 = x - sigma * velocity
        step = {"sigma": sigma.item(), "next_sigma": next_sigma.item(), "index": index}
        if capture_trajectory:
            step.update(latent=x.cpu(), velocity=velocity.cpu(), predicted_x0=x0.cpu())
        trajectory.append(step)
        x = x + (next_sigma - sigma) * velocity
    return initial, x.cpu(), trajectory


def save_rollout(directory, record, result, decoder, metadata):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    initial, final, trajectory = result
    torch.save(initial, directory / "initial_noise.pt")
    torch.save(final, directory / "final_x0_latent.pt")
    previews, statistics = [], []
    for step in trajectory:
        if "latent" not in step:
            statistics.append(step)
            continue
        step_dir = directory / "steps" / f"{step['index']:02d}"
        step_dir.mkdir(parents=True)
        stat = {k: step[k] for k in ("index", "sigma", "next_sigma")}
        for name in ("latent", "velocity", "predicted_x0"):
            torch.save(step[name], step_dir / f"{name}.pt")
            stat[name] = tensor_stats(step[name])
        image, _ = decoder.decode(step["predicted_x0"], record["height"], record["width"])
        image.save(step_dir / "predicted_x0.png")
        previews.append((image, f"step {step['index']+1} sigma={step['sigma']:.4f}"))
        statistics.append(stat)
    image, _ = decoder.decode(final, record["height"], record["width"])
    image.save(directory / "image.png")
    previews.append((image, "final"))
    if len(previews) > 1:
        labeled_grid(previews, directory / "trajectory.png")
    for preview, _ in previews:
        preview.close()
    files = {str(p.relative_to(directory)): sha256(p) for p in directory.rglob("*") if p.is_file()}
    atomic_json(directory / "metadata.json", {**metadata, "record": record, "steps": statistics,
                "initial_noise": tensor_stats(initial), "final_x0": tensor_stats(final),
                "nfe": len(trajectory), "cfg": "conditional_only", "kv_cache": False,
                "sha256": files})


def debug_event(output, state, model, snapshots, evaluation, store, schedule, decoder,
                device, rank, world_size, reference_root, prompts_csv, contract,
                prompt_count=64, comparison_steps=0):
    """Every rank enters every evaluation round, including deterministic padding rounds."""
    import time

    rows = evaluation["records"][:prompt_count]
    if not rows or len(rows) != prompt_count:
        raise ValueError("Debug prompt count exceeds evaluation manifest")
    event = Path(output) / "debug" / debug_event_name(state)
    rng = capture_rng_state()
    was_training = model.training
    started = time.monotonic()
    try:
        def reserve():
            if rank == 0:
                event.mkdir(parents=True, exist_ok=False)
                (event / "prompts_snapshot.csv").write_bytes(Path(prompts_csv).read_bytes())
                atomic_json(event / "event.json", {"rollout_updates": state.state_dict(),
                    "contract": contract, "generator_terminal": .4,
                    "prompt_ids": [r["id"] for r in rows],
                    "comparison_steps": comparison_steps, "cfg": "conditional_only",
                    "training_snapshots": "actual pre-optimizer forward tensors",
                    "created_at": time.time()})
        collective_call("reserve debug event (never overwrite)", reserve)
        if rank == 0:
            logger.info("Debug %s: starting %d paired rollouts (6 / %d steps)",
                        event.name, len(rows), comparison_steps)
        model.eval()
        for phase in ("fake_score", "generator"):
            def save_local():
                for snapshot in snapshots.get(phase, []):
                    row = snapshot["record"]
                    ga = snapshot["metadata"]["ga_index"]
                    directory = event / ("train_fake" if phase == "fake_score" else "train_dmd")
                    directory = directory / f"rank_{rank:03d}" / f"ga_{ga:02d}" / row["id"]
                    save_train_sample(directory, snapshot, decoder, reference_root)
            collective_call(f"write {phase} debug tensors", save_local)
        local_count = 0
        for start in range(0, len(rows), world_size):
            index = start + rank
            row = rows[index] if index < len(rows) else rows[0]
            condition = collective_call("load rollout condition", lambda: store.get(row, device))
            sample_start = time.monotonic()
            # All ranks execute six FSDP forwards, even the non-writing padding ranks.
            result = rollout_sample(model, condition, row, schedule, device)
            def save_local_rollout():
                if index < len(rows):
                    save_rollout(event / "rollout" / row["id"], row, result, decoder,
                                 {"updates": state.state_dict(), "rank": rank,
                                  "seconds_before_decode": time.monotonic() - sample_start,
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                  "contract": contract})
            collective_call("write rollout and trajectory", save_local_rollout)
            if comparison_steps:
                comparison = rollout_sample(model, condition, row, schedule, device,
                                            steps=comparison_steps, official_schedule=True,
                                            capture_trajectory=False)
                def save_comparison():
                    if index < len(rows):
                        if not torch.equal(result[0], comparison[0]):
                            raise RuntimeError("Comparison initial noise differs")
                        save_rollout(event / f"rollout_{comparison_steps}step" / row["id"],
                                     row, comparison, decoder,
                                     {"updates": state.state_dict(), "rank": rank,
                                      "scheduler": schedule.config,
                                      "schedule": "official_unmodified", "contract": contract})
                collective_call("write official-schedule comparison", save_comparison)
            if rank == 0:
                logger.info("Debug %s: rollout %d/%d written", event.name,
                            min(start + world_size, len(rows)), len(rows))
            local_count += int(index < len(rows))
        counts = [None] * world_size
        dist.all_gather_object(counts, local_count)
        if sum(counts) != len(rows):
            raise RuntimeError("Debug event did not cover selected prompts")
        def commit():
            if rank == 0:
                atomic_json(event / "summary.json", {"rollout_completed": sum(counts), "failed": 0,
                    "comparison_completed": sum(counts) if comparison_steps else 0,
                    "comparison_steps": comparison_steps,
                    "training_samples_per_phase": {k: len(v) * world_size for k, v in snapshots.items()},
                    "elapsed_seconds": time.monotonic() - started})
                atomic_json(event / "COMPLETE", {"updates": state.state_dict()})
        collective_call("commit debug event", commit)
    except Exception as exc:
        # Each worker records its own failure path; absence of COMPLETE is authoritative.
        try:
            atomic_json(event / f"failure-rank-{rank:03d}.json", {"error": repr(exc)})
        except OSError:
            pass
        raise
    finally:
        try:
            decoder.offload()
        finally:
            model.train(was_training)
            restore_rng_state(rng)
