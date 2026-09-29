"""Read-only model probe: paired Teacher CFG geometry and official 40-step images.

Never updates weights or starts training. An immutable probe manifest assigns
one sample per worker, with the same H/F/noise/sigma across CFG strengths.
"""
import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image

from verl_distill.data.qwen_image21 import atomic_json, canonical_hash, sha256, within
from verl_distill.models.qwen_image21.guidance import (
    combine_cfg, encode_negative_condition, geometry_metrics,
)
from verl_distill.models.qwen_image21.modeling import (
    ConditionStore, model_identity, predict_velocity, require_qwen_runtime,
)
from verl_distill.trainers.qwen_image21_debug import labeled_grid, tensor_stats


@torch.inference_mode()
def decode(pipe, packed, row):
    z = pipe._unpack_latents(packed, row["height"], row["width"], 16).to("cuda", dtype=pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, 64, 1, 1, 1)
    std = torch.tensor(pipe.vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, 64, 1, 1, 1)
    pixels = pipe.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0]
    if not torch.isfinite(pixels).all():
        raise FloatingPointError("Nonfinite VAE decode")
    return pipe.image_processor.postprocess(pixels.float(), output_type="pil")[0]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--sample-index", required=True, type=int)
    p.add_argument("--output", required=True)
    p.add_argument("--reference-root", default="/tmp/qwen21-dataset")
    p.add_argument("--geometry-only", action="store_true")
    args = p.parse_args()
    require_qwen_runtime()
    torch.cuda.set_device(0)
    manifest = json.loads(Path(args.manifest).read_text())
    selected = manifest["samples"][args.sample_index]
    metadata_path = Path(selected["metadata_path"])
    if sha256(metadata_path) != selected["metadata_sha256"]:
        raise ValueError("Source debug metadata changed")
    meta = json.loads(metadata_path.read_text())
    row = meta["record"]
    source = metadata_path.parent
    output = Path(args.output) / f"sample_{args.sample_index:02d}"
    output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    try:
        identity = model_identity(args.model)
        if canonical_hash(identity) != manifest["model_identity"]:
            raise ValueError("Probe teacher/model identity differs from training")
        store = ConditionStore(manifest["condition_cache"], identity, manifest["manifest_hashes"])

        def load(name):
            path = source / f"{name}.pt"
            if sha256(path) != meta["sha256"][path.name]:
                raise ValueError(f"Source tensor changed: {path}")
            return torch.load(path, map_location="cuda", weights_only=True)

        h, f, noisy, sigma = [load(name) for name in
                              ("generator_x0", "fake_x0", "score_noisy_latent", "score_sigma")]
        original_real = load("real_x0")
        from diffusers import QwenImage21Pipeline

        pipe = QwenImage21Pipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                   local_files_only=True)
        pipe.enable_model_cpu_offload(gpu_id=0)
        pipe.set_progress_bar_config(disable=True)
        positive = store.get(row, "cuda")
        negative = encode_negative_condition(pipe, row, positive, args.reference_root, "cuda")
        pipe.maybe_free_model_hooks()
        conditional_v = predict_velocity(pipe.transformer, noisy, sigma, positive)
        negative_v = predict_velocity(pipe.transformer, noisy, sigma, negative)
        pipe.maybe_free_model_hooks()
        rows, previews = [], []
        for name, value in {"generator_x0": h, "fake_x0": f, "noisy": noisy,
                            "score_sigma": sigma, "conditional_velocity": conditional_v,
                            "negative_velocity": negative_v}.items():
            torch.save(value.detach().cpu(), output / f"{name}.pt")
        for scale in manifest["cfg_scales"]:
            velocity = combine_cfg(conditional_v, negative_v, scale)
            real = noisy.double() - sigma.double().reshape(-1, 1, 1) * velocity.double()
            torch.save(real.cpu(), output / f"real_x0_cfg{scale:g}.pt")
            info = {"cfg": scale, **geometry_metrics(h, f, real), "real_x0": tensor_stats(real)}
            if scale == 1:
                info["conditional_replay_rmse_vs_training"] = (real - original_real.double()).square().mean().sqrt().item()
            rows.append(info)
            preview = decode(pipe, real, row)
            preview.save(output / f"real_x0_cfg{scale:g}.png")
            previews.append((preview, f"Teacher x0 CFG={scale:g}"))
            print(json.dumps({"sample": args.sample_index, **info}), flush=True)
        labeled_grid(previews, output / "score_comparison.png")
        for image, _ in previews:
            image.close()
        atomic_json(output / "geometry.json", {"source": str(source), "record": row,
                    "metrics": rows, "cfg_definition": "v_negative + scale*(v_conditional-v_negative)",
                    "negative_prompt": "", "reference_images_preserved": True,
                    "source_forward_backend": "training flash2_segmented", "probe_backend": "official_sdpa",
                    "caveat": "Differences include BF16/backend replay error; geometry alone is not quality."})
        del h, f, noisy, sigma, original_real, positive, negative, conditional_v, negative_v, real, velocity
        pipe.maybe_free_model_hooks()
        torch.cuda.empty_cache()
        if not args.geometry_only:
            images = []
            try:
                for relative, digest in zip(row["reference_images"], row["reference_sha256"], strict=True):
                    path = within(args.reference_root, relative)
                    if sha256(path) != digest:
                        raise ValueError("Reference image changed")
                    with Image.open(path) as im:
                        images.append(im.convert("RGBA"))
                previews = []
                for scale in manifest["cfg_scales"]:
                    print(f"rollout sample={args.sample_index} CFG={scale} start", flush=True)
                    torch.cuda.reset_peak_memory_stats()
                    tick = time.time()
                    packed = pipe(prompt=row["prompt"], image=images or None, negative_prompt="" if scale > 1 else None,
                                  true_cfg_scale=scale, width=row["width"], height=row["height"],
                                  output_resolution=2048, num_inference_steps=40, use_kv_cache=True,
                                  generator=torch.Generator("cuda").manual_seed(row["seed"]),
                                  output_type="latent").images
                    torch.save(packed.cpu(), output / f"rollout_cfg{scale:g}.pt")
                    im = decode(pipe, packed, row)
                    im.save(output / f"rollout_cfg{scale:g}.png")
                    previews.append((im, f"40 steps CFG={scale:g}"))
                    atomic_json(output / f"rollout_cfg{scale:g}.json", {
                        "cfg": scale, "steps": 40, "seed": row["seed"], "seconds": time.time() - tick,
                        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                        "final_latent_stats": tensor_stats(packed), "scheduler": dict(pipe.scheduler.config)})
                    del packed
                    pipe.maybe_free_model_hooks()
                    torch.cuda.empty_cache()
                labeled_grid(previews, output / "rollout_comparison.png", cell=640)
                for im, _ in previews:
                    im.close()
            finally:
                for im in images:
                    im.close()
        atomic_json(output / "COMPLETE", {"elapsed_seconds": time.time() - started,
                    "manifest_sha256": sha256(args.manifest), "geometry_only": args.geometry_only})
    except Exception as exc:
        atomic_json(output / "failure.json", {"error": repr(exc)})
        raise


if __name__ == "__main__":
    main()
