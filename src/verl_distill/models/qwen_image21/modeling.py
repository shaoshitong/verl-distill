"""Official Qwen conditioning and flow adapter; training never uses a persistent KV cache."""
from __future__ import annotations

import importlib.metadata as metadata
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from verl_distill.data.qwen_image21 import canonical_hash, sha256, within
from .configuration import CACHE_SCHEMA, DIFFUSERS_REVISION, MODEL_REVISION


def require_qwen_runtime():
    expected = {"torch": "2.8.0", "torchvision": "0.23.0", "transformers": "5.17.0"}
    for package, version in expected.items():
        actual = metadata.version(package)
        if actual.split("+")[0] != version:
            raise RuntimeError(f"Qwen requires {package}=={version}; found {actual}")
    direct = metadata.distribution("diffusers").read_text("direct_url.json")
    source = json.loads(direct or "{}")
    if source.get("vcs_info", {}).get("commit_id") != DIFFUSERS_REVISION:
        raise RuntimeError(f"Install Diffusers from Git commit {DIFFUSERS_REVISION}")


def model_identity(root):
    """Hash actual model assets, not just a user supplied revision string."""
    root = Path(root)
    required = ("transformer", "vae", "text_encoder", "processor", "scheduler")
    for name in required:
        if not (root / name).is_dir():
            raise FileNotFoundError(root / name)
    hashes = {}
    for path in sorted(root.rglob("*")):
        if path.is_file() and ".cache" not in path.parts:
            if path.suffix in (".json", ".safetensors", ".txt", ".model", ".jinja"):
                hashes[str(path.relative_to(root))] = sha256(path)
    if not any(k.endswith(".safetensors") for k in hashes):
        raise ValueError("No model weights found")
    return {"revision": MODEL_REVISION, "diffusers_revision": DIFFUSERS_REVISION,
            "transformers": metadata.version("transformers"), "files": hashes}


def condition_key(record, identity):
    return canonical_hash({"schema": CACHE_SCHEMA, "model": canonical_hash(identity),
                           "record": record, "reference_resolution": 2048})


def save_condition(cache_root, record, identity, condition):
    key = condition_key(record, identity)
    path = Path(cache_root) / key[:2] / f"{key}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    torch.save({"schema": CACHE_SCHEMA, "key": key, "condition": condition}, tmp)
    os.replace(tmp, path)
    return key, {"file": str(path.relative_to(cache_root)), "sha256": sha256(path)}


def open_reused_cache(descriptor, identity):
    if descriptor is None:
        return None
    root = Path(descriptor["root"])
    if sha256(root / "index.json") != descriptor["index_sha256"]:
        raise ValueError("Reused condition index digest changed")
    index = json.loads((root / "index.json").read_text())
    if index["model_identity"] != identity or index.get("reused_cache"):
        raise ValueError("Reused cache must be a matching self-contained cache")
    return root, index


def condition_entry_path(root, key, entry, reused):
    if entry.get("source") == "base":
        if reused is None:
            raise ValueError("Missing reused-cache provenance")
        base_root, base_index = reused
        if {k: v for k, v in entry.items() if k != "source"} != base_index["entries"].get(key):
            raise ValueError("Reused condition entry differs from pinned source index")
        return within(base_root, entry["file"])
    if "source" in entry:
        raise ValueError("Unknown condition source")
    return within(root, entry["file"])


class ConditionStore:
    def __init__(self, root, identity, manifest_hashes):
        self.root, self.identity = Path(root), identity
        self.index = json.loads((self.root / "index.json").read_text())
        if self.index.get("schema") != CACHE_SCHEMA:
            raise ValueError("Invalid condition cache schema")
        if self.index["model_identity"] != identity:
            raise ValueError("Condition cache/model runtime mismatch")
        if self.index["manifest_hashes"] != manifest_hashes:
            raise ValueError("Condition cache was built for different manifests")
        self.reused = open_reused_cache(self.index.get("reused_cache"), identity)
        self.verified = set()

    def validate_record(self, record):
        key = condition_key(record, self.identity)
        if key not in self.index["entries"]:
            raise FileNotFoundError(f"No cached condition for {record['id']}; prepare the cache first")
        return key

    def get(self, record, device, dtype=torch.bfloat16):
        key = self.validate_record(record)
        entry = self.index["entries"][key]
        path = condition_entry_path(self.root, key, entry, self.reused)
        if key not in self.verified:
            if sha256(path) != entry["sha256"]:
                raise ValueError(f"Condition cache digest mismatch: {path}")
            self.verified.add(key)
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["schema"] != CACHE_SCHEMA or saved["key"] != key:
            raise ValueError("Condition cache identity mismatch")
        result = {}
        for name, value in saved["condition"].items():
            if isinstance(value, torch.Tensor):
                value = value.to(device=device, dtype=dtype if value.is_floating_point() else value.dtype)
                if value.is_floating_point() and not torch.isfinite(value).all():
                    raise ValueError(f"Nonfinite condition tensor: {path} {name}")
            result[name] = value
        return result


def load_condition_pipeline(model_root, device):
    from diffusers import QwenImage21Pipeline

    pipe = QwenImage21Pipeline.from_pretrained(
        model_root, transformer=None, torch_dtype=torch.bfloat16, local_files_only=True)
    pipe.text_encoder.requires_grad_(False).eval()
    pipe.vae.requires_grad_(False).eval()
    pipe.text_encoder.to(device)
    pipe.vae.to(device)
    return pipe


@torch.no_grad()
def encode_condition(pipe, record, reference_root, device):
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    originals, prompt_images, vae_images, shapes = [], [], [], []
    try:
        for relative, digest in zip(record["reference_images"], record["reference_sha256"], strict=True):
            path = within(reference_root, relative)
            if sha256(path) != digest:
                raise ValueError(f"Reference image changed: {path}")
            with Image.open(path) as image:
                original = image.convert("RGBA")
            originals.append(original)
            w, h, _ = calculate_dimensions(2048 * 2048, original.width / original.height)
            prompt_images.append(pipe.image_processor.resize(original, width=w, height=h))
            vae_images.append(pipe.image_processor.preprocess(original, width=w, height=h).unsqueeze(2))
            shapes.append((1, h // 16, w // 16))
        embeds, mask, image_mask = pipe.encode_prompt(
            prompt=record["prompt"], image=prompt_images or None, device=device)
        target_tokens = (record["height"] // 16) * (record["width"] // 16)
        # Pass existing latents to avoid consuming RNG while encoding reference images.
        dummy = torch.zeros((1, target_tokens, 64), device=device, dtype=embeds.dtype)
        _, reference_latents = pipe.prepare_latents(
            vae_images or None, 1, 64, record["height"], record["width"],
            embeds.dtype, device, None, latents=dummy)
        shapes.append((1, record["height"] // 16, record["width"] // 16))
        image_mask = torch.cat([image_mask, image_mask.new_ones(1, target_tokens // 4)], dim=1)
        result = {"encoder_hidden_states": embeds, "encoder_hidden_states_mask": mask,
                  "img_mask": image_mask, "img_shapes": [shapes],
                  "reference_latents": reference_latents}
        return {k: v.detach().cpu().contiguous() if isinstance(v, torch.Tensor) else v
                for k, v in result.items()}
    finally:
        for image in originals + prompt_images:
            image.close()


class QwenSchedule:
    def __init__(self, model_root, terminal=0.4):
        from diffusers import FlowMatchEulerDiscreteScheduler

        self.config = dict(FlowMatchEulerDiscreteScheduler.from_pretrained(
            model_root, subfolder="scheduler", local_files_only=True).config)
        self.terminal = terminal
        self.cache = {}

    def levels(self, height, width, steps=6, *, generator=True, device="cpu"):
        from diffusers import FlowMatchEulerDiscreteScheduler
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_shift

        key = (height, width, steps, generator)
        if key not in self.cache:
            config = dict(self.config)
            if generator:
                config["shift_terminal"] = self.terminal
            scheduler = FlowMatchEulerDiscreteScheduler.from_config(config)
            tokens = (height // 16) * (width // 16)
            mu = calculate_shift(tokens, config["base_image_seq_len"], config["max_image_seq_len"],
                                 config["base_shift"], config["max_shift"])
            scheduler.set_timesteps(steps, device="cpu", sigmas=np.linspace(1, 1 / steps, steps), mu=mu)
            self.cache[key] = scheduler.sigmas.detach().float().clone()
        return self.cache[key].to(device)

    def score_sigma(self, height, width, low, high, device):
        levels = self.levels(height, width, 1000, generator=False, device=device)[:-1]
        # float32 terminal stretching can land just below .02: tolerate its rounding only.
        levels = levels[(levels >= low - 1e-7) & (levels <= high + 1e-7)].clamp(low, high)
        if not len(levels):
            raise ValueError("No score timesteps inside the requested sigma interval")
        return levels[torch.randint(len(levels), (1,), device=device)]


def predict_velocity(model, target_latents, sigma, condition):
    reference = condition["reference_latents"]
    hidden = target_latents.to(torch.bfloat16)
    if reference is not None:
        hidden = torch.cat([reference.to(hidden), hidden], dim=1)
    kwargs = {k: v for k, v in condition.items() if k != "reference_latents"}
    predicted = model(hidden_states=hidden, timestep=sigma.float().reshape(-1),
                      **kwargs, return_dict=False)[0]
    return predicted[:, -target_latents.shape[1]:].float()


def load_transformer(model_root, attention_backend="sdpa"):
    from diffusers import QwenImage21Transformer2DModel

    model = QwenImage21Transformer2DModel.from_pretrained(
        model_root, subfolder="transformer", torch_dtype=torch.bfloat16,
        local_files_only=True, low_cpu_mem_usage=True)
    if not model.config.causal_condition:
        raise ValueError("Expected the Qwen-Image-2.1 causal-condition checkpoint")
    from .attention import configure_attention
    return configure_attention(model, attention_backend)


class QwenDecoder:
    """Keep only the VAE on CPU between debug events; no production pipeline mutation."""
    def __init__(self, model_root, device):
        from diffusers import AutoencoderKLQwenImage21
        from diffusers.image_processor import VaeImageProcessor

        self.vae = AutoencoderKLQwenImage21.from_pretrained(
            model_root, subfolder="vae", torch_dtype=torch.bfloat16, local_files_only=True)
        self.vae.requires_grad_(False).eval()
        self.device = device
        self.processor = VaeImageProcessor(vae_scale_factor=16, vae_latent_channels=64)

    @torch.no_grad()
    def decode(self, packed, height, width):
        self.vae.to(self.device)
        z = packed.transpose(1, 2).reshape(1, 64, 1, height // 16, width // 16).to(
            device=self.device, dtype=self.vae.dtype)
        mean = torch.tensor(self.vae.config.latents_mean, device=z.device, dtype=z.dtype).view(1, 64, 1, 1, 1)
        std = torch.tensor(self.vae.config.latents_std, device=z.device, dtype=z.dtype).view(1, 64, 1, 1, 1)
        pixels = self.vae.decode(z * std + mean, return_dict=False)[0][:, :, 0].float()
        if not torch.isfinite(pixels).all():
            raise FloatingPointError("Nonfinite VAE output")
        image = self.processor.postprocess(pixels, output_type="pil")[0]
        return image, pixels.cpu()

    def offload(self):
        self.vae.cpu()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
