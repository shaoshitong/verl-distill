"""Teacher CFG with empty text and unchanged reference images."""

import math

import torch
from PIL import Image

from verl_distill.data.qwen_image21 import sha256, within


def combine_cfg(conditional, negative, scale):
    if not math.isfinite(scale) or scale < 1:
        raise ValueError("CFG scale must be finite and >=1")
    if scale == 1:
        return conditional
    if negative is None or negative.shape != conditional.shape:
        raise ValueError("CFG requires a matching negative prediction")
    # Promote before subtraction. Do not normalize away the strength of guidance.
    return negative.double() + scale * (conditional.double() - negative.double())


@torch.no_grad()
def encode_negative_condition(pipe, record, positive, reference_root, device, prompt=""):
    """Remove only text conditioning; retain ordered reference images and their latents."""
    from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

    images = []
    try:
        for relative, digest in zip(
            record["reference_images"], record["reference_sha256"], strict=True
        ):
            path = within(reference_root, relative)
            if sha256(path) != digest:
                raise ValueError(f"Reference image changed: {path}")
            with Image.open(path) as source:
                original = source.convert("RGBA")
            w, h, _ = calculate_dimensions(2048 * 2048, original.width / original.height)
            images.append(pipe.image_processor.resize(original, width=w, height=h))
            original.close()
        embeds, mask, image_mask = pipe.encode_prompt(
            prompt=prompt, image=images or None, device=device
        )
        tokens = record["height"] // 16 * (record["width"] // 16)
        image_mask = torch.cat([image_mask, image_mask.new_ones(1, tokens // 4)], 1)
        return {
            **positive,
            "encoder_hidden_states": embeds,
            "encoder_hidden_states_mask": mask,
            "img_mask": image_mask,
        }
    finally:
        for image in images:
            image.close()


def geometry_metrics(generated, fake, real):
    """Three-point geometry in FP64; a larger angle is not a quality objective."""
    h, f, r = [x.detach().double() for x in (generated, fake, real)]
    a, b, direction = (r - h).flatten(), (f - h).flatten(), (r - f).flatten()

    def cosine(x, y):
        denominator = x.norm() * y.norm()
        return (x.dot(y) / denominator).clamp(-1, 1).item() if denominator > 0 else None

    cos_h = cosine(a, b)
    return {
        "mse_HR": a.square().mean().item(),
        "mse_HF": b.square().mean().item(),
        "mse_RF": direction.square().mean().item(),
        "cos_HR_HF": cos_h,
        "angle_H_degrees": math.degrees(math.acos(cos_h)) if cos_h is not None else None,
        "cos_DMD_HR": cosine(direction, a),
        "direction_rms": direction.square().mean().sqrt().item(),
    }
