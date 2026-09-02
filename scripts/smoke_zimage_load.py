#!/usr/bin/env python3
import argparse

import torch

from verl_distill.models.zimage import load_zimage
from verl_distill.models.zimage.compatibility import require_zimage_diffusers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--aux-time-embed", action="store_true")
    args = parser.parse_args()
    require_zimage_diffusers()
    ZImage = load_zimage()
    model = ZImage(
        model_id=args.model,
        aux_time_embed=args.aux_time_embed,
        text_dtype=torch.bfloat16,
        imgs_dtype=torch.bfloat16,
        device=args.device,
    )
    parameter_count = sum(parameter.numel() for parameter in model.transformer.parameters())
    prompt, prompt_mask, uncond, uncond_mask = model.encode_prompt(
        ["a small red cube"], do_cfg=True
    )
    print(
        "ZIMAGE_LOAD_OK "
        f"parameters={parameter_count} prompt={tuple(prompt.shape)} "
        f"mask={tuple(prompt_mask.shape)} uncond={tuple(uncond.shape)} "
        f"uncond_mask={tuple(uncond_mask.shape)}"
    )


if __name__ == "__main__":
    main()
