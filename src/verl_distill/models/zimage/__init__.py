def load_zimage():
    from verl_distill.models.zimage.compatibility import require_zimage_diffusers

    require_zimage_diffusers()
    from verl_distill.models.zimage.modeling import ZImage

    return ZImage


__all__ = ["load_zimage"]
