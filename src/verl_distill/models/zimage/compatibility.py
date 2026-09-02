def require_zimage_diffusers() -> None:
    import diffusers

    required = ("ZImagePipeline", "ZImageTransformer2DModel")
    missing = [name for name in required if not hasattr(diffusers, name)]
    if missing:
        version = getattr(diffusers, "__version__", "unknown")
        names = ", ".join(missing)
        raise RuntimeError(
            "The installed diffusers build does not provide Z-Image support: "
            f"version={version}, missing={names}. Install verl-distill[train] "
            "to use the supported diffusers version."
        )
