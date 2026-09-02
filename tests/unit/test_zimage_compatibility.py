import pytest

from verl_distill.models.zimage.compatibility import require_zimage_diffusers


def test_compatibility_error_names_missing_exports(monkeypatch):
    import diffusers

    monkeypatch.delattr(diffusers, "ZImagePipeline", raising=False)
    monkeypatch.delattr(diffusers, "ZImageTransformer2DModel", raising=False)
    with pytest.raises(RuntimeError, match="does not provide Z-Image support"):
        require_zimage_diffusers()
