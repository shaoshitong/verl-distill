import sys
from types import ModuleType

import pytest

from verl_distill.models.zimage.compatibility import require_zimage_diffusers


def test_compatibility_error_names_missing_exports(monkeypatch):
    # A real diffusers lazy module would recreate attributes after delattr.
    monkeypatch.setitem(sys.modules, "diffusers", ModuleType("diffusers"))
    with pytest.raises(RuntimeError, match="does not provide Z-Image support"):
        require_zimage_diffusers()
