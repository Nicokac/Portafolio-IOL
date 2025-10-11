"""Package bootstrapping the legacy cache module with new core primitives."""

from __future__ import annotations

import importlib.util
import logging
import sys
from pathlib import Path
from types import ModuleType
from typing import Iterable

from .core import CacheService, PredictiveCacheState


def _load_legacy_module() -> ModuleType:
    module_path = Path(__file__).resolve().parent.parent / "cache.py"
    spec = importlib.util.spec_from_file_location("services._cache_legacy", module_path)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive guard
        raise ImportError(f"Cannot load legacy cache module from {module_path!s}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["services._cache_legacy"] = module
    spec.loader.exec_module(module)
    return module


_legacy_module = _load_legacy_module()
_legacy_module.__name__ = __name__
_legacy_module.__package__ = __name__
_legacy_module.__file__ = str(Path(__file__).resolve())
_legacy_module.__path__ = [str(Path(__file__).resolve().parent)]
_legacy_module.CacheService = CacheService
_legacy_module.PredictiveCacheState = PredictiveCacheState
_legacy_module.logger = logging.getLogger(__name__)

sys.modules[__name__] = _legacy_module


def _exported_names(module: ModuleType) -> Iterable[str]:
    declared = getattr(module, "__all__", None)
    if declared:
        return declared
    return [name for name in module.__dict__ if not name.startswith("__")]


__all__ = sorted(set(_exported_names(_legacy_module)) | {"CacheService", "PredictiveCacheState"})
