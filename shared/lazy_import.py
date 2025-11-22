"""Generic lazy import helper built on :mod:`importlib`."""

from __future__ import annotations

import importlib
from types import ModuleType

_MODULE_CACHE: dict[str, ModuleType] = {}


def lazy_import(module_name: str) -> ModuleType:
    """Import a module on demand and cache the result.

    Parameters
    ----------
    module_name:
        Fully-qualified module path to import.

    Returns
    -------
    ModuleType
        The imported module, cached for subsequent lookups.
    """

    module = _MODULE_CACHE.get(module_name)
    if module is not None:
        return module

    imported = importlib.import_module(module_name)
    _MODULE_CACHE[module_name] = imported
    return imported


__all__: list[str] = ["lazy_import"]
