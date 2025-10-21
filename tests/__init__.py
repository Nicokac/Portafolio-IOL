"""Test package to ensure deterministic import paths during collection."""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
from pathlib import Path

os.environ.setdefault(
    "FASTAPI_TOKENS_KEY", "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA="
)
os.environ.setdefault(
    "IOL_TOKENS_KEY", "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE="
)


def _ensure_ui_settings_compat() -> None:
    """Restore critical attributes when tests stub ui.ui_settings."""

    try:
        module = importlib.import_module("ui.ui_settings")
    except ModuleNotFoundError:  # pragma: no cover - best-effort guard
        return

    if getattr(module, "apply_settings", None) is not None:
        return

    module_path = Path(__file__).resolve().parents[1] / "ui" / "ui_settings.py"
    spec = importlib.util.spec_from_file_location("ui._ui_settings_compat", module_path)
    if not spec or not spec.loader:  # pragma: no cover - filesystem missing
        return

    compat_module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(spec.name, compat_module)
    spec.loader.exec_module(compat_module)

    for attr in ("apply_settings", "init_ui", "render_ui_controls", "UISettings", "get_settings"):
        value = getattr(compat_module, attr, None)
        if value is not None and not hasattr(module, attr):
            setattr(module, attr, value)


_ensure_ui_settings_compat()

