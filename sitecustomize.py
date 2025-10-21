"""Optional lightweight test doubles for heavy UI dependencies."""

from __future__ import annotations

import importlib.machinery
import os
import sys
import types
from dataclasses import dataclass
from typing import Any


def _ensure_streamlit_javascript_stub() -> None:
    if "streamlit_javascript" in sys.modules:
        return

    stub = types.ModuleType("streamlit_javascript")

    def _noop(*_: Any, **__: Any) -> None:  # pragma: no cover - trivial shim
        return None

    stub.st_javascript = _noop  # type: ignore[attr-defined]
    sys.modules[stub.__name__] = stub


def _ensure_ui_settings_stub() -> None:
    if "ui" in sys.modules and hasattr(sys.modules["ui"], "ui_settings"):
        return

    ui_pkg = sys.modules.get("ui")
    if ui_pkg is None:
        ui_pkg = types.ModuleType("ui")
        spec = importlib.machinery.ModuleSpec("ui", loader=None, is_package=True)
        spec.submodule_search_locations = []  # type: ignore[attr-defined]
        ui_pkg.__spec__ = spec  # type: ignore[attr-defined]
        ui_pkg.__path__ = []  # type: ignore[attr-defined]
        sys.modules["ui"] = ui_pkg

    if "ui.ui_settings" in sys.modules:
        ui_settings = sys.modules["ui.ui_settings"]
    else:
        ui_settings = types.ModuleType("ui.ui_settings")
        spec = importlib.machinery.ModuleSpec("ui.ui_settings", loader=None, is_package=False)
        ui_settings.__spec__ = spec  # type: ignore[attr-defined]
        sys.modules["ui.ui_settings"] = ui_settings

    @dataclass
    class UISettings:  # pragma: no cover - simple container
        layout: str = "wide"
        theme: str = "dark"

    def get_settings() -> UISettings:  # pragma: no cover - deterministic values
        return UISettings()

    def apply_settings(settings: UISettings) -> None:  # pragma: no cover
        _ = settings

    def init_ui() -> UISettings:  # pragma: no cover
        settings = get_settings()
        apply_settings(settings)
        return settings

    def render_ui_controls(container: Any = None) -> UISettings:  # pragma: no cover
        _ = container
        return get_settings()

    ui_settings.UISettings = UISettings  # type: ignore[attr-defined]
    ui_settings.get_settings = get_settings  # type: ignore[attr-defined]
    ui_settings.apply_settings = apply_settings  # type: ignore[attr-defined]
    ui_settings.init_ui = init_ui  # type: ignore[attr-defined]
    ui_settings.render_ui_controls = render_ui_controls  # type: ignore[attr-defined]

    setattr(sys.modules["ui"], "ui_settings", ui_settings)


if os.environ.get("FAST_TEST_STUBS"):
    _ensure_streamlit_javascript_stub()
    _ensure_ui_settings_stub()
