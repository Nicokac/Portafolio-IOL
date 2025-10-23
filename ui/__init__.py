"""Módulo de interfaz de usuario para Portafolio-IOL."""

import importlib
import importlib.util
from pathlib import Path

from shared.version import __build_signature__ as _APP_BUILD_SIGNATURE
from shared.version import __version__ as _APP_VERSION

from .actions import render_action_menu
from .footer import render_footer
from .fundamentals import (
    render_fundamental_data,
    render_fundamental_ranking,
    render_sector_comparison,
)
from .fx_panels import render_fx_history, render_spreads
from .header import render_header
from .palette import get_active_palette, get_palette
from .sidebar_controls import render_sidebar
from .summary_metrics import render_summary_metrics
from .tables import render_table
from .ui_settings import UISettings, init_ui, render_ui_controls

__version__ = _APP_VERSION
__build_signature__ = _APP_BUILD_SIGNATURE

__all__ = [
    "header",
    "tables",
    "fx_panels",
    "sidebar_controls",
    "fundamentals",
    "ui_settings",
    "actions",
    "palette",
    "footer",
    "__version__",
    "__build_signature__",
]

_ui_settings_module = None
_apply_settings = None

try:  # pragma: no cover - defensive compatibility shim
    _ui_settings_module = importlib.import_module(f"{__name__}.ui_settings")
except ModuleNotFoundError:  # pragma: no cover - legacy fallback
    _ui_settings_module = None
else:
    _apply_settings = getattr(_ui_settings_module, "apply_settings", None)

if _apply_settings is None:
    module_path = Path(__file__).with_name("ui_settings.py")
    spec = (
        importlib.util.spec_from_file_location(f"{__name__}._ui_settings_compat", module_path)
        if module_path.exists()
        else None
    )
    if spec and spec.loader:  # pragma: no cover - filesystem fallback
        compat_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(compat_module)
        if _ui_settings_module is not None:
            for attr in (
                "apply_settings",
                "init_ui",
                "render_ui_controls",
                "UISettings",
                "get_settings",
            ):
                value = getattr(compat_module, attr, None)
                if value is not None and not hasattr(_ui_settings_module, attr):
                    setattr(_ui_settings_module, attr, value)
        _apply_settings = getattr(_ui_settings_module, "apply_settings", None)
        if _apply_settings is None:
            _apply_settings = getattr(compat_module, "apply_settings", None)

if _apply_settings is None:
    try:  # pragma: no cover - legacy fallback
        from .orchestrator import apply_settings as _apply_settings
    except (
        ModuleNotFoundError,
        ImportError,
        AttributeError,
    ) as exc:  # pragma: no cover - fatal if neither import works
        raise ImportError("The UI apply_settings helper is unavailable") from exc
    if _ui_settings_module is not None:
        setattr(_ui_settings_module, "apply_settings", _apply_settings)

apply_settings = _apply_settings
del _apply_settings
del _ui_settings_module


__all__ += [
    "render_header",
    "render_summary_metrics",
    "render_table",
    "render_spreads",
    "render_fx_history",
    "render_sidebar",
    "render_fundamental_data",
    "render_fundamental_ranking",
    "render_sector_comparison",
    "apply_settings",
    "init_ui",
    "render_ui_controls",
    "UISettings",
    "render_action_menu",
    "get_palette",
    "get_active_palette",
    "render_footer",
]
