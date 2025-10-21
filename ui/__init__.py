"""Módulo de interfaz de usuario para Portafolio-IOL."""

from shared.version import __build_signature__ as _APP_BUILD_SIGNATURE
from shared.version import __version__ as _APP_VERSION

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

from .header import render_header
from .tables import render_totals, render_table
from .fx_panels import render_spreads, render_fx_history
from .sidebar_controls import render_sidebar
from .fundamentals import (
    render_fundamental_data,
    render_fundamental_ranking,
    render_sector_comparison,
)
from .ui_settings import init_ui, render_ui_controls, UISettings
from .palette import get_palette, get_active_palette
from .actions import render_action_menu
from .footer import render_footer

__all__ += [
    "render_header",
    "render_totals",
    "render_table",
    "render_spreads",
    "render_fx_history",
    "render_sidebar",
    "render_fundamental_data",
    "render_fundamental_ranking",
    "render_sector_comparison",
    "init_ui",
    "render_ui_controls",
    "UISettings",
    "render_action_menu",
    "get_palette",
    "get_active_palette",
    "render_footer",
]
