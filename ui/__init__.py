# ui/__init__.py
# Podés dejarlo mínimo:
__all__ = ["header", "tables", "fx_panels", "sidebar_controls", "fundamentals", "ui_settings", "actions", "palette"]

# (Opcional) Re-exportes cómodos:
from .header import render_header
from .tables import render_totals, render_table
from .fx_panels import render_fx_panel, render_spreads, render_fx_history
from .sidebar_controls import render_sidebar
from .fundamentals import render_fundamental_data
from .ui_settings import init_ui, render_ui_controls, UISettings
from .palette import get_palette, get_active_palette
from .actions import render_action_menu

__all__ += [
    "render_header",
    "render_totals",
    "render_table",
    "render_fx_panel",
    "render_spreads",
    "render_fx_history",
    "render_sidebar",
    "render_fundamental_data",
    "init_ui",
    "render_ui_controls",
    "UISettings",
    "render_action_menu",
    "get_palette",
    "get_active_palette",
]