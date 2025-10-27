"""Public panels available to the UI shell."""

from .about import render_about_panel
from .diagnostics import render_diagnostics_panel
from .portfolio_comparison import render_portfolio_comparison_panel
from .system_diagnostics import render_system_diagnostics_panel
from .system_status import render_system_status_panel

__all__ = [
    "render_about_panel",
    "render_diagnostics_panel",
    "render_portfolio_comparison_panel",
    "render_system_diagnostics_panel",
    "render_system_status_panel",
]
