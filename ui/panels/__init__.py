"""Public panels available to the UI shell."""

from .about import render_about_panel
from .diagnostics import render_diagnostics_panel

__all__ = ["render_about_panel", "render_diagnostics_panel"]
