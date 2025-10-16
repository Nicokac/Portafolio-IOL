from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "render_portfolio_section",
    "build_portfolio_viewmodel",
    "get_portfolio_tabs",
    "PortfolioMetrics",
    "PortfolioViewModel",
    "load_portfolio_data",
    "apply_filters",
    "generate_basic_charts",
    "render_basic_section",
    "render_advanced_analysis",
    "compute_risk_metrics",
    "render_risk_analysis",
    "render_fundamental_analysis",
]


_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "render_portfolio_section": ("controllers.portfolio.portfolio", "render_portfolio_section"),
    "build_portfolio_viewmodel": ("application.portfolio_viewmodel", "build_portfolio_viewmodel"),
    "get_portfolio_tabs": ("application.portfolio_viewmodel", "get_portfolio_tabs"),
    "PortfolioMetrics": ("application.portfolio_viewmodel", "PortfolioMetrics"),
    "PortfolioViewModel": ("application.portfolio_viewmodel", "PortfolioViewModel"),
    "load_portfolio_data": ("controllers.portfolio.load_data", "load_portfolio_data"),
    "apply_filters": ("controllers.portfolio.filters", "apply_filters"),
    "generate_basic_charts": ("controllers.portfolio.charts", "generate_basic_charts"),
    "render_basic_section": ("controllers.portfolio.charts", "render_basic_section"),
    "render_advanced_analysis": ("controllers.portfolio.charts", "render_advanced_analysis"),
    "compute_risk_metrics": ("controllers.portfolio.risk", "compute_risk_metrics"),
    "render_risk_analysis": ("controllers.portfolio.risk", "render_risk_analysis"),
    "render_fundamental_analysis": (
        "controllers.portfolio.fundamentals",
        "render_fundamental_analysis",
    ),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module 'controllers.portfolio' has no attribute {name!r}")

    module_name, attr_name = _LAZY_IMPORTS[name]
    module = import_module(module_name)
    attr = getattr(module, attr_name)
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(set(__all__))
