from __future__ import annotations

"""Streamlit-facing controller helpers for the portfolio dashboard."""

from typing import Any, Callable

from controllers.portfolio.portfolio import render_portfolio_section
from services.performance_timer import performance_timer


def render_portfolio_ui(
    container: Any,
    cli: Any,
    fx_rates: Any,
    *,
    view_model_service_factory: Callable[[], Any] | None = None,
    notifications_service_factory: Callable[[], Any] | None = None,
) -> Any:
    """Render the portfolio tab while measuring UI latency."""

    telemetry: dict[str, object] = {
        "status": "success",
        "has_cli": cli is not None,
    }
    with performance_timer("render_portfolio_ui", extra=telemetry):
        try:
            refresh_secs = render_portfolio_section(
                container,
                cli,
                fx_rates,
                view_model_service_factory=view_model_service_factory,
                notifications_service_factory=notifications_service_factory,
            )
        except Exception:
            telemetry["status"] = "error"
            raise
        if isinstance(refresh_secs, (int, float)):
            telemetry["refresh_secs"] = float(refresh_secs)
        return refresh_secs


__all__ = ["render_portfolio_ui"]
