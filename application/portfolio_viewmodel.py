"""View-model builders for the portfolio section."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import pandas as pd

from application.portfolio_service import PortfolioTotals, calculate_totals
from domain.models import Controls
from services.portfolio_view import PortfolioViewSnapshot


_DEFAULT_TABS: tuple[str, ...] = (
    "📂 Portafolio",
    "📊 Análisis avanzado",
    "🎲 Análisis de Riesgo",
    "📑 Análisis fundamental",
    "🔎 Análisis de activos",
)


@dataclass(frozen=True)
class PortfolioMetrics:
    """Aggregated metrics for the portfolio section."""

    refresh_secs: int
    ccl_rate: float | None
    all_symbols: tuple[str, ...]
    has_positions: bool


@dataclass(frozen=True)
class PortfolioViewModel:
    """Container with all data required by the portfolio UI."""

    positions: pd.DataFrame
    totals: PortfolioTotals
    controls: Controls
    metrics: PortfolioMetrics
    tab_options: tuple[str, ...]


def get_portfolio_tabs() -> tuple[str, ...]:
    """Return the tuple with the available portfolio tabs."""

    return _DEFAULT_TABS


def build_portfolio_viewmodel(
    *,
    snapshot: PortfolioViewSnapshot | None,
    controls: Controls,
    fx_rates: Mapping[str, float] | None,
    all_symbols: Sequence[str] | None,
) -> PortfolioViewModel:
    """Build the portfolio view-model based on cached portfolio data."""

    if snapshot is None:
        df_view = pd.DataFrame()
        totals = calculate_totals(None)
    else:
        df_view = snapshot.df_view if isinstance(snapshot.df_view, pd.DataFrame) else pd.DataFrame()
        totals = snapshot.totals if isinstance(snapshot.totals, PortfolioTotals) else calculate_totals(df_view)

    ccl_rate = None
    if fx_rates:
        ccl_rate = fx_rates.get("ccl")

    metrics = PortfolioMetrics(
        refresh_secs=controls.refresh_secs,
        ccl_rate=ccl_rate,
        all_symbols=tuple(all_symbols or ()),
        has_positions=not df_view.empty,
    )

    return PortfolioViewModel(
        positions=df_view,
        totals=totals,
        controls=controls,
        metrics=metrics,
        tab_options=get_portfolio_tabs(),
    )
