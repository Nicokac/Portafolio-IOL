"""View-model builders for the portfolio section."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import logging
from typing import Mapping, Sequence, Any

import pandas as pd

from application.portfolio_service import PortfolioTotals, calculate_totals
from domain.models import Controls
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewSnapshot,
)
from services import snapshots as snapshot_service


logger = logging.getLogger(__name__)


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
    historical_total: pd.DataFrame
    contributions: PortfolioContributionMetrics
    snapshot_id: str | None = None
    snapshot_catalog: Mapping[str, tuple["SnapshotSummary", ...]] = field(default_factory=dict)


@dataclass(frozen=True)
class SnapshotSummary:
    """Compact representation of stored snapshots for selection widgets."""

    id: str
    kind: str
    label: str
    created_at: float
    metadata: Mapping[str, Any] = field(default_factory=dict)


def get_portfolio_tabs() -> tuple[str, ...]:
    """Return the tuple with the available portfolio tabs."""

    return _DEFAULT_TABS


def build_portfolio_viewmodel(
    *,
    snapshot: PortfolioViewSnapshot | None,
    controls: Controls,
    fx_rates: Mapping[str, float] | None,
    all_symbols: Sequence[str] | None,
    snapshots_module: Any | None = None,
) -> PortfolioViewModel:
    """Build the portfolio view-model based on cached portfolio data."""

    snapshots_module = snapshots_module or snapshot_service
    snapshot_catalog = _load_snapshot_catalog(snapshots_module)

    if snapshot is None:
        df_view = pd.DataFrame()
        totals = calculate_totals(None)
        historical_total = pd.DataFrame(
            columns=["timestamp", "total_value", "total_cost", "total_pl"]
        )
        contributions = PortfolioContributionMetrics.empty()
        snapshot_id = None
    else:
        df_view = snapshot.df_view if isinstance(snapshot.df_view, pd.DataFrame) else pd.DataFrame()
        totals = snapshot.totals if isinstance(snapshot.totals, PortfolioTotals) else calculate_totals(df_view)
        historical_total = (
            snapshot.historical_total
            if isinstance(snapshot.historical_total, pd.DataFrame)
            else pd.DataFrame(columns=["timestamp", "total_value", "total_cost", "total_pl"])
        )
        contributions = (
            snapshot.contribution_metrics
            if isinstance(snapshot.contribution_metrics, PortfolioContributionMetrics)
            else PortfolioContributionMetrics.empty()
        )
        snapshot_id = getattr(snapshot, "storage_id", None)

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
        historical_total=historical_total,
        contributions=contributions,
        snapshot_id=snapshot_id,
        snapshot_catalog=snapshot_catalog,
    )


def _load_snapshot_catalog(module: Any) -> Mapping[str, tuple[SnapshotSummary, ...]]:
    kinds = ("portfolio", "technical", "risk")
    catalog: dict[str, tuple[SnapshotSummary, ...]] = {}
    list_fn = getattr(module, "list_snapshots", None)
    if not callable(list_fn):
        return catalog

    for kind in kinds:
        try:
            records = list_fn(kind, limit=50, order="desc")
        except Exception:
            logger.exception("No se pudieron cargar snapshots para %s", kind)
            continue

        summaries: list[SnapshotSummary] = []
        for record in records or []:
            summary = _snapshot_summary_from_record(kind, record)
            if summary is not None:
                summaries.append(summary)
        if summaries:
            catalog[kind] = tuple(summaries)
    return catalog


def _snapshot_summary_from_record(kind: str, record: Any) -> SnapshotSummary | None:
    if not isinstance(record, Mapping):
        return None
    snapshot_id = str(record.get("id") or "").strip()
    if not snapshot_id:
        return None
    created_at = float(record.get("created_at") or 0.0)
    metadata = record.get("metadata") if isinstance(record.get("metadata"), Mapping) else {}
    label = _format_snapshot_label(created_at, metadata)
    return SnapshotSummary(
        id=snapshot_id,
        kind=kind,
        label=label,
        created_at=created_at,
        metadata=metadata,
    )


def _format_snapshot_label(created_at: float, metadata: Mapping[str, Any]) -> str:
    if created_at:
        try:
            dt = datetime.fromtimestamp(created_at)
            label = dt.strftime("%Y-%m-%d %H:%M")
        except (OSError, OverflowError, ValueError):
            label = "Snapshot"
    else:
        label = "Snapshot"

    dataset_key = metadata.get("dataset_key") if isinstance(metadata, Mapping) else None
    if dataset_key:
        label = f"{label} · {str(dataset_key)[:8]}"
    return label
