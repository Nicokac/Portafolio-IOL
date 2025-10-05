"""Helpers to transform portfolio snapshots into enriched exports."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from shared.export import fig_to_png_bytes

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricSpec:
    """Metadata describing a KPI available for export."""

    key: str
    label: str
    description: str
    compute: Callable[["PortfolioSnapshotExport"], float | int | None]
    formatter: Callable[[float | int | None], str]


@dataclass(frozen=True)
class ChartSpec:
    """Metadata describing an available chart for enriched exports."""

    key: str
    title: str
    description: str
    builder: Callable[["PortfolioSnapshotExport", int], go.Figure | None]


@dataclass(frozen=True)
class RankingTable:
    """Structured ranking used by CSV/Excel generators."""

    key: str
    title: str
    value_label: str
    share_label: str | None
    description: str
    dataframe: pd.DataFrame


@dataclass(frozen=True)
class PortfolioSnapshotExport:
    """Normalized payload with portfolio analytics for export routines."""

    name: str
    generated_at: datetime | None
    positions: pd.DataFrame
    totals: dict[str, float | None]
    history: pd.DataFrame
    contributions_by_symbol: pd.DataFrame
    contributions_by_type: pd.DataFrame

    @classmethod
    def from_snapshot(cls, snapshot, name: str = "snapshot") -> "PortfolioSnapshotExport":
        """Create a normalized payload from :class:`PortfolioViewSnapshot`."""

        try:
            from application.portfolio_service import PortfolioTotals
            from services.portfolio_view import PortfolioContributionMetrics
        except ImportError:  # pragma: no cover - defensive, tests patch modules
            PortfolioTotals = object  # type: ignore[assignment]
            PortfolioContributionMetrics = object  # type: ignore[assignment]

        df_view = _ensure_dataframe(getattr(snapshot, "df_view", None))
        totals = getattr(snapshot, "totals", None)
        totals_dict: dict[str, float | None]
        if totals is None:
            totals_dict = {}
        elif isinstance(totals, dict):
            totals_dict = {k: _safe_float(v) for k, v in totals.items()}
        elif "PortfolioTotals" in str(type(totals)):
            totals_dict = {k: _safe_float(v) for k, v in asdict(totals).items()}
        else:
            totals_dict = {
                key: _safe_float(getattr(totals, key, None))
                for key in ("total_value", "total_cost", "total_pl", "total_pl_pct", "total_cash")
            }

        generated_at_ts = getattr(snapshot, "generated_at", None)
        generated_at_dt = _parse_timestamp(generated_at_ts)

        history_df = _ensure_dataframe(getattr(snapshot, "historical_total", None))
        contrib = getattr(snapshot, "contribution_metrics", None)
        if contrib is not None and "PortfolioContributionMetrics" in str(type(contrib)):
            by_symbol = _ensure_dataframe(getattr(contrib, "by_symbol", None))
            by_type = _ensure_dataframe(getattr(contrib, "by_type", None))
        else:
            by_symbol = _ensure_dataframe(getattr(contrib, "by_symbol", None))
            by_type = _ensure_dataframe(getattr(contrib, "by_type", None))

        return cls(
            name=name,
            generated_at=generated_at_dt,
            positions=df_view.reset_index(drop=True),
            totals=totals_dict,
            history=history_df.reset_index(drop=True),
            contributions_by_symbol=by_symbol.reset_index(drop=True),
            contributions_by_type=by_type.reset_index(drop=True),
        )

    @classmethod
    def from_payload(
        cls,
        payload: dict,
        name: str | None = None,
    ) -> "PortfolioSnapshotExport":
        """Create a payload from a JSON-compatible snapshot representation."""

        name = name or str(payload.get("name") or payload.get("snapshot") or "snapshot")
        positions = payload.get("positions")
        if positions is None:
            positions = payload.get("df_view")
        positions_df = _ensure_dataframe(positions)

        totals_raw = payload.get("totals") or {}
        if isinstance(totals_raw, str):
            try:
                totals_raw = json.loads(totals_raw)
            except json.JSONDecodeError:
                totals_raw = {}
        totals = {k: _safe_float(v) for k, v in dict(totals_raw).items()} if totals_raw else {}

        generated_at = payload.get("generated_at") or payload.get("timestamp")
        generated_at_dt = _parse_timestamp(generated_at)

        history_raw = payload.get("history") or payload.get("historical_total") or []
        history_df = _ensure_dataframe(history_raw)

        contributions = payload.get("contributions") or {}
        by_symbol = contributions.get("by_symbol") or payload.get("contribution_by_symbol") or []
        by_type = contributions.get("by_type") or payload.get("contribution_by_type") or []

        return cls(
            name=name,
            generated_at=generated_at_dt,
            positions=positions_df.reset_index(drop=True),
            totals=totals,
            history=history_df.reset_index(drop=True),
            contributions_by_symbol=_ensure_dataframe(by_symbol).reset_index(drop=True),
            contributions_by_type=_ensure_dataframe(by_type).reset_index(drop=True),
        )

    @classmethod
    def from_path(cls, path: Path) -> "PortfolioSnapshotExport":
        """Load a snapshot payload from disk."""

        suffix = path.suffix.lower()
        if suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(payload, dict):
                raise ValueError(f"Snapshot {path} is not a JSON object")
            return cls.from_payload(payload, name=path.stem)
        raise ValueError(f"Unsupported snapshot format: {path.suffix}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ensure_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if data is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(data)
    except ValueError:
        return pd.DataFrame()


def _safe_float(value) -> float | None:
    if value is None:
        return None
    try:
        fval = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(fval):  # type: ignore[arg-type]
        return None
    return fval


def _parse_timestamp(value) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (OverflowError, OSError, ValueError):
            return None
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            try:
                return datetime.fromtimestamp(float(value))
            except (ValueError, OverflowError, OSError):
                return None
    return None


def _format_currency(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"${value:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _format_integer(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{int(value):,}".replace(",", ".")


def _format_percentage(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.2f}%"


def _format_ratio(value: float | int | None) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{float(value):.2f}"


def _compute_ratio(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None:
        return None
    if np.isclose(denominator, 0.0):
        return None
    return float(numerator) / float(denominator)


def _unique_symbols(df: pd.DataFrame) -> int:
    if df is None or df.empty or "simbolo" not in df.columns:
        return 0
    return int(df["simbolo"].astype(str).nunique())


def _normalize_history(history: pd.DataFrame) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=["timestamp", "total_value", "total_cost", "total_pl"])
    df = history.copy()
    if "timestamp" in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", errors="coerce")
        df = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metric and chart configuration
# ---------------------------------------------------------------------------


def _total_value(snapshot: PortfolioSnapshotExport) -> float | None:
    return snapshot.totals.get("total_value")


def _total_cost(snapshot: PortfolioSnapshotExport) -> float | None:
    return snapshot.totals.get("total_cost")


def _total_pl(snapshot: PortfolioSnapshotExport) -> float | None:
    return snapshot.totals.get("total_pl")


def _total_pl_pct(snapshot: PortfolioSnapshotExport) -> float | None:
    return snapshot.totals.get("total_pl_pct")


def _total_cash(snapshot: PortfolioSnapshotExport) -> float | None:
    return snapshot.totals.get("total_cash")


def _num_positions(snapshot: PortfolioSnapshotExport) -> int:
    return len(snapshot.positions.index)


def _num_symbols(snapshot: PortfolioSnapshotExport) -> int:
    return _unique_symbols(snapshot.positions)


def _avg_position(snapshot: PortfolioSnapshotExport) -> float | None:
    total_val = _total_value(snapshot)
    if total_val is None:
        return None
    count = _num_positions(snapshot)
    if count <= 0:
        return None
    return total_val / count


def _cash_ratio(snapshot: PortfolioSnapshotExport) -> float | None:
    total_val = _total_value(snapshot)
    cash = _total_cash(snapshot)
    ratio = _compute_ratio(cash, total_val)
    if ratio is None:
        return None
    return ratio * 100.0


METRIC_SPECS: list[MetricSpec] = [
    MetricSpec(
        key="total_value",
        label="Valor total (ARS)",
        description="Valorizado actual del portafolio.",
        compute=_total_value,
        formatter=_format_currency,
    ),
    MetricSpec(
        key="total_cost",
        label="Costo total (ARS)",
        description="Inversión histórica realizada.",
        compute=_total_cost,
        formatter=_format_currency,
    ),
    MetricSpec(
        key="total_pl",
        label="P/L acumulado (ARS)",
        description="Ganancia o pérdida acumulada en moneda local.",
        compute=_total_pl,
        formatter=_format_currency,
    ),
    MetricSpec(
        key="total_pl_pct",
        label="P/L acumulado (%)",
        description="Rentabilidad acumulada respecto del costo.",
        compute=_total_pl_pct,
        formatter=_format_percentage,
    ),
    MetricSpec(
        key="total_cash",
        label="Cash disponible (ARS)",
        description="Liquidez identificada en cuentas o parking.",
        compute=_total_cash,
        formatter=_format_currency,
    ),
    MetricSpec(
        key="positions",
        label="Cantidad de posiciones",
        description="Total de filas en el snapshot exportado.",
        compute=lambda snap: float(_num_positions(snap)),
        formatter=_format_integer,
    ),
    MetricSpec(
        key="symbols",
        label="Símbolos únicos",
        description="Cantidad de tickers diferentes presentes.",
        compute=lambda snap: float(_num_symbols(snap)),
        formatter=_format_integer,
    ),
    MetricSpec(
        key="avg_position",
        label="Valorizado promedio",
        description="Promedio de valuación por posición.",
        compute=_avg_position,
        formatter=_format_currency,
    ),
    MetricSpec(
        key="cash_ratio",
        label="Cash sobre total (%)",
        description="Porcentaje del portafolio asignado a liquidez.",
        compute=_cash_ratio,
        formatter=_format_percentage,
    ),
]

METRIC_LOOKUP = {spec.key: spec for spec in METRIC_SPECS}


def _chart_pl_top(snapshot: PortfolioSnapshotExport, limit: int) -> go.Figure | None:
    df = snapshot.positions
    if df is None or df.empty or "pl" not in df.columns:
        return None
    data = df.copy()
    data["pl"] = pd.to_numeric(data["pl"], errors="coerce")
    data = data.dropna(subset=["pl"]).sort_values("pl", ascending=False).head(limit)
    if data.empty:
        return None
    fig = px.bar(
        data,
        x="simbolo",
        y="pl",
        hover_data={"tipo": True, "pl": ":,.0f"},
        color="simbolo",
        title="Top P/L acumulado",
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _chart_composition(snapshot: PortfolioSnapshotExport, limit: int) -> go.Figure | None:
    df = snapshot.positions
    if df is None or df.empty:
        return None
    if "valor_actual" not in df.columns or "tipo" not in df.columns:
        return None
    data = df.copy()
    data["valor_actual"] = pd.to_numeric(data["valor_actual"], errors="coerce")
    data = data.dropna(subset=["valor_actual"]).groupby("tipo", dropna=False)["valor_actual"].sum().reset_index()
    data = data.sort_values("valor_actual", ascending=False)
    if data.empty:
        return None
    fig = px.pie(
        data,
        values="valor_actual",
        names="tipo",
        hole=0.5,
        title="Composición por tipo",
    )
    fig.update_traces(textinfo="percent+label")
    fig.update_layout(template="plotly_white")
    return fig


def _chart_distribution(snapshot: PortfolioSnapshotExport, limit: int) -> go.Figure | None:
    df = snapshot.positions
    if df is None or df.empty:
        return None
    if "valor_actual" not in df.columns or "tipo" not in df.columns:
        return None
    data = df.copy()
    data["valor_actual"] = pd.to_numeric(data["valor_actual"], errors="coerce")
    data = data.dropna(subset=["valor_actual"]).groupby("tipo", dropna=False)["valor_actual"].sum().reset_index()
    data = data.sort_values("valor_actual", ascending=False)
    if data.empty:
        return None
    fig = px.bar(
        data,
        x="tipo",
        y="valor_actual",
        color="tipo",
        hover_data={"valor_actual": ":,.0f"},
        title="Distribución por tipo",
    )
    fig.update_traces(texttemplate="%{y:,.0f}", textposition="outside", cliponaxis=False)
    fig.update_layout(template="plotly_white", showlegend=False)
    return fig


def _chart_timeline(snapshot: PortfolioSnapshotExport, limit: int) -> go.Figure | None:
    history = _normalize_history(snapshot.history)
    if history.empty:
        return None
    value_cols = [
        col
        for col in ("total_value", "total_cost", "total_pl")
        if col in history.columns
    ]
    if not value_cols:
        return None
    melted = history.melt(
        id_vars=["timestamp"],
        value_vars=value_cols,
        var_name="metric",
        value_name="value",
    )
    melted = melted.dropna(subset=["value"])
    if melted.empty:
        return None
    fig = px.line(
        melted,
        x="timestamp",
        y="value",
        color="metric",
        markers=True,
        title="Evolución histórica",
    )
    fig.update_layout(template="plotly_white")
    return fig


def _chart_heatmap(snapshot: PortfolioSnapshotExport, limit: int) -> go.Figure | None:
    df = snapshot.contributions_by_symbol
    if df is None or df.empty:
        return None
    if "tipo" not in df.columns or "simbolo" not in df.columns:
        return None
    value_col = "valor_actual_pct" if "valor_actual_pct" in df.columns else None
    if value_col is None and "pl_pct" in df.columns:
        value_col = "pl_pct"
    if value_col is None:
        return None
    data = df.copy()
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    pivot = data.pivot_table(index="tipo", columns="simbolo", values=value_col, aggfunc="sum")
    pivot = pivot.sort_index().fillna(0.0)
    if pivot.empty:
        return None
    fig = go.Figure(data=go.Heatmap(z=pivot.values, x=list(pivot.columns), y=list(pivot.index), colorscale="Blues"))
    fig.update_layout(template="plotly_white", title="Mapa de calor por símbolo y tipo")
    return fig


CHART_SPECS: list[ChartSpec] = [
    ChartSpec(
        key="pl_top",
        title="Top P/L acumulado",
        description="Ranking de ganancias acumuladas por símbolo.",
        builder=_chart_pl_top,
    ),
    ChartSpec(
        key="composition",
        title="Composición por tipo",
        description="Participación porcentual de cada tipo de activo.",
        builder=_chart_composition,
    ),
    ChartSpec(
        key="distribution",
        title="Distribución valorizada",
        description="Valorizado total por tipo de instrumento.",
        builder=_chart_distribution,
    ),
    ChartSpec(
        key="timeline",
        title="Evolución histórica",
        description="Serie temporal con valor, costo y P/L.",
        builder=_chart_timeline,
    ),
    ChartSpec(
        key="heatmap",
        title="Mapa de calor por símbolo/tipo",
        description="Matriz de contribución porcentual por símbolo.",
        builder=_chart_heatmap,
    ),
]

CHART_LOOKUP = {spec.key: spec for spec in CHART_SPECS}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compute_kpis(
    snapshot: PortfolioSnapshotExport,
    metric_keys: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a table with the selected metrics for the provided snapshot."""

    keys = list(metric_keys or [spec.key for spec in METRIC_SPECS[:5]])
    rows = []
    for key in keys:
        spec = METRIC_LOOKUP.get(key)
        if spec is None:
            continue
        raw = spec.compute(snapshot)
        if raw is not None and not np.isfinite(raw):
            raw = None
        rows.append(
            {
                "metric": spec.key,
                "label": spec.label,
                "value": spec.formatter(raw),
                "raw_value": raw,
                "description": spec.description,
            }
        )
    return pd.DataFrame(rows)


def build_rankings(
    snapshot: PortfolioSnapshotExport,
    *,
    limit: int = 10,
) -> list[RankingTable]:
    """Return ranking tables for cumulative P/L and valorizado metrics."""

    df = snapshot.positions
    if df is None or df.empty:
        return []

    rankings: list[RankingTable] = []

    def _prepare(column: str, ascending: bool) -> pd.DataFrame:
        data = df.copy()
        data[column] = pd.to_numeric(data[column], errors="coerce")
        data = data.dropna(subset=[column]).sort_values(column, ascending=ascending).head(limit)
        if data.empty:
            return pd.DataFrame()
        total = data[column].sum(min_count=1)
        total_ref = df[column].sum(min_count=1)
        share = None
        if np.isfinite(total_ref) and not np.isclose(total_ref, 0.0):
            share = (data[column] / total_ref) * 100.0
        data = data.assign(
            Rank=range(1, len(data) + 1),
            Símbolo=data.get("simbolo", pd.Series(dtype=str)).astype(str),
            Tipo=data.get("tipo", pd.Series(dtype=str)).astype(str),
            Valor=data[column],
        )
        if share is not None:
            data = data.assign(Participación=share)
        cols = ["Rank", "Símbolo", "Tipo", "Valor"]
        if "Participación" in data.columns:
            cols.append("Participación")
        return data[cols]

    if "pl" in df.columns:
        top_df = _prepare("pl", ascending=False)
        if not top_df.empty:
            rankings.append(
                RankingTable(
                    key="pl_top",
                    title="Ranking P/L (Top)",
                    value_label="Valor P/L",
                    share_label="Participación (%)" if "Participación" in top_df.columns else None,
                    description="Símbolos con mayor ganancia acumulada.",
                    dataframe=top_df,
                )
            )
        bottom_df = _prepare("pl", ascending=True)
        if not bottom_df.empty:
            rankings.append(
                RankingTable(
                    key="pl_bottom",
                    title="Ranking P/L (Bottom)",
                    value_label="Valor P/L",
                    share_label="Participación (%)" if "Participación" in bottom_df.columns else None,
                    description="Símbolos con mayor pérdida acumulada.",
                    dataframe=bottom_df,
                )
            )

    if "valor_actual" in df.columns:
        val_df = _prepare("valor_actual", ascending=False)
        if not val_df.empty:
            rankings.append(
                RankingTable(
                    key="valor_actual",
                    title="Ranking valorizado",
                    value_label="Valor actual",
                    share_label="Participación (%)" if "Participación" in val_df.columns else None,
                    description="Posiciones con mayor peso en el portafolio.",
                    dataframe=val_df,
                )
            )

    return rankings


def assemble_tables(
    snapshot: PortfolioSnapshotExport,
    *,
    metric_keys: Sequence[str] | None = None,
    include_rankings: bool = True,
    include_history: bool = True,
    limit: int = 10,
) -> dict[str, pd.DataFrame]:
    """Return a mapping of table name → DataFrame for CSV/Excel exports."""

    tables: dict[str, pd.DataFrame] = {}

    kpis = compute_kpis(snapshot, metric_keys)
    if not kpis.empty:
        kpis.insert(0, "snapshot", snapshot.name)
        if snapshot.generated_at is not None:
            kpis.insert(1, "generated_at", snapshot.generated_at.isoformat())
    tables["kpis"] = kpis

    positions = snapshot.positions.copy()
    if not positions.empty:
        positions.insert(0, "snapshot", snapshot.name)
    tables["positions"] = positions

    by_symbol = snapshot.contributions_by_symbol.copy()
    if not by_symbol.empty:
        by_symbol.insert(0, "snapshot", snapshot.name)
    tables["contribution_by_symbol"] = by_symbol

    by_type = snapshot.contributions_by_type.copy()
    if not by_type.empty:
        by_type.insert(0, "snapshot", snapshot.name)
    tables["contribution_by_type"] = by_type

    if include_history:
        history = _normalize_history(snapshot.history)
        if not history.empty:
            history.insert(0, "snapshot", snapshot.name)
            tables["history"] = history

    if include_rankings:
        rankings = build_rankings(snapshot, limit=limit)
        for ranking in rankings:
            df = ranking.dataframe.copy()
            df.insert(0, "snapshot", snapshot.name)
            tables[f"ranking_{ranking.key}"] = df

    return tables


def create_csv_bundle(
    snapshot: PortfolioSnapshotExport,
    *,
    metric_keys: Sequence[str] | None = None,
    include_rankings: bool = True,
    include_history: bool = True,
    limit: int = 10,
) -> bytes:
    """Return a ZIP with CSV exports for the snapshot."""

    from zipfile import ZIP_DEFLATED, ZipFile

    tables = assemble_tables(
        snapshot,
        metric_keys=metric_keys,
        include_rankings=include_rankings,
        include_history=include_history,
        limit=limit,
    )

    buffer = BytesIO()
    with ZipFile(buffer, "w", compression=ZIP_DEFLATED) as zf:
        for name, df in tables.items():
            csv_bytes = df.to_csv(index=False).encode("utf-8")
            zf.writestr(f"{name}.csv", csv_bytes)
    return buffer.getvalue()


def write_tables_to_directory(
    snapshot: PortfolioSnapshotExport,
    directory: Path,
    *,
    metric_keys: Sequence[str] | None = None,
    include_rankings: bool = True,
    include_history: bool = True,
    limit: int = 10,
) -> dict[str, Path]:
    """Persist CSV tables to ``directory`` and return mapping name → path."""

    directory.mkdir(parents=True, exist_ok=True)
    tables = assemble_tables(
        snapshot,
        metric_keys=metric_keys,
        include_rankings=include_rankings,
        include_history=include_history,
        limit=limit,
    )
    written: dict[str, Path] = {}
    for name, df in tables.items():
        path = directory / f"{name}.csv"
        df.to_csv(path, index=False)
        written[name] = path
    return written


def build_chart_figures(
    snapshot: PortfolioSnapshotExport,
    chart_keys: Iterable[str],
    *,
    limit: int = 10,
) -> dict[str, go.Figure]:
    """Generate Plotly figures for the requested chart keys."""

    figures: dict[str, go.Figure] = {}
    for key in chart_keys:
        spec = CHART_LOOKUP.get(key)
        if spec is None:
            continue
        try:
            fig = spec.builder(snapshot, limit)
        except Exception:  # pragma: no cover - safeguard for plotly edge-cases
            logger.exception("No se pudo generar el gráfico %s", key)
            continue
        if fig is not None:
            figures[key] = fig
    return figures


def create_excel_workbook(
    snapshot: PortfolioSnapshotExport,
    *,
    metric_keys: Sequence[str] | None = None,
    chart_keys: Sequence[str] | None = None,
    include_rankings: bool = True,
    include_history: bool = True,
    limit: int = 10,
) -> bytes:
    """Create an Excel workbook containing KPIs, tables and embedded charts."""

    tables = assemble_tables(
        snapshot,
        metric_keys=metric_keys,
        include_rankings=include_rankings,
        include_history=include_history,
        limit=limit,
    )

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
        workbook = writer.book

        sheet_order: list[tuple[str, pd.DataFrame]] = []

        for name in ["kpis", "positions", "contribution_by_symbol", "contribution_by_type", "history"]:
            if name in tables and not tables[name].empty:
                sheet_order.append((name, tables[name]))

        ranking_names = [name for name in tables if name.startswith("ranking_")]
        for name in ranking_names:
            sheet_order.append((name, tables[name]))

        # Persist tables to individual sheets
        for name, df in sheet_order:
            sheet_name = _sheet_title(name)
            df.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.sheets[sheet_name]
            for idx, col in enumerate(df.columns):
                lengths = df[col].astype(str).map(len).tolist()
                max_len = max([14] + lengths) if lengths else 14
                width = max(14, min(60, max_len + 4))
                worksheet.set_column(idx, idx, width)

        # Add charts if available
        chart_keys = list(chart_keys or [])
        if chart_keys:
            charts_sheet = workbook.add_worksheet("Gráficos")
            charts_sheet.set_column(0, 4, 48)
            row = 0
            figures = build_chart_figures(snapshot, chart_keys, limit=limit)
            for key in chart_keys:
                if key not in figures:
                    continue
                spec = CHART_LOOKUP[key]
                charts_sheet.write(row, 0, spec.title)
                row += 1
                try:
                    img_bytes = fig_to_png_bytes(figures[key])
                except ValueError:
                    logger.warning("No se pudo convertir el gráfico %s a PNG", key)
                    charts_sheet.write(row, 0, "No se pudo exportar el gráfico (kaleido ausente)")
                    row += 18
                    continue
                if not img_bytes or len(img_bytes) < 8:
                    logger.warning("⛔ Imagen omitida: %s (PNG vacío o inválido)", key)
                    charts_sheet.write(row, 0, "⛔ Imagen omitida")
                    row += 2
                    continue
                charts_sheet.insert_image(
                    row,
                    0,
                    f"{key}.png",
                    {"image_data": BytesIO(img_bytes), "x_scale": 0.9, "y_scale": 0.9},
                )
                row += 20

    return buffer.getvalue()


def _sheet_title(name: str) -> str:
    mapping = {
        "kpis": "KPIs",
        "positions": "Posiciones",
        "contribution_by_symbol": "Contribución símbolo",
        "contribution_by_type": "Contribución tipo",
        "history": "Histórico",
    }
    if name.startswith("ranking_"):
        return name.replace("ranking_", "Ranking ").title()[:31]
    return mapping.get(name, name.replace("_", " ").title())[:31]


__all__ = [
    "MetricSpec",
    "ChartSpec",
    "RankingTable",
    "PortfolioSnapshotExport",
    "METRIC_SPECS",
    "METRIC_LOOKUP",
    "CHART_SPECS",
    "CHART_LOOKUP",
    "compute_kpis",
    "build_rankings",
    "assemble_tables",
    "build_chart_figures",
    "create_csv_bundle",
    "create_excel_workbook",
    "write_tables_to_directory",
]
