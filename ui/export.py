from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from typing import Sequence

import pandas as pd
import streamlit as st

from shared.export import df_to_csv_bytes
from shared.portfolio_export import (
    CHART_LOOKUP,
    CHART_SPECS,
    METRIC_LOOKUP,
    METRIC_SPECS,
    PortfolioSnapshotExport,
    create_csv_bundle,
    create_excel_workbook,
)


# Configuración común de Plotly para habilitar captura a PNG desde la barra de herramientas
PLOTLY_CONFIG = {"modeBarButtonsToAdd": ["toImage"]}


def download_csv(df: pd.DataFrame, filename: str, *, label: str = "⬇️ Exportar CSV") -> None:
    """Renderice un botón de descarga para exportar DataFrame como CSV."""
    st.download_button(label, df_to_csv_bytes(df), file_name=filename, mime="text/csv")


def render_portfolio_exports(
    *,
    snapshot,
    df_view: pd.DataFrame,
    totals,
    historical_total: pd.DataFrame | None,
    contribution_metrics,
    filename_prefix: str = "portafolio",
    default_metrics: Sequence[str] | None = None,
    default_charts: Sequence[str] | None = None,
    ranking_limit: int = 10,
) -> None:
    """Renderiza controles para exportar reportes enriquecidos del portafolio."""

    if df_view is None or df_view.empty:
        return

    with st.expander("📦 Exportar análisis enriquecido", expanded=False):
        st.write(
            "Generá un paquete de CSV y un Excel con KPIs, rankings y gráficos listos para compartir."
        )

        metrics_options = [spec.key for spec in METRIC_SPECS]
        default_metric_keys = [
            key for key in (default_metrics or metrics_options[:5]) if key in METRIC_LOOKUP
        ]
        metric_selection = st.multiselect(
            "Métricas a incluir",
            metrics_options,
            default=default_metric_keys,
            format_func=lambda key: METRIC_LOOKUP[key].label,
            key=f"metrics_{filename_prefix}",
        )

        chart_options = [spec.key for spec in CHART_SPECS]
        default_chart_keys = [
            key for key in (default_charts or chart_options[:3]) if key in CHART_LOOKUP
        ]
        chart_selection = st.multiselect(
            "Gráficos a embeber en el Excel",
            chart_options,
            default=default_chart_keys,
            format_func=lambda key: CHART_LOOKUP[key].title,
            key=f"charts_{filename_prefix}",
        )

        include_rankings = st.checkbox(
            "Incluir rankings de P/L y valorizado",
            value=True,
            key=f"rankings_{filename_prefix}",
        )
        include_history = st.checkbox(
            "Incluir historial de totales",
            value=True,
            key=f"history_{filename_prefix}",
        )
        limit = st.slider(
            "Elementos por ranking",
            min_value=5,
            max_value=50,
            value=ranking_limit,
            step=5,
            key=f"limit_{filename_prefix}",
        )

        export_snapshot = _build_snapshot_export(
            snapshot,
            df_view=df_view,
            totals=totals,
            historical_total=historical_total,
            contribution_metrics=contribution_metrics,
            name=filename_prefix,
        )

        metric_keys = metric_selection or default_metric_keys or metrics_options[:5]

        csv_bundle = create_csv_bundle(
            export_snapshot,
            metric_keys=metric_keys,
            include_rankings=include_rankings,
            include_history=include_history,
            limit=limit,
        )
        st.download_button(
            "⬇️ Descargar CSV (ZIP)",
            csv_bundle,
            file_name=f"{filename_prefix}_analisis.zip",
            mime="application/zip",
            key=f"csv_bundle_{filename_prefix}",
        )

        try:
            excel_bytes = create_excel_workbook(
                export_snapshot,
                metric_keys=metric_keys,
                chart_keys=chart_selection,
                include_rankings=include_rankings,
                include_history=include_history,
                limit=limit,
            )
        except ValueError:
            st.warning(
                "No se pudieron generar los gráficos (depende de kaleido). Instálalo para habilitar la exportación a Excel."
            )
        else:
            st.download_button(
                "⬇️ Descargar Excel (.xlsx)",
                excel_bytes,
                file_name=f"{filename_prefix}_analisis.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"excel_bundle_{filename_prefix}",
            )


def _build_snapshot_export(
    snapshot,
    *,
    df_view: pd.DataFrame,
    totals,
    historical_total: pd.DataFrame | None,
    contribution_metrics,
    name: str,
) -> PortfolioSnapshotExport:
    """Coerce los datos visibles en la UI a un payload exportable."""

    if snapshot is not None:
        return PortfolioSnapshotExport.from_snapshot(snapshot, name=name)

    totals_dict: dict[str, float | None] = {}
    if totals is not None:
        try:
            totals_dict = {
                key: float(val) if val is not None else None
                for key, val in asdict(totals).items()
            }
        except TypeError:
            totals_dict = {}

    history_df = df_to_frame(historical_total)
    contrib_symbol = df_to_frame(getattr(contribution_metrics, "by_symbol", None))
    contrib_type = df_to_frame(getattr(contribution_metrics, "by_type", None))

    return PortfolioSnapshotExport(
        name=name,
        generated_at=datetime.now(),
        positions=df_to_frame(df_view),
        totals=totals_dict,
        history=history_df,
        contributions_by_symbol=contrib_symbol,
        contributions_by_type=contrib_type,
    )


def df_to_frame(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    if data is None:
        return pd.DataFrame()
    try:
        return pd.DataFrame(data)
    except ValueError:
        return pd.DataFrame()
