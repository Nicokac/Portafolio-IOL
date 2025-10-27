"""Streamlit panel to compare the local portfolio view with IOL's CSV format."""

from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from application.portfolio_service import to_iol_format
from shared.time_provider import TimeProvider

logger = logging.getLogger(__name__)

_TAB_TITLE = "📊 Comparativa IOL"


def _get_positions_dataframe() -> pd.DataFrame:
    state = getattr(st, "session_state", None)
    if isinstance(state, dict):
        df = state.get("portfolio_last_positions")
    else:
        getter = getattr(state, "get", None)
        df = getter("portfolio_last_positions") if callable(getter) else None
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def render_portfolio_comparison_panel() -> None:
    """Render the comparison dashboard and expose the CSV export helper."""

    st.header(_TAB_TITLE)

    df_positions = _get_positions_dataframe()
    if df_positions.empty:
        st.info(
            "Aún no hay datos del portafolio disponibles para exportar. "
            "Actualizá la vista principal e intentá nuevamente.",
        )
        return

    df_iol_format = to_iol_format(df_positions)
    today = TimeProvider.now_datetime().date().isoformat()
    csv_text = df_iol_format.to_csv(index=False, encoding="utf-8-sig")
    csv_buffer = csv_text.encode("utf-8-sig") if isinstance(csv_text, str) else csv_text
    file_name = f"portafolio_iol_{today}.csv"

    column_config = None
    column_factory = getattr(st, "column_config", None)
    if column_factory is not None:
        column_config = {
            "Activo": column_factory.TextColumn("Activo", width="medium"),
            "Cantidad": column_factory.TextColumn("Cantidad", width="small"),
            "Variación diaria": column_factory.TextColumn(
                "Variación diaria",
                width="small",
            ),
            "Último precio": column_factory.TextColumn("Último precio"),
            "Precio promedio de compra": column_factory.TextColumn(
                "Precio promedio de compra",
            ),
            "Rendimiento Porcentaje": column_factory.TextColumn(
                "Rendimiento Porcentaje",
                width="small",
            ),
            "Rendimiento Monto": column_factory.TextColumn(
                "Rendimiento Monto",
            ),
            "Valorizado": column_factory.TextColumn("Valorizado"),
        }

    st.dataframe(
        df_iol_format,
        hide_index=True,
        column_config=column_config,
    )

    triggered = st.download_button(
        label="💾 Exportar a CSV (Formato IOL)",
        data=csv_buffer,
        file_name=file_name,
        mime="text/csv",
    )

    if triggered:
        logger.info(
            "[Export] portfolio_comparison generated — %s rows — format=IOL — file=%s",
            len(df_iol_format.index),
            file_name,
        )


__all__ = ["render_portfolio_comparison_panel"]
