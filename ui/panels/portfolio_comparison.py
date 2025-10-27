"""Streamlit panel to compare the local portfolio view with IOL's CSV format."""

from __future__ import annotations

import inspect
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
    text_column_factory = (
        getattr(column_factory, "TextColumn", None)
        if column_factory is not None
        else None
    )
    if callable(text_column_factory):
        supports_alignment = False
        try:
            signature = inspect.signature(text_column_factory)
        except (TypeError, ValueError):
            signature = None
        else:
            supports_alignment = "alignment" in signature.parameters

        column_options: dict[str, dict[str, str]] = {
            "Activo": {"width": "medium", "alignment": "left"},
            "Cantidad": {"width": "small", "alignment": "right"},
            "Variación diaria": {"width": "small", "alignment": "right"},
            "Último precio": {"alignment": "right"},
            "Precio promedio de compra": {"alignment": "right"},
            "Rendimiento Porcentaje": {"width": "small", "alignment": "right"},
            "Rendimiento Monto": {"alignment": "right"},
            "Valorizado": {"alignment": "right"},
        }

        column_config = {}
        for column_name, options in column_options.items():
            config_kwargs = dict(options)
            if not supports_alignment:
                config_kwargs.pop("alignment", None)
            column_config[column_name] = text_column_factory(
                column_name,
                **config_kwargs,
            )

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
