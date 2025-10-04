from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
import logging

logger = logging.getLogger(__name__)


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Devuelve un DataFrame codificado como bytes CSV UTF-8 sin índice."""
    return df.to_csv(index=False).encode("utf-8")


@st.cache_resource
def _get_kaleido_scope():
    """Obtiene un alcance de kaleido cacheado para reutilizar el proceso de Chromium."""
    return pio.kaleido.scope


def fig_to_png_bytes(fig: go.Figure) -> bytes:
    """Devuelve la figura renderizada como bytes PNG usando kaleido."""
    try:
        _get_kaleido_scope()
        return pio.to_image(fig, format="png")
    except (ValueError, RuntimeError, TypeError) as e:  # pragma: no cover - depende de librerías externas
        logger.exception("kaleido no disponible")
        raise ValueError("kaleido no disponible") from e
