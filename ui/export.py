from __future__ import annotations

import streamlit as st
import pandas as pd
from shared.export import df_to_csv_bytes


# Configuración común de Plotly para habilitar captura a PNG desde la barra de herramientas
PLOTLY_CONFIG = {"modeBarButtonsToAdd": ["toImage"]}


def download_csv(df: pd.DataFrame, filename: str, *, label: str = "⬇️ Exportar CSV") -> None:
    """Renderice un botón de descarga para exportar DataFrame como CSV."""
    st.download_button(label, df_to_csv_bytes(df), file_name=filename, mime="text/csv")
