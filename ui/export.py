from __future__ import annotations
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from shared.export import df_to_csv_bytes, fig_to_png_bytes


def download_csv(df: pd.DataFrame, filename: str, *, label: str = "⬇️ Exportar CSV") -> None:
    """Renderice un botón de descarga para exportar DataFrame como CSV."""
    st.download_button(label, df_to_csv_bytes(df), file_name=filename, mime="text/csv")


def download_chart(fig: go.Figure, filename: str, *, label: str = "⬇️ Exportar PNG") -> None:
    """Renderice un botón de descarga para exportar una figura de Plotly como imagen PNG."""
    st.download_button(label, fig_to_png_bytes(fig), file_name=filename, mime="image/png")