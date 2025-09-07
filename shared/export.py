from __future__ import annotations
import pandas as pd
import plotly.graph_objects as go


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Devuelve un DataFrame codificado como bytes CSV UTF-8 sin índice."""
    return df.to_csv(index=False).encode("utf-8")


def fig_to_png_bytes(fig: go.Figure) -> bytes:
    """Devuelve la figura renderizada como bytes PNG usando kaleido."""
    return fig.to_image(format="png")