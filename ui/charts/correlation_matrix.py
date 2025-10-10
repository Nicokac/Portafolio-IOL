from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.charts import FONT_FAMILY, _apply_layout
from ui.palette import get_active_palette


def _collect_labels(*matrices: pd.DataFrame | None, beta_shift: pd.Series | None = None) -> list[str]:
    labels: set[str] = set()
    for matrix in matrices:
        if isinstance(matrix, pd.DataFrame) and not matrix.empty:
            labels.update(str(col) for col in matrix.columns)
            labels.update(str(idx) for idx in matrix.index)
    if isinstance(beta_shift, pd.Series) and not beta_shift.empty:
        labels.update(str(idx) for idx in beta_shift.index)
    return sorted(labels)


def _align_matrix(matrix: pd.DataFrame | None, labels: list[str]) -> pd.DataFrame:
    if not isinstance(matrix, pd.DataFrame) or matrix.empty:
        return pd.DataFrame(index=labels, columns=labels, dtype=float)
    aligned = matrix.copy()
    aligned.index = aligned.index.map(str)
    aligned.columns = aligned.columns.map(str)
    aligned = aligned.reindex(index=labels, columns=labels)
    aligned = aligned.fillna(0.0)
    if not aligned.empty:
        np.fill_diagonal(aligned.values, 1.0)
    return aligned


def _heatmap_trace(matrix: pd.DataFrame, *, showscale: bool, coloraxis: str | None = None) -> go.Heatmap:
    pal = get_active_palette()
    colorscale = [
        [0.0, pal.negative],
        [0.5, pal.plot_bg],
        [1.0, pal.positive],
    ]
    return go.Heatmap(
        z=matrix.values,
        x=matrix.columns.tolist(),
        y=matrix.index.tolist(),
        colorscale=colorscale,
        zmin=-1,
        zmax=1,
        showscale=showscale,
        coloraxis=coloraxis,
        hovertemplate="Sector %{y} → %{x}<br>ρ=%{z:.2f}<extra></extra>",
    )


def _add_empty_annotation(fig: go.Figure, *, row: int, col: int) -> None:
    pal = get_active_palette()
    fig.add_annotation(
        text="Sin datos",
        showarrow=False,
        font=dict(color=pal.text, size=12, family=FONT_FAMILY),
        xref=f"x{'' if col == 1 else col}",
        yref=f"y{'' if row == 1 else row}",
        x=0.5,
        y=0.5,
        xanchor="center",
        yanchor="middle",
        row=row,
        col=col,
    )


def build_correlation_figure(
    historical: pd.DataFrame | None,
    rolling: pd.DataFrame | None,
    adaptive: pd.DataFrame | None,
    *,
    beta_shift: pd.Series | None = None,
    title: str | None = None,
) -> go.Figure:
    labels = _collect_labels(historical, rolling, adaptive, beta_shift=beta_shift)
    historical_aligned = _align_matrix(historical, labels)
    rolling_aligned = _align_matrix(rolling, labels)
    adaptive_aligned = _align_matrix(adaptive, labels)

    fig = make_subplots(rows=1, cols=3, subplot_titles=("Histórica", "Rolling", "Adaptativa"))

    matrices = [historical_aligned, rolling_aligned, adaptive_aligned]
    for idx, matrix in enumerate(matrices, start=1):
        if matrix.empty:
            _add_empty_annotation(fig, row=1, col=idx)
            continue
        trace = _heatmap_trace(matrix, showscale=(idx == 3), coloraxis="coloraxis")
        fig.add_trace(trace, row=1, col=idx)

    if isinstance(beta_shift, pd.Series) and not beta_shift.empty:
        pal = get_active_palette()
        for sector, value in beta_shift.items():
            if str(sector) not in labels:
                continue
            fig.add_annotation(
                text=f"βΔ {value:+.2f}",
                showarrow=False,
                font=dict(color=pal.text, size=11, family=FONT_FAMILY),
                x=sector,
                y=sector,
                row=1,
                col=3,
            )

    fig.update_layout(coloraxis=dict(colorscale=[
        [0.0, get_active_palette().negative],
        [0.5, get_active_palette().plot_bg],
        [1.0, get_active_palette().positive],
    ]))

    return _apply_layout(fig, title=title, show_legend=False)


__all__ = ["build_correlation_figure"]
