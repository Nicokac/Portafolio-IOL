"""Regression tests for modernised Streamlit layout components."""

from __future__ import annotations

import pandas as pd

from ui import export
from ui.tabs import opportunities


def test_export_summary_dataframe_uses_stretch_width(streamlit_stub) -> None:
    df = pd.DataFrame({"ticker": ["AA", "BB"], "peso": [0.4, 0.6]})

    export._render_export_summary(
        metric_keys=("rentabilidad",),
        chart_keys=("allocations",),
        include_rankings=True,
        include_history=False,
        ranking_limit=5,
        df_view=df,
    )

    frames = streamlit_stub.get_records("dataframe")
    assert frames, "Se esperaba al menos un dataframe renderizado"
    assert frames[0]["width"] == "stretch"


def test_opportunities_charts_use_stretch_width(streamlit_stub) -> None:
    table = pd.DataFrame(
        {
            "sector": ["Tecnología", "Finanzas", "Tecnología"],
            "score_compuesto": [0.8, 0.6, 0.9],
        }
    )

    opportunities._render_sector_score_chart(table)

    charts = streamlit_stub.get_records("altair_chart")
    assert charts, "Se esperaba renderizar el gráfico de sectores"
    assert charts[0]["width"] == "stretch"
