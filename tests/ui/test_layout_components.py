"""Regression tests for modernised Streamlit layout components."""

from __future__ import annotations

import pandas as pd

from ui import export


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
