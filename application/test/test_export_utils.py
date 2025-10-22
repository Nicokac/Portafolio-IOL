import pandas as pd
import plotly.graph_objects as go
import pytest

from shared.export import df_to_csv_bytes, fig_to_png_bytes


def test_df_to_csv_bytes():
    df = pd.DataFrame({"a": [1, 2]})
    data = df_to_csv_bytes(df)
    assert b"a" in data and b"1" in data


def test_fig_to_png_bytes_starts_with_png_header():
    fig = go.Figure(data=[go.Bar(x=[1], y=[2])])
    try:
        data = fig_to_png_bytes(fig)
    except ValueError:
        pytest.skip("kaleido not installed")
    assert data.startswith(b"\x89PNG")
