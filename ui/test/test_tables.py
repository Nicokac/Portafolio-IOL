import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pandas as pd
import pytest

from ui import tables


@pytest.fixture
def mock_st(monkeypatch):
    mock = MagicMock()
    mock.columns.return_value = [MagicMock() for _ in range(4)]
    mock.text_input.return_value = ""
    mock.dataframe = MagicMock()
    mock.metric = MagicMock()
    mock.info = MagicMock()
    mock.markdown = MagicMock()
    mock.subheader = MagicMock()
    mock.number_input.side_effect = lambda *a, **k: k.get("value")
    mock.session_state = {}
    mock.column_config = SimpleNamespace(
        Column=lambda *a, **k: None,
        NumberColumn=lambda *a, **k: None,
        LineChartColumn=lambda *a, **k: None,
    )
    monkeypatch.setattr(tables, "st", mock)
    return mock
def test_render_totals_without_ccl_rate(mock_st):
    df = pd.DataFrame({"valor_actual": [100, 200], "costo": [40, 110]})
    cols = [MagicMock() for _ in range(4)]
    mock_st.columns.return_value = cols
    with patch.object(tables, "format_money", lambda v, currency="ARS": f"{v}-{currency}"):
        tables.render_totals(df)
    expected_calls = [
        call("Valorizado", "300.0-ARS"),
        call("Costo", "150.0-ARS"),
        call("P/L", "150.0-ARS", delta="100.00%"),
        call("P/L %", "100.00%"),
    ]
    actual_calls = [c.metric.call_args for c in cols]
    assert actual_calls == expected_calls


def test_render_totals_with_ccl_rate(mock_st):
    df = pd.DataFrame({"valor_actual": [100, 200], "costo": [40, 110]})
    cols1 = [MagicMock() for _ in range(4)]
    cols2 = [MagicMock() for _ in range(4)]
    mock_st.columns.side_effect = [cols1, cols2]
    with patch.object(tables, "format_money", lambda v, currency="ARS": f"{v}-{currency}"):
        tables.render_totals(df, ccl_rate=100)
    base_calls = [
        call("Valorizado", "300.0-ARS"),
        call("Costo", "150.0-ARS"),
        call("P/L", "150.0-ARS", delta="100.00%"),
        call("P/L %", "100.00%"),
    ]
    usd_calls = [
        call("Valorizado (USD CCL)", "3.0-USD"),
        call("Costo (USD CCL)", "1.5-USD"),
        call("P/L (USD CCL)", "1.5-USD"),
        call("CCL usado", "100.0-ARS"),
    ]
    assert [c.metric.call_args for c in cols1] == base_calls
    assert [c.metric.call_args for c in cols2] == usd_calls


def test_render_table_empty_df_shows_info(mock_st):
    empty_df = pd.DataFrame()
    tables.render_table(empty_df, order_by="pl", desc=False)
    mock_st.info.assert_called_once_with("Sin datos para mostrar.")
    mock_st.dataframe.assert_not_called()


def test_render_table_invalid_order(mock_st):
    df = pd.DataFrame({
        "simbolo": ["ALUA"],
        "tipo": ["ACC"],
        "valor_actual": [100.0],
        "costo": [80.0],
        "pl": [20.0],
        "pl_%": [25.0],
        "pl_d": [1.0],
        "chg_%": [0.5],
    })
    mock_st.text_input.return_value = ""
    palette = SimpleNamespace(bg="", text="", highlight_bg="", highlight_text="", negative="red", positive="green")
    with patch.object(tables, "get_active_palette", return_value=palette), \
         patch.object(tables, "download_csv"), \
         patch.object(logging, "getLogger") as mock_logger:
        tables.render_table(df, order_by="invalid", desc=False)
    assert mock_st.dataframe.called
    mock_logger.return_value.warning.assert_called()


def test_render_table_search_and_usd_projection(mock_st):
    df = pd.DataFrame({
        "simbolo": ["ALUA", "BBAR"],
        "tipo": ["ACC", "ACC"],
        "cantidad": [10, 5],
        "ultimo": [100.0, 200.0],
        "valor_actual": [1000.0, 1000.0],
        "costo": [800.0, 900.0],
        "pl": [200.0, 100.0],
        "pl_%": [25.0, 11.1],
        "pl_d": [10.0, -5.0],
        "chg_%": [1.0, -0.5],
    })
    mock_st.text_input.return_value = "alua"
    palette = SimpleNamespace(bg="", text="", highlight_bg="", highlight_text="", negative="red", positive="green")
    with patch.object(tables, "get_active_palette", return_value=palette), \
         patch.object(tables, "download_csv"):
        tables.render_table(df, order_by="simbolo", desc=False, ccl_rate=100, show_usd=True)
    styler = mock_st.dataframe.call_args[0][0]
    df_rendered = styler.data
    assert len(df_rendered) == 1
    assert "val_usd_num" in df_rendered.columns
    assert df_rendered.iloc[0]["val_usd_num"] == 10


def test_color_pl_styles(mock_st):
    df = pd.DataFrame({
        "simbolo": ["AAA", "BBB"],
        "tipo": ["ACC", "ACC"],
        "cantidad": [1, 1],
        "ultimo": [1.0, 1.0],
        "valor_actual": [1.0, 1.0],
        "costo": [0.5, 2.0],
        "pl": [0.5, -1.0],
        "pl_%": [50.0, -50.0],
        "pl_d": [0.1, -0.2],
        "chg_%": [0.1, -0.2],
    })
    mock_st.text_input.return_value = ""
    palette = SimpleNamespace(bg="", text="", highlight_bg="", highlight_text="", negative="red", positive="green")
    with patch.object(tables, "get_active_palette", return_value=palette), \
         patch.object(tables, "download_csv"):
        tables.render_table(df, order_by="simbolo", desc=False)
    styler = mock_st.dataframe.call_args[0][0]
    styler._compute()
    col_idx = list(styler.data.columns).index("pl_num")
    pos_style = dict(styler.ctx[(0, col_idx)])
    neg_style = dict(styler.ctx[(1, col_idx)])
    assert pos_style.get("color") == palette.positive
    assert neg_style.get("color") == palette.negative

