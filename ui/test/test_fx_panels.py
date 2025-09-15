import pandas as pd
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from ui.fx_panels import render_spreads, render_fx_history


def test_render_spreads_shows_info_when_no_rates():
    mock_st = SimpleNamespace(
        subheader=MagicMock(),
        info=MagicMock(),
        dataframe=MagicMock(),
        caption=MagicMock(),
    )
    with patch("ui.fx_panels.st", mock_st):
        render_spreads({})

    mock_st.info.assert_called_once_with("Sin cotizaciones para calcular brechas.")
    mock_st.dataframe.assert_not_called()
    mock_st.caption.assert_not_called()


def test_render_spreads_includes_mayorista_when_available():
    rates = {
        "ccl": 1,
        "oficial": 2,
        "blue": 3,
        "mep": 4,
        "mayorista": 5,
    }
    captured_df = {}
    mock_st = SimpleNamespace(
        subheader=MagicMock(),
        info=MagicMock(),
        dataframe=MagicMock(side_effect=lambda df, **kwargs: captured_df.update(df=df)),
        caption=MagicMock(),
    )
    with patch("ui.fx_panels.st", mock_st):
        render_spreads(rates)

    mock_st.info.assert_not_called()
    assert "Mayorista vs CCL" in captured_df["df"]["Par"].tolist()
    mock_st.caption.assert_called_once_with(
        "Muestra la diferencia porcentual entre distintas cotizaciones del dólar."
    )


def test_render_fx_history_info_when_no_valid_columns():
    hist = pd.DataFrame({"ts_dt": pd.date_range("2024", periods=2), "foo": [1, 2]})
    mock_st = SimpleNamespace(
        subheader=MagicMock(),
        info=MagicMock(),
        line_chart=MagicMock(),
        plotly_chart=MagicMock(),
    )
    with patch("ui.fx_panels.st", mock_st):
        render_fx_history(hist)

    mock_st.info.assert_called_once_with("No hay series disponibles para graficar.")
    mock_st.line_chart.assert_not_called()
    mock_st.plotly_chart.assert_not_called()


def test_render_fx_history_plots_when_valid_series():
    hist = pd.DataFrame(
        {
            "ts_dt": pd.date_range("2024", periods=2),
            "ccl": [1.0, 2.0],
            "mep": [1.5, 2.5],
        }
    )
    mock_st = SimpleNamespace(
        subheader=MagicMock(),
        info=MagicMock(),
        line_chart=MagicMock(),
        plotly_chart=MagicMock(),
    )
    fig = SimpleNamespace(update_layout=MagicMock())
    mock_px = SimpleNamespace(line=MagicMock(return_value=fig))
    with patch("ui.fx_panels.st", mock_st), patch("ui.fx_panels.px", mock_px):
        render_fx_history(hist)

    mock_st.info.assert_not_called()
    mock_st.line_chart.assert_called_once()
    mock_st.plotly_chart.assert_called_once_with(fig, width="stretch")
