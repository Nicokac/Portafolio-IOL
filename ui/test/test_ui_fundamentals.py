import pandas as pd
from unittest.mock import MagicMock, patch, call

from ui import fundamentals


def test_render_fundamental_data_none():
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        fundamentals.render_fundamental_data(None)
    st.warning.assert_called_once_with("Datos fundamentales no disponibles.")


def test_render_fundamental_data_error():
    data = {"error": "Algo salió mal"}
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        fundamentals.render_fundamental_data(data)
    st.warning.assert_called_once_with("Algo salió mal")


def test_render_fundamental_data_normal_flow():
    data = {
        "name": "ACME",
        "sector": "Tech",
        "website": "https://acme.com",
        "market_cap": 1000,
        "pe_ratio": 10,
        "dividend_yield": 0.02,
        "price_to_book": 1.5,
        "return_on_equity": 0.1,
        "profit_margin": 0.15,
        "debt_to_equity": 0.5,
    }
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        fundamentals.render_fundamental_data(data)
    st.warning.assert_not_called()
    st.subheader.assert_called_once_with("Análisis Fundamental: ACME")
    st.caption.assert_called_once_with("**Sector:** Tech | **Web:** https://acme.com")
    df_arg = st.table.call_args[0][0]
    assert isinstance(df_arg, pd.DataFrame)
    assert len(df_arg) == len(fundamentals.INDICATORS)
    st.divider.assert_called_once()


def test_render_fundamental_ranking_empty_df():
    df = pd.DataFrame()
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        fundamentals.render_fundamental_ranking(df)
    st.info.assert_called_once_with("No se pudieron obtener datos fundamentales.")
    st.selectbox.assert_not_called()


def test_render_fundamental_ranking_filters_sector_and_detects_warnings():
    df = pd.DataFrame(
        {
            "sector": ["Tech", "Tech", "Finance"],
            "market_cap": [100, 200, 150],
            "pe_ratio": [10, 20, 15],
            "revenue_growth": [5, 10, 8],
            "earnings_growth": [-1, 2, 3],
            "esg_score": [25, 40, 50],
        }
    )
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        st.selectbox.side_effect = ["Tech", "market_cap"]
        fundamentals.render_fundamental_ranking(df)
    st.dataframe.assert_called_once()
    filtered = st.dataframe.call_args[0][0]
    assert (filtered["sector"] == "Tech").all()
    assert list(filtered["market_cap"]) == [200, 100]
    st.warning.assert_has_calls(
        [
            call("Alerta: crecimiento de ganancias negativo en algunos activos."),
            call("Alerta ESG: puntajes ESG bajos detectados."),
        ]
    )


def test_render_sector_comparison_no_metric_data():
    df = pd.DataFrame(
        {
            "sector": ["Tech", "Finance"],
            "symbol": ["A", "B"],
            "pe_ratio": [float("nan"), float("nan")],
        }
    )
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        st.selectbox.return_value = "pe_ratio"
        fundamentals.render_sector_comparison(df)
    st.info.assert_called_once_with("No hay datos disponibles para la métrica seleccionada.")
    st.plotly_chart.assert_not_called()
