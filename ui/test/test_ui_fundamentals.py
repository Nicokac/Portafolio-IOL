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
        "dividend_yield": 2.5,
        "price_to_book": 1.5,
        "return_on_equity": 12.0,
        "profit_margin": 15.0,
        "return_on_assets": 9.0,
        "operating_margin": 30.0,
        "fcf_yield": 5.0,
        "interest_coverage": 4.5,
        "debt_to_equity": 0.5,
    }
    with patch.object(fundamentals, "st", new=MagicMock()) as st:
        fundamentals.render_fundamental_data(data)
    st.warning.assert_not_called()
    st.subheader.assert_called_once_with("Análisis Fundamental: ACME")
    st.caption.assert_has_calls(
        [
            call("**Sector:** Tech | **Web:** https://acme.com"),
            call("Resumen de indicadores fundamentales básicos."),
        ]
    )
    df_arg = st.table.call_args[0][0]
    assert isinstance(df_arg, pd.DataFrame)
    assert len(df_arg) == len(fundamentals.INDICATORS)
    rows = {row["Indicador"]: row["Valor"] for row in df_arg.to_dict("records")}
    assert rows["ROE"] == "12.00 %"
    assert rows["Margen Neto"] == "15.00 %"
    assert rows["ROA"] == "9.00 %"
    assert rows["Margen Operativo"] == "30.00 %"
    assert rows["FCF Yield"] == "5.00 %"
    assert rows["Cobertura de Intereses"] == "4.50×"
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
            "return_on_assets": [9.0, 6.0, 7.0],
            "operating_margin": [25.0, 15.0, 10.0],
            "fcf_yield": [4.0, 3.0, 2.0],
            "interest_coverage": [4.0, 5.0, 6.0],
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
    metric_options = st.selectbox.call_args_list[1].args[1]
    assert "return_on_assets" in metric_options
    assert "fcf_yield" in metric_options
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
