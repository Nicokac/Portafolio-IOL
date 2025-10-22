"""Tests for risk analysis asset type alignment."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from controllers.portfolio import risk as risk_mod
from shared.favorite_symbols import FavoriteSymbols


def test_heatmap_excludes_local_symbols_in_cedear_group(monkeypatch, streamlit_stub):
    """Ensure CEDEAR heatmap excludes local tickers like LOMA and YPFD."""

    df = pd.DataFrame(
        {
            "simbolo": ["AAPL", "NVDA", "LOMA", "YPFD"],
            "valor_actual": [1000.0, 950.0, 400.0, 380.0],
            "mercado": ["nyse", "nyse", "bcba", "bcba"],
            "tipo": ["CEDEAR", "CEDEAR", None, None],
        }
    )

    streamlit_stub.reset()
    streamlit_stub.session_state["selected_asset_types"] = ["CEDEAR"]
    monkeypatch.setattr(risk_mod, "st", streamlit_stub)
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    class _ContainerCtx:
        def __enter__(self):
            return streamlit_stub

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(streamlit_stub, "container", lambda: _ContainerCtx(), raising=False)

    original_selectbox = streamlit_stub.selectbox

    def fake_selectbox(label, options, *, index=0, key=None, help=None, format_func=None):
        return original_selectbox(label, options, index=index, key=key, help=help)

    monkeypatch.setattr(streamlit_stub, "selectbox", fake_selectbox)

    history_calls: list[list[str]] = []

    def fake_history(*, simbolos, period):
        history_calls.append(list(simbolos))
        if len(history_calls) >= 2:
            return pd.DataFrame()
        idx = pd.date_range("2024-01-01", periods=8, freq="B")
        data = {sym: np.linspace(100.0 + i, 105.0 + i, len(idx)) for i, sym in enumerate(simbolos)}
        return pd.DataFrame(data, index=idx)

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    heatmap_columns: list[list[str]] = []

    def fake_heatmap(prices_df: pd.DataFrame, *, title: str | None = None):
        heatmap_columns.append(list(prices_df.columns))
        return MagicMock()

    monkeypatch.setattr(risk_mod, "plot_correlation_heatmap", fake_heatmap)

    def fake_compute_returns(df_hist: pd.DataFrame) -> pd.DataFrame:
        if df_hist.empty:
            return pd.DataFrame()
        returns = df_hist.pct_change(fill_method=None).dropna(how="all")
        return returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)

    favorites = FavoriteSymbols({})

    risk_mod.render_risk_analysis(df, tasvc, favorites=favorites)

    assert history_calls, "portfolio_history should be invoked at least once"
    assert set(history_calls[0]) == {"AAPL", "NVDA"}
    assert all(sym not in {"LOMA", "YPFD", "TECO2"} for call in history_calls for sym in call)
    assert heatmap_columns, "Expected heatmap to be rendered"
    assert set(heatmap_columns[0]) == {"AAPL", "NVDA"}


def test_accion_local_tab_renders_independent_heatmap(monkeypatch, streamlit_stub):
    """The analysis should render a dedicated tab for local equities."""

    df = pd.DataFrame(
        {
            "simbolo": ["AAPL", "NVDA", "LOMA", "YPFD", "TECO2"],
            "valor_actual": [1000.0, 950.0, 400.0, 380.0, 360.0],
            "mercado": ["nyse", "nyse", "bcba", "bcba", "bcba"],
            "tipo": ["CEDEAR", "CEDEAR", "Acción", "Acción", None],
        }
    )

    streamlit_stub.reset()
    streamlit_stub.session_state["selected_asset_types"] = []
    monkeypatch.setattr(risk_mod, "st", streamlit_stub)
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    original_selectbox = streamlit_stub.selectbox

    def fake_selectbox(label, options, *, index=0, key=None, help=None, format_func=None):
        return original_selectbox(label, options, index=index, key=key, help=help)

    monkeypatch.setattr(streamlit_stub, "selectbox", fake_selectbox, raising=False)

    history_calls: list[list[str]] = []

    def fake_history(*, simbolos, period):
        symbols_list = list(simbolos)
        history_calls.append(symbols_list)
        if len(history_calls) >= 2:
            return pd.DataFrame()
        idx = pd.date_range("2024-01-01", periods=10, freq="B")
        data = {sym: np.linspace(100.0 + i * 2, 110.0 + i * 2, len(idx)) for i, sym in enumerate(symbols_list)}
        return pd.DataFrame(data, index=idx)

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    heatmap_payloads: list[tuple[str, list[str]]] = []

    def fake_heatmap(prices_df: pd.DataFrame, *, title: str | None = None):
        heatmap_payloads.append((title or "", list(prices_df.columns)))
        return MagicMock()

    monkeypatch.setattr(risk_mod, "plot_correlation_heatmap", fake_heatmap)

    def fake_compute_returns(df_hist: pd.DataFrame) -> pd.DataFrame:
        returns = df_hist.pct_change(fill_method=None).dropna(how="all")
        return returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)

    favorites = FavoriteSymbols({})

    risk_mod.render_risk_analysis(df, tasvc, favorites=favorites)

    assert history_calls, "Expected portfolio_history to be invoked"
    corr_call = history_calls[0]
    assert set(corr_call) == {"AAPL", "NVDA", "LOMA", "YPFD", "TECO2"}

    tab_entries = streamlit_stub.get_records("tabs")
    assert tab_entries, "Tabs should be rendered when multiple types exist"
    rendered_labels = tab_entries[0]["labels"]
    assert any("CEDEAR" in label for label in rendered_labels)
    assert any("Acciones locales" in label for label in rendered_labels)

    assert len(heatmap_payloads) >= 2, "Expected separate heatmaps for each type"
    cedear_heatmap = next(
        (cols for title, cols in heatmap_payloads if "CEDEAR" in title),
        None,
    )
    local_heatmap = next(
        (cols for title, cols in heatmap_payloads if "Acciones locales" in title),
        None,
    )
    assert cedear_heatmap == ["AAPL", "NVDA"]
    assert set(local_heatmap or []) == {"LOMA", "YPFD", "TECO2"}


def test_build_type_metadata_respects_catalog_overrides():
    """Local tickers are forced to ACCION_LOCAL even if raw type mislabels them."""

    df = pd.DataFrame(
        {
            "simbolo": ["AAPL", "LOMA", "TECO2"],
            "tipo": ["CEDEAR", "CEDEAR", None],
        }
    )

    normalized, display_map, symbol_map = risk_mod._build_type_metadata(df)

    assert normalized.tolist() == ["CEDEAR", "ACCION_LOCAL", "ACCION_LOCAL"]
    assert symbol_map["LOMA"] == "ACCION_LOCAL"
    assert symbol_map["TECO2"] == "ACCION_LOCAL"
    assert display_map["ACCION_LOCAL"].lower().startswith("accion")


def test_all_asset_tabs_render_with_warnings_for_insufficient_data(monkeypatch, streamlit_stub):
    """Every asset type present in the portfolio should render a tab with friendly labels."""

    df = pd.DataFrame(
        {
            "simbolo": [
                "AAPL",
                "NVDA",
                "LOMA",
                "GD30",
                "S31Y5",
                "FIMA",
                "SPY",
                "BTCUSDT",
            ],
            "valor_actual": [
                1000.0,
                950.0,
                400.0,
                300.0,
                200.0,
                150.0,
                500.0,
                250.0,
            ],
            "tipo": [
                "CEDEAR",
                "CEDEAR",
                "Acciones Argentinas",
                "Bonos Dólar",
                "Letras del Tesoro",
                "Fondo Money Market",
                "ETF",
                "Otros",
            ],
        }
    )

    streamlit_stub.reset()
    streamlit_stub.session_state["selected_asset_types"] = []
    monkeypatch.setattr(risk_mod, "st", streamlit_stub)
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)

    original_selectbox = streamlit_stub.selectbox

    def fake_selectbox(label, options, *, index=0, key=None, help=None, format_func=None):
        return original_selectbox(label, options, index=index, key=key, help=help)

    monkeypatch.setattr(streamlit_stub, "selectbox", fake_selectbox, raising=False)

    history_calls: list[list[str]] = []

    def fake_history(*, simbolos, period):
        symbols_list = list(simbolos)
        history_calls.append(symbols_list)
        idx = pd.date_range("2024-02-01", periods=6, freq="B")
        available = {
            "AAPL": np.linspace(100.0, 104.0, len(idx)),
            "NVDA": np.linspace(200.0, 208.0, len(idx)),
            "LOMA": np.linspace(50.0, 55.0, len(idx)),
        }
        data = {sym: available[sym] for sym in symbols_list if sym in available}
        return pd.DataFrame(data, index=idx)

    tasvc = SimpleNamespace(portfolio_history=fake_history)

    heatmap_payloads: list[tuple[str, list[str]]] = []

    def fake_heatmap(prices_df: pd.DataFrame, *, title: str | None = None):
        heatmap_payloads.append((title or "", list(prices_df.columns)))
        return MagicMock()

    monkeypatch.setattr(risk_mod, "plot_correlation_heatmap", fake_heatmap)

    def fake_compute_returns(df_hist: pd.DataFrame) -> pd.DataFrame:
        if df_hist.empty:
            return pd.DataFrame()
        returns = df_hist.pct_change(fill_method=None).dropna(how="all")
        return returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)

    favorites = FavoriteSymbols({})

    risk_mod.render_risk_analysis(
        df,
        tasvc,
        favorites=favorites,
        available_types=[
            "CEDEAR",
            "ACCION_LOCAL",
            "BONO",
            "LETRA",
            "FCI",
            "ETF",
            "OTRO",
        ],
    )

    assert history_calls, "portfolio_history should be invoked"
    assert heatmap_payloads, "At least one heatmap should be rendered"
    heatmap_title, heatmap_columns = heatmap_payloads[0]
    assert "Matriz de Correlación — CEDEARs" in heatmap_title
    assert set(heatmap_columns) == {"AAPL", "NVDA"}

    tab_entries = streamlit_stub.get_records("tabs")
    assert tab_entries, "Tabs should be rendered for all asset types"
    expected_labels = [
        "CEDEARs",
        "Acciones locales",
        "Bonos",
        "Letras",
        "Fondos comunes (FCI)",
        "ETFs",
        "Otros",
    ]
    assert tab_entries[0]["labels"] == expected_labels

    warnings = [entry["text"] for entry in streamlit_stub.get_records("warning")]
    for label in expected_labels[1:]:
        assert any(label in warning for warning in warnings), f"Missing warning for {label}"
