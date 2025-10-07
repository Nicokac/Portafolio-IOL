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
