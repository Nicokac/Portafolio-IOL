from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from controllers.portfolio import risk as risk_mod
from shared.favorite_symbols import FavoriteSymbols


def test_factor_analysis_section_renders_with_metrics(monkeypatch, streamlit_stub):
    df = pd.DataFrame(
        {
            "simbolo": ["AAPL", "MSFT", "GGAL"],
            "valor_actual": [1200.0, 1100.0, 800.0],
            "tipo": ["CEDEAR", "CEDEAR", "Accion"],
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

    original_columns = streamlit_stub.columns

    def fake_columns(spec, *, gap=None):
        cols = original_columns(spec, gap=gap)

        for col in cols:

            def _plotly_chart(self, fig, **kwargs):
                record = self._core._record("plotly_chart", fig=fig, kwargs=kwargs)
                self._entry.setdefault("children", []).append(record)
                return record

            def _info(self, text):
                record = self._core._record("info", text=str(text))
                self._entry.setdefault("children", []).append(record)
                return record

            col.plotly_chart = _plotly_chart.__get__(col, col.__class__)
            col.info = _info.__get__(col, col.__class__)

        return cols

    monkeypatch.setattr(streamlit_stub, "columns", fake_columns, raising=False)

    monkeypatch.setattr(
        streamlit_stub,
        "bar_chart",
        lambda data, *, x=None, y=None: streamlit_stub._record("bar_chart", data=data, x=x, y=y),
        raising=False,
    )

    base_idx = pd.date_range("2024-01-01", periods=60, freq="B")

    price_map = {
        "AAPL": pd.Series(np.linspace(100, 130, len(base_idx)), index=base_idx),
        "MSFT": pd.Series(np.linspace(80, 110, len(base_idx)), index=base_idx),
        "GGAL": pd.Series(np.linspace(50, 62, len(base_idx)), index=base_idx),
        "^GSPC": pd.Series(np.linspace(4000, 4300, len(base_idx)), index=base_idx),
    }

    def fake_history(*, simbolos, period):
        data = {}
        for sym in simbolos:
            series = price_map.get(sym)
            if series is None:
                base = np.linspace(90, 95, len(base_idx))
                series = pd.Series(base, index=base_idx)
            data[sym] = series
        return pd.DataFrame(data)

    def fake_factors(*, period="1y", benchmark=None):
        return pd.DataFrame(
            {
                "Tasa": np.linspace(0.01, 0.015, len(base_idx)),
                "Inflación": np.linspace(0.005, 0.01, len(base_idx)),
            },
            index=base_idx,
        )

    tasvc = SimpleNamespace(
        portfolio_history=fake_history,
        factor_history=fake_factors,
    )

    heatmaps: list[tuple[str, list[str]]] = []

    def fake_heatmap(prices_df: pd.DataFrame, *, title: str | None = None):
        heatmaps.append((title or "", list(prices_df.columns)))
        return MagicMock()

    monkeypatch.setattr(risk_mod, "plot_correlation_heatmap", fake_heatmap)

    factor_plots: list[tuple[dict[str, float], float]] = []

    def fake_plot_factor_betas(betas, r_squared):
        factor_plots.append((betas, r_squared))
        return MagicMock()

    monkeypatch.setattr(risk_mod, "plot_factor_betas", fake_plot_factor_betas)

    def fake_compute_returns(df_hist: pd.DataFrame) -> pd.DataFrame:
        if df_hist.empty:
            return pd.DataFrame()
        returns = df_hist.pct_change(fill_method=None).dropna(how="all")
        return returns.replace([np.inf, -np.inf], np.nan).dropna(axis=1, how="all")

    def fake_compute_risk_metrics(returns_df, bench_ret, weights, *, var_confidence=0.95):
        port_ret = returns_df.mean(axis=1) if not returns_df.empty else pd.Series(dtype=float)
        asset_vols = pd.Series(dtype=float)
        asset_drawdowns = pd.Series(dtype=float)
        opt_w = (
            pd.Series(1 / max(len(weights), 1), index=weights.index) if not weights.empty else pd.Series(dtype=float)
        )
        return (
            0.18,
            1.05,
            0.04,
            0.06,
            opt_w,
            port_ret,
            asset_vols,
            asset_drawdowns,
            -0.12,
        )

    monkeypatch.setattr(risk_mod, "compute_returns", fake_compute_returns)
    monkeypatch.setattr(risk_mod, "compute_risk_metrics", fake_compute_risk_metrics)

    favorites = FavoriteSymbols({})

    risk_mod.render_risk_analysis(df, tasvc, favorites=favorites)

    subheaders = [entry["text"] for entry in streamlit_stub.get_records("subheader")]
    assert any("Análisis de Factores" in text for text in subheaders)

    def _gather_metrics(nodes):
        found: list[dict[str, object]] = []
        stack = list(nodes)
        while stack:
            entry = stack.pop()
            if isinstance(entry, dict):
                if entry.get("type") == "metric":
                    found.append(entry)
                stack.extend(entry.get("children", []))
        return found

    metric_labels = [entry["label"] for entry in _gather_metrics(streamlit_stub._calls)]
    assert "Tracking Error" in metric_labels
    assert "Information Ratio" in metric_labels

    assert factor_plots, "Expected factor beta plot to be rendered"

    download_keys = {entry["key"] for entry in streamlit_stub.get_records("download_button")}
    assert {"factor_analysis_csv", "factor_analysis_xlsx"}.issubset(download_keys)
