"""Integration-style tests exercising the portfolio tabs with real view-models."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Sequence
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.portfolio_service import PortfolioTotals, calculate_totals
from controllers.portfolio import portfolio as portfolio_mod
from domain.models import Controls
from services.portfolio_view import PortfolioViewSnapshot


class _Column:
    def __init__(self, owner: "_FakeStreamlit") -> None:
        self._owner = owner

    def __enter__(self) -> "_Column":  # noqa: D401 - plain context manager
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - plain context manager
        return None

    def number_input(self, *args, **kwargs):  # noqa: ANN002 - proxy helper
        return self._owner.number_input(*args, **kwargs)

    def selectbox(self, *args, **kwargs):  # noqa: ANN002 - proxy helper
        return self._owner.selectbox(*args, **kwargs)

    def checkbox(self, *args, **kwargs):  # noqa: ANN002 - proxy helper
        return self._owner.checkbox(*args, **kwargs)

    def metric(self, *args, **kwargs):  # noqa: ANN002 - proxy helper
        return self._owner.metric(*args, **kwargs)


class _Spinner:
    def __enter__(self) -> None:  # noqa: D401 - plain context manager
        return None

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - plain context manager
        return None


class _FakeStreamlit:
    """Streamlit stub covering interactions across all tabs."""

    def __init__(self, *, radio_sequence: Iterable[int]) -> None:
        self.session_state: dict[str, object] = {}
        self._radio_iter = iter(radio_sequence)
        self.radio_calls: list[dict[str, object]] = []
        self.selectboxes: list[dict[str, object]] = []
        self.number_inputs: list[dict[str, object]] = []
        self.checkbox_calls: list[dict[str, object]] = []
        self.subheaders: list[str] = []
        self.infos: list[str] = []
        self.warnings: list[str] = []
        self.errors: list[str] = []
        self.successes: list[str] = []
        self.captions: list[str] = []
        self.plot_calls: list[dict[str, object]] = []
        self.line_charts: list[pd.DataFrame] = []
        self.metrics: list[tuple[str, object, object | None]] = []
        self.spinner_messages: list[str] = []
        self.text_inputs: list[dict[str, object]] = []
        self.bar_charts: list[pd.DataFrame] = []

    # -- Layout helpers -------------------------------------------------
    def radio(
        self,
        label: str,
        *,
        options: Sequence[int],
        format_func,
        index: int = 0,
        horizontal: bool,
        key: str,
    ) -> int:
        try:
            value = next(self._radio_iter)
        except StopIteration:
            value = options[index]
        self.radio_calls.append({"label": label, "options": list(options), "key": key})
        self.session_state[key] = value
        return value

    def selectbox(self, label: str, options, index: int = 0, key: str | None = None, **_: object):
        self.selectboxes.append({"label": label, "options": list(options), "key": key})
        if not options:
            return None
        try:
            return options[index]
        except IndexError:
            return options[0]

    def number_input(
        self,
        label: str,
        *,
        min_value: object,
        max_value: object,
        value: object,
        step: object,
    ) -> object:
        self.number_inputs.append(
            {
                "label": label,
                "min_value": min_value,
                "max_value": max_value,
                "value": value,
                "step": step,
            }
        )
        return value

    def checkbox(self, label: str, key: str, **_: object) -> bool:
        self.checkbox_calls.append({"label": label, "key": key})
        return False

    def columns(self, spec):  # noqa: ANN001 - mimic streamlit signature
        if isinstance(spec, int):
            return [_Column(self) for _ in range(spec)]
        return [_Column(self) for _ in spec]

    def expander(self, label: str):  # noqa: ANN001 - mimic streamlit signature
        self.subheaders.append(f"expander:{label}")
        return _Column(self)

    def spinner(self, message: str) -> _Spinner:
        self.spinner_messages.append(message)
        return _Spinner()

    # -- Display primitives --------------------------------------------
    def subheader(self, text: str) -> None:
        self.subheaders.append(text)

    def info(self, message: str) -> None:
        self.infos.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def success(self, message: str) -> None:
        self.successes.append(message)

    def caption(self, message: str) -> None:
        self.captions.append(message)

    def markdown(self, message: str, **_: object) -> None:
        self.captions.append(message)

    def plotly_chart(self, fig, **kwargs) -> None:  # noqa: ANN001 - mimic streamlit signature
        self.plot_calls.append({"fig": fig, "kwargs": kwargs})

    def dataframe(self, data, **_: object) -> None:  # noqa: ANN001 - mimic streamlit signature
        self.table = data

    def line_chart(self, data: pd.DataFrame) -> None:
        self.line_charts.append(data)

    def metric(self, label: str, value: object, delta: object | None = None) -> None:
        self.metrics.append((label, value, delta))

    def text_input(self, label: str, value: str = "", **_: object) -> str:
        self.text_inputs.append({"label": label, "value": value})
        return value

    def bar_chart(self, data: pd.DataFrame, **_: object) -> None:
        self.bar_charts.append(data)

    def write(self, message: object) -> None:
        self.captions.append(str(message))

    # -- Session helpers ------------------------------------------------
    def stop(self) -> None:  # pragma: no cover - should not trigger
        raise RuntimeError("streamlit.stop should not be called in integration tests")

    def rerun(self) -> None:  # pragma: no cover - should not trigger
        raise RuntimeError("streamlit.rerun should not be called in integration tests")


class _DummyFavorites:
    def __init__(self, items: Iterable[str] | None = None) -> None:
        self._items = [str(item) for item in (items or [])]

    def list(self) -> list[str]:
        return list(self._items)

    def sort_options(self, options: Sequence[str]) -> list[str]:
        return list(options)

    def default_index(self, options: Sequence[str]) -> int:
        return 0 if options else 0

    def format_symbol(self, symbol: str) -> str:
        return str(symbol)

    def normalize(self, symbol: str | None) -> str:
        return str(symbol or "").upper()

    def is_favorite(self, symbol: str | None) -> bool:
        sym = self.normalize(symbol)
        return sym in self._items

    def add(self, symbol: str | None) -> None:
        sym = self.normalize(symbol)
        if sym and sym not in self._items:
            self._items.append(sym)

    def remove(self, symbol: str | None) -> None:
        sym = self.normalize(symbol)
        if sym in self._items:
            self._items.remove(sym)


class _DummyContainer:
    def __enter__(self) -> "_DummyContainer":  # noqa: D401 - plain context manager
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - plain context manager
        return None


class _DummyTAService:
    """Synthetic TA service providing deterministic datasets."""

    def portfolio_history(self, *, simbolos: list[str], period: str = "1y") -> pd.DataFrame:
        idx = pd.date_range("2023-01-01", periods=60, freq="D")
        data = {}
        for sym in simbolos:
            base = np.linspace(100.0, 120.0, len(idx))
            if sym == "^GSPC":
                base = np.linspace(3_800.0, 4_200.0, len(idx))
                data[sym] = base
            else:
                data[sym] = base + (hash(sym) % 5)
        return pd.DataFrame(data, index=idx)

    def fundamentals(self, us_ticker: str) -> dict[str, object]:
        return {"ticker": us_ticker, "pe_ratio": 15.2}

    def indicators_for(self, *_, **__) -> pd.DataFrame:
        idx = pd.date_range("2024-01-01", periods=40, freq="D")
        prices = np.linspace(100.0, 120.0, len(idx))
        data = {
            "Open": prices + 1,
            "High": prices + 2,
            "Low": prices - 2,
            "Close": prices,
            "Volume": np.linspace(1_000_000, 1_200_000, len(idx)),
            "SMA_FAST": prices * 0.98,
            "SMA_SLOW": prices * 0.99,
            "EMA": prices * 0.985,
            "BB_L": prices - 5,
            "BB_M": prices,
            "BB_U": prices + 5,
            "RSI": np.linspace(45, 65, len(idx)),
            "MACD": np.linspace(-0.5, 0.5, len(idx)),
            "MACD_SIGNAL": np.zeros(len(idx)),
            "MACD_HIST": np.linspace(-0.2, 0.2, len(idx)),
            "ATR": np.linspace(1.0, 1.5, len(idx)),
            "STOCH_K": np.linspace(20, 80, len(idx)),
            "STOCH_D": np.linspace(25, 75, len(idx)),
            "ICHI_CONV": prices * 0.97,
            "ICHI_BASE": prices * 0.96,
            "ICHI_A": prices * 0.95,
            "ICHI_B": prices * 0.94,
        }
        return pd.DataFrame(data, index=idx)

    def alerts_for(self, df_ind: pd.DataFrame) -> list[str]:
        if df_ind.empty:
            return []
        return ["⚡ Cruce alcista detectado", "📈 Precio sobre banda superior"]

    def backtest(self, df_ind: pd.DataFrame, *, strategy: str = "sma") -> pd.DataFrame:
        if df_ind.empty:
            return pd.DataFrame()
        idx = pd.date_range("2024-02-01", periods=10, freq="D")
        equity = np.linspace(1.0, 1.1, len(idx))
        return pd.DataFrame({"equity": equity}, index=idx)


def _snapshot(df_view: pd.DataFrame) -> PortfolioViewSnapshot:
    totals: PortfolioTotals = calculate_totals(df_view)
    return PortfolioViewSnapshot(
        df_view=df_view,
        totals=totals,
        apply_elapsed=0.001,
        totals_elapsed=0.0005,
        generated_at=0.0,
    )


def _base_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "tipo": "ACCION",
                "cantidad": 10,
                "valor_actual": 1500.0,
                "costo": 1000.0,
                "pl": 500.0,
                "pl_%": 50.0,
                "pl_d": 25.0,
                "pld_%": 1.5,
                "costo_unitario": 100.0,
            },
            {
                "simbolo": "AAPL",
                "mercado": "nyse",
                "tipo": "CEDEAR",
                "cantidad": 5,
                "valor_actual": 1100.0,
                "costo": 900.0,
                "pl": 200.0,
                "pl_%": 22.2,
                "pl_d": -10.0,
                "pld_%": -0.8,
                "costo_unitario": 150.0,
            },
        ]
    )


def _run_for_tab(tab_index: int, monkeypatch: pytest.MonkeyPatch) -> _FakeStreamlit:
    df = _base_dataframe()
    controls = Controls(refresh_secs=45, top_n=3)
    favorites = _DummyFavorites(["GGAL"])
    ta_stub = _DummyTAService()

    from controllers.portfolio import charts as charts_mod
    from controllers.portfolio import risk as risk_mod
    from ui import tables as tables_mod
    from ui import favorites as favorites_mod

    fake_st = _FakeStreamlit(radio_sequence=[tab_index])
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(charts_mod, "st", fake_st)
    monkeypatch.setattr(risk_mod, "st", fake_st)
    monkeypatch.setattr(tables_mod, "st", fake_st)
    monkeypatch.setattr(favorites_mod, "st", fake_st)
    monkeypatch.setattr(charts_mod, "render_table", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_totals", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_mod, "PortfolioService", lambda: object())
    monkeypatch.setattr(portfolio_mod, "TAService", lambda: ta_stub)
    monkeypatch.setattr(portfolio_mod, "load_portfolio_data", lambda cli, svc: (df, ["GGAL", "AAPL"], ["ACCION", "CEDEAR"]))
    monkeypatch.setattr(portfolio_mod, "render_sidebar", lambda syms, types: controls)
    monkeypatch.setattr(portfolio_mod, "render_ui_controls", lambda: None)
    monkeypatch.setattr(portfolio_mod, "get_persistent_favorites", lambda: favorites)
    monkeypatch.setattr(portfolio_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(charts_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_badges", lambda *a, **k: None)
    monkeypatch.setattr(risk_mod, "render_favorite_toggle", lambda *a, **k: None)
    monkeypatch.setattr(
        risk_mod,
        "markowitz_optimize",
        lambda returns: pd.Series(
            1.0 / len(returns.columns), index=returns.columns
        )
        if not returns.empty
        else pd.Series(dtype=float),
    )
    monkeypatch.setattr(portfolio_mod, "render_fundamental_data", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_mod, "render_fundamental_analysis", lambda *a, **k: None)
    monkeypatch.setattr(portfolio_mod, "view_model_service", SimpleNamespace(get_portfolio_view=lambda **_: _snapshot(df)))
    monkeypatch.setattr(portfolio_mod, "map_to_us_ticker", lambda sym: f"{sym}.US")

    refresh = portfolio_mod.render_portfolio_section(_DummyContainer(), cli=object(), fx_rates={"ccl": 890.0})
    assert refresh == controls.refresh_secs
    assert not fake_st.errors
    return fake_st


def test_portfolio_tab_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _run_for_tab(0, monkeypatch)
    keys = {call["kwargs"].get("key") for call in fake_st.plot_calls}
    assert {"pl_topn", "donut_tipo", "dist_tipo", "pl_diario"}.issubset(keys)


def test_advanced_tab_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _run_for_tab(1, monkeypatch)
    keys = {call["kwargs"].get("key") for call in fake_st.plot_calls}
    assert "bubble_chart" in keys


def test_risk_tab_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _run_for_tab(2, monkeypatch)
    assert any(label.startswith("Volatilidad") for label, *_ in fake_st.metrics)
    assert any("VaR" in label for label, *_ in fake_st.metrics)


def test_technical_tab_renders(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_st = _run_for_tab(4, monkeypatch)
    keys = {call["kwargs"].get("key") for call in fake_st.plot_calls}
    assert "ta_chart" in keys
    assert fake_st.line_charts, "Expected backtest chart to be rendered"
