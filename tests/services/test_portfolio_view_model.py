"""Tests for the portfolio view service risk metric helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from application.risk_service import compute_returns
from services import portfolio_view as pv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class DummyHistoryService:
    """Minimal stub exposing the ``portfolio_history`` interface."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self._frame = frame

    def portfolio_history(self, *, simbolos, period):  # noqa: ANN001 - signature mimic
        # ``compute_symbol_risk_metrics`` only requires a DataFrame with the
        # requested columns, therefore returning the pre-built frame is enough.
        return self._frame.copy()


def test_compute_symbol_risk_metrics_uses_risk_service_max_drawdown(monkeypatch):
    dates = pd.date_range("2024-01-01", periods=4, freq="D")
    prices = pd.DataFrame(
        {
            "AL30": [100.0, 102.0, 101.0, 103.0],
            "GGAL": [50.0, 51.0, 53.0, 55.0],
            "MERV": [200.0, 198.0, 202.0, 205.0],
        },
        index=dates,
    )

    history_service = DummyHistoryService(prices)

    drawdown_calls: list[pd.Series] = []

    def fake_drawdown(series: pd.Series) -> float:
        drawdown_calls.append(series)
        return -0.5

    monkeypatch.setattr(pv, "max_drawdown", fake_drawdown)
    monkeypatch.setattr(pv, "annualized_volatility", lambda series: float(series.std()))
    monkeypatch.setattr(pv, "beta", lambda sym, bench: 0.75)

    result = pv.compute_symbol_risk_metrics(
        history_service,
        symbols=["AL30", "GGAL"],
        benchmark="MERV",
        period="1mo",
    )

    assert not result.empty
    assert set(result["simbolo"]) == {"AL30", "GGAL", "MERV"}
    assert all(result["drawdown"] == -0.5)

    expected_returns = compute_returns(prices)
    assert len(drawdown_calls) == len(prices.columns)
    assert drawdown_calls[0].equals(expected_returns["AL30"])
