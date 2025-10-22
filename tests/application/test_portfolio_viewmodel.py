"""Tests for the portfolio view-model and cached helpers."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

import application.portfolio_service as portfolio_mod
from application.portfolio_service import PortfolioService, calc_rows

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


@pytest.fixture(autouse=True)
def _reset_classify_cache():
    """Ensure classify cache is clean before each test."""
    portfolio_mod._classify_sym_cache.cache_clear()
    yield
    portfolio_mod._classify_sym_cache.cache_clear()


def test_calc_rows_generates_viewmodel_with_expected_metrics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`calc_rows` should combine quotes and positions into a rich view-model."""

    df_pos = pd.DataFrame(
        [
            {
                "simbolo": "GGAL",
                "mercado": "bcba",
                "cantidad": 10,
                "costo_unitario": 100,
            },
            {
                "simbolo": "AAPL",
                "mercado": "nyse",
                "cantidad": 5,
                "costo_unitario": 150,
            },
        ]
    )

    quotes = {
        ("bcba", "GGAL"): {"last": 120.0, "chg_pct": 1.5, "cierreAnterior": 118.0},
        ("nyse", "AAPL"): {"last": 160.0, "chg_pct": -0.5, "cierreAnterior": 161.0},
    }

    def fake_quote(mercado: str, simbolo: str) -> dict:
        return quotes[(mercado, simbolo)]

    monkeypatch.setattr(
        portfolio_mod,
        "classify_symbol",
        lambda sym: "CEDEAR" if sym == "AAPL" else "ACCION",
    )
    monkeypatch.setattr(portfolio_mod, "scale_for", lambda sym, tipo: 1.0)

    view = calc_rows(fake_quote, df_pos, exclude_syms=[])

    assert list(view.columns) == [
        "simbolo",
        "mercado",
        "tipo",
        "cantidad",
        "ppc",
        "ultimo",
        "valor_actual",
        "costo",
        "pl",
        "pl_%",
        "pl_d",
        "pld_%",
    ]
    assert len(view) == 2

    ggal = view.loc[view["simbolo"] == "GGAL"].iloc[0]
    assert ggal["tipo"] == "ACCION"
    assert ggal["ultimo"] == pytest.approx(120.0)
    assert ggal["valor_actual"] == pytest.approx(10 * 120.0)
    assert ggal["costo"] == pytest.approx(10 * 100.0)
    assert ggal["pl"] == pytest.approx(200.0)
    assert ggal["pl_%"] == pytest.approx(20.0)
    # Daily P/L accounts for change percentage relative to today's value
    expected_daily_pl = (ggal["valor_actual"] * 0.015) / (1 + 0.015)
    assert ggal["pl_d"] == pytest.approx(expected_daily_pl)
    assert ggal["pld_%"] == pytest.approx(1.5)

    aapl = view.loc[view["simbolo"] == "AAPL"].iloc[0]
    assert aapl["tipo"] == "CEDEAR"
    assert aapl["pl"] == pytest.approx(5 * (160.0 - 150.0))
    assert aapl["pl_%"] == pytest.approx((160.0 - 150.0) / 150.0 * 100)


def test_classify_asset_cached_uses_lru_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    """`PortfolioService.classify_asset_cached` should hit the expensive classifier once."""

    calls: list[str] = []

    def fake_classify(row: dict) -> str:
        calls.append(row["simbolo"])
        return "ETF"

    monkeypatch.setattr(portfolio_mod, "classify_asset", fake_classify)

    svc = PortfolioService()
    first = svc.classify_asset_cached("GGAL")
    second = svc.classify_asset_cached("GGAL")

    assert first == "ETF"
    assert second == "ETF"
    assert calls == ["GGAL"], "Expected classify_asset to be invoked once thanks to caching"
