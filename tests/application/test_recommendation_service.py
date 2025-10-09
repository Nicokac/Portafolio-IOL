from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from application.recommendation_service import RecommendationService


@pytest.fixture()
def portfolio_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL", "YPFD", "AL30", "BMA", "KO"],
            "valor_actual": [350_000, 220_000, 180_000, 140_000, 110_000],
            "tipo": ["ACCION", "ACCION", "BONO", "ACCION", "CEDEAR"],
            "sector": [
                "Financial Services",
                "Energy",
                "Government",
                "Financial Services",
                "Consumer Defensive",
            ],
            "moneda": ["ARS", "ARS", "ARS", "ARS", "USD"],
        }
    )


@pytest.fixture()
def opportunities_df() -> pd.DataFrame:
    rows = [
        {
            "ticker": "JNJ",
            "sector": "Healthcare",
            "score_compuesto": 82.0,
            "cagr": 7.0,
            "dividend_yield": 2.7,
            "market_cap": 430_000,
            "pe_ratio": 21.0,
        },
        {
            "ticker": "XLU",
            "sector": "Utilities",
            "score_compuesto": 71.0,
            "cagr": 5.5,
            "dividend_yield": 3.0,
            "market_cap": 15_000,
            "pe_ratio": 18.0,
        },
        {
            "ticker": "SMH",
            "sector": "Technology",
            "score_compuesto": 78.0,
            "cagr": 16.0,
            "dividend_yield": 1.0,
            "market_cap": 12_000,
            "pe_ratio": 28.0,
        },
        {
            "ticker": "VIG",
            "sector": "Consumer Defensive",
            "score_compuesto": 69.0,
            "cagr": 8.0,
            "dividend_yield": 1.9,
            "market_cap": 70_000,
            "pe_ratio": 24.0,
        },
        {
            "ticker": "IGIB",
            "sector": "Financial Services",
            "score_compuesto": 66.0,
            "cagr": 4.0,
            "dividend_yield": 3.4,
            "market_cap": 8_500,
            "pe_ratio": 18.0,
        },
        {
            "ticker": "XLRE",
            "sector": "Real Estate",
            "score_compuesto": 68.0,
            "cagr": 6.5,
            "dividend_yield": 3.2,
            "market_cap": 6_000,
            "pe_ratio": 25.0,
        },
    ]
    return pd.DataFrame(rows)


@pytest.fixture()
def risk_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "simbolo": ["GGAL", "YPFD", "AL30", "BMA", "KO"],
            "beta": [1.4, 1.2, 0.4, 1.1, 0.6],
        }
    )


def test_diversify_mode_prioritises_underrepresented_sectors(
    portfolio_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    svc = RecommendationService(
        portfolio_df=portfolio_df,
        opportunities_df=opportunities_df,
        risk_metrics_df=risk_df,
    )
    result = svc.recommend(100_000, mode="diversify")

    assert list(result.columns) == ["symbol", "allocation_%", "allocation_amount", "rationale"]
    assert len(result) == 5
    top_symbol = result.iloc[0]["symbol"]
    assert top_symbol in {"JNJ", "XLU"}, "Expected defensive sectors to lead diversification"
    assert (result["allocation_%"] >= 10.0 - 1e-6).all()
    assert (result["allocation_%"] <= 40.0 + 1e-6).all()
    assert result["allocation_amount"].sum() == pytest.approx(100_000.0)
    assert result["allocation_%"].sum() == pytest.approx(100.0)
    assert set(result["symbol"]) & set(portfolio_df["simbolo"]) != set()
    assert any("Healthcare" in rationale for rationale in result["rationale"])


def test_low_risk_mode_favours_defensive_assets(
    portfolio_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    svc = RecommendationService(
        portfolio_df=portfolio_df,
        opportunities_df=opportunities_df,
        risk_metrics_df=risk_df,
    )
    result = svc.recommend(50_000, mode="low_risk")

    assert len(result) == 5
    assert result.iloc[0]["symbol"] in {"JNJ", "XLU", "XLRE"}
    rationale = " ".join(result["rationale"].tolist())
    assert "defensivo" in rationale or "defensivo" in rationale.lower()


def test_max_return_respects_supported_modes(
    portfolio_df: pd.DataFrame,
    opportunities_df: pd.DataFrame,
    risk_df: pd.DataFrame,
) -> None:
    svc = RecommendationService(
        portfolio_df=portfolio_df,
        opportunities_df=opportunities_df,
        risk_metrics_df=risk_df,
    )
    result = svc.recommend(200_000, mode="max_return")

    assert len(result) == 5
    assert np.all(result["allocation_%"] > 0)
    assert result["allocation_%"].sum() == pytest.approx(100.0)
    assert any("retorno" in rationale.lower() for rationale in result["rationale"])

