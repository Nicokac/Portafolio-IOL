"""Integration tests for the predictive engine FastAPI router."""

from __future__ import annotations

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app
from predictive_engine.models import (
    AdaptiveForecastResult,
    CorrelationBundle,
    ModelMetrics,
    SectorPrediction,
    SectorPredictionSet,
)


@pytest.fixture()
def client() -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_engine_predict_endpoint_returns_predictions(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The /engine/predict endpoint should expose computed sector predictions."""

    fake_predictions = SectorPredictionSet(
        rows=[
            SectorPrediction(
                sector="Energy",
                predicted_return=5.5,
                sample_size=3,
                avg_correlation=0.2,
                confidence=0.85,
            )
        ]
    )

    monkeypatch.setattr(
        "api.routers.engine.compute_sector_predictions",
        lambda *args, **kwargs: fake_predictions,
    )
    monkeypatch.setattr("api.routers.engine.BacktestingService", lambda: object())

    response = client.post(
        "/engine/predict",
        json={
            "opportunities": [
                {"symbol": "GGAL", "sector": "Finance"},
            ],
            "span": 8,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["predictions"]
    assert payload["predictions"][0]["sector"] == "Energy"
    assert payload["predictions"][0]["predicted_return"] == pytest.approx(5.5)


def test_engine_forecast_adaptive_returns_forecast_payload(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The adaptive forecast endpoint should serialise the forecast result."""

    metrics = ModelMetrics(
        mae=0.12,
        rmse=0.2,
        bias=0.05,
        raw_mae=0.22,
        raw_rmse=0.31,
        raw_bias=0.08,
        beta_shift_avg=0.04,
        correlation_mean=0.67,
        sector_dispersion=0.15,
    )
    correlations = CorrelationBundle(
        correlation_matrix=pd.DataFrame({"Energy": [1.0]}, index=["Energy"]),
        historical_correlation=pd.DataFrame({"Energy": [1.0]}, index=["Energy"]),
        rolling_correlation=pd.DataFrame({"Energy": [1.0]}, index=["Energy"]),
    )
    steps = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-01-01"),
                "sector": "Energy",
                "raw_prediction": 5.0,
                "adjusted_prediction": 5.4,
                "actual_return": 4.8,
                "beta_adjustment": 0.1,
            }
        ]
    )
    forecast_result = AdaptiveForecastResult(
        metrics=metrics,
        beta_shift=pd.Series({"Energy": 0.15}),
        correlations=correlations,
        steps=steps,
        cache_metadata={"hit_ratio": 0.5},
        summary={"text": "ok"},
    )

    def fake_run_adaptive_forecast(**_kwargs):
        return {
            "forecast": forecast_result,
            "cache_hit": True,
            "cache_metadata": {"hit_ratio": 0.5, "last_updated": "2024-01-01"},
        }

    monkeypatch.setattr("api.routers.engine.run_adaptive_forecast", fake_run_adaptive_forecast)

    response = client.post(
        "/engine/forecast/adaptive",
        json={"history": []},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["cache_hit"] is True
    assert payload["cache_metadata"]["hit_ratio"] == pytest.approx(0.5)
    assert payload["forecast"]["beta_shift"]["Energy"] == pytest.approx(0.15)
    assert payload["forecast"]["mae"] == pytest.approx(0.12)


def test_engine_history_endpoint_returns_records(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The history endpoint should return persisted adaptive history rows."""

    history = pd.DataFrame(
        [
            {
                "timestamp": pd.Timestamp("2024-05-01"),
                "sector": "Energy",
                "predicted_return": 5.5,
                "actual_return": 5.1,
            }
        ]
    )
    monkeypatch.setattr("api.routers.engine.load_forecast_history", lambda: history)

    response = client.get("/engine/history")

    assert response.status_code == 200
    payload = response.json()
    assert payload["history"]
    assert payload["history"][0]["sector"] == "Energy"
