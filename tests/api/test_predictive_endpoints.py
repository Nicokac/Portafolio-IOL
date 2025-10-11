"""Tests for FastAPI predictive and cache endpoints."""

from __future__ import annotations

from typing import Generator

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    """Provide a TestClient instance for each test."""

    with TestClient(app) as test_client:
        yield test_client


def test_predict_endpoint_returns_payload(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    """The /predict endpoint should expose predictions from the service."""

    def fake_predict(*_args, **_kwargs):
        return [
            {
                "sector": "Tecnologia",
                "predicted_return": 0.12,
                "sample_size": 3,
                "avg_correlation": 0.15,
                "confidence": 0.82,
            }
        ]

    monkeypatch.setattr(
        "api.routers.predictive.predict_sector_performance",
        fake_predict,
    )

    response = client.post(
        "/predict",
        json={
            "opportunities": [
                {"symbol": "GGAL", "sector": "Finanzas"},
            ]
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert "predictions" in payload
    assert payload["predictions"][0]["sector"] == "Tecnologia"
    assert pytest.approx(payload["predictions"][0]["predicted_return"], rel=1e-3) == 0.12


def test_adaptive_forecast_endpoint_returns_metrics(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The /forecast/adaptive endpoint should include key forecast metrics."""

    def fake_forecast(*_args, **_kwargs):
        return {
            "mae": 0.05,
            "rmse": 0.07,
            "bias": -0.01,
            "raw_mae": 0.08,
            "raw_rmse": 0.11,
            "raw_bias": -0.03,
            "beta_shift": pd.Series({"Tech": 0.12}),
            "correlation_matrix": [],
            "historical_correlation": [],
            "rolling_correlation": [],
            "summary": {"message": "ok"},
            "steps": [],
            "cache_metadata": {"last_updated": "2024-01-01T00:00:00Z"},
        }

    monkeypatch.setattr(
        "api.routers.predictive.simulate_adaptive_forecast",
        fake_forecast,
    )

    response = client.post("/forecast/adaptive", json={"history": []})

    assert response.status_code == 200
    payload = response.json()
    assert payload["mae"] == pytest.approx(0.05)
    assert payload["cache_metadata"]["last_updated"] == "2024-01-01T00:00:00Z"
    assert payload["beta_shift"] == {"Tech": 0.12}


def test_cache_status_endpoint_reports_counters(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
) -> None:
    """The /cache/status endpoint should expose cache counters."""

    def fake_cache_stats():
        return {
            "namespace": "predictive",
            "hits": 5,
            "misses": 2,
            "hit_ratio": 71.43,
            "last_updated": "2024-01-02 15:30:00",
            "ttl_seconds": 3600.0,
            "remaining_ttl": 1200.0,
        }

    monkeypatch.setattr("api.routers.cache.get_cache_stats", fake_cache_stats)

    response = client.get("/cache/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["namespace"] == "predictive"
    assert payload["hits"] == 5
    assert payload["remaining_ttl"] == pytest.approx(1200.0)
