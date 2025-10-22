"""Tests for shared predictive schema definitions."""

from __future__ import annotations

import pytest

from api.routers.engine import (
    AdaptiveForecastRequest as EngineAdaptiveForecastRequest,
)
from api.routers.engine import (
    PredictResponse as EnginePredictResponse,
)
from api.schemas.predictive import (
    AdaptiveForecastRequest,
    AdaptiveHistoryEntry,
    PredictRequest,
)


def test_predict_request_allows_extra_fields() -> None:
    payload = PredictRequest(
        opportunities=[{"symbol": "AAPL", "sector": "Technology", "ignored": True}],
        span=5,
        ttl_hours=2.5,
        extra_field="discarded",
    )

    assert payload.span == 5
    assert payload.ttl_hours == 2.5
    assert payload.opportunities[0].symbol == "AAPL"
    assert "extra_field" not in payload.model_dump()
    assert "ignored" not in payload.opportunities[0].model_dump()


def test_adaptive_forecast_request_limits_unique_symbols() -> None:
    history = [
        AdaptiveHistoryEntry(
            sector=f"Sector-{index}",
            predicted_return=0.1,
            actual_return=0.2,
        )
        for index in range(31)
    ]

    with pytest.raises(ValueError):
        AdaptiveForecastRequest(history=history)


def test_engine_adaptive_request_extends_base() -> None:
    payload = EngineAdaptiveForecastRequest(
        history=[
            {
                "sector": "Finance",
                "predicted_return": 0.2,
                "actual_return": 0.18,
            }
        ],
        predictions=[{"sector": "Finance", "predicted_return": 0.2}],
        actuals=[{"sector": "Finance", "actual_return": 0.18}],
    )

    assert payload.history
    assert payload.predictions
    assert payload.actuals


def test_engine_predict_response_uses_shared_predictions() -> None:
    response = EnginePredictResponse(
        predictions=[
            {
                "sector": "Energy",
                "predicted_return": 0.15,
                "sample_size": 10,
                "avg_correlation": 0.5,
            }
        ]
    )

    assert response.predictions[0].sector == "Energy"
    assert response.predictions[0].sample_size == 10
