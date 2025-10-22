import pandas as pd
import pytest

from application.predictive_service import PredictiveSnapshot
from controllers import recommendations_controller


@pytest.fixture()
def sample_snapshot() -> PredictiveSnapshot:
    return PredictiveSnapshot(
        hits=5,
        misses=2,
        last_updated="2024-06-01",
        ttl_hours=1.5,
        remaining_ttl=7200.0,
    )


def test_load_sector_performance_view_returns_view_model(
    monkeypatch: pytest.MonkeyPatch, sample_snapshot: PredictiveSnapshot
) -> None:
    opportunities = pd.DataFrame({"symbol": ["TEST"], "sector": ["Tecnología"]})
    predictions = pd.DataFrame(
        {
            "sector": ["Tecnología"],
            "predicted_return": [0.12],
        }
    )

    def fake_predict(frame: pd.DataFrame | None, **kwargs):
        assert frame is opportunities
        return predictions

    monkeypatch.setattr(recommendations_controller, "predict_sector_performance", fake_predict)
    monkeypatch.setattr(recommendations_controller, "get_cache_stats", lambda: sample_snapshot)

    view = recommendations_controller.load_sector_performance_view(opportunities)

    assert view.predictions is not predictions
    assert view.predictions.equals(predictions)
    cache_dict = view.cache.to_dict()
    assert cache_dict["hits"] == sample_snapshot.hits
    assert cache_dict["misses"] == sample_snapshot.misses
    assert cache_dict["last_updated"] == sample_snapshot.last_updated

    cache_view = recommendations_controller.get_predictive_cache_view()
    assert cache_view.to_dict()["hits"] == sample_snapshot.hits


def test_run_adaptive_forecast_view_sanitizes_payload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    history = pd.DataFrame(
        {
            "timestamp": ["2024-01-01"],
            "sector": ["Tech"],
            "predicted_return": [0.1],
            "actual_return": [0.08],
        }
    )
    summary = {"mae": 0.02, "rmse": 0.03}
    payload = {
        "summary": summary,
        "historical_correlation": pd.DataFrame(),
        "rolling_correlation": pd.DataFrame(),
        "correlation_matrix": pd.DataFrame([[1.0]]),
        "beta_shift": pd.Series([0.1]),
        "cache_metadata": {"hit_ratio": 80.0},
        "steps": pd.DataFrame({"timestamp": ["2024-01-01"], "mae": [0.02]}),
    }

    def fake_simulate(frame: pd.DataFrame | None, **kwargs):
        assert frame is history
        return payload

    monkeypatch.setattr(recommendations_controller, "simulate_adaptive_forecast", fake_simulate)

    view = recommendations_controller.run_adaptive_forecast_view(history, ema_span=6, persist=False)

    assert view.summary == summary
    assert view.summary is not summary
    assert view.payload is not payload
    assert view.payload["summary"] is not summary
    assert view.payload["cache_metadata"] is not payload["cache_metadata"]
    assert isinstance(view.payload["correlation_matrix"], pd.DataFrame)
