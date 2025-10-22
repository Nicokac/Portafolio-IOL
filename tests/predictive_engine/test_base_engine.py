import numpy as np
import pandas as pd
import pytest

from predictive_engine.base import (
    calculate_adaptive_forecast,
    compute_sector_predictions,
    evaluate_model_metrics,
    update_adaptive_state,
)
from predictive_engine.models import AdaptiveState, empty_history_frame


def test_evaluate_model_metrics_returns_expected_values():
    adjusted = [0.5, -0.5, 1.0]
    raw = [1.0, -1.0, 2.0]
    corr_matrix = pd.DataFrame([[1.0, 0.2], [0.2, 1.0]], index=["A", "B"], columns=["A", "B"])
    metrics = evaluate_model_metrics(
        adjusted,
        raw,
        beta_shift_avg=0.3,
        correlation_matrix=corr_matrix,
        sector_dispersion=1.2,
    )
    assert metrics.mae == pytest.approx(2.0 / 3.0, rel=1e-6)
    assert metrics.rmse == pytest.approx(np.sqrt((0.25 + 0.25 + 1.0) / 3.0), rel=1e-6)
    assert metrics.raw_mae == pytest.approx(4.0 / 3.0, rel=1e-6)
    assert metrics.beta_shift_avg == pytest.approx(0.3, rel=1e-9)
    assert metrics.correlation_mean == pytest.approx(0.2, rel=1e-9)


def test_calculate_adaptive_forecast_serialisation():
    timestamps = pd.date_range("2024-04-01", periods=4, freq="D")
    history = pd.DataFrame(
        {
            "timestamp": np.tile(timestamps, 2),
            "sector": ["Tech", "Finance"] * 4,
            "predicted_return": [8.0, 4.0, 7.5, 4.2, 7.8, 4.1, 7.6, 4.3],
            "actual_return": [6.5, 3.8, 6.9, 4.0, 7.0, 3.9, 6.7, 4.2],
        }
    )

    state = AdaptiveState(history=empty_history_frame(), last_updated=None)

    def _updater(preds: pd.DataFrame, acts: pd.DataFrame, ts: pd.Timestamp):
        nonlocal state
        result = update_adaptive_state(
            preds,
            acts,
            state=state,
            ema_span=3,
            timestamp=ts,
            max_history_rows=720,
        )
        state = result.state
        return result

    result = calculate_adaptive_forecast(
        history,
        ema_span=3,
        rolling_window=2,
        model_updater=_updater,
        cache_metadata={"hit_ratio": 0.4, "last_updated": "-"},
    )

    payload = result.as_dict()
    assert payload["raw_mae"] >= payload["mae"]
    assert isinstance(payload["beta_shift"], pd.Series)
    assert not payload["beta_shift"].empty
    assert "Tech" in payload["beta_shift"].index
    assert isinstance(payload["summary"], dict)
    assert "MAE adaptativo" in payload["summary"].get("text", "")
    assert payload["cache_metadata"]["hit_ratio"] == pytest.approx(0.4)
    assert isinstance(payload["steps"], pd.DataFrame)
    assert not payload["steps"].empty


def test_compute_sector_predictions_outputs_dataframe():
    opportunities = pd.DataFrame(
        {
            "symbol": ["AAA", "BBB"],
            "sector": ["Energy", "Energy"],
        }
    )

    backtest_data = {
        "AAA": pd.DataFrame({"strategy_ret": [0.01, 0.02, 0.03]}),
        "BBB": pd.DataFrame({"strategy_ret": [0.015, 0.018, 0.02]}),
    }

    class DummyService:
        def __init__(self, data):
            self.data = data

    service = DummyService(backtest_data)

    def run_service(svc, symbol):
        return svc.data.get(symbol)

    def extract_column(backtest, column):
        return pd.Series(backtest[column])

    def ema_predictor(series: pd.Series, span: int) -> float:
        return float(series.ewm(span=span, adjust=False).mean().iloc[-1] * 100.0)

    def zero_correlation(symbol_returns: dict[str, pd.Series]) -> pd.Series:
        return pd.Series(0.0, index=list(symbol_returns))

    result_set = compute_sector_predictions(
        opportunities,
        backtesting_service=service,
        run_backtest=run_service,
        extract_series=extract_column,
        ema_predictor=ema_predictor,
        average_correlation=zero_correlation,
        span=3,
    )

    frame = result_set.to_dataframe()
    assert list(frame.columns) == [
        "sector",
        "predicted_return",
        "sample_size",
        "avg_correlation",
        "confidence",
    ]
    assert frame.iloc[0]["sample_size"] == 2
    expected_a = ema_predictor(backtest_data["AAA"]["strategy_ret"], 3)
    expected_b = ema_predictor(backtest_data["BBB"]["strategy_ret"], 3)
    expected = np.mean([expected_a, expected_b])
    assert frame.iloc[0]["predicted_return"] == pytest.approx(expected, rel=1e-6)
