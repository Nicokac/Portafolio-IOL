from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def test_compute_adaptive_payload_passes_profile_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ui.tabs.recommendations.correlation_tab as correlation_tab
    from controllers import recommendations_controller

    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=3, freq="D"),
            "sector": ["Tech", "Finance", "Tech"],
            "predicted_return": [5.0, 4.0, 5.1],
            "actual_return": [4.5, 3.8, 4.9],
        }
    )

    captured_profile: dict[str, Any] = {}
    captured_context: dict[str, Any] = {}

    def fake_build(opportunities, recommendations, *, profile=None, **_kwargs):
        captured_profile.update(profile or {})
        return history.copy(), True

    def fake_run(history_frame, **kwargs):
        captured_context.update(kwargs.get("context", {}))
        return recommendations_controller.AdaptiveForecastViewModel(
            payload={"summary": {}, "history_frame": history_frame},
            summary={},
            cache_metadata={},
        )

    monkeypatch.setattr(
        recommendations_controller,
        "build_adaptive_history_view",
        fake_build,
    )
    monkeypatch.setattr(
        recommendations_controller,
        "run_adaptive_forecast_view",
        fake_run,
    )

    recommendations = pd.DataFrame(
        [
            {"symbol": "AAA", "sector": "Tech"},
            {"symbol": "BBB", "sector": "Finance"},
        ]
    )
    opportunities = recommendations.copy()
    profile = {"preferred_mode": "diversify", "risk_tolerance": "medio"}

    payload = correlation_tab._compute_adaptive_payload(
        recommendations,
        opportunities,
        profile=profile,
    )

    assert payload is not None
    assert payload.get("synthetic") is True
    assert captured_profile["preferred_mode"] == "diversify"
    assert "symbols" in captured_context and captured_context["symbols"]
    assert captured_context.get("profile", {}).get("preferred_mode") == "diversify"


def test_compute_adaptive_payload_returns_none_when_history_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ui.tabs.recommendations.correlation_tab as correlation_tab
    from controllers import recommendations_controller

    def fake_build(*_args, **_kwargs):
        return pd.DataFrame(), False

    monkeypatch.setattr(
        recommendations_controller,
        "build_adaptive_history_view",
        fake_build,
    )

    recommendations = pd.DataFrame([{"symbol": "AAA", "sector": "Tech", "predicted_return_pct": 0.1}])
    opportunities = pd.DataFrame([{"symbol": "AAA", "sector": "Tech"}])

    payload = correlation_tab._compute_adaptive_payload(
        recommendations,
        opportunities,
        profile={"preferred_mode": "low_risk"},
    )

    assert payload is None
