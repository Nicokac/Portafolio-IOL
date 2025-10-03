from __future__ import annotations

import pytest

from services.notifications import build_notification_badges


@pytest.mark.parametrize(
    "risk_value,threshold,expected",
    [
        (0.85, 0.75, True),
        (0.60, 0.75, False),
    ],
)
def test_risk_badge_activation(risk_value: float, threshold: float, expected: bool) -> None:
    badges = build_notification_badges(
        risk_metrics={"score": risk_value},
        technical_indicators=None,
        earnings_calendar=None,
        risk_threshold=threshold,
    )
    assert badges["risk"]["active"] is expected
    assert badges["risk"]["value"] == pytest.approx(risk_value)
    assert badges["risk"]["threshold"] == pytest.approx(threshold)


def test_technical_badge_prefers_bullish_direction() -> None:
    badges = build_notification_badges(
        risk_metrics=None,
        technical_indicators={"bullish": 3, "bearish": 1},
        earnings_calendar=None,
        technical_threshold=2,
    )
    tech = badges["technical"]
    assert tech["active"] is True
    assert tech["direction"] == "bullish"
    assert tech["counts"] == {"bullish": 3, "bearish": 1}


def test_technical_badge_handles_bearish_signals() -> None:
    badges = build_notification_badges(
        risk_metrics=None,
        technical_indicators={"signals": {"bullish": ["rsi"], "bearish": ["macd", "sma"]}},
        earnings_calendar=None,
        technical_threshold=2,
    )
    tech = badges["technical"]
    assert tech["active"] is True
    assert tech["direction"] == "bearish"
    assert tech["counts"] == {"bullish": 1, "bearish": 2}


def test_earnings_badge_activates_for_upcoming_event(monkeypatch: pytest.MonkeyPatch) -> None:
    # Freeze "today" by overriding TimeProvider.now_datetime so the delta is deterministic.
    from shared import time_provider

    fixed_now = time_provider.datetime(2024, 1, 1, tzinfo=time_provider.ZoneInfo(time_provider.TIMEZONE))

    monkeypatch.setattr(time_provider.TimeProvider, "now_datetime", classmethod(lambda cls: fixed_now))

    badges = build_notification_badges(
        risk_metrics=None,
        technical_indicators=None,
        earnings_calendar=[{"symbol": "AAPL", "date": "2024-01-03"}, {"symbol": "MSFT", "days_until": 10}],
        earnings_days_threshold=5,
    )
    earnings = badges["earnings"]
    assert earnings["active"] is True
    assert earnings["next_event"]["symbol"] == "AAPL"
    assert earnings["next_event"]["days_until"] == 2


def test_earnings_badge_ignores_distant_events() -> None:
    badges = build_notification_badges(
        risk_metrics=None,
        technical_indicators=None,
        earnings_calendar=[{"symbol": "AAPL", "days_until": 12}],
        earnings_days_threshold=5,
    )
    earnings = badges["earnings"]
    assert earnings["active"] is False
    assert earnings["next_event"]["symbol"] == "AAPL"
    assert earnings["next_event"]["days_until"] == 12
