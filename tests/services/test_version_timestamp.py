from __future__ import annotations

import pytest

from services import update_checker


def test_format_last_check_returns_nunca() -> None:
    assert update_checker.format_last_check(None) == "Nunca"


def test_format_last_check_recent_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = 1_000_000.0
    monkeypatch.setattr(update_checker.time, "time", lambda: reference)

    result = update_checker.format_last_check(reference - 120)

    assert result == "hace 2 min"


def test_format_last_check_hours_and_minutes(monkeypatch: pytest.MonkeyPatch) -> None:
    reference = 2_000_000.0
    monkeypatch.setattr(update_checker.time, "time", lambda: reference)

    delta_minutes = (3 * 60) + 17
    timestamp = reference - (delta_minutes * 60)

    result = update_checker.format_last_check(timestamp)

    assert result == "hace 3 h 17 min"
