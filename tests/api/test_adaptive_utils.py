"""Tests for adaptive forecast validation utilities."""

from __future__ import annotations

import pytest
from fastapi import HTTPException

from api.schemas.adaptive_utils import validate_adaptive_limits
from api.schemas.predictive import AdaptiveHistoryEntry


def _make_entry(symbol: str | None, *, sector: str = "TECH") -> AdaptiveHistoryEntry:
    return AdaptiveHistoryEntry(
        timestamp="2024-01-01T00:00:00Z",
        sector=sector,
        predicted_return=0.1,
        actual_return=0.05,
        symbol=symbol,
    )


def test_validate_adaptive_limits_accepts_unique_symbols() -> None:
    history = [_make_entry("AAA"), _make_entry("BBB"), _make_entry("CCC")]

    assert validate_adaptive_limits(history, max_size=5) is True


def test_validate_adaptive_limits_rejects_duplicate_symbols() -> None:
    history = [_make_entry("AAA"), _make_entry("AAA"), _make_entry("BBB")]

    with pytest.raises(HTTPException) as exc_info:
        validate_adaptive_limits(history, max_size=5)

    assert exc_info.value.detail == "Duplicate symbols detected in adaptive forecast history."


def test_validate_adaptive_limits_rejects_history_overflow() -> None:
    history = [_make_entry("AAA"), _make_entry("BBB")]

    with pytest.raises(HTTPException) as exc_info:
        validate_adaptive_limits(history, max_size=1)

    assert exc_info.value.detail == "History length 2 exceeds limit 1"
