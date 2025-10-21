"""Tests for :func:`shared.utils.format_percent`."""

from __future__ import annotations

import math

import pytest

from shared.utils import format_percent


@pytest.mark.parametrize(
    ("value", "spaced", "expected"),
    [
        (12.3, False, "12.30%"),
        (12.3, True, "12.30 %"),
        (-7.891, False, "-7.89%"),
        (0, True, "0.00 %"),
        (123456.789, False, "123456.79%"),
    ],
)
def test_format_percent_numeric_values(value: float, spaced: bool, expected: str) -> None:
    assert format_percent(value, spaced=spaced) == expected


@pytest.mark.parametrize("value", [None, math.nan, math.inf, -math.inf])
def test_format_percent_invalid_values(value: float | None) -> None:
    assert format_percent(value) == "—"
