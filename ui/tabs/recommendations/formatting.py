"""Formatting helpers for the recommendations tab."""

from __future__ import annotations

from ui.utils.formatters import (
    format_currency,
    format_currency_delta,
    format_float,
    format_float_delta,
    format_percent,
    format_percent_delta,
)

__all__ = [
    "_format_currency",
    "_format_percent",
    "_format_float",
    "_format_currency_delta",
    "_format_percent_delta",
    "_format_float_delta",
]

_format_currency = format_currency
_format_percent = format_percent
_format_float = format_float
_format_currency_delta = format_currency_delta
_format_percent_delta = format_percent_delta
_format_float_delta = format_float_delta
