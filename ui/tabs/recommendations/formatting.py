from __future__ import annotations

import numpy as np


__all__ = [
    "_format_currency",
    "_format_percent",
    "_format_float",
    "_format_currency_delta",
    "_format_percent_delta",
    "_format_float_delta",
]


def _format_currency(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"${value:,.0f}".replace(",", ".")


def _format_percent(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}%"


def _format_float(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}"


def _format_currency_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.0f}".replace(",", ".")


def _format_percent_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def _format_float_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}"
