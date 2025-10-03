"""Helpers to compute notification badges across the application.

The :func:`build_notification_badges` helper centralises the heuristics
used by controllers to highlight risk, technical and earnings alerts.  It
keeps the UI decoupled from the configuration format by working with simple
Python primitives and returning a dictionary of flags ready for rendering.

The returned payload has the following structure::

    {
        "risk": {
            "active": bool,                # Whether the risk badge should be shown.
            "value": Optional[float],      # Normalised risk metric used for the comparison.
            "threshold": float,            # Threshold applied.
        },
        "technical": {
            "active": bool,
            "direction": Optional[str],    # "bullish" | "bearish" when active.
            "counts": {                    # Aggregated signal counts.
                "bullish": int,
                "bearish": int,
            },
            "threshold": float,
        },
        "earnings": {
            "active": bool,
            "next_event": Optional[dict],  # Shallow copy of the closest event with ``days_until`` normalised.
            "threshold_days": int,
        },
    }

Controllers can rely on the keys above without caring about the shape of the
raw data provided by the analytics layer.  Each badge becomes active when the
normalised value crosses the configured thresholds exposed by
``shared.settings``.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date, datetime
from typing import Any, Iterable, Mapping, MutableMapping

from shared.settings import (
    earnings_upcoming_days as _default_earnings_threshold,
    risk_badge_threshold as _default_risk_threshold,
    technical_signal_threshold as _default_technical_threshold,
)
from shared.time_provider import TimeProvider


def build_notification_badges(
    *,
    risk_metrics: Mapping[str, Any] | float | int | None,
    technical_indicators: Mapping[str, Any] | None,
    earnings_calendar: Iterable[Mapping[str, Any]] | None,
    risk_threshold: float | None = None,
    technical_threshold: float | None = None,
    earnings_days_threshold: int | None = None,
) -> dict[str, Any]:
    """Return badge flags derived from analytics data.

    Parameters
    ----------
    risk_metrics:
        Either a numeric score or a mapping containing keys such as ``"score"``
        or ``"value"``.  The first numeric value found is compared against the
        configured threshold.
    technical_indicators:
        Mapping containing counts or iterables of signals under ``"bullish"``
        and ``"bearish"`` keys (optionally nested under ``"signals"``).  The
        badge becomes active when either side meets the threshold.
    earnings_calendar:
        Iterable of mappings that may include a ``"days_until"`` precomputed
        value or an ``"date"`` entry (``datetime``/``date``/ISO string).  The
        closest upcoming event is compared against the threshold.
    risk_threshold, technical_threshold, earnings_days_threshold:
        Optional overrides, useful for tests.  When omitted the values defined
        in ``config.json``/environment (via :mod:`shared.settings`) are used.
    """

    risk_threshold = float(
        risk_threshold if risk_threshold is not None else _default_risk_threshold
    )
    technical_threshold = float(
        technical_threshold
        if technical_threshold is not None
        else _default_technical_threshold
    )
    earnings_days_threshold = int(
        earnings_days_threshold
        if earnings_days_threshold is not None
        else _default_earnings_threshold
    )

    risk_value = _extract_risk_value(risk_metrics)
    risk_active = bool(
        risk_value is not None and _is_greater_or_equal(risk_value, risk_threshold)
    )

    bullish_count, bearish_count = _extract_signal_counts(technical_indicators)
    technical_active = bool(
        max(bullish_count, bearish_count) >= technical_threshold
    )
    if technical_active:
        if bullish_count >= bearish_count and bullish_count >= technical_threshold:
            technical_direction: str | None = "bullish"
        elif bearish_count >= technical_threshold:
            technical_direction = "bearish"
        else:  # pragma: no cover - defensive guard, should not be hit
            technical_direction = None
            technical_active = False
    else:
        technical_direction = None

    next_event = _find_next_earnings_event(earnings_calendar)
    earnings_active = bool(
        next_event is not None
        and next_event.get("days_until") is not None
        and next_event["days_until"] <= earnings_days_threshold
    )

    return {
        "risk": {
            "active": risk_active,
            "value": risk_value,
            "threshold": risk_threshold,
        },
        "technical": {
            "active": technical_active,
            "direction": technical_direction,
            "counts": {"bullish": bullish_count, "bearish": bearish_count},
            "threshold": technical_threshold,
        },
        "earnings": {
            "active": earnings_active,
            "next_event": next_event,
            "threshold_days": earnings_days_threshold,
        },
    }


def _extract_risk_value(data: Mapping[str, Any] | float | int | None) -> float | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        for key in ("score", "value", "volatility", "risk", "beta"):
            raw = data.get(key)
            numeric = _to_float(raw)
            if numeric is not None:
                return numeric
        return None
    return _to_float(data)


def _extract_signal_counts(data: Mapping[str, Any] | None) -> tuple[int, int]:
    if not isinstance(data, Mapping):
        return 0, 0

    signals: Mapping[str, Any]
    if "signals" in data and isinstance(data["signals"], Mapping):
        signals = data["signals"]
    else:
        signals = data

    bullish = _to_int(signals.get("bullish"))
    bearish = _to_int(signals.get("bearish"))
    return bullish, bearish


def _find_next_earnings_event(
    calendar: Iterable[Mapping[str, Any]] | None,
) -> MutableMapping[str, Any] | None:
    if not calendar:
        return None

    closest: MutableMapping[str, Any] | None = None
    min_days: float | None = None
    for entry in calendar:
        if not isinstance(entry, Mapping):
            continue
        days = _coerce_days_until(entry)
        if days is None:
            continue
        if days < 0:
            continue
        if min_days is None or days < min_days:
            closest = deepcopy(dict(entry))
            closest["days_until"] = int(days)
            min_days = days
    return closest


def _coerce_days_until(entry: Mapping[str, Any]) -> float | None:
    raw = entry.get("days_until")
    numeric = _to_float(raw)
    if numeric is not None:
        return numeric

    raw_date = entry.get("date")
    event_date = _parse_date(raw_date)
    if event_date is None:
        return None
    now = TimeProvider.now_datetime().date()
    delta = event_date - now
    return float(delta.days)


def _parse_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed.date()
    return None


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _is_greater_or_equal(value: float, threshold: float) -> bool:
    try:
        return float(value) >= float(threshold)
    except (TypeError, ValueError):  # pragma: no cover - defensive guard
        return False


__all__ = ["build_notification_badges"]
