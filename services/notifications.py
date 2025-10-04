"""Notification services and helpers for section badges."""
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Iterable, Mapping, MutableMapping

import requests

from infrastructure.http.session import build_session
from shared.settings import (
    settings,
    earnings_upcoming_days as _default_earnings_threshold,
    risk_badge_threshold as _default_risk_threshold,
    technical_signal_threshold as _default_technical_threshold,
)
from shared.time_provider import TimeProvider
from shared.utils import _as_float_or_none, _to_float

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers para coerción y parsing
# ---------------------------------------------------------------------------

def _coerce_bool(value: Any) -> bool:
    """Best-effort conversion of arbitrary payload values to ``bool``."""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return False
        return text in {"1", "true", "yes", "on", "active", "enabled"}
    if isinstance(value, Mapping):
        for key in ("active", "enabled", "flag", "value", "status"):
            if key in value:
                return _coerce_bool(value[key])
        return any(_coerce_bool(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_coerce_bool(item) for item in value)
    return False


def _to_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, (list, tuple, set)):
        return len(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


# ---------------------------------------------------------------------------
# Flags simples basados en payload remoto
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class NotificationFlags:
    """Boolean flags signalling additional attention required in each section."""

    risk_alert: bool = False
    technical_signal: bool = False
    upcoming_earnings: bool = False

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any] | None) -> "NotificationFlags":
        """Create flags from an API payload."""
        if not isinstance(payload, Mapping):
            return cls()

        raw_flags: Any = payload
        for key in ("flags", "data", "results"):
            if isinstance(payload.get(key), Mapping):
                raw_flags = payload[key]
                break

        if not isinstance(raw_flags, Mapping):
            return cls()

        risk = _coerce_bool(
            raw_flags.get("risk_alert", raw_flags.get("risk", raw_flags.get("riesgo")))
        )
        technical = _coerce_bool(
            raw_flags.get("technical_signal", raw_flags.get("technical", raw_flags.get("tecnico")))
        )
        earnings = _coerce_bool(
            raw_flags.get("upcoming_earnings", raw_flags.get("earnings", raw_flags.get("earnings_upcoming")))
        )

        return cls(
            risk_alert=risk,
            technical_signal=technical,
            upcoming_earnings=earnings,
        )


class NotificationsService:
    """Client to fetch notification flags from the configured endpoint."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float | None = None,
    ) -> None:
        self._base_url = base_url if base_url is not None else getattr(settings, "NOTIFICATIONS_URL", None)
        self._timeout = timeout if timeout is not None else float(getattr(settings, "NOTIFICATIONS_TIMEOUT", 3.0))
        self._session = session

    def _get_session(self) -> requests.Session:
        if self._session is None:
            user_agent = getattr(settings, "USER_AGENT", "Portafolio-IOL/1.0")
            self._session = build_session(user_agent)
        return self._session

    def get_flags(self) -> NotificationFlags:
        """Return the latest notification flags.

        Network or parsing errors are swallowed after logging and return empty flags
        so the UI can continue rendering without disruption.
        """
        if not self._base_url:
            return NotificationFlags()

        try:
            response = self._get_session().get(self._base_url, timeout=self._timeout)
        except Exception:
            logger.exception("Error fetching notification flags from %s", self._base_url)
            return NotificationFlags()

        if response.status_code >= 400:
            logger.warning("Notification endpoint responded with status %s", response.status_code)
            return NotificationFlags()

        try:
            payload = response.json()
        except ValueError:
            logger.exception("Invalid JSON payload from notification endpoint")
            return NotificationFlags()

        return NotificationFlags.from_payload(payload)


# ---------------------------------------------------------------------------
# Builder de badges enriquecidos desde métricas locales
# ---------------------------------------------------------------------------

def build_notification_badges(
    *,
    risk_metrics: Mapping[str, Any] | float | int | None,
    technical_indicators: Mapping[str, Any] | None,
    earnings_calendar: Iterable[Mapping[str, Any]] | None,
    risk_threshold: float | None = None,
    technical_threshold: float | None = None,
    earnings_days_threshold: int | None = None,
) -> dict[str, Any]:
    """Return badge flags derived from analytics data."""

    risk_threshold = float(risk_threshold if risk_threshold is not None else _default_risk_threshold)
    technical_threshold = float(technical_threshold if technical_threshold is not None else _default_technical_threshold)
    earnings_days_threshold = int(earnings_days_threshold if earnings_days_threshold is not None else _default_earnings_threshold)

    risk_value = _extract_risk_value(risk_metrics)
    risk_numeric = _as_float_or_none(risk_value)
    risk_threshold_numeric = _as_float_or_none(risk_threshold)
    risk_active = bool(
        risk_numeric is not None
        and risk_threshold_numeric is not None
        and risk_numeric >= risk_threshold_numeric
    )

    bullish_count, bearish_count = _extract_signal_counts(technical_indicators)
    technical_active = bool(max(bullish_count, bearish_count) >= technical_threshold)
    if technical_active:
        if bullish_count >= bearish_count and bullish_count >= technical_threshold:
            technical_direction: str | None = "bullish"
        elif bearish_count >= technical_threshold:
            technical_direction = "bearish"
        else:
            technical_direction = None
            technical_active = False
    else:
        technical_direction = None

    next_event = _find_next_earnings_event(earnings_calendar)
    earnings_active = bool(
        next_event is not None and next_event.get("days_until") is not None and next_event["days_until"] <= earnings_days_threshold
    )

    return {
        "risk": {"active": risk_active, "value": risk_value, "threshold": risk_threshold},
        "technical": {
            "active": technical_active,
            "direction": technical_direction,
            "counts": {"bullish": bullish_count, "bearish": bearish_count},
            "threshold": technical_threshold,
        },
        "earnings": {"active": earnings_active, "next_event": next_event, "threshold_days": earnings_days_threshold},
    }


def _extract_risk_value(data: Mapping[str, Any] | float | int | None) -> float | None:
    if data is None:
        return None
    if isinstance(data, Mapping):
        for key in ("score", "value", "volatility", "risk", "beta"):
            raw = data.get(key)
            numeric = _to_float(raw, log=False)
            if numeric is not None:
                return numeric
        return None
    return _to_float(data, log=False)


def _extract_signal_counts(data: Mapping[str, Any] | None) -> tuple[int, int]:
    if not isinstance(data, Mapping):
        return 0, 0
    signals: Mapping[str, Any] = data["signals"] if "signals" in data and isinstance(data["signals"], Mapping) else data
    bullish = _to_int(signals.get("bullish"))
    bearish = _to_int(signals.get("bearish"))
    return bullish, bearish


def _find_next_earnings_event(calendar: Iterable[Mapping[str, Any]] | None) -> MutableMapping[str, Any] | None:
    if not calendar:
        return None
    closest: MutableMapping[str, Any] | None = None
    min_days: float | None = None
    for entry in calendar:
        if not isinstance(entry, Mapping):
            continue
        days = _coerce_days_until(entry)
        if days is None or days < 0:
            continue
        if min_days is None or days < min_days:
            closest = deepcopy(dict(entry))
            closest["days_until"] = int(days)
            min_days = days
    return closest


def _coerce_days_until(entry: Mapping[str, Any]) -> float | None:
    raw = entry.get("days_until")
    numeric = _to_float(raw, log=False)
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


__all__ = [
    "NotificationFlags",
    "NotificationsService",
    "build_notification_badges",
]
