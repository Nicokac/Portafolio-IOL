"""Notification service integration for section badges."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import requests

from infrastructure.http.session import build_session
from shared.settings import settings

logger = logging.getLogger(__name__)


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

        # Some APIs wrap the actual flags under a ``"flags"`` or ``"data"`` key.
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
            raw_flags.get(
                "technical_signal",
                raw_flags.get("technical", raw_flags.get("tecnico")),
            )
        )
        earnings = _coerce_bool(
            raw_flags.get(
                "upcoming_earnings",
                raw_flags.get("earnings", raw_flags.get("earnings_upcoming")),
            )
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

        Network or parsing errors are swallowed after logging and return empty flags so
        the UI can continue rendering without disruption.
        """

        if not self._base_url:
            return NotificationFlags()

        try:
            response = self._get_session().get(self._base_url, timeout=self._timeout)
        except Exception:
            logger.exception("Error fetching notification flags from %s", self._base_url)
            return NotificationFlags()

        if response.status_code >= 400:
            logger.warning(
                "Notification endpoint responded with status %s", response.status_code
            )
            return NotificationFlags()

        try:
            payload = response.json()
        except ValueError:
            logger.exception("Invalid JSON payload from notification endpoint")
            return NotificationFlags()

        return NotificationFlags.from_payload(payload)


__all__ = [
    "NotificationFlags",
    "NotificationsService",
]
