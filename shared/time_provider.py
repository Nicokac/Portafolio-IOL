"""Centralised time utilities for the application."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo
from typing import ClassVar


class TimeProvider:
    """Provide timezone-aware timestamps formatted consistently."""

    TIMEZONE: ClassVar[ZoneInfo] = ZoneInfo("America/Argentina/Buenos_Aires")
    DATETIME_FORMAT: ClassVar[str] = "%Y-%m-%d %H:%M:%S"

    @classmethod
    def now_datetime(cls) -> datetime:
        """Return the current datetime aware of the configured timezone."""
        return datetime.now(tz=cls.TIMEZONE)

    @classmethod
    def now(cls) -> str:
        """Return the current datetime formatted with the standard format."""
        return cls.now_datetime().strftime(cls.DATETIME_FORMAT)

    @classmethod
    def from_timestamp(cls, ts: float | int | str) -> str:
        """Format a numeric timestamp using the configured timezone and format."""
        moment = datetime.fromtimestamp(float(ts), tz=cls.TIMEZONE)
        return moment.strftime(cls.DATETIME_FORMAT)


__all__ = ["TimeProvider"]
