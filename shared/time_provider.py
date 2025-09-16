from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

TIMEZONE = "America/Argentina/Buenos_Aires"
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"


@dataclass(frozen=True)
class TimeSnapshot:
    """Container for a formatted timestamp and its datetime representation."""

    text: str
    moment: datetime


class TimeProvider:
    """Centralised time provider to generate formatted timestamps."""

    _zone = ZoneInfo(TIMEZONE)

    @classmethod
    def timezone(cls) -> ZoneInfo:
        """Return the zoneinfo instance used for timestamp generation."""
        return cls._zone

    @classmethod
    def now(cls) -> TimeSnapshot:
        """Return the current time in the configured timezone."""
        moment = datetime.now(cls._zone)
        return TimeSnapshot(moment.strftime(TIME_FORMAT), moment)

    @classmethod
    def from_timestamp(cls, ts: Optional[float | int | str]) -> Optional[TimeSnapshot]:
        """Convert a POSIX timestamp into a formatted snapshot.

        Invalid or missing values yield ``None`` to mirror previous behaviour
        in formatting helpers.
        """

        if ts is None or ts == 0:
            return None
        try:
            raw = float(ts)
        except (TypeError, ValueError):
            return None
        try:
            moment = datetime.fromtimestamp(raw, tz=cls._zone)
        except (OverflowError, OSError, ValueError):
            return None
        return TimeSnapshot(moment.strftime(TIME_FORMAT), moment)


__all__ = ["TIMEZONE", "TIME_FORMAT", "TimeProvider", "TimeSnapshot"]
