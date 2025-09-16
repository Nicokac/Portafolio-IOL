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

    def __post_init__(self) -> None:  # pragma: no cover - defensive normalisation
        text, moment = self.text, self.moment
        if isinstance(text, datetime) and isinstance(moment, str):
            object.__setattr__(self, "text", moment)
            object.__setattr__(self, "moment", text)

    def __str__(self) -> str:
        return self.text


class TimeProvider:
    """Centralised time provider to generate formatted timestamps."""

    _zone = ZoneInfo(TIMEZONE)

    @classmethod
    def timezone(cls) -> ZoneInfo:
        """Return the zoneinfo instance used for timestamp generation."""
        return cls._zone

    @classmethod
    def now_datetime(cls) -> datetime:
        """Return the current datetime in the configured timezone."""
        return datetime.now(cls._zone)

    @classmethod
    def now(cls) -> str:
        """Return the current timestamp as a formatted string.

        Call :meth:`now_datetime` when the :class:`~datetime.datetime` object is required.
        """

        return cls.now_datetime().strftime(TIME_FORMAT)

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
