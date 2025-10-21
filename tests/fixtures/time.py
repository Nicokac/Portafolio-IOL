"""Time-related test fixtures and utilities."""

from __future__ import annotations


class FakeTime:
    """Deterministic fake time controller for metric tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)

    def advance(self, seconds: float) -> None:  # pragma: no cover - legacy alias
        """Compatibility alias for older tests still using ``advance``."""

        self.sleep(seconds)
