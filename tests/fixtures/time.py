"""Time-related test fixtures and utilities."""

from __future__ import annotations


class FakeTime:
    """Deterministic fake time controller for metric tests."""

    def __init__(self, start: float = 1000.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        """Allow using the instance as a callable time source."""

        return self.time()

    def time(self) -> float:
        return self.now

    def set(self, value: float) -> None:
        """Set the fake clock to an explicit timestamp."""

        self.now = float(value)

    def sleep(self, seconds: float) -> None:
        self.now += float(seconds)

    def advance(self, seconds: float) -> None:  # pragma: no cover - legacy alias
        """Compatibility alias for older tests still using ``advance``."""

        self.sleep(seconds)
