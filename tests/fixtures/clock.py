"""Shared fake monotonic clock utilities for tests."""

from __future__ import annotations


class FakeClock:
    """Deterministic monotonic callable for TTL and cache expiry tests."""

    def __init__(self, start: float = 0.0) -> None:
        self._t = float(start)

    def __call__(self) -> float:
        """Return the current simulated monotonic time."""
        return self._t

    def advance(self, seconds: float) -> None:
        """Advance the clock by a given number of seconds."""
        self._t += float(seconds)
