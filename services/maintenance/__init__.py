"""Maintenance helpers for long-running background tasks."""


def ensure_sqlite_maintenance_started() -> bool:
    """Lazy import to prevent circular dependency when starting the scheduler."""

    from .sqlite_maintenance import ensure_sqlite_maintenance_started as _impl

    return _impl()


def run_sqlite_maintenance_now(
    *, reason: str = "manual", now: float | None = None, vacuum: bool = True
):
    """Execute the maintenance cycle immediately using a lazy import."""

    from .sqlite_maintenance import run_sqlite_maintenance_now as _impl

    return _impl(reason=reason, now=now, vacuum=vacuum)


__all__ = [
    "ensure_sqlite_maintenance_started",
    "run_sqlite_maintenance_now",
]
