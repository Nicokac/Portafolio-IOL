"""Automatic maintenance loop for SQLite-backed stores."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

from shared.settings import (
    enable_prometheus,
    performance_store_ttl_days,
    sqlite_maintenance_interval_hours,
    sqlite_maintenance_size_threshold_mb,
)

LOGGER = logging.getLogger(__name__)
_BYTES_PER_MB = 1024.0 * 1024.0

try:  # pragma: no cover - prometheus_client optional at runtime
    if enable_prometheus:
        from prometheus_client import Counter, Summary  # type: ignore import
    else:  # pragma: no cover - disabled via settings
        Counter = Summary = None  # type: ignore[assignment]
except Exception:  # pragma: no cover - dependency missing or misconfigured
    Counter = Summary = None  # type: ignore[assignment]

_CACHE_CLEANUP_TOTAL = (
    Counter(
        "cache_cleanup_total",
        "Total cache rows deleted during maintenance.",
        labelnames=("database", "reason"),
    )
    if Counter is not None
    else None
)
_VACUUM_DURATION_SECONDS = (
    Summary(
        "vacuum_duration_seconds",
        "Duration of SQLite VACUUM operations.",
        labelnames=("database", "reason"),
    )
    if Summary is not None
    else None
)


@dataclass(frozen=True)
class _MaintenanceTarget:
    name: str
    path_resolver: Callable[[], Path | None]
    maintenance_callable: Callable[[float | None, bool], dict[str, float | int] | None]


def _market_cache_path() -> Path | None:
    from services.cache.market_data_cache import get_sqlite_backend_path

    return get_sqlite_backend_path()


def _market_cache_maintenance(now: float | None, vacuum: bool) -> dict[str, float | int] | None:
    from services.cache.market_data_cache import run_persistent_cache_maintenance

    return run_persistent_cache_maintenance(now=now, vacuum=vacuum)


def _performance_store_path() -> Path | None:
    from services.performance_store import get_database_path

    path = get_database_path()
    return path if path.exists() else None


def _performance_store_maintenance(now: float | None, vacuum: bool) -> dict[str, float | int] | None:
    from services.performance_store import run_maintenance

    retention = performance_store_ttl_days
    if retention <= 0:
        retention = None
    return run_maintenance(retention_days=retention, now=now, vacuum=vacuum)


_TARGETS: tuple[_MaintenanceTarget, ...] = (
    _MaintenanceTarget(
        name="market_data_cache",
        path_resolver=_market_cache_path,
        maintenance_callable=_market_cache_maintenance,
    ),
    _MaintenanceTarget(
        name="performance_store",
        path_resolver=_performance_store_path,
        maintenance_callable=_performance_store_maintenance,
    ),
)


def _safe_file_size(path: Path | None) -> float:
    if path is None:
        return 0.0
    try:
        return float(path.stat().st_size)
    except OSError:
        return 0.0


def _run_targets(
    *, reason: str, now: float | None, vacuum: bool, threshold_bytes: float
) -> list[dict[str, float | int | str]]:
    reports: list[dict[str, float | int | str]] = []
    timestamp = float(now if now is not None else time.time())

    for target in _TARGETS:
        path = target.path_resolver()
        size_before = _safe_file_size(path)
        if path is None:
            LOGGER.debug(
                "sqlite-maintenance skipping database=%s: backend sin ruta disponible",
                target.name,
            )
            continue
        LOGGER.info(
            "sqlite-maintenance start database=%s reason=%s size_before=%.2fMB path=%s",
            target.name,
            reason,
            size_before / _BYTES_PER_MB,
            path,
        )
        if threshold_bytes > 0 and size_before >= threshold_bytes:
            LOGGER.warning(
                "sqlite-maintenance size threshold exceeded database=%s size=%.2fMB threshold=%.2fMB",
                target.name,
                size_before / _BYTES_PER_MB,
                threshold_bytes / _BYTES_PER_MB,
            )
        try:
            stats = target.maintenance_callable(timestamp, vacuum)
        except Exception:  # pragma: no cover - defensive logging
            LOGGER.exception(
                "sqlite-maintenance failed database=%s reason=%s", target.name, reason
            )
            continue
        if not stats:
            LOGGER.debug(
                "sqlite-maintenance no-op database=%s reason=%s", target.name, reason
            )
            continue
        size_after = float(stats.get("size_after", _safe_file_size(path)))
        deleted = int(stats.get("deleted", 0))
        vacuum_duration = float(stats.get("vacuum_duration", 0.0))

        if _CACHE_CLEANUP_TOTAL is not None and deleted:
            _CACHE_CLEANUP_TOTAL.labels(database=target.name, reason=reason).inc(deleted)
        if _VACUUM_DURATION_SECONDS is not None:
            _VACUUM_DURATION_SECONDS.labels(database=target.name, reason=reason).observe(
                vacuum_duration
            )

        LOGGER.info(
            "sqlite-maintenance done database=%s reason=%s deleted=%s size_before=%.2fMB size_after=%.2fMB vacuum=%.4fs path=%s",
            target.name,
            reason,
            deleted,
            size_before / _BYTES_PER_MB,
            size_after / _BYTES_PER_MB,
            vacuum_duration,
            path,
        )
        if threshold_bytes > 0 and size_after >= threshold_bytes:
            LOGGER.warning(
                "sqlite-maintenance size threshold still exceeded database=%s size=%.2fMB threshold=%.2fMB",
                target.name,
                size_after / _BYTES_PER_MB,
                threshold_bytes / _BYTES_PER_MB,
            )

        reports.append(
            {
                "database": target.name,
                "reason": reason,
                "path": str(path),
                "deleted": deleted,
                "size_before": size_before,
                "size_after": size_after,
                "vacuum_duration": vacuum_duration,
            }
        )

    return reports


class _SQLiteMaintenanceScheduler:
    def __init__(self) -> None:
        self.interval_seconds = max(0.0, float(sqlite_maintenance_interval_hours) * 3600.0)
        self.threshold_bytes = max(
            0.0, float(sqlite_maintenance_size_threshold_mb) * _BYTES_PER_MB
        )
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run = 0.0

    def ensure_running(self) -> None:
        if self.interval_seconds <= 0 and self.threshold_bytes <= 0:
            LOGGER.info(
                "sqlite-maintenance disabled: interval=%.2fh threshold=%.2fMB",
                sqlite_maintenance_interval_hours,
                sqlite_maintenance_size_threshold_mb,
            )
            return
        with self._lock:
            if self._thread and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_loop, name="sqlite-maintenance", daemon=True
            )
            self._thread.start()
            LOGGER.info(
                "sqlite-maintenance scheduler started interval=%.2fs threshold=%.2fMB",
                self.interval_seconds,
                self.threshold_bytes / _BYTES_PER_MB,
            )

    def run_once(
        self, reason: str, *, now: float | None = None, vacuum: bool = True
    ) -> list[dict[str, float | int | str]]:
        run_at = float(now if now is not None else time.time())
        reports = _run_targets(
            reason=reason,
            now=run_at,
            vacuum=vacuum,
            threshold_bytes=self.threshold_bytes,
        )
        with self._lock:
            self._last_run = run_at
        return reports

    def _run_loop(self) -> None:
        poll_seconds = self._compute_poll_interval()
        while not self._stop_event.wait(poll_seconds):
            reason = self._determine_reason()
            if reason is None:
                continue
            try:
                self.run_once(reason)
            except Exception:  # pragma: no cover - defensive logging
                LOGGER.exception("sqlite-maintenance loop failed")

    def _determine_reason(self) -> str | None:
        now = time.time()
        if self.threshold_bytes > 0:
            for target in _TARGETS:
                path = target.path_resolver()
                size = _safe_file_size(path)
                if size >= self.threshold_bytes:
                    LOGGER.warning(
                        "sqlite-maintenance threshold trigger database=%s size=%.2fMB threshold=%.2fMB",
                        target.name,
                        size / _BYTES_PER_MB,
                        self.threshold_bytes / _BYTES_PER_MB,
                    )
                    return "size-threshold"
        if self.interval_seconds > 0:
            with self._lock:
                elapsed = now - self._last_run
            if elapsed >= self.interval_seconds:
                return "interval"
        return None

    def _compute_poll_interval(self) -> float:
        if self.interval_seconds > 0:
            return max(60.0, min(self.interval_seconds / 4.0, 900.0))
        return 300.0


_SCHEDULER = _SQLiteMaintenanceScheduler()


def ensure_sqlite_maintenance_started() -> None:
    """Ensure the background scheduler is running."""

    _SCHEDULER.ensure_running()


def run_sqlite_maintenance_now(
    *, reason: str = "manual", now: float | None = None, vacuum: bool = True
) -> list[dict[str, float | int | str]]:
    """Execute the maintenance cycle immediately and return collected reports."""

    return _SCHEDULER.run_once(reason, now=now, vacuum=vacuum)


__all__ = ["ensure_sqlite_maintenance_started", "run_sqlite_maintenance_now"]
