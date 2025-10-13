"""Automatic maintenance loop for SQLite-backed stores."""

from __future__ import annotations

import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
_BYTES_PER_MB = 1024.0 * 1024.0

_CACHE_CLEANUP_TOTAL: Any | None = None
_VACUUM_DURATION_SECONDS: Any | None = None
_METRICS_INITIALIZED = False


@dataclass(frozen=True)
class SQLiteMaintenanceConfiguration:
    """Runtime configuration required to schedule and report maintenance."""

    interval_hours: float
    size_threshold_mb: float
    performance_store_ttl_days: float | None
    enable_prometheus: bool


_CONFIGURATION: SQLiteMaintenanceConfiguration | None = None


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

    retention = _current_configuration().performance_store_ttl_days
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


def _ensure_metrics_initialized() -> None:
    global _CACHE_CLEANUP_TOTAL, _VACUUM_DURATION_SECONDS, _METRICS_INITIALIZED

    if _METRICS_INITIALIZED:
        return

    _METRICS_INITIALIZED = True
    configuration = _current_configuration()

    try:  # pragma: no cover - prometheus_client optional at runtime
        if not configuration.enable_prometheus:  # pragma: no cover
            _CACHE_CLEANUP_TOTAL = None
            _VACUUM_DURATION_SECONDS = None
            return

        from prometheus_client import Counter, Summary  # type: ignore import

        _CACHE_CLEANUP_TOTAL = Counter(
            "cache_cleanup_total",
            "Total cache rows deleted during maintenance.",
            labelnames=("database", "reason"),
        )
        _VACUUM_DURATION_SECONDS = Summary(
            "vacuum_duration_seconds",
            "Duration of SQLite VACUUM operations.",
            labelnames=("database", "reason"),
        )
    except Exception:  # pragma: no cover - dependency missing or misconfigured
        _CACHE_CLEANUP_TOTAL = None
        _VACUUM_DURATION_SECONDS = None


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

    _ensure_metrics_initialized()
    cache_counter = _CACHE_CLEANUP_TOTAL
    vacuum_summary = _VACUUM_DURATION_SECONDS

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

        if cache_counter is not None and deleted:
            cache_counter.labels(database=target.name, reason=reason).inc(deleted)
        if vacuum_summary is not None:
            vacuum_summary.labels(database=target.name, reason=reason).observe(
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
    def __init__(self, configuration: SQLiteMaintenanceConfiguration) -> None:
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_run = 0.0
        self.interval_seconds = 0.0
        self.threshold_bytes = 0.0
        self.reload_configuration(configuration)

    def ensure_running(self) -> bool:
        if self.interval_seconds <= 0 and self.threshold_bytes <= 0:
            LOGGER.info(
                "sqlite-maintenance disabled: interval=%.2fh threshold=%.2fMB",
                self.interval_seconds / 3600.0,
                self.threshold_bytes / _BYTES_PER_MB,
            )
            return False
        with self._lock:
            if self._thread and self._thread.is_alive():
                return False
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
            return True

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

    def reload_configuration(self, configuration: SQLiteMaintenanceConfiguration) -> None:
        with self._lock:
            self.interval_seconds = max(0.0, float(configuration.interval_hours) * 3600.0)
            self.threshold_bytes = max(0.0, float(configuration.size_threshold_mb) * _BYTES_PER_MB)


_SCHEDULER: _SQLiteMaintenanceScheduler | None = None


def _coerce_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_positive_float_or_none(value: Any) -> float | None:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if numeric > 0 else None


def _coerce_bool(value: Any, default: bool) -> bool:
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
        if normalized in {"", "none", "null"}:
            return default
    if value is None:
        return default
    return bool(value)


def _load_default_configuration() -> SQLiteMaintenanceConfiguration:
    try:  # pragma: no cover - during early startup shared.config may fail
        from shared.config import settings as config_settings
    except Exception:  # pragma: no cover - fallback to environment values
        config_settings = None

    if config_settings is not None:
        interval = _coerce_float(
            getattr(config_settings, "SQLITE_MAINTENANCE_INTERVAL_HOURS", 6.0),
            6.0,
        )
        threshold = _coerce_float(
            getattr(config_settings, "SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB", 256.0),
            256.0,
        )
        raw_retention = getattr(config_settings, "PERFORMANCE_STORE_TTL_DAYS", None)
        retention = _coerce_positive_float_or_none(raw_retention)
        if raw_retention is None:
            fallback = _coerce_positive_float_or_none(
                getattr(config_settings, "LOG_RETENTION_DAYS", 7)
            )
            retention = fallback if fallback is not None else 7.0
        enable_prom = bool(getattr(config_settings, "ENABLE_PROMETHEUS", True))
    else:
        interval = _coerce_float(
            os.getenv("SQLITE_MAINTENANCE_INTERVAL_HOURS"),
            6.0,
        )
        threshold = _coerce_float(
            os.getenv("SQLITE_MAINTENANCE_SIZE_THRESHOLD_MB"),
            256.0,
        )
        retention_candidate = os.getenv("PERFORMANCE_STORE_TTL_DAYS")
        retention = _coerce_positive_float_or_none(retention_candidate)
        if retention_candidate is None:
            fallback_candidate = os.getenv("LOG_RETENTION_DAYS")
            fallback = _coerce_positive_float_or_none(fallback_candidate)
            retention = fallback if fallback is not None else 7.0
        enable_prom = _coerce_bool(os.getenv("ENABLE_PROMETHEUS"), True)

    return SQLiteMaintenanceConfiguration(
        interval_hours=interval,
        size_threshold_mb=threshold,
        performance_store_ttl_days=retention,
        enable_prometheus=enable_prom,
    )


def _current_configuration() -> SQLiteMaintenanceConfiguration:
    global _CONFIGURATION

    if _CONFIGURATION is None:
        _CONFIGURATION = _load_default_configuration()
    return _CONFIGURATION


def configure_sqlite_maintenance(
    configuration: SQLiteMaintenanceConfiguration | None = None,
    **overrides: Any,
) -> None:
    """Update runtime configuration for maintenance without forcing imports."""

    global _CONFIGURATION, _CACHE_CLEANUP_TOTAL, _VACUUM_DURATION_SECONDS, _METRICS_INITIALIZED

    if configuration is not None and overrides:
        msg = "Cannot mix configuration object with keyword overrides"
        raise ValueError(msg)

    if configuration is None:
        if overrides:
            configuration = SQLiteMaintenanceConfiguration(**overrides)
        else:
            configuration = _load_default_configuration()

    _CONFIGURATION = configuration
    _CACHE_CLEANUP_TOTAL = None
    _VACUUM_DURATION_SECONDS = None
    _METRICS_INITIALIZED = False

    if _SCHEDULER is not None:
        _SCHEDULER.reload_configuration(_CONFIGURATION)


def _get_scheduler() -> _SQLiteMaintenanceScheduler:
    global _SCHEDULER

    configuration = _current_configuration()
    if _SCHEDULER is None:
        _SCHEDULER = _SQLiteMaintenanceScheduler(configuration)
    else:
        _SCHEDULER.reload_configuration(configuration)
    return _SCHEDULER


def ensure_sqlite_maintenance_started() -> bool:
    """Ensure the background scheduler is running."""

    return _get_scheduler().ensure_running()


def run_sqlite_maintenance_now(
    *, reason: str = "manual", now: float | None = None, vacuum: bool = True
) -> list[dict[str, float | int | str]]:
    """Execute the maintenance cycle immediately and return collected reports."""

    return _get_scheduler().run_once(reason, now=now, vacuum=vacuum)


__all__ = [
    "SQLiteMaintenanceConfiguration",
    "configure_sqlite_maintenance",
    "ensure_sqlite_maintenance_started",
    "run_sqlite_maintenance_now",
]
