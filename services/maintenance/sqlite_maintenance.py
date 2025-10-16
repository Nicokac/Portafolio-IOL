"""Automatic maintenance loop for SQLite-backed stores."""

from __future__ import annotations

import csv
import logging
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

LOGGER = logging.getLogger(__name__)
_BYTES_PER_MB = 1024.0 * 1024.0

_ASYNC_COOLDOWN_SECONDS = 6 * 3600.0
_LOG_FILE_PATH = Path("services/maintenance/logs/sqlite_maintenance.log")
_METRICS_FILE_PATH = Path("performance_metrics_13.csv")
_LOG_HANDLER: logging.Handler | None = None


def _ensure_log_handler() -> None:
    """Attach a file handler to LOGGER to persist maintenance runs."""

    global _LOG_HANDLER

    if _LOG_HANDLER is not None:
        return

    _LOG_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.FileHandler(_LOG_FILE_PATH, encoding="utf-8")
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    LOGGER.addHandler(handler)
    _LOG_HANDLER = handler


def _dynamic_threshold(size_mb: float) -> float:
    """Compute an adaptive threshold for maintenance triggers."""

    return max(128.0, size_mb * 0.75)

_CACHE_CLEANUP_TOTAL: Any | None = None
_VACUUM_DURATION_SECONDS: Any | None = None
_METRICS_INITIALIZED = False


def _safe_file_size(path: Path | None) -> float:
    if path is None:
        return 0.0
    try:
        return float(path.stat().st_size)
    except OSError:
        return 0.0


def _release_sqlite_locks(path: Path) -> None:
    """Attempt to release any outstanding SQLite locks on the cache file."""

    try:
        with sqlite3.connect(path, timeout=5.0) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    except Exception:  # pragma: no cover - best effort cleanup
        LOGGER.exception("sqlite-maintenance failed to release WAL locks", exc_info=True)


def _append_metrics(row: dict[str, Any]) -> None:
    """Persist maintenance telemetry to ``performance_metrics_13.csv``."""

    _METRICS_FILE_PATH.parent.mkdir(parents=True, exist_ok=True)
    file_exists = _METRICS_FILE_PATH.exists()
    with _METRICS_FILE_PATH.open("a", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "run_timestamp",
            "db_size_before_mb",
            "db_size_after_mb",
            "vacuum_time_s",
            "freed_space_mb",
            "tables_cleaned",
            "was_async",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def _log_large_free_space(freed_space_mb: float, path: Path) -> None:
    if freed_space_mb >= 50.0:
        LOGGER.info(
            "sqlite-maintenance freed-space path=%s freed=%.2fMB", path, freed_space_mb
        )


def _execute_cache_maintenance(*, force: bool, asynchronous: bool) -> dict[str, Any] | None:
    """Execute adaptive maintenance for the persistent cache backend."""

    _ensure_log_handler()

    cache_path = _market_cache_path()
    if cache_path is None:
        LOGGER.info("sqlite-maintenance cache database unavailable — skipping run")
        return None

    size_before_bytes = _safe_file_size(cache_path)
    size_before_mb = size_before_bytes / _BYTES_PER_MB
    threshold = _dynamic_threshold(size_before_mb)

    if not force and size_before_mb <= threshold:
        LOGGER.info(
            "sqlite-maintenance skip threshold=%.2fMB size=%.2fMB path=%s",
            threshold,
            size_before_mb,
            cache_path,
        )
        return None

    _release_sqlite_locks(cache_path)

    start_ts = datetime.now(timezone.utc)
    now_seconds = time.time()
    deleted_rows = 0
    vacuum_time = 0.0

    stats = _market_cache_maintenance(now_seconds, vacuum=False)
    if not stats:
        LOGGER.info(
            "sqlite-maintenance skip-no-stats path=%s threshold=%.2fMB",
            cache_path,
            threshold,
        )
        return None
    deleted_rows = int(stats.get("deleted", 0))

    tmp_copy = cache_path.with_suffix(".maintenance.tmp")
    if tmp_copy.exists():
        try:
            tmp_copy.unlink()
        except OSError:
            LOGGER.warning("sqlite-maintenance could not remove previous temp file path=%s", tmp_copy)

    try:
        with sqlite3.connect(cache_path, timeout=5.0, isolation_level=None) as conn:
            conn.execute("PRAGMA busy_timeout=5000")
            start_vacuum = time.perf_counter()
            safe_target = tmp_copy.as_posix().replace("'", "''")
            conn.execute(f"VACUUM INTO '{safe_target}'")
            vacuum_time = float(time.perf_counter() - start_vacuum)
    except sqlite3.Error:
        LOGGER.exception("sqlite-maintenance vacuum failed path=%s", cache_path)
        return None
    else:
        try:
            tmp_copy.replace(cache_path)
        except OSError:
            LOGGER.exception("sqlite-maintenance failed replacing vacuum output path=%s", cache_path)
            return None

    size_after_bytes = _safe_file_size(cache_path)
    size_after_mb = size_after_bytes / _BYTES_PER_MB
    freed_space_mb = max(size_before_mb - size_after_mb, 0.0)
    tables_cleaned = 1 if deleted_rows or vacuum_time > 0 else 0

    LOGGER.info(
        "sqlite-maintenance complete async=%s size_before=%.2fMB size_after=%.2fMB freed=%.2fMB vacuum=%.2fs deleted=%s path=%s",
        asynchronous,
        size_before_mb,
        size_after_mb,
        freed_space_mb,
        vacuum_time,
        deleted_rows,
        cache_path,
    )

    _log_large_free_space(freed_space_mb, cache_path)

    if size_before_mb > 512.0 or vacuum_time > 60.0:
        LOGGER.warning("SQLite maintenance threshold exceeded — consider manual cleanup.")

    row = {
        "run_timestamp": start_ts.isoformat(),
        "db_size_before_mb": f"{size_before_mb:.6f}",
        "db_size_after_mb": f"{size_after_mb:.6f}",
        "vacuum_time_s": f"{vacuum_time:.6f}",
        "freed_space_mb": f"{freed_space_mb:.6f}",
        "tables_cleaned": tables_cleaned,
        "was_async": asynchronous,
    }
    _append_metrics(row)

    return row


@dataclass(frozen=True)
class SQLiteMaintenanceConfiguration:
    """Runtime configuration required to schedule and report maintenance."""

    interval_hours: float
    size_threshold_mb: float
    performance_store_ttl_days: float | None
    enable_prometheus: bool


_CONFIGURATION: SQLiteMaintenanceConfiguration | None = None


class _AsyncCacheMaintenance:
    """Background task runner dedicated to market cache maintenance."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._completion: threading.Event | None = None
        self._cooldown_until = 0.0
        self._last_result: dict[str, Any] | None = None

    def schedule(self, *, force: bool) -> bool:
        """Schedule the maintenance run on a background daemon thread."""

        with self._lock:
            now = time.time()
            if self._thread and self._thread.is_alive():
                LOGGER.debug("sqlite-maintenance async run already in progress")
                return False
            if not force and now < self._cooldown_until:
                LOGGER.debug(
                    "sqlite-maintenance async run skipped due to cooldown remaining=%.2fs",
                    self._cooldown_until - now,
                )
                return False
            self._completion = threading.Event()
            self._thread = threading.Thread(
                target=self._run_worker,
                args=(force,),
                name="sqlite-maintenance-async",
                daemon=True,
            )
            self._thread.start()
            return True

    def run_sync(self, *, force: bool) -> dict[str, Any] | None:
        result = _execute_cache_maintenance(force=force, asynchronous=False)
        if result is not None:
            with self._lock:
                self._cooldown_until = time.time() + _ASYNC_COOLDOWN_SECONDS
                self._last_result = result
        return result

    def wait(self, timeout: float | None = None) -> dict[str, Any] | None:
        event: threading.Event | None
        with self._lock:
            event = self._completion
        if event is None:
            return self._last_result
        event.wait(timeout)
        with self._lock:
            return self._last_result

    def is_running(self) -> bool:
        with self._lock:
            return bool(self._thread and self._thread.is_alive())

    def reset_for_tests(self) -> None:
        with self._lock:
            self._cooldown_until = 0.0
            self._completion = None
            self._thread = None
            self._last_result = None

    def _run_worker(self, force: bool) -> None:
        try:
            result = _execute_cache_maintenance(force=force, asynchronous=True)
            if result is not None:
                with self._lock:
                    self._cooldown_until = time.time() + _ASYNC_COOLDOWN_SECONDS
                    self._last_result = result
        finally:
            with self._lock:
                if self._completion is not None:
                    self._completion.set()


@dataclass(frozen=True)
class _MaintenanceTarget:
    name: str
    path_resolver: Callable[[], Path | None]
    maintenance_callable: Callable[[float | None, bool], dict[str, float | int] | None]


def _market_cache_path() -> Path | None:
    try:
        from services.cache.market_data_cache import get_sqlite_backend_path
    except ImportError:
        LOGGER.warning("⚠️ MarketDataCache unavailable — running with fallback cache")
        return None

    return get_sqlite_backend_path()


def _market_cache_maintenance(now: float | None, vacuum: bool) -> dict[str, float | int] | None:
    try:
        from services.cache.market_data_cache import run_persistent_cache_maintenance
    except ImportError:
        LOGGER.warning("⚠️ MarketDataCache unavailable — running with fallback cache")
        return None

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
_ASYNC_CACHE_TASK = _AsyncCacheMaintenance()


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


def run_sqlite_maintenance(*, force: bool = False) -> bool:
    """Schedule asynchronous maintenance for the SQLite cache backend."""

    return _ASYNC_CACHE_TASK.schedule(force=force)


def wait_for_sqlite_maintenance(timeout: float | None = None) -> dict[str, Any] | None:
    """Block until the last asynchronous maintenance run has finished."""

    return _ASYNC_CACHE_TASK.wait(timeout)


def is_sqlite_maintenance_running() -> bool:
    """Return ``True`` when the async maintenance task is still running."""

    return _ASYNC_CACHE_TASK.is_running()


def run_sqlite_maintenance_now(
    *, reason: str = "manual", now: float | None = None, vacuum: bool = True
) -> list[dict[str, float | int | str]]:
    """Execute the maintenance cycle immediately and return collected reports."""

    _ASYNC_CACHE_TASK.run_sync(force=True)
    return _get_scheduler().run_once(reason, now=now, vacuum=vacuum)


def _reset_async_cache_state_for_tests() -> None:
    """Reset async cache state (only used within tests)."""

    _ASYNC_CACHE_TASK.reset_for_tests()


__all__ = [
    "SQLiteMaintenanceConfiguration",
    "configure_sqlite_maintenance",
    "ensure_sqlite_maintenance_started",
    "is_sqlite_maintenance_running",
    "run_sqlite_maintenance",
    "run_sqlite_maintenance_now",
    "wait_for_sqlite_maintenance",
]
