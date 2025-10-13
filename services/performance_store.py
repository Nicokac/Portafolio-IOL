from __future__ import annotations

"""SQLite persistence helpers for performance telemetry."""

import json
import logging
import os
import sqlite3
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING

from shared.settings import app_env

if TYPE_CHECKING:  # pragma: no cover - only used for type checkers
    from services.performance_timer import PerformanceEntry

LOGGER = logging.getLogger(__name__)

_DB_ENV = "PERFORMANCE_DB_PATH"
_DEFAULT_DB_PATH = Path("logs/performance/performance_metrics.db")
_LOCK = RLock()
_SCHEMA_READY = False
_DB_PATH: Path | None = None


def _resolve_db_path() -> Path:
    raw_path = os.getenv(_DB_ENV, str(_DEFAULT_DB_PATH))
    path = Path(raw_path).expanduser()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    except Exception:
        fallback = Path.cwd() / Path(raw_path).name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def _get_db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        _DB_PATH = _resolve_db_path()
    return _DB_PATH


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS performance_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            label TEXT NOT NULL,
            duration_s REAL NOT NULL,
            cpu_pct REAL,
            mem_pct REAL,
            extra_json TEXT
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_label
            ON performance_metrics(label)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_performance_metrics_timestamp
            ON performance_metrics(timestamp)
        """
    )


def _connect() -> sqlite3.Connection:
    path = _get_db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    uri = f"file:{path}" if path.is_absolute() else f"file:{path.resolve()}"
    return sqlite3.connect(uri, uri=True)


def store_entry(entry: "PerformanceEntry") -> None:
    """Persist a telemetry entry into the SQLite store when in production."""

    if app_env != "prod":
        return
    payload = dict(entry.extras)
    payload.setdefault("module", entry.module)
    payload.setdefault("success", entry.success)
    extra_json = json.dumps(payload, ensure_ascii=False)
    with _LOCK:
        conn = _connect()
        try:
            global _SCHEMA_READY
            if not _SCHEMA_READY:
                _ensure_schema(conn)
                _SCHEMA_READY = True
            conn.execute(
                """
                INSERT INTO performance_metrics (
                    timestamp, label, duration_s, cpu_pct, mem_pct, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    entry.timestamp,
                    entry.label,
                    float(entry.duration_s),
                    None if entry.cpu_percent is None else float(entry.cpu_percent),
                    None if entry.ram_percent is None else float(entry.ram_percent),
                    extra_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def get_database_path() -> Path:
    """Expose the resolved SQLite database path."""

    return _get_db_path()


def run_maintenance(
    *,
    vacuum: bool = True,
    retention_days: float | None = None,
    now: float | None = None,
) -> dict[str, float | int] | None:
    """Compact the SQLite store and purge entries older than the retention window."""

    path = _get_db_path()
    if not path.exists():
        LOGGER.debug(
            "Omitiendo mantenimiento de performance_store: la base %s no existe", path
        )
        return None

    reference_time = float(now if now is not None else time.time())
    cutoff_iso: str | None = None
    if retention_days is not None and retention_days > 0:
        cutoff = datetime.fromtimestamp(reference_time, tz=timezone.utc) - timedelta(
            days=float(retention_days)
        )
        cutoff_iso = cutoff.isoformat()

    size_before = float(path.stat().st_size if path.exists() else 0)
    deleted = 0
    vacuum_duration = 0.0

    with sqlite3.connect(path) as conn:
        _ensure_schema(conn)
        if cutoff_iso is not None:
            before_changes = conn.total_changes
            conn.execute(
                "DELETE FROM performance_metrics WHERE timestamp <= ?",
                (cutoff_iso,),
            )
            conn.commit()
            delta = conn.total_changes - before_changes
            if delta > 0:
                deleted = int(delta)
        if vacuum:
            start = time.perf_counter()
            conn.execute("VACUUM")
            vacuum_duration = float(time.perf_counter() - start)

    size_after = float(path.stat().st_size if path.exists() else 0)
    return {
        "deleted": deleted,
        "size_before": size_before,
        "size_after": size_after,
        "vacuum_duration": vacuum_duration,
    }


__all__ = ["store_entry", "get_database_path", "run_maintenance"]
