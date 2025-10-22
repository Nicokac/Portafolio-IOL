"""Background predictive job management with shared cache metadata."""

from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping

try:  # pragma: no cover - pandas es opcional en runtime mínimos
    import pandas as pd
except Exception:  # pragma: no cover - fallback cuando pandas no está disponible
    pd = None  # type: ignore[assignment]


LOGGER = logging.getLogger(__name__)

_MAX_WORKERS = 2


@dataclass
class _JobRecord:
    job_id: str
    key: str
    ttl_seconds: float
    status: str = "pending"
    created_at: float = field(default_factory=time.time)
    started_at: float | None = None
    finished_at: float | None = None
    duration: float | None = None
    error: str | None = None
    future: Future[Any] | None = None


@dataclass
class _LatestResult:
    value: Any
    metadata: dict[str, Any]
    expires_at: float | None


_LOCK = threading.RLock()
_EXECUTOR: ThreadPoolExecutor | None = None
_JOBS: dict[str, _JobRecord] = {}
_JOBS_BY_KEY: dict[str, str] = {}
_LATEST: dict[str, _LatestResult] = {}


def _get_executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    with _LOCK:
        if _EXECUTOR is None or getattr(_EXECUTOR, "_shutdown", False):  # pragma: no cover - defensive
            _EXECUTOR = ThreadPoolExecutor(max_workers=_MAX_WORKERS, thread_name_prefix="predictive")
    return _EXECUTOR


def _clone_value(value: Any) -> Any:
    if value is None:
        return None
    if pd is not None:
        if isinstance(value, pd.DataFrame):
            return value.copy(deep=True)
        if isinstance(value, pd.Series):
            return value.copy(deep=True)
    return value


def _ensure_latest_locked(key: str, *, status: str, job_id: str | None, submitted_at: float) -> None:
    existing = _LATEST.get(key)
    if existing is None:
        metadata = {
            "job_id": job_id,
            "status": status,
            "result_ready": False,
            "submitted_at": submitted_at,
            "cache_key": key,
        }
        _LATEST[key] = _LatestResult(value=None, metadata=metadata, expires_at=None)
        return

    metadata = dict(existing.metadata)
    metadata["status"] = status
    metadata.setdefault("cache_key", key)
    if job_id:
        metadata["pending_job_id"] = job_id
    metadata["submitted_at"] = submitted_at
    metadata.setdefault("result_ready", bool(existing.metadata.get("result_ready")))
    _LATEST[key] = _LatestResult(value=existing.value, metadata=metadata, expires_at=existing.expires_at)


def _store_latest_locked(key: str, value: Any, metadata: Mapping[str, Any], ttl_seconds: float) -> None:
    expires_at: float | None = None
    ttl = max(float(ttl_seconds or 0.0), 0.0)
    if ttl > 0:
        expires_at = time.time() + ttl
    payload = dict(metadata)
    payload.setdefault("cache_key", key)
    _LATEST[key] = _LatestResult(value=_clone_value(value), metadata=payload, expires_at=expires_at)


def _run_job(
    job_id: str,
    key: str,
    func: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> None:
    record = _JOBS[job_id]
    with _LOCK:
        record.status = "running"
        record.started_at = time.time()
        _ensure_latest_locked(key, status="running", job_id=job_id, submitted_at=record.created_at)

    telemetry = kwargs.get("telemetry")
    if telemetry is not None and not isinstance(telemetry, Mapping):
        telemetry = None

    value: Any = None
    status = "finished"
    error_message: str | None = None

    try:
        value = func(*args, **kwargs)
    except Exception as exc:  # pragma: no cover - fallbacks defensivos
        status = "failed"
        error_message = str(exc)
        LOGGER.exception("Predictive job %s failed", job_id, exc_info=exc)

    finished_at = time.time()
    started_at = record.started_at or finished_at
    duration = max(finished_at - started_at, 0.0)

    metadata: dict[str, Any] = {
        "job_id": job_id,
        "status": status,
        "submitted_at": record.created_at,
        "started_at": record.started_at,
        "completed_at": finished_at,
        "duration": duration,
        "result_ready": status == "finished" and value is not None,
    }

    if telemetry is not None:
        metadata.update(dict(telemetry))

    if status != "finished":
        metadata["result_ready"] = False
        if error_message:
            metadata["error"] = error_message

    with _LOCK:
        record.status = status
        record.finished_at = finished_at
        record.duration = duration
        record.error = error_message
        _store_latest_locked(key, value if status == "finished" else None, metadata, record.ttl_seconds)
        _JOBS_BY_KEY.pop(key, None)


def submit(
    key: str,
    func: Callable[..., Any],
    *args: Any,
    ttl_seconds: float | int | None = None,
    **kwargs: Any,
) -> str:
    """Submit a predictive job to the shared executor.

    Returns the ``job_id`` associated with the computation. When another job with
    the same ``key`` is already running, the current job identifier is returned
    without scheduling a new execution.
    """

    ttl = float(ttl_seconds) if ttl_seconds is not None else 0.0

    with _LOCK:
        existing_id = _JOBS_BY_KEY.get(key)
        if existing_id:
            existing = _JOBS.get(existing_id)
            if existing and existing.status in {"pending", "running"}:
                future = existing.future
                if future is None or not future.done():
                    return existing_id

        job_id = uuid.uuid4().hex
        record = _JobRecord(job_id=job_id, key=key, ttl_seconds=ttl)
        _JOBS[job_id] = record
        _JOBS_BY_KEY[key] = job_id
        _ensure_latest_locked(key, status="pending", job_id=job_id, submitted_at=record.created_at)

    executor = _get_executor()
    future = executor.submit(_run_job, job_id, key, func, args, dict(kwargs))
    with _LOCK:
        record.future = future

    return job_id


def get_latest(key: str) -> tuple[Any, dict[str, Any]] | None:
    """Return the latest cached value and metadata for ``key`` if available."""

    now = time.time()
    with _LOCK:
        entry = _LATEST.get(key)
        if entry is None:
            return None
        metadata = dict(entry.metadata)
        expires_at = entry.expires_at
        if expires_at is not None and expires_at <= now:
            metadata.setdefault("status", "finished")
            metadata["result_ready"] = False
            metadata["expired"] = True
            entry.metadata = metadata
            entry.value = None
            entry.expires_at = None
            return None, dict(metadata)
        return _clone_value(entry.value), metadata


def status(job_id: str | None) -> dict[str, Any]:
    """Expose the current status for a given ``job_id``."""

    if not job_id:
        return {"job_id": None, "status": "idle"}

    with _LOCK:
        record = _JOBS.get(job_id)
        if record is None:
            return {"job_id": job_id, "status": "unknown"}
        snapshot = {
            "job_id": job_id,
            "status": record.status,
            "submitted_at": record.created_at,
            "started_at": record.started_at,
            "completed_at": record.finished_at,
            "duration": record.duration,
            "result_ready": record.status == "finished" and record.finished_at is not None,
            "cache_key": record.key,
            "ttl_seconds": record.ttl_seconds,
        }
        if record.error:
            snapshot["error"] = record.error
        future = record.future
        if record.status in {"pending", "running"} and future is not None and future.done():
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - defensive sync update
                snapshot["status"] = "failed"
                snapshot["error"] = str(exc)
                snapshot["result_ready"] = False
        return snapshot


def reset() -> None:
    """Reset the worker state. Intended for tests."""

    global _EXECUTOR
    with _LOCK:
        for record in _JOBS.values():
            future = record.future
            if future is not None:
                future.cancel()
        _JOBS.clear()
        _JOBS_BY_KEY.clear()
        _LATEST.clear()
        if _EXECUTOR is not None:
            _EXECUTOR.shutdown(wait=True, cancel_futures=True)
            _EXECUTOR = None


__all__ = ["submit", "get_latest", "status", "reset"]
