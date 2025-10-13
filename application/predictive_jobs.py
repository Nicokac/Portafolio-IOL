"""Background worker that orchestrates predictive computation jobs."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, MutableMapping, Tuple

__all__ = ["submit", "get_latest", "status", "reset"]


_EXECUTOR: ThreadPoolExecutor | None = None
_EXECUTOR_LOCK = threading.Lock()


def _executor() -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is not None:
        return _EXECUTOR
    with _EXECUTOR_LOCK:
        if _EXECUTOR is None:
            _EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="predictive-worker")
    return _EXECUTOR


@dataclass
class _JobRecord:
    job_id: str
    key: str
    ttl_seconds: float | None
    submitted_at: float = field(default_factory=time.time)
    started_at: float | None = None
    completed_at: float | None = None
    status: str = "pending"
    result: Any = None
    error: BaseException | None = None
    expires_at: float | None = None
    future: Future[Any] | None = None

    def snapshot(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "job_id": self.job_id,
            "key": self.key,
            "status": self.status,
            "submitted_at": self.submitted_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "ttl_seconds": self.ttl_seconds,
            "expires_at": self.expires_at,
        }
        if self.error is not None:
            payload["error"] = repr(self.error)
        ready = False
        if self.result is not None and (self.expires_at is None or self.expires_at > time.time()):
            ready = True
        payload["result_ready"] = ready
        return payload


_LOCK = threading.RLock()
_JOBS: MutableMapping[str, _JobRecord] = {}
_PENDING_BY_KEY: MutableMapping[str, _JobRecord] = {}
_LATEST_BY_KEY: MutableMapping[str, _JobRecord] = {}


def _clone_value(value: Any) -> Any:
    if hasattr(value, "copy"):
        try:
            return value.copy()  # type: ignore[return-value]
        except TypeError:
            return value
    return value


def _register_completion(record: _JobRecord, future: Future[Any]) -> None:
    now = time.time()
    record.completed_at = record.completed_at or now
    record.future = future
    if future.cancelled():
        record.status = "cancelled"
        record.error = RuntimeError("cancelled")
        return
    error = future.exception()
    if error is not None:
        record.status = "failed"
        record.error = error
        return
    try:
        value = future.result()
    except Exception as exc:  # pragma: no cover - defensive
        record.status = "failed"
        record.error = exc
        return
    record.status = "finished"
    cloned = _clone_value(value)
    record.result = cloned
    if record.ttl_seconds is not None:
        record.expires_at = now + float(record.ttl_seconds)
    else:
        record.expires_at = None
    _LATEST_BY_KEY[record.key] = record


def submit(
    key: str,
    func: Callable[..., Any],
    /,
    *args: Any,
    ttl_seconds: float | None = None,
    **kwargs: Any,
) -> str:
    """Schedule a predictive job in the background."""

    if not callable(func):
        raise TypeError("func debe ser invocable")
    normalized_key = str(key)
    with _LOCK:
        existing = _PENDING_BY_KEY.get(normalized_key)
        if existing is not None and existing.future is not None and not existing.future.done():
            return existing.job_id
        job_id = str(uuid.uuid4())
        record = _JobRecord(job_id=job_id, key=normalized_key, ttl_seconds=ttl_seconds)
        _JOBS[job_id] = record
        _PENDING_BY_KEY[normalized_key] = record

    def _wrapper(*wrapper_args: Any, **wrapper_kwargs: Any) -> Any:
        with _LOCK:
            record.started_at = time.time()
            record.status = "running"
        return func(*wrapper_args, **wrapper_kwargs)

    future = _executor().submit(_wrapper, *args, **kwargs)
    record.future = future

    def _callback(fut: Future[Any]) -> None:
        with _LOCK:
            _register_completion(record, fut)
            pending = _PENDING_BY_KEY.get(normalized_key)
            if pending is record:
                _PENDING_BY_KEY.pop(normalized_key, None)

    future.add_done_callback(_callback)
    return job_id


def _resolve_record(key: str) -> _JobRecord | None:
    record = _PENDING_BY_KEY.get(key)
    if record is not None:
        return record
    record = _LATEST_BY_KEY.get(key)
    if record is not None:
        return record
    return None


def get_latest(key: str) -> Tuple[Any, Dict[str, Any]] | None:
    """Return the most recent result for ``key`` and its metadata."""

    normalized_key = str(key)
    with _LOCK:
        record = _resolve_record(normalized_key)
        if record is None:
            return None
        metadata = record.snapshot()
        value = None
        expires_at = record.expires_at
        if record.result is not None and (expires_at is None or expires_at > time.time()):
            value = _clone_value(record.result)
        elif record.key in _LATEST_BY_KEY:
            # Expired result should be removed to avoid stale usage
            if expires_at is not None and expires_at <= time.time():
                _LATEST_BY_KEY.pop(record.key, None)
        return value, metadata


def status(job_id: str) -> Dict[str, Any]:
    """Expose job metadata for external consumers."""

    with _LOCK:
        record = _JOBS.get(str(job_id))
        if record is None:
            return {"job_id": job_id, "status": "unknown"}
        metadata = record.snapshot()
        future = record.future
        if future is not None and future.done():
            if future.cancelled():
                metadata["status"] = "cancelled"
            elif future.exception() is not None:
                metadata["status"] = "failed"
        if metadata.get("result_ready") and record.key in _LATEST_BY_KEY:
            metadata["latest_key"] = record.key
        return metadata


def reset() -> None:
    """Clear bookkeeping without shutting down the executor (test helper)."""

    with _LOCK:
        _JOBS.clear()
        _PENDING_BY_KEY.clear()
        _LATEST_BY_KEY.clear()
