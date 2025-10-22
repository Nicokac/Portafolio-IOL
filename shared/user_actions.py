"""Asynchronous user action telemetry writer."""

from __future__ import annotations

import atexit
import csv
import json
import logging
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Iterable

try:  # pragma: no cover - optional dependency for certain test environments
    import streamlit as st  # type: ignore
except Exception:  # pragma: no cover - defensive fallback
    st = None  # type: ignore

from infrastructure.iol.auth import get_current_user_id
from shared.telemetry import is_hydration_locked, log_default_telemetry

logger = logging.getLogger(__name__)

_USER_ACTION_COLUMNS = (
    "timestamp",
    "action",
    "detail",
    "dataset_hash",
    "scope",
    "user_id",
    "latency_ms",
)

_LOG_PATH = Path("logs/user_actions.csv")
_QUEUE_MAXSIZE = 4096
_QUEUE_TIMEOUT = 0.25
_FLUSH_INTERVAL = 0.75
_MAX_BATCH_SIZE = 128
_DEDUP_WINDOW_SECONDS = 0.1


@dataclass(frozen=True)
class UserActionEvent:
    """Structured representation of a user interaction."""

    timestamp: float
    action: str
    detail: str
    dataset_hash: str
    scope: str
    user_id: str
    latency_ms: str

    def as_row(self) -> dict[str, str]:
        recorded = datetime.fromtimestamp(self.timestamp, tz=timezone.utc)
        return {
            "timestamp": recorded.isoformat(),
            "action": self.action,
            "detail": self.detail,
            "dataset_hash": self.dataset_hash,
            "scope": self.scope,
            "user_id": self.user_id,
            "latency_ms": self.latency_ms,
        }


_QUEUE: Queue[UserActionEvent | object] = Queue(maxsize=_QUEUE_MAXSIZE)
_WORKER: threading.Thread | None = None
_WORKER_LOCK = threading.Lock()
_STOP_MARKER: object = object()
_LAST_EVENT_TIMES: dict[tuple[str, str, str, str, str], float] = {}
_LAST_EVENT_LOCK = threading.Lock()
_PENDING_ERROR_LOCK = threading.Lock()
_LAST_ERROR_TS: float | None = None


def _ensure_worker() -> None:
    global _WORKER
    if _WORKER and _WORKER.is_alive():
        return
    with _WORKER_LOCK:
        if _WORKER and _WORKER.is_alive():
            return
        _WORKER = threading.Thread(target=_worker_loop, name="user-action-logger", daemon=True)
        _WORKER.start()


def _worker_loop() -> None:
    batch: deque[UserActionEvent] = deque()
    last_flush = time.monotonic()
    while True:
        try:
            item = _QUEUE.get(timeout=_QUEUE_TIMEOUT)
        except Empty:
            item = None
        if item is _STOP_MARKER:
            _QUEUE.task_done()
            if batch:
                _flush_batch(batch)
            break
        if isinstance(item, UserActionEvent):
            batch.append(item)
        if not batch:
            continue
        now = time.monotonic()
        should_flush = (
            len(batch) >= _MAX_BATCH_SIZE
            or (item is None and now - last_flush >= _FLUSH_INTERVAL)
            or (item is not None and now - last_flush >= _FLUSH_INTERVAL)
        )
        if should_flush:
            _flush_batch(batch)
            last_flush = time.monotonic()


def _flush_batch(batch: deque[UserActionEvent]) -> None:
    if not batch:
        return
    events: list[UserActionEvent] = [batch.popleft() for _ in range(len(batch))]
    try:
        _write_rows(events)
    except Exception as exc:  # pragma: no cover - best effort diagnostics
        _report_worker_error(exc)
    finally:
        for _ in events:
            _QUEUE.task_done()


def _write_rows(events: Iterable[UserActionEvent]) -> None:
    path = _LOG_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=_USER_ACTION_COLUMNS)
        if not file_exists:
            writer.writeheader()
        for event in events:
            writer.writerow(event.as_row())


def _report_worker_error(exc: Exception) -> None:
    global _LAST_ERROR_TS
    logger.debug("User action logger failure", exc_info=exc)
    with _PENDING_ERROR_LOCK:
        now = time.monotonic()
        if _LAST_ERROR_TS is not None and now - _LAST_ERROR_TS < 1.0:
            return
        _LAST_ERROR_TS = now
    try:
        log_default_telemetry(
            phase="user_action_logger_error",
            elapsed_s=None,
            extra={"error": repr(exc)},
        )
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("Fallback telemetry for user action logger failed", exc_info=True)


def _coerce_detail(detail: Any) -> str:
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    try:
        return json.dumps(detail, sort_keys=True, default=str)
    except Exception:
        return str(detail)


def _coerce_latency(latency_ms: Any) -> str:
    if latency_ms is None:
        return ""
    try:
        value = float(latency_ms)
    except (TypeError, ValueError):
        return ""
    if value < 0:
        value = 0.0
    return f"{value:.2f}"


def _resolve_scope() -> str:
    module = sys.modules.get("ui.lazy.runtime")
    if module is None:
        return "global"
    current_scope = getattr(module, "current_scope", None)
    if not callable(current_scope):
        return "global"
    try:
        scope = current_scope()
    except Exception:  # pragma: no cover - runtime resilience
        return "global"
    if not scope:
        return "global"
    return str(scope)


def _resolve_user_id() -> str:
    try:
        user_id = get_current_user_id()
    except Exception:  # pragma: no cover - defensive safeguard
        user_id = None
    if not user_id and st is not None:
        state = getattr(st, "session_state", None)
        if isinstance(state, dict):
            user_id = state.get("last_user_id")
        elif state is not None:
            try:
                candidate = state.get("last_user_id")  # type: ignore[attr-defined]
            except Exception:
                candidate = None
            else:
                user_id = candidate
    return str(user_id or "anon")


def _resolve_dataset_hash(value: Any) -> str:
    if isinstance(value, str) and value:
        return value
    if st is not None:
        state = getattr(st, "session_state", None)
        if state is not None:
            try:
                candidate = state.get("dataset_hash")  # type: ignore[attr-defined]
            except Exception:
                candidate = None
            else:
                if isinstance(candidate, str) and candidate:
                    return candidate
    return ""


def _dedupe_signature(action: str, detail: str, dataset_hash: str, scope: str, user_id: str) -> bool:
    signature = (action, detail, dataset_hash, scope, user_id)
    now = time.perf_counter()
    with _LAST_EVENT_LOCK:
        last = _LAST_EVENT_TIMES.get(signature)
        if last is not None and now - last < _DEDUP_WINDOW_SECONDS:
            return False
        _LAST_EVENT_TIMES[signature] = now
        # Opportunistically prune outdated entries to avoid unbounded growth.
        if len(_LAST_EVENT_TIMES) > 1024:
            expired = [key for key, ts in list(_LAST_EVENT_TIMES.items()) if now - ts > 5.0]
            for key in expired:
                _LAST_EVENT_TIMES.pop(key, None)
    return True


def log_user_action(
    action: str,
    detail: Any,
    *,
    dataset_hash: str | None = None,
    latency_ms: Any | None = None,
) -> None:
    """Queue an interaction event for asynchronous persistence."""

    safe_action = str(action or "").strip()
    if not safe_action:
        return
    if is_hydration_locked():
        logger.debug("Hydration locked; skipping user action %s", safe_action)
        return
    safe_detail = _coerce_detail(detail)
    resolved_dataset = _resolve_dataset_hash(dataset_hash)
    scope = _resolve_scope()
    user_id = _resolve_user_id()
    if not _dedupe_signature(safe_action, safe_detail, resolved_dataset, scope, user_id):
        return
    event = UserActionEvent(
        timestamp=time.time(),
        action=safe_action,
        detail=safe_detail,
        dataset_hash=resolved_dataset,
        scope=scope,
        user_id=user_id,
        latency_ms=_coerce_latency(latency_ms),
    )
    _ensure_worker()
    try:
        _QUEUE.put_nowait(event)
    except Exception:  # pragma: no cover - drop on queue saturation
        logger.debug("User action logger queue saturated; dropping event")


def wait_for_flush(timeout: float = 2.0) -> bool:
    """Best-effort helper for tests to await queue draining."""

    deadline = time.monotonic() + max(timeout, 0.0)
    while time.monotonic() < deadline:
        if _QUEUE.unfinished_tasks == 0:  # type: ignore[attr-defined]
            return True
        time.sleep(0.01)
    return _QUEUE.unfinished_tasks == 0  # type: ignore[attr-defined]


def _shutdown_worker() -> None:
    thread = _WORKER
    if not thread:
        return
    if not thread.is_alive():
        return
    try:
        _QUEUE.put_nowait(_STOP_MARKER)
    except Exception:
        try:
            _QUEUE.put(_STOP_MARKER, timeout=0.5)
        except Exception:  # pragma: no cover - defensive
            return
    thread.join(timeout=1.0)


def _reset_for_tests() -> None:
    """Reset internal state between tests."""

    global _QUEUE, _WORKER
    _shutdown_worker()
    with _WORKER_LOCK:
        _QUEUE = Queue(maxsize=_QUEUE_MAXSIZE)
        _WORKER = None
    with _LAST_EVENT_LOCK:
        _LAST_EVENT_TIMES.clear()
    with _PENDING_ERROR_LOCK:
        global _LAST_ERROR_TS
        _LAST_ERROR_TS = None
    _ensure_worker()


atexit.register(_shutdown_worker)

__all__ = ["log_user_action", "wait_for_flush"]
