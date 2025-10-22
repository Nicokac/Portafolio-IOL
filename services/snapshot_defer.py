"""Helpers to defer snapshot persistence until the UI is idle."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Deque, Mapping, MutableSet

from shared import snapshot as snapshot_async

logger = logging.getLogger(__name__)

PersistCallable = snapshot_async.PersistCallable
ListCallable = snapshot_async.ListCallable
TelemetryCallable = snapshot_async.TelemetryCallable


@dataclass(slots=True)
class _DeferredSnapshot:
    kind: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] | None
    dataset_hash: str | None
    persist_fn: PersistCallable
    list_fn: ListCallable | None
    on_complete: Callable[[snapshot_async.SnapshotResult], None] | None
    telemetry_fn: TelemetryCallable | None
    phase: str
    max_batch_size: int | None
    enqueued_at: float


_QUEUE: Deque[_DeferredSnapshot] = deque()
_PENDING_HASHES: MutableSet[str] = set()
_QUEUE_COND = threading.Condition()
_UI_IDLE_EVENT = threading.Event()
_WORKER_THREAD: threading.Thread | None = None
_WORKER_STARTED = False
_FIRST_DEFERRED_AT: float | None = None
_LAST_IDLE_LATENCY_MS: float = 0.0
_IDLE_BATCH_INTERVAL = 5.0
_BACKLOG_WARNING_THRESHOLD = 5


def _ensure_worker_started() -> None:
    global _WORKER_STARTED, _WORKER_THREAD
    if _WORKER_STARTED:
        return
    with _QUEUE_COND:
        if _WORKER_STARTED:
            return
        _WORKER_STARTED = True
        thread = threading.Thread(
            target=_drain_loop,
            name="snapshot-defer-dispatch",
            daemon=True,
        )
        _WORKER_THREAD = thread
        thread.start()


def _drain_loop() -> None:
    while True:
        with _QUEUE_COND:
            while not (_QUEUE and _UI_IDLE_EVENT.is_set()):
                _QUEUE_COND.wait()
            tasks = list(_QUEUE)
            _QUEUE.clear()
            hashes_to_release = [task.dataset_hash for task in tasks if task.dataset_hash]
            for dataset_hash in hashes_to_release:
                _PENDING_HASHES.discard(dataset_hash)
            deferred_count = len(tasks)
            idle_latency_ms = _LAST_IDLE_LATENCY_MS
        for task in tasks:
            _dispatch_task(task, deferred_count=deferred_count, idle_latency_ms=idle_latency_ms)


def _dispatch_task(
    task: _DeferredSnapshot,
    *,
    deferred_count: int,
    idle_latency_ms: float,
) -> None:
    telemetry_fn = task.telemetry_fn
    wrapped_telemetry: TelemetryCallable | None = None
    if telemetry_fn is not None:

        def _wrapped(
            phase: str,
            elapsed_s: float | None,
            dataset_hash: str | None,
            extra: Mapping[str, object] | None,
        ) -> None:
            data = dict(extra or {})
            data["snapshot_deferred_count"] = deferred_count
            data["ui_idle_latency_ms"] = idle_latency_ms
            telemetry_fn(phase, elapsed_s, dataset_hash, data)

        wrapped_telemetry = _wrapped

    snapshot_async.persist_async(
        kind=task.kind,
        payload=task.payload,
        metadata=task.metadata,
        persist_fn=task.persist_fn,
        list_fn=task.list_fn,
        dataset_hash=task.dataset_hash,
        on_complete=task.on_complete,
        telemetry_fn=wrapped_telemetry,
        phase=task.phase,
        max_batch_size=task.max_batch_size,
        max_batch_interval=_IDLE_BATCH_INTERVAL,
    )


def queue_snapshot_persistence(
    *,
    kind: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    persist_fn: PersistCallable,
    list_fn: ListCallable | None = None,
    dataset_hash: str | None = None,
    on_complete: Callable[[snapshot_async.SnapshotResult], None] | None = None,
    telemetry_fn: TelemetryCallable | None = None,
    phase: str = "snapshot.persist_async",
    max_batch_size: int | None = None,
) -> None:
    """Defer snapshot persistence until the UI reports being idle."""

    global _FIRST_DEFERRED_AT

    _ensure_worker_started()
    now = time.perf_counter()
    task = _DeferredSnapshot(
        kind=kind,
        payload=dict(payload),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else metadata,
        dataset_hash=dataset_hash,
        persist_fn=persist_fn,
        list_fn=list_fn,
        on_complete=on_complete,
        telemetry_fn=telemetry_fn,
        phase=phase,
        max_batch_size=max_batch_size,
        enqueued_at=now,
    )

    with _QUEUE_COND:
        if dataset_hash:
            if dataset_hash in _PENDING_HASHES:
                logger.debug(
                    "Skipping deferred snapshot for dataset %s because it is already queued",
                    dataset_hash,
                )
                return
            _PENDING_HASHES.add(dataset_hash)
        if _FIRST_DEFERRED_AT is None:
            _FIRST_DEFERRED_AT = now
        _QUEUE.append(task)
        queue_size = len(_QUEUE)
        if queue_size > _BACKLOG_WARNING_THRESHOLD:
            logger.warning(
                "Snapshot defer queue backlog is high (pending=%s)",
                queue_size,
            )
        _QUEUE_COND.notify_all()


def mark_ui_idle(*, timestamp: float | None = None) -> None:
    """Signal that the UI finished rendering and the queue can be flushed."""

    global _FIRST_DEFERRED_AT, _LAST_IDLE_LATENCY_MS

    ts = timestamp if timestamp is not None else time.perf_counter()
    with _QUEUE_COND:
        if _FIRST_DEFERRED_AT is None:
            _LAST_IDLE_LATENCY_MS = 0.0
        else:
            _LAST_IDLE_LATENCY_MS = max((ts - _FIRST_DEFERRED_AT) * 1000.0, 0.0)
        _FIRST_DEFERRED_AT = None
        _UI_IDLE_EVENT.set()
        _QUEUE_COND.notify_all()


def mark_ui_busy() -> None:
    """Clear the UI idle flag so new requests wait for the next idle window."""

    global _FIRST_DEFERRED_AT, _LAST_IDLE_LATENCY_MS

    with _QUEUE_COND:
        _FIRST_DEFERRED_AT = None
        _LAST_IDLE_LATENCY_MS = 0.0
        _UI_IDLE_EVENT.clear()
        _QUEUE_COND.notify_all()


def reset_for_tests() -> None:
    """Reset internal state for deterministic unit tests."""

    global _FIRST_DEFERRED_AT, _LAST_IDLE_LATENCY_MS

    with _QUEUE_COND:
        _QUEUE.clear()
        _PENDING_HASHES.clear()
        _FIRST_DEFERRED_AT = None
        _LAST_IDLE_LATENCY_MS = 0.0
        _UI_IDLE_EVENT.clear()
        _QUEUE_COND.notify_all()


__all__ = [
    "queue_snapshot_persistence",
    "mark_ui_busy",
    "mark_ui_idle",
    "reset_for_tests",
]
