"""Asynchronous snapshot persistence helpers with batching and compression."""

from __future__ import annotations

import base64
import json
import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from hashlib import sha1
from typing import Any, Callable, Deque, Mapping, MutableMapping, Sequence

try:  # pragma: no cover - optional dependency
    import lz4.frame as _lz4
except Exception:  # pragma: no cover - fallback when lz4 is unavailable
    _lz4 = None

import zlib

logger = logging.getLogger(__name__)


_SNAPSHOT_COMPRESSED_FLAG = "__snapshot_compressed__"
_SNAPSHOT_COMPRESSED_DATA = "data"
_SNAPSHOT_COMPRESSION_CODEC = "codec"
_SNAPSHOT_ORIGINAL_SIZE = "original_size"
_SUPPORTED_CODECS = ("lz4", "zlib")
_DEFAULT_CODEC = "lz4" if _lz4 is not None else "zlib"

_MAX_BATCH_SIZE = 4
_MAX_BATCH_INTERVAL = 1.0  # seconds


PersistCallable = Callable[[str, Mapping[str, Any], Mapping[str, Any] | None], Mapping[str, Any] | None]
ListCallable = Callable[[], Sequence[Mapping[str, Any]]]
TelemetryCallable = Callable[[str, float | None, str | None, Mapping[str, object] | None], None]


@dataclass(slots=True)
class SnapshotResult:
    """Outcome of an asynchronous snapshot persistence request."""

    saved: Mapping[str, Any] | None = None
    history: Sequence[Mapping[str, Any]] | None = None
    skipped: bool = False
    error: Exception | None = None
    payload_hash: str | None = None
    write_ms: float = 0.0


@dataclass(slots=True)
class _SnapshotTask:
    """Internal queue element representing a snapshot persistence request."""

    kind: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] | None
    dataset_hash: str | None
    persist_fn: PersistCallable
    list_fn: ListCallable | None
    on_complete: Callable[[SnapshotResult], None] | None
    telemetry_fn: TelemetryCallable | None
    phase: str = "snapshot.persist_async"
    enqueued_at: float = field(default_factory=time.perf_counter)


_TASK_QUEUE: Deque[_SnapshotTask] = deque()
_QUEUE_COND = threading.Condition()
_WORK_QUEUE: Deque[tuple[list[_SnapshotTask], float]] = deque()
_WORK_QUEUE_COND = threading.Condition()
_WORKERS_STARTED = False
_QUEUE_THREAD: threading.Thread | None = None
_LAST_HASH_BY_KIND: MutableMapping[str, str] = {}
_LAST_HASH_LOCK = threading.Lock()
_current_batch_size = _MAX_BATCH_SIZE
_current_batch_interval = _MAX_BATCH_INTERVAL


def _ensure_workers_started() -> None:
    global _WORKERS_STARTED, _QUEUE_THREAD
    if _WORKERS_STARTED:
        return
    with _WORK_QUEUE_COND:
        if _WORKERS_STARTED:
            return
        _WORKERS_STARTED = True
        for index in range(2):
            worker = threading.Thread(
                target=_worker_loop,
                name=f"snapshot-persister-{index}",
                daemon=True,
            )
            worker.start()
        _QUEUE_THREAD = threading.Thread(
            target=_dispatcher_loop,
            name="snapshot-persister-dispatcher",
            daemon=True,
        )
        _QUEUE_THREAD.start()


def persist_async(
    *,
    kind: str,
    payload: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    persist_fn: PersistCallable,
    list_fn: ListCallable | None = None,
    dataset_hash: str | None = None,
    on_complete: Callable[[SnapshotResult], None] | None = None,
    telemetry_fn: TelemetryCallable | None = None,
    phase: str = "snapshot.persist_async",
    max_batch_size: int | None = None,
    max_batch_interval: float | None = None,
) -> None:
    """Queue a snapshot persistence request to be executed in background."""

    if not callable(persist_fn):
        raise TypeError("persist_fn must be callable")

    task = _SnapshotTask(
        kind=kind,
        payload=dict(payload),
        metadata=dict(metadata) if isinstance(metadata, Mapping) else metadata,
        dataset_hash=dataset_hash,
        persist_fn=persist_fn,
        list_fn=list_fn,
        on_complete=on_complete,
        telemetry_fn=telemetry_fn,
        phase=phase,
    )

    batch_size = max_batch_size or _MAX_BATCH_SIZE
    batch_interval = max_batch_interval or _MAX_BATCH_INTERVAL

    _ensure_workers_started()
    with _QUEUE_COND:
        global _current_batch_size, _current_batch_interval
        _current_batch_size = max(1, int(batch_size))
        _current_batch_interval = max(0.01, float(batch_interval))
        _TASK_QUEUE.append(task)
        _QUEUE_COND.notify()


def compress_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a compressed representation of ``payload`` using zlib or lz4."""

    if not isinstance(payload, Mapping):
        return {}
    if payload.get(_SNAPSHOT_COMPRESSED_FLAG):
        return dict(payload)

    json_bytes = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    codec = _DEFAULT_CODEC
    if codec == "lz4":  # pragma: no branch - evaluated at runtime
        compressed = _lz4.compress(json_bytes) if _lz4 is not None else zlib.compress(json_bytes, level=6)
    else:
        compressed = zlib.compress(json_bytes, level=6)
    encoded = base64.b64encode(compressed).decode("ascii")
    return {
        _SNAPSHOT_COMPRESSED_FLAG: True,
        _SNAPSHOT_COMPRESSION_CODEC: codec,
        _SNAPSHOT_COMPRESSED_DATA: encoded,
        _SNAPSHOT_ORIGINAL_SIZE: len(json_bytes),
    }


def decompress_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    """Expand ``payload`` previously compressed with :func:`compress_payload`."""

    if not isinstance(payload, Mapping):
        return {}
    if not payload.get(_SNAPSHOT_COMPRESSED_FLAG):
        return dict(payload)

    codec = str(payload.get(_SNAPSHOT_COMPRESSION_CODEC) or "zlib").lower()
    data = payload.get(_SNAPSHOT_COMPRESSED_DATA)
    if not isinstance(data, str):
        logger.warning("Invalid compressed payload representation: missing data")
        return {}
    try:
        compressed = base64.b64decode(data.encode("ascii"))
    except (ValueError, TypeError) as exc:  # pragma: no cover - defensive path
        logger.warning("Invalid base64 payload: %s", exc)
        return {}

    try:
        if codec == "lz4" and _lz4 is not None:
            raw = _lz4.decompress(compressed)
        else:
            raw = zlib.decompress(compressed)
    except Exception as exc:  # pragma: no cover - defensive path
        logger.warning("Failed to decompress snapshot payload: %s", exc)
        return {}

    try:
        return json.loads(raw.decode("utf-8"))
    except (
        json.JSONDecodeError,
        UnicodeDecodeError,
    ) as exc:  # pragma: no cover - defensive
        logger.warning("Invalid JSON payload after decompression: %s", exc)
        return {}


def _dispatcher_loop() -> None:
    """Aggregate queued tasks and submit batches to worker threads."""

    while True:
        with _QUEUE_COND:
            while not _TASK_QUEUE:
                _QUEUE_COND.wait()
            batch_size = _current_batch_size
            batch_interval = _current_batch_interval
            deadline = _TASK_QUEUE[0].enqueued_at + batch_interval
            while len(_TASK_QUEUE) < batch_size and time.perf_counter() < deadline:
                remaining = max(deadline - time.perf_counter(), 0.0)
                if remaining <= 0:
                    break
                _QUEUE_COND.wait(timeout=remaining)
            tasks: list[_SnapshotTask] = []
            while _TASK_QUEUE and len(tasks) < batch_size:
                tasks.append(_TASK_QUEUE.popleft())
        started_at = time.perf_counter()
        with _WORK_QUEUE_COND:
            _WORK_QUEUE.append((tasks, started_at))
            _WORK_QUEUE_COND.notify()


def _worker_loop() -> None:
    """Process batches of snapshot persistence requests."""

    while True:
        with _WORK_QUEUE_COND:
            while not _WORK_QUEUE:
                _WORK_QUEUE_COND.wait()
            tasks, started_at = _WORK_QUEUE.popleft()
        if not tasks:
            continue
        try:
            _process_batch(tasks, started_at)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.exception("Unexpected error while processing snapshot batch")
        time.sleep(0.005)  # yield execution to keep the pool low priority


def _process_batch(tasks: Sequence[_SnapshotTask], started_at: float) -> None:
    if not tasks:
        return

    earliest = min(task.enqueued_at for task in tasks)
    batch_wait_ms = max((started_at - earliest) * 1000.0, 0.0)
    for task in tasks:
        payload_hash = _hash_payload(task.payload)
        if _is_duplicate(task.kind, payload_hash):
            result = SnapshotResult(skipped=True, payload_hash=payload_hash)
            _deliver_result(task, result, batch_wait_ms)
            continue

        compressed_payload = compress_payload(task.payload)
        write_start = time.perf_counter()
        saved: Mapping[str, Any] | None = None
        error: Exception | None = None
        try:
            saved = task.persist_fn(task.kind, compressed_payload, task.metadata)
        except Exception as exc:  # pragma: no cover - defensive safeguard
            error = exc
            logger.exception("Failed to persist snapshot asynchronously")
        write_ms = (time.perf_counter() - write_start) * 1000.0

        if error is None and isinstance(saved, Mapping):
            saved = dict(saved)
            saved_payload = saved.get("payload")
            saved["payload"] = decompress_payload(saved_payload)

        history: Sequence[Mapping[str, Any]] | None = None
        if error is None and task.list_fn is not None:
            try:
                raw_history = task.list_fn()
                history_rows: list[Mapping[str, Any]] = []
                for row in raw_history:
                    if isinstance(row, Mapping):
                        normalized = dict(row)
                        normalized["payload"] = decompress_payload(row.get("payload"))
                        history_rows.append(normalized)
                history = history_rows
            except Exception as exc:  # pragma: no cover - defensive safeguard
                logger.debug("Failed to collect snapshot history after write", exc_info=True)
                error = exc

        if error is None:
            _store_hash(task.kind, payload_hash)

        result = SnapshotResult(
            saved=saved,
            history=history,
            skipped=False,
            error=error,
            payload_hash=payload_hash,
            write_ms=write_ms,
        )
        _deliver_result(task, result, batch_wait_ms)


def _deliver_result(task: _SnapshotTask, result: SnapshotResult, batch_wait_ms: float) -> None:
    if task.on_complete is not None:
        try:
            task.on_complete(result)
        except Exception:  # pragma: no cover - defensive safeguard
            logger.exception("Snapshot on_complete callback failed")
    if task.telemetry_fn is None:
        return
    if result.skipped or result.error is not None:
        return
    extra = {
        "snapshot_batch_ms": batch_wait_ms,
        "snapshot_write_ms": result.write_ms,
    }
    elapsed_s = result.write_ms / 1000.0
    try:
        task.telemetry_fn(task.phase, elapsed_s, task.dataset_hash, extra)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("Failed to emit snapshot telemetry", exc_info=True)


def _hash_payload(payload: Mapping[str, Any]) -> str:
    try:
        serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    except (TypeError, ValueError):  # pragma: no cover - fallback to string conversion
        serialized = json.dumps(_stringify_payload(payload), sort_keys=True, separators=(",", ":"))
    return sha1(serialized.encode("utf-8")).hexdigest()


def _stringify_payload(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    return {str(key): _coerce_value(value) for key, value in payload.items()}


def _coerce_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _stringify_payload(value)
    if isinstance(value, (list, tuple)):
        return [_coerce_value(item) for item in value]
    return value


def _is_duplicate(kind: str, payload_hash: str) -> bool:
    with _LAST_HASH_LOCK:
        return _LAST_HASH_BY_KIND.get(kind) == payload_hash


def _store_hash(kind: str, payload_hash: str) -> None:
    with _LAST_HASH_LOCK:
        _LAST_HASH_BY_KIND[kind] = payload_hash


__all__ = [
    "SnapshotResult",
    "compress_payload",
    "decompress_payload",
    "persist_async",
]
