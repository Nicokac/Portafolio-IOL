"""Utilities to prewarm the portfolio visual cache for frequent datasets."""

from __future__ import annotations

import csv
import json
import logging
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import streamlit as st

from shared.cache import visual_cache_registry
from shared.settings import settings
from shared.telemetry import DEFAULT_TELEMETRY_FILES, log_default_telemetry

logger = logging.getLogger(__name__)

_PREWARM_STATE_KEY = "__visual_cache_prewarmed__"
_PREWARM_DATA_KEY = "__visual_cache_prewarm__"
_PREWARM_TARGETS_KEY = "__visual_cache_prewarm_targets__"
_SESSION_LOCK = threading.Lock()
_COMPONENTS = ("summary", "table", "charts")
_TELEMETRY_PHASE = "portfolio.visual_cache"


def _coerce_positive_int(value: int | None, default: int) -> int:
    try:
        if value is None:
            raise ValueError
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced > 0 else default


def _parse_timestamp(raw: str | None) -> float:
    if not raw:
        return 0.0
    try:
        return datetime.fromisoformat(raw).timestamp()
    except (ValueError, TypeError):
        return 0.0


def _iter_recent_rows(path: Path, *, max_rows: int = 500) -> Iterable[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            window: deque[dict[str, str]] = deque(maxlen=max_rows)
            for row in reader:
                if isinstance(row, dict):
                    window.append(row)
    except FileNotFoundError:
        return ()
    except OSError:
        logger.debug("No se pudo leer el archivo de telemetría %s", path, exc_info=True)
        return ()
    return tuple(window)


def _row_context(row: Mapping[str, str]) -> dict[str, object]:
    raw_context = row.get("context", "")
    if raw_context:
        try:
            parsed = json.loads(raw_context)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            logger.debug(
                "No se pudo decodificar el contexto de telemetría: %s", raw_context
            )
    dataset = row.get("dataset_hash")
    if dataset:
        return {"dataset_hash": dataset}
    return {}


def resolve_top_datasets(
    *,
    max_preload_count: int | None = None,
    telemetry_files: Sequence[Path | str] | None = None,
) -> list[str]:
    """Return the most frequent dataset hashes found in telemetry logs."""

    configured = getattr(settings, "VISUAL_CACHE_MAX_PRELOAD_COUNT", 3)
    limit = _coerce_positive_int(max_preload_count, configured)
    if limit <= 0:
        return []

    files = telemetry_files or DEFAULT_TELEMETRY_FILES
    stats: dict[str, dict[str, float | int]] = {}
    for file_path in files:
        if not file_path:
            continue
        path = Path(file_path)
        for row in _iter_recent_rows(path):
            context = _row_context(row)
            dataset = str(context.get("dataset_hash", "")).strip()
            if not dataset or dataset.lower() == "none":
                continue
            bucket = stats.setdefault(dataset, {"count": 0, "last_ts": 0.0})
            bucket["count"] = int(bucket.get("count", 0)) + 1
            timestamp = _parse_timestamp(row.get("timestamp"))
            if timestamp > float(bucket.get("last_ts", 0.0)):
                bucket["last_ts"] = timestamp

    ordered = sorted(
        stats.items(),
        key=lambda item: (-int(item[1].get("count", 0)), -float(item[1].get("last_ts", 0.0))),
    )
    return [dataset for dataset, _ in ordered[:limit]]


def _ensure_store() -> dict[str, dict[str, object]]:
    with _SESSION_LOCK:
        store = st.session_state.get(_PREWARM_DATA_KEY)
        if not isinstance(store, dict):
            store = {}
            st.session_state[_PREWARM_DATA_KEY] = store
        return store


def _record_dataset(dataset_hash: str) -> None:
    store = _ensure_store()
    with _SESSION_LOCK:
        entry = store.setdefault(dataset_hash, {})
        timestamp = time.time()
        entry["dataset_hash"] = dataset_hash
        entry["prefetched_at"] = timestamp
        components = entry.setdefault("components", {})
        for name in _COMPONENTS:
            component = components.setdefault(name, {})
            component["status"] = "prefetched"
            component["prefetched_at"] = timestamp
        visual_cache_registry.record(
            "prewarm",
            dataset_hash=dataset_hash,
            reused=True,
            signature="prewarm",
        )


def _prewarm_dataset(dataset_hash: str) -> None:
    try:
        _record_dataset(dataset_hash)
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo precalentar el dataset %s", dataset_hash, exc_info=True)


def prewarm_visual_cache(
    *,
    max_preload_count: int | None = None,
    telemetry_files: Sequence[Path | str] | None = None,
    force: bool = False,
) -> list[str]:
    """Warm the visual cache for the most common datasets using background threads."""

    with _SESSION_LOCK:
        if not force and st.session_state.get(_PREWARM_STATE_KEY):
            return []

    datasets = resolve_top_datasets(
        max_preload_count=max_preload_count,
        telemetry_files=telemetry_files,
    )
    if not datasets:
        with _SESSION_LOCK:
            st.session_state[_PREWARM_STATE_KEY] = True
        return []

    with _SESSION_LOCK:
        st.session_state[_PREWARM_TARGETS_KEY] = list(datasets)

    start = time.perf_counter()
    threads: list[threading.Thread] = []
    for dataset_hash in datasets:
        thread = threading.Thread(
            target=_prewarm_dataset,
            args=(dataset_hash,),
            name=f"visual-cache-prewarm-{dataset_hash}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    elapsed_ms = max((time.perf_counter() - start) * 1000.0, 0.0)
    log_default_telemetry(
        phase=_TELEMETRY_PHASE,
        dataset_hash="prewarm",
        elapsed_s=elapsed_ms / 1000.0,
        extra={"visual_cache_prewarm_ms": round(elapsed_ms, 2)},
    )

    logger.info("[Warmup] Visual cache preloaded for top datasets.")

    with _SESSION_LOCK:
        st.session_state[_PREWARM_STATE_KEY] = True

    return list(datasets)


__all__ = [
    "prewarm_visual_cache",
    "resolve_top_datasets",
]

