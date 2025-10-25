from __future__ import annotations

import csv
import hashlib
import json
import logging
import os
import pickle
import threading
import time
from collections import Counter, deque
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Mapping, MutableMapping, Sequence

import numpy as np
import pandas as pd

from application.portfolio_service import (
    PORTFOLIO_TOTALS_VERSION,
    PortfolioTotals,
    calculate_totals,
)
from application.risk_service import (
    annualized_volatility,
    beta,
    compute_returns,
    max_drawdown,
)
from services import health, snapshot_defer
from services import snapshots as snapshot_service
from services.cache import CacheService
from services.cache.market_data_cache import get_market_data_cache
from shared import snapshot as snapshot_async
from shared.telemetry import log_default_telemetry

logger = logging.getLogger(__name__)

_BOPREAL_ARS_SYMBOLS = frozenset({"BPOA7", "BPOB7", "BPOC7", "BPOD7"})

try:  # pragma: no cover - optional dependency for Redis adapter
    import redis  # type: ignore[import-not-found]
except Exception:  # pragma: no cover - redis is optional
    redis = None

_INCREMENTAL_METRICS_PATH = Path("performance_metrics_12.csv")
_INCREMENTAL_METRICS_FIELDS = (
    "dataset_hash",
    "filters_changed",
    "reused_blocks",
    "recomputed_blocks",
    "total_duration_s",
    "memoization_hit_ratio",
)


_DATASET_CACHE_TTL_SECONDS = 300.0
_DATASET_CACHE_MAX_ENTRIES = 8
_DATASET_CACHE_LOCK = threading.Lock()
_DATASET_CACHE_ADAPTER: "PortfolioDatasetCacheAdapter | None" = None


class PortfolioDatasetCacheAdapter:
    """Abstract adapter used to persist dataset-level aggregates."""

    def get(self, key: str) -> Any | None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:  # pragma: no cover
        raise NotImplementedError

    def invalidate(self, key: str) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError

    def clear(self) -> None:  # pragma: no cover - interface definition
        raise NotImplementedError


@dataclass
class PortfolioDatasetCacheEntry:
    """Snapshot persisted in the dataset-level cache."""

    snapshot: "PortfolioViewSnapshot"
    incremental_cache: dict[str, Any]
    fingerprints: dict[str, str]
    stats: dict[str, Any]
    dataset_key: str
    filters_key: str
    created_at: float
    pending_metrics: tuple[str, ...]
    history_records: tuple[dict[str, float], ...]
    timestamp_bucket: str | None = None


class InMemoryDatasetCacheAdapter(PortfolioDatasetCacheAdapter):
    """Simple adapter backed by ``CacheService`` using process memory."""

    def __init__(
        self,
        *,
        ttl_seconds: float | None = None,
        max_entries: int = _DATASET_CACHE_MAX_ENTRIES,
        namespace: str = "portfolio_dataset",
    ) -> None:
        self._cache = CacheService(namespace=namespace)
        self._ttl_seconds = float(ttl_seconds) if ttl_seconds is not None else None
        self._max_entries = max(int(max_entries or 0), 0)
        self._order: Deque[str] = deque()
        self._lock = threading.Lock()

    def get(self, key: str) -> Any | None:
        return self._cache.get(key)

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        ttl_value = self._ttl_seconds if ttl is None else ttl
        with self._lock:
            if key in self._order:
                try:
                    self._order.remove(key)
                except ValueError:  # pragma: no cover - defensive guard
                    pass
            self._order.append(key)
            while self._max_entries and len(self._order) > self._max_entries:
                evicted = self._order.popleft()
                if evicted == key:
                    break
                self._cache.invalidate(evicted)
        self._cache.set(key, value, ttl=ttl_value)

    def invalidate(self, key: str) -> None:
        with self._lock:
            try:
                self._order.remove(key)
            except ValueError:
                pass
        self._cache.invalidate(key)

    def clear(self) -> None:
        with self._lock:
            self._order.clear()
        self._cache.clear()


class RedisDatasetCacheAdapter(PortfolioDatasetCacheAdapter):
    """Redis-backed adapter storing pickled dataset cache entries."""

    def __init__(
        self,
        client: Any,
        *,
        ttl_seconds: float | None = None,
        namespace: str = "portfolio_dataset",
    ) -> None:
        if client is None:  # pragma: no cover - defensive guard
            raise ValueError("Redis client must not be None")
        self._client = client
        self._ttl_seconds = float(ttl_seconds) if ttl_seconds is not None else None
        self._namespace = namespace.strip()

    def _full_key(self, key: str) -> str:
        if not self._namespace:
            return key
        return f"{self._namespace}:{key}"

    def get(self, key: str) -> Any | None:
        try:
            payload = self._client.get(self._full_key(key))
        except Exception:  # pragma: no cover - redis errors should not bubble up
            logger.debug("Redis dataset cache get failed for %s", key, exc_info=True)
            return None
        if payload is None:
            return None
        try:
            return pickle.loads(payload)
        except Exception:  # pragma: no cover - corrupted payloads are ignored
            logger.warning("Invalid dataset cache payload for key %s", key, exc_info=True)
            return None

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> None:
        ttl_value = self._ttl_seconds if ttl is None else ttl
        payload = pickle.dumps(value)
        full_key = self._full_key(key)
        try:
            if ttl_value is not None and float(ttl_value) > 0:
                self._client.setex(full_key, int(float(ttl_value)), payload)
            else:
                self._client.set(full_key, payload)
        except Exception:  # pragma: no cover - redis errors are logged
            logger.debug("Redis dataset cache set failed for %s", key, exc_info=True)

    def invalidate(self, key: str) -> None:
        try:
            self._client.delete(self._full_key(key))
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Redis dataset cache invalidate failed for %s", key, exc_info=True)

    def clear(self) -> None:
        if not self._namespace:
            try:
                self._client.flushdb()
            except Exception:  # pragma: no cover - defensive guard
                logger.debug("Redis dataset cache flush failed", exc_info=True)
            return
        pattern = f"{self._namespace}:*"
        try:
            for key in self._client.scan_iter(pattern):
                self._client.delete(key)
        except Exception:  # pragma: no cover - defensive guard
            logger.debug("Redis dataset cache clear failed", exc_info=True)


def _initialize_dataset_cache_adapter() -> PortfolioDatasetCacheAdapter:
    url = os.getenv("PORTFOLIO_DATASET_CACHE_URL")
    if url and redis is not None:
        try:
            client = redis.Redis.from_url(url)
            logger.info(
                "portfolio_dataset_cache configured with Redis backend",
                extra={"url": url},
            )
            return RedisDatasetCacheAdapter(client, ttl_seconds=_DATASET_CACHE_TTL_SECONDS)
        except Exception:  # pragma: no cover - fall back to in-memory adapter
            logger.warning("Falling back to in-memory dataset cache adapter", exc_info=True)
    return InMemoryDatasetCacheAdapter(
        ttl_seconds=_DATASET_CACHE_TTL_SECONDS,
        max_entries=_DATASET_CACHE_MAX_ENTRIES,
    )


def configure_portfolio_dataset_cache(
    adapter: PortfolioDatasetCacheAdapter | None,
) -> None:
    """Globally override the dataset cache adapter used by the portfolio service."""

    global _DATASET_CACHE_ADAPTER
    with _DATASET_CACHE_LOCK:
        if adapter is None:
            _DATASET_CACHE_ADAPTER = _initialize_dataset_cache_adapter()
        else:
            _DATASET_CACHE_ADAPTER = adapter


def _get_dataset_cache_adapter() -> PortfolioDatasetCacheAdapter:
    global _DATASET_CACHE_ADAPTER
    with _DATASET_CACHE_LOCK:
        if _DATASET_CACHE_ADAPTER is None:
            _DATASET_CACHE_ADAPTER = _initialize_dataset_cache_adapter()
        return _DATASET_CACHE_ADAPTER


def _normalize_timestamp_bucket(value: Any) -> str | None:
    if value is None:
        return None
    try:
        timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    except Exception:
        return None
    if timestamp is None or pd.isna(timestamp):
        return None
    floored = timestamp.floor("T")
    return floored.isoformat()


def _extract_quotes_timestamp(*sources: Any) -> str | None:
    candidates = (
        "quotes_timestamp",
        "quotes_ts",
        "quotes_updated_at",
        "quotes_last_sync",
        "last_quotes_refresh",
        "last_quotes_timestamp",
    )
    for source in sources:
        if source is None:
            continue
        for attr in candidates:
            value = getattr(source, attr, None)
            bucket = _normalize_timestamp_bucket(value)
            if bucket:
                return bucket
    return None


_VOLATILE_CONTROL_KEYS = {
    "refresh_secs",
    "refresh_interval",
    "last_refresh_ts",
    "last_refresh_at",
    "quotes_refresh_ts",
    "quotes_timestamp",
    "quotes_ts",
    "quotes_updated_at",
    "quotes_last_sync",
    "ui_cache_token",
}


def _sanitize_control_attributes(attributes: Mapping[str, Any]) -> dict[str, Any]:
    sanitized: dict[str, Any] = {}
    for key, value in attributes.items():
        normalized_key = key.lower()
        if normalized_key in _VOLATILE_CONTROL_KEYS:
            continue
        if normalized_key.startswith("ui_"):
            continue
        if (
            "timestamp" in normalized_key
            or normalized_key.endswith("_ts")
            or normalized_key.endswith("_at")
            or "last_refresh" in normalized_key
        ):
            bucket = _normalize_timestamp_bucket(value)
            if bucket is None:
                continue
            sanitized[key] = bucket
            continue
        sanitized[key] = value
    return sanitized


def _dataset_cache_key(
    dataset_key: str,
    filters_key: str,
    fingerprints: Mapping[str, str],
    timestamp_bucket: str | None,
) -> str:
    components = [str(dataset_key or "empty"), str(filters_key or "none")]
    components.append(fingerprints.get("filters.time", "none"))
    components.append(fingerprints.get("filters.fx", "none"))
    if timestamp_bucket:
        components.append(str(timestamp_bucket))
    return "|".join(components)


def _log_dataset_cache_event(event: str, dataset_hash: str | None, **extra: Any) -> None:
    payload: dict[str, Any] = {
        "event": str(event),
        "dataset_hash": str(dataset_hash or "unknown"),
    }
    for key, value in extra.items():
        payload[key] = value
    try:
        message = json.dumps(payload, sort_keys=True, default=_coerce_json)
    except TypeError:  # pragma: no cover - defensive fallback
        message = json.dumps({k: str(v) for k, v in payload.items()}, sort_keys=True)
    logger.info("portfolio_dataset_cache %s", message)


def _extract_totals_version(dataset_key: str | None) -> str | None:
    """Return the totals version embedded in ``dataset_key`` if present."""

    if not dataset_key:
        return None
    marker = "|totals_v"
    if marker not in dataset_key:
        return None
    _, _, version = dataset_key.partition(marker)
    version = version.strip()
    return version or None


def _extract_dataset_base(dataset_key: str | None) -> str | None:
    """Return the dataset component without the totals version suffix."""

    if not dataset_key:
        return None
    marker = "|totals_v"
    if marker not in dataset_key:
        return dataset_key
    base, _, _ = dataset_key.partition(marker)
    base = base.strip()
    return base or None


def _extract_quotes_hash(dataset_key: str | None) -> str | None:
    """Return the quotes hash embedded in the dataset identifier if present."""

    base = _extract_dataset_base(dataset_key)
    if not base:
        return None
    marker = "|quotes:"
    if marker not in base:
        if base.startswith("quotes:"):
            _, _, remainder = base.partition("quotes:")
            quotes, _, _ = remainder.partition("|")
            return quotes or None
        return None
    _, _, remainder = base.partition(marker)
    quotes, _, _ = remainder.partition("|")
    quotes = quotes.strip()
    return quotes or None


def _extract_positions_fingerprint(dataset_key: str | None) -> str | None:
    """Return the positions fingerprint without quotes metadata."""

    base = _extract_dataset_base(dataset_key)
    if not base:
        return None
    marker = "|quotes:"
    if marker in base:
        positions, _, _ = base.partition(marker)
        positions = positions.strip()
        return positions or None
    if base.startswith("quotes:"):
        return None
    return base


@dataclass(frozen=True)
class PortfolioCacheMetricsSnapshot:
    """Resumen inmutable del estado del memoizador del portafolio."""

    portfolio_view_render_s: float
    portfolio_cache_hit_ratio: float
    pipeline_cache_hit_ratio: float
    portfolio_cache_miss_count: int
    hits: int
    misses: int
    render_invocations: int
    fingerprint_invalidations: Dict[str, int]
    cache_miss_reasons: Dict[str, int]
    recent_misses: tuple[Dict[str, Any], ...]
    recent_invalidations: tuple[Dict[str, Any], ...]

    def total_invalidations(self) -> int:
        return sum(self.fingerprint_invalidations.values())

    def unnecessary_misses(self) -> int:
        return int(self.cache_miss_reasons.get("unchanged_fingerprint", 0) or 0)


class _PortfolioCacheTelemetry:
    """Recolecta métricas de uso del memoizador del portafolio."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._recent_misses: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._recent_invalidations: Deque[Dict[str, Any]] = deque(maxlen=25)
        self._reset_locked()

    def _reset_locked(self) -> None:
        self._render_total = 0.0
        self._render_invocations = 0
        self._hits = 0
        self._misses = 0
        self._pipeline_hits = 0
        self._pipeline_misses = 0
        self._invalidations: Counter[str] = Counter()
        self._miss_reasons: Counter[str] = Counter()
        self._recent_misses.clear()
        self._recent_invalidations.clear()

    def reset(self) -> None:
        with self._lock:
            self._reset_locked()

    @staticmethod
    def _categorize_reason(dataset_changed: bool, filters_changed: bool) -> str:
        if dataset_changed and filters_changed:
            return "dataset_and_filters"
        if dataset_changed:
            return "dataset_changed"
        if filters_changed:
            return "filters_changed"
        return "unchanged_fingerprint"

    def record_hit(
        self,
        *,
        elapsed_s: float,
        dataset_changed: bool,
        filters_changed: bool,
    ) -> None:
        del dataset_changed, filters_changed
        with self._lock:
            self._render_total += max(float(elapsed_s), 0.0)
            self._render_invocations += 1
            self._hits += 1

    def record_miss(
        self,
        *,
        elapsed_s: float,
        dataset_changed: bool,
        filters_changed: bool,
        apply_elapsed: float,
        totals_elapsed: float,
    ) -> None:
        reason = self._categorize_reason(dataset_changed, filters_changed)
        event = {
            "ts": time.time(),
            "reason": reason,
            "dataset_changed": bool(dataset_changed),
            "filters_changed": bool(filters_changed),
            "apply_elapsed": max(float(apply_elapsed), 0.0),
            "totals_elapsed": max(float(totals_elapsed), 0.0),
            "render_elapsed": max(float(elapsed_s), 0.0),
        }
        with self._lock:
            self._render_total += max(float(elapsed_s), 0.0)
            self._render_invocations += 1
            self._misses += 1
            self._miss_reasons[reason] += 1
            self._recent_misses.append(event)

    def record_invalidation(self, reason: str, *, detail: str | None = None) -> None:
        reason_key = str(reason or "unknown").strip() or "unknown"
        detail_text = None
        if detail is not None:
            detail_text = str(detail).strip()
            if len(detail_text) > 120:
                detail_text = detail_text[:117] + "..."
        event = {"ts": time.time(), "reason": reason_key}
        if detail_text:
            event["detail"] = detail_text
        with self._lock:
            self._invalidations[reason_key] += 1
            self._recent_invalidations.append(event)

    def record_pipeline_event(self, *, hit: bool) -> None:
        with self._lock:
            if hit:
                self._pipeline_hits += 1
            else:
                self._pipeline_misses += 1

    def snapshot(self) -> PortfolioCacheMetricsSnapshot:
        with self._lock:
            total = self._hits + self._misses
            hit_ratio = (self._hits / total) * 100.0 if total else 0.0
            pipeline_total = self._pipeline_hits + self._pipeline_misses
            pipeline_ratio = (self._pipeline_hits / pipeline_total) * 100.0 if pipeline_total else 0.0
            return PortfolioCacheMetricsSnapshot(
                portfolio_view_render_s=self._render_total,
                portfolio_cache_hit_ratio=hit_ratio,
                pipeline_cache_hit_ratio=pipeline_ratio,
                portfolio_cache_miss_count=self._misses,
                hits=self._hits,
                misses=self._misses,
                render_invocations=self._render_invocations,
                fingerprint_invalidations=dict(self._invalidations),
                cache_miss_reasons=dict(self._miss_reasons),
                recent_misses=tuple(dict(item) for item in self._recent_misses),
                recent_invalidations=tuple(dict(item) for item in self._recent_invalidations),
            )


_PORTFOLIO_CACHE_TELEMETRY = _PortfolioCacheTelemetry()


def reset_portfolio_cache_metrics() -> None:
    """Reinicia las métricas recopiladas del memoizador del portafolio."""

    _PORTFOLIO_CACHE_TELEMETRY.reset()
    try:
        adapter = _get_dataset_cache_adapter()
    except Exception:  # pragma: no cover - defensive safeguard
        adapter = None
    if adapter is not None:
        try:
            adapter.clear()
        except Exception:  # pragma: no cover - cache errors should not break flow
            logger.debug(
                "No se pudo limpiar el dataset cache tras reset_portfolio_cache_metrics",
                exc_info=True,
            )


def get_portfolio_cache_metrics_snapshot() -> PortfolioCacheMetricsSnapshot:
    """Devuelve un snapshot del estado actual del memoizador."""

    return _PORTFOLIO_CACHE_TELEMETRY.snapshot()


def _normalize_fingerprint_value(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return _normalize_fingerprint_value(value.tolist())
    if isinstance(value, Mapping):
        items = sorted(value.items(), key=lambda item: str(item[0]))
        return {str(k): _normalize_fingerprint_value(v) for k, v in items}
    if isinstance(value, (set, frozenset)):
        normalized = [_normalize_fingerprint_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return normalized
    if isinstance(value, (list, tuple)):
        normalized = [_normalize_fingerprint_value(v) for v in value]
        try:
            return sorted(normalized)
        except TypeError:
            return normalized
    return value


def _fingerprint_from_payload(payload: Mapping[str, Any]) -> str:
    try:
        normalized = {str(key): _normalize_fingerprint_value(value) for key, value in payload.items()}
        serialized = json.dumps(normalized, sort_keys=True, default=_coerce_json)
    except Exception:
        serialized = json.dumps(
            {str(key): str(value) for key, value in payload.items()},
            sort_keys=True,
        )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def _extract_controls_attributes(controls: Any) -> dict[str, Any]:
    if controls is None:
        return {}
    if hasattr(controls, "__dict__"):
        return {key: value for key, value in vars(controls).items() if not key.startswith("_")}
    result: dict[str, Any] = {}
    for attr in dir(controls):
        if attr.startswith("_"):
            continue
        try:
            value = getattr(controls, attr)
        except Exception:
            continue
        if callable(value):
            continue
        result[attr] = value
    return result


def _build_incremental_fingerprints(dataset_key: str, controls: Any, filters_key: str) -> dict[str, str]:
    raw_attributes = _extract_controls_attributes(controls)
    attributes = _sanitize_control_attributes(raw_attributes)
    fingerprints: dict[str, str] = {
        "dataset": str(dataset_key or "empty"),
        "filters.base": str(filters_key or "none"),
    }

    time_payload = {
        key: attributes[key]
        for key in attributes
        if any(token in key.lower() for token in ("date", "range", "window", "period"))
    }
    fx_payload = {
        key: attributes[key]
        for key in attributes
        if any(token in key.lower() for token in ("fx", "currency", "exchange"))
    }
    misc_payload = {
        key: attributes[key]
        for key in attributes
        if key not in time_payload
        and key not in fx_payload
        and key not in {"selected_syms", "selected_types", "symbol_query"}
    }

    fingerprints["filters.time"] = _fingerprint_from_payload(time_payload) if time_payload else "none"
    fingerprints["filters.fx"] = _fingerprint_from_payload(fx_payload) if fx_payload else "none"
    fingerprints["filters.misc"] = _fingerprint_from_payload(misc_payload) if misc_payload else "none"

    return fingerprints


def _compute_returns_block(df_view: pd.DataFrame) -> pd.DataFrame:
    if df_view is None or df_view.empty:
        return pd.DataFrame(columns=["simbolo", "return_pct"])

    df = df_view.copy(deep=False)
    if "pl_pct" in df.columns:
        returns = pd.DataFrame(
            {
                "simbolo": df.get("simbolo", pd.Series(dtype=str)),
                "return_pct": pd.to_numeric(df["pl_pct"], errors="coerce"),
            }
        )
        return returns.reset_index(drop=True)

    if {"pl", "costo"}.issubset(df.columns):
        costo = pd.to_numeric(df["costo"], errors="coerce")
        pl = pd.to_numeric(df["pl"], errors="coerce")
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(
                np.isfinite(costo) & (np.abs(costo) > 1e-9),
                (pl / costo) * 100.0,
                np.nan,
            )
        returns = pd.DataFrame({"return_pct": ratio})
        if "simbolo" in df.columns:
            returns["simbolo"] = df["simbolo"].values
        columns = [col for col in ("simbolo", "return_pct") if col in returns.columns]
        return returns.loc[:, columns].reset_index(drop=True)

    return pd.DataFrame(columns=["simbolo", "return_pct"])


def _should_reuse_block(
    *,
    dataset_changed: bool,
    previous_snapshot: PortfolioViewSnapshot | None,
    previous_fingerprints: Mapping[str, str],
    current_fingerprints: Mapping[str, str],
    dependencies: Sequence[str],
) -> bool:
    if dataset_changed or previous_snapshot is None:
        return False
    for dep in dependencies:
        if current_fingerprints.get(dep) != previous_fingerprints.get(dep):
            return False
    return True


def compute_incremental_view(
    *,
    dataset_changed: bool,
    fingerprints: Mapping[str, str],
    previous_snapshot: PortfolioViewSnapshot | None,
    previous_blocks: Mapping[str, Any] | None,
    load_positions: Callable[[], tuple[pd.DataFrame, float]],
    compute_totals_fn: Callable[[pd.DataFrame], PortfolioTotals],
    compute_contributions_fn: Callable[[pd.DataFrame], PortfolioContributionMetrics],
    update_history_fn: Callable[[PortfolioTotals], pd.DataFrame],
    build_returns_fn: Callable[[pd.DataFrame], pd.DataFrame] = _compute_returns_block,
    include_extended: bool = True,
    force_totals_recompute: bool = False,
    allow_dataset_reuse: bool = False,
) -> IncrementalComputationResult:
    start = time.perf_counter()
    prev_blocks: MutableMapping[str, Any] = dict(previous_blocks or {})
    prev_fingerprints: Mapping[str, str] = prev_blocks.get("fingerprints", {})
    reused_blocks: set[str] = set()
    recomputed_blocks: set[str] = set()
    previous_pending = tuple(getattr(previous_snapshot, "pending_metrics", ()) or ())

    positions_df: pd.DataFrame | None = None
    apply_elapsed = 0.0
    reuse_guard_changed = dataset_changed and not allow_dataset_reuse
    if _should_reuse_block(
        dataset_changed=reuse_guard_changed,
        previous_snapshot=previous_snapshot,
        previous_fingerprints=prev_fingerprints,
        current_fingerprints=fingerprints,
        dependencies=("dataset", "filters.base", "filters.misc"),
    ):
        cached_positions = prev_blocks.get("positions_df")
        if isinstance(cached_positions, pd.DataFrame):
            positions_df = cached_positions
            reused_blocks.add("positions_df")

    if positions_df is None:
        positions_df, apply_elapsed = load_positions()
        if positions_df is None:
            positions_df = pd.DataFrame()
        recomputed_blocks.add("positions_df")

    if not isinstance(positions_df, pd.DataFrame):
        positions_df = pd.DataFrame(positions_df)

    totals: PortfolioTotals | None = None
    totals_elapsed = 0.0
    if (
        not force_totals_recompute
        and _should_reuse_block(
            dataset_changed=reuse_guard_changed,
            previous_snapshot=previous_snapshot,
            previous_fingerprints=prev_fingerprints,
            current_fingerprints=fingerprints,
            dependencies=(
                "dataset",
                "filters.base",
                "filters.time",
                "filters.fx",
                "filters.misc",
            ),
        )
        and "positions_df" in reused_blocks
    ):
        totals = previous_snapshot.totals
        reused_blocks.add("totals")

    if totals is None:
        totals_start = time.perf_counter()
        totals = compute_totals_fn(positions_df)
        totals_elapsed = time.perf_counter() - totals_start
        recomputed_blocks.add("totals")

    contribution_metrics: PortfolioContributionMetrics | None = None
    history_df: pd.DataFrame | None = None
    returns_df: pd.DataFrame | None = None
    extended_computed = False

    if include_extended:
        if (
            not previous_pending
            and _should_reuse_block(
                dataset_changed=reuse_guard_changed,
                previous_snapshot=previous_snapshot,
                previous_fingerprints=prev_fingerprints,
                current_fingerprints=fingerprints,
                dependencies=("dataset", "filters.time", "filters.fx"),
            )
            and "positions_df" in reused_blocks
        ):
            cached_returns = prev_blocks.get("returns_df")
            if isinstance(cached_returns, pd.DataFrame):
                returns_df = cached_returns
                reused_blocks.add("returns_df")

        if returns_df is None:
            returns_df = build_returns_fn(positions_df)
            if returns_df is None:
                returns_df = pd.DataFrame()
            recomputed_blocks.add("returns_df")

        if (
            not previous_pending
            and _should_reuse_block(
                dataset_changed=reuse_guard_changed,
                previous_snapshot=previous_snapshot,
                previous_fingerprints=prev_fingerprints,
                current_fingerprints=fingerprints,
                dependencies=("dataset", "filters.base", "filters.misc"),
            )
            and "positions_df" in reused_blocks
        ):
            contribution_metrics = previous_snapshot.contribution_metrics
            reused_blocks.add("contribution_metrics")

        if contribution_metrics is None:
            contribution_metrics = compute_contributions_fn(positions_df)
            recomputed_blocks.add("contribution_metrics")

        history_df = update_history_fn(totals)
        extended_computed = True
    else:
        if (
            previous_snapshot is not None
            and not reuse_guard_changed
            and not getattr(previous_snapshot, "pending_metrics", ())
        ):
            contribution_metrics = previous_snapshot.contribution_metrics
            history_df = previous_snapshot.historical_total
            cached_returns = prev_blocks.get("returns_df")
            if isinstance(cached_returns, pd.DataFrame):
                returns_df = cached_returns
        if contribution_metrics is None:
            contribution_metrics = PortfolioContributionMetrics.empty()
        if history_df is None:
            history_df = _empty_history_dataframe()
        if returns_df is None:
            cached_returns = prev_blocks.get("returns_df")
            if isinstance(cached_returns, pd.DataFrame):
                returns_df = cached_returns
        if returns_df is None:
            returns_df = pd.DataFrame()

    duration = time.perf_counter() - start

    return IncrementalComputationResult(
        df_view=positions_df,
        totals=totals,
        contribution_metrics=contribution_metrics,
        historical_total=history_df,
        returns_df=returns_df,
        apply_elapsed=apply_elapsed,
        totals_elapsed=totals_elapsed,
        reused_blocks=tuple(sorted(reused_blocks)),
        recomputed_blocks=tuple(sorted(recomputed_blocks)),
        duration=duration,
        extended_computed=extended_computed,
    )


def _append_incremental_metric(
    *,
    dataset_hash: str,
    filters_changed: bool,
    reused_blocks: Sequence[str],
    recomputed_blocks: Sequence[str],
    total_duration: float,
    memoization_hit_ratio: float,
) -> None:
    payload = {
        "dataset_hash": str(dataset_hash),
        "filters_changed": "true" if filters_changed else "false",
        "reused_blocks": ";".join(sorted(reused_blocks)),
        "recomputed_blocks": ";".join(sorted(recomputed_blocks)),
        "total_duration_s": f"{max(float(total_duration), 0.0):.6f}",
        "memoization_hit_ratio": f"{max(min(memoization_hit_ratio, 1.0), 0.0):.3f}",
    }
    try:
        _INCREMENTAL_METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
        file_exists = _INCREMENTAL_METRICS_PATH.exists()
        with _INCREMENTAL_METRICS_PATH.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=_INCREMENTAL_METRICS_FIELDS)
            if not file_exists:
                writer.writeheader()
            writer.writerow(payload)
    except Exception:  # pragma: no cover - best effort logging
        logger.debug(
            "No se pudo actualizar %s con métricas incrementales",
            _INCREMENTAL_METRICS_PATH,
            exc_info=True,
        )


_SYMBOL_RISK_CACHE: dict[tuple[str, str, str], tuple[dict[str, Any], str, float]] = {}
_SYMBOL_RISK_CACHE_MAX = 512


def _series_signature(series: pd.Series | None) -> str:
    if series is None or series.empty:
        return "empty"
    try:
        hashed = pd.util.hash_pandas_object(series, index=True, categorize=False)
        return hashlib.sha1(hashed.values.tobytes()).hexdigest()
    except Exception:
        try:
            values = np.asarray(series.to_numpy(copy=False))
            return hashlib.sha1(values.tobytes()).hexdigest()
        except Exception:
            tail_value = series.iloc[-1] if len(series) else None
            return f"len:{len(series)}|tail:{tail_value}"


def _prune_symbol_risk_cache() -> None:
    if len(_SYMBOL_RISK_CACHE) <= _SYMBOL_RISK_CACHE_MAX:
        return
    surplus = len(_SYMBOL_RISK_CACHE) - _SYMBOL_RISK_CACHE_MAX
    if surplus <= 0:
        return
    ordered_keys = sorted(_SYMBOL_RISK_CACHE.items(), key=lambda item: item[1][2])
    for key, _ in ordered_keys[:surplus]:
        _SYMBOL_RISK_CACHE.pop(key, None)


def compute_symbol_risk_metrics(
    tasvc,
    symbols: list[str],
    *,
    benchmark: str,
    period: str,
) -> pd.DataFrame:
    """Return risk metrics (volatility, drawdown, beta) for each symbol.

    Parameters
    ----------
    tasvc:
        Service exposing ``portfolio_history``.
    symbols:
        Portfolio symbols to evaluate.
    benchmark:
        Benchmark symbol used to compare beta and relative metrics.
    period:
        Historical period used to compute metrics (e.g. ``"6mo"``).
    """

    if tasvc is None or not symbols:
        return pd.DataFrame()

    request_symbols = list({*symbols, benchmark})
    if not request_symbols:
        return pd.DataFrame()

    cache = get_market_data_cache()
    try:
        prices = cache.get_history(
            request_symbols,
            loader=lambda symbols=request_symbols: tasvc.portfolio_history(simbolos=list(symbols), period=period),
            period=period,
            benchmark=benchmark,
        )
    except Exception:
        logger.exception("Error fetching portfolio history for risk metrics")
        return pd.DataFrame()

    if prices is None or prices.empty:
        return pd.DataFrame()

    returns = compute_returns(prices)
    if returns.empty:
        return pd.DataFrame()

    metrics: list[dict[str, Any]] = []

    bench_returns = returns.get(benchmark)
    if bench_returns is None:
        bench_returns = pd.Series(dtype=float)

    benchmark_key = str(benchmark or "").strip().upper()
    period_key = str(period or "").strip()
    bench_signature = _series_signature(prices.get(benchmark))

    for sym in prices.columns:
        sym_returns = returns.get(sym)
        if sym_returns is None or sym_returns.empty:
            continue

        norm_symbol = str(sym or "").strip().upper()
        cache_key = (norm_symbol, benchmark_key, period_key)
        sym_signature = _series_signature(prices.get(sym))
        combined_signature = f"{sym_signature}|{bench_signature}"
        cached_entry = _SYMBOL_RISK_CACHE.get(cache_key)
        if cached_entry and cached_entry[1] == combined_signature:
            payload = dict(cached_entry[0])
            metrics.append(payload)
            _SYMBOL_RISK_CACHE[cache_key] = (
                dict(payload),
                combined_signature,
                time.time(),
            )
            continue

        vol = annualized_volatility(sym_returns)
        dd = max_drawdown(sym_returns)

        is_benchmark = sym == benchmark
        if is_benchmark:
            sym_beta = 1.0 if len(sym_returns) else float("nan")
        elif bench_returns.empty:
            sym_beta = float("nan")
        else:
            aligned_sym, aligned_bench = sym_returns.align(bench_returns, join="inner")
            sym_beta = beta(aligned_sym, aligned_bench)

        record = {
            "simbolo": sym,
            "volatilidad": vol,
            "drawdown": dd,
            "beta": sym_beta,
            "es_benchmark": is_benchmark,
        }
        metrics.append(record)
        _SYMBOL_RISK_CACHE[cache_key] = (
            dict(record),
            combined_signature,
            time.time(),
        )

    _prune_symbol_risk_cache()

    if not metrics:
        return pd.DataFrame()

    metrics_df = pd.DataFrame(metrics)
    for col in ("volatilidad", "drawdown", "beta"):
        if col in metrics_df.columns:
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce").astype("float32")
    return metrics_df


@dataclass(frozen=True)
class PortfolioContributionMetrics:
    """Aggregate contribution metrics for charts and analytics."""

    by_symbol: pd.DataFrame
    by_type: pd.DataFrame

    @classmethod
    def empty(cls) -> "PortfolioContributionMetrics":
        cols = [
            "tipo",
            "simbolo",
            "valor_actual",
            "costo",
            "pl",
            "pl_d",
            "valor_actual_pct",
            "pl_pct",
        ]
        by_symbol = pd.DataFrame(columns=cols)
        by_type = pd.DataFrame(
            columns=[
                "tipo",
                "valor_actual",
                "costo",
                "pl",
                "pl_d",
                "valor_actual_pct",
                "pl_pct",
            ]
        )
        return cls(by_symbol=by_symbol, by_type=by_type)


def _empty_history_dataframe() -> pd.DataFrame:
    return pd.DataFrame(columns=["timestamp", "total_value", "total_cost", "total_pl"])


@dataclass(frozen=True)
class PortfolioViewSnapshot:
    """Resultado cacheado del portafolio."""

    df_view: pd.DataFrame
    totals: PortfolioTotals
    apply_elapsed: float
    totals_elapsed: float
    generated_at: float
    historical_total: pd.DataFrame
    contribution_metrics: PortfolioContributionMetrics
    storage_id: str | None = None
    pending_metrics: tuple[str, ...] = field(default_factory=tuple)
    dataset_hash: str = ""
    soft_refresh_guard: bool = False
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class IncrementalComputationResult:
    """Resultado de `compute_incremental_view` con telemetría asociada."""

    df_view: pd.DataFrame
    totals: PortfolioTotals
    contribution_metrics: PortfolioContributionMetrics
    historical_total: pd.DataFrame
    returns_df: pd.DataFrame
    apply_elapsed: float
    totals_elapsed: float
    reused_blocks: tuple[str, ...]
    recomputed_blocks: tuple[str, ...]
    duration: float
    extended_computed: bool


class PortfolioViewModelService:
    """Wrapper cacheado alrededor de ``apply_filters``.

    Memoriza el último resultado junto con los totales derivados para evitar
    recomputar mientras no cambien ni las posiciones ni los filtros
    relevantes.
    """

    def __init__(self, *, snapshot_backend: Any | None = None) -> None:
        self._snapshot: PortfolioViewSnapshot | None = None
        self._dataset_key: str | None = None
        self._current_dataset_hash: str | None = None
        self._filters_key: str | None = None
        self._history_records: list[dict[str, float]] = []
        self._incremental_cache: dict[str, Any] = {}
        self._last_incremental_stats: dict[str, Any] | None = None
        self._snapshot_lock = threading.Lock()
        self._snapshot_kind = "portfolio"
        self._dataset_cache_adapter: PortfolioDatasetCacheAdapter | None = _get_dataset_cache_adapter()
        self._soft_refresh_guard_active = False
        if self._dataset_cache_adapter is not None:
            try:
                self._dataset_cache_adapter.clear()
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo limpiar el dataset cache durante la inicialización",
                    exc_info=True,
                )
        self.configure_snapshot_backend(snapshot_backend)

    def configure_snapshot_backend(self, snapshot_backend: Any | None) -> None:
        """Configure the storage backend used to persist portfolio snapshots."""

        if snapshot_backend is None:
            self._snapshot_storage = snapshot_service
        else:
            self._snapshot_storage = snapshot_backend

    def configure_dataset_cache_adapter(self, adapter: PortfolioDatasetCacheAdapter | None) -> None:
        """Override the dataset cache adapter for this service instance."""

        if adapter is None:
            adapter = _get_dataset_cache_adapter()
        self._dataset_cache_adapter = adapter

    def _snapshot_backend_details(self) -> Dict[str, Any]:
        backend = getattr(self, "_snapshot_storage", None)
        details: Dict[str, Any] = {}
        if backend is None:
            return details

        backend_name: str | None = None
        getter = getattr(backend, "current_backend_name", None)
        if callable(getter):
            try:
                backend_name = getter()
            except Exception:
                logger.debug(
                    "No se pudo determinar el backend activo de snapshots",
                    exc_info=True,
                )
        if isinstance(backend_name, str):
            backend_name = backend_name.strip() or None
        elif backend_name is not None:
            backend_name = str(backend_name)

        if not backend_name:
            raw_name = getattr(backend, "backend_name", None)
            if isinstance(raw_name, str):
                backend_name = raw_name.strip() or None
            elif raw_name is not None:
                backend_name = str(raw_name)

        if not backend_name:
            module_name = getattr(backend, "__name__", None)
            if isinstance(module_name, str) and module_name.strip():
                backend_name = module_name.strip()

        if not backend_name:
            backend_name = backend.__class__.__name__

        if backend_name:
            details["name"] = str(backend_name)

        for attr in ("path", "storage_path", "location"):
            value = getattr(backend, attr, None)
            if value:
                details[attr] = str(value)

        return details

    def _record_snapshot_event(
        self,
        *,
        action: str,
        status: str,
        storage_id: str | None = None,
        detail: str | None = None,
    ) -> None:
        try:
            health.record_snapshot_event(
                kind=self._snapshot_kind,
                action=action,
                status=status,
                storage_id=storage_id,
                detail=detail,
                backend=self._snapshot_backend_details(),
            )
        except Exception:
            logger.debug("No se pudo registrar el evento de snapshot", exc_info=True)

    def _should_invalidate_cache(self, dataset_hash: str | None, skip_invalidation: bool) -> bool:
        """Return whether cache invalidation should run for this cycle."""

        skip_flag = bool(skip_invalidation)
        current_hash = self._current_dataset_hash or self._dataset_key
        candidate_hash: str | None = str(dataset_hash) if dataset_hash else None
        same_dataset = bool(candidate_hash and current_hash and candidate_hash == current_hash)

        if skip_flag or same_dataset:
            self._soft_refresh_guard_active = True
            logger.info("[PortfolioView] Skipped early invalidate (dataset stable)")
            logger.info(
                "portfolio_view.skip_invalidation_guarded "
                'event="skip_invalidation_guarded" dataset=%s '
                "skip_invalidation=%s current=%s",
                candidate_hash,
                skip_flag,
                current_hash,
            )
            _log_dataset_cache_event(
                "skip_invalidation_guarded",
                candidate_hash,
                skip_invalidation=skip_flag,
                previous_dataset=current_hash,
            )
            try:
                log_default_telemetry(
                    phase="portfolio_view.skip_invalidation_guarded",
                    elapsed_s=0.0,
                    dataset_hash=candidate_hash,
                    extra={
                        "skip_invalidation": skip_flag,
                        "previous_dataset": current_hash,
                    },
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar telemetría para portfolio_view.skip_invalidation_guarded",
                    exc_info=True,
                )
            return False

        self._soft_refresh_guard_active = False
        return True

    def _update_history(self, totals: PortfolioTotals) -> pd.DataFrame:
        entry = _history_row(time.time(), totals)
        self._history_records = _append_history(self._history_records, entry, maxlen=500)
        return _normalize_history_df(self._history_records)

    @staticmethod
    def _hash_dataset(df: pd.DataFrame | None) -> str:
        if df is None or df.empty:
            return "empty"
        try:
            hashed = pd.util.hash_pandas_object(df, index=True, categorize=True)
            return hashlib.sha1(hashed.values.tobytes()).hexdigest()
        except TypeError:
            payload = json.dumps(df.to_dict(orient="list"), sort_keys=True, default=_coerce_json).encode("utf-8")
            return hashlib.sha1(payload).hexdigest()

    @staticmethod
    def _filters_key_from(controls: Any) -> str:
        payload = {
            "selected_syms": sorted(map(str, getattr(controls, "selected_syms", []))),
            "selected_types": sorted(map(str, getattr(controls, "selected_types", []))),
            "symbol_query": (getattr(controls, "symbol_query", "") or "").strip(),
        }
        return json.dumps(payload, sort_keys=True)

    def _persist_snapshot_sync(
        self,
        *,
        df_view: pd.DataFrame,
        totals: PortfolioTotals,
        controls: Any,
        dataset_key: str,
        filters_key: str,
        generated_at: float,
        contribution_metrics: PortfolioContributionMetrics,
        historical_total: pd.DataFrame,
    ) -> tuple[str | None, pd.DataFrame | None]:
        backend = getattr(self._snapshot_storage, "save_snapshot", None)
        list_fn = getattr(self._snapshot_storage, "list_snapshots", None)
        if not callable(backend):
            return None, None

        payload, metadata = self._snapshot_payload_and_metadata(
            df_view=df_view,
            totals=totals,
            controls=controls,
            dataset_key=dataset_key,
            filters_key=filters_key,
            generated_at=generated_at,
            contribution_metrics=contribution_metrics,
            historical_total=historical_total,
        )

        try:
            saved = backend(self._snapshot_kind, payload, metadata)
        except Exception as exc:
            logger.exception("No se pudo persistir el snapshot del portafolio")
            self._record_snapshot_event(
                action="save",
                status="error",
                detail=str(exc),
            )
            return None, None

        storage_id = saved.get("id") if isinstance(saved, Mapping) else None
        self._record_snapshot_event(
            action="save",
            status="saved",
            storage_id=str(storage_id) if storage_id else None,
        )
        persisted_history: pd.DataFrame | None = None
        if callable(list_fn):
            try:
                records = list_fn(self._snapshot_kind, limit=500, order="asc")
                persisted_history = _history_df_from_snapshot_records(records)
                if isinstance(persisted_history, pd.DataFrame) and persisted_history.empty:
                    persisted_history = None
            except Exception:
                logger.exception("No se pudo construir la historia persistida del portafolio")

        return storage_id, persisted_history

    def _snapshot_payload_and_metadata(
        self,
        *,
        df_view: pd.DataFrame,
        totals: PortfolioTotals,
        controls: Any,
        dataset_key: str,
        filters_key: str,
        generated_at: float,
        contribution_metrics: PortfolioContributionMetrics,
        historical_total: pd.DataFrame,
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        payload = _serialize_snapshot_payload(
            df_view=df_view,
            totals=totals,
            generated_at=generated_at,
            history=historical_total,
            contribution_metrics=contribution_metrics,
        )
        metadata = {
            "dataset_key": dataset_key,
            "filters_key": filters_key,
            "controls": _safe_asdict(controls),
            "totals_version": f"v{PORTFOLIO_TOTALS_VERSION}",
        }
        return payload, metadata

    def _schedule_snapshot_persistence(
        self,
        *,
        snapshot: PortfolioViewSnapshot,
        controls: Any,
        dataset_key: str,
        filters_key: str,
    ) -> None:
        backend = getattr(self._snapshot_storage, "save_snapshot", None)
        if not callable(backend):
            return

        list_fn = getattr(self._snapshot_storage, "list_snapshots", None)
        history_fetcher: Callable[[], Sequence[Mapping[str, Any]]] | None = None
        if callable(list_fn):

            def _fetch_history() -> Sequence[Mapping[str, Any]]:
                return list_fn(self._snapshot_kind, limit=500, order="asc")

            history_fetcher = _fetch_history

        payload, metadata = self._snapshot_payload_and_metadata(
            df_view=snapshot.df_view,
            totals=snapshot.totals,
            controls=controls,
            dataset_key=dataset_key,
            filters_key=filters_key,
            generated_at=snapshot.generated_at,
            contribution_metrics=snapshot.contribution_metrics,
            historical_total=snapshot.historical_total,
        )

        def _telemetry_hook(
            phase: str,
            elapsed_s: float | None,
            dataset_hash: str | None,
            extra: Mapping[str, object] | None,
        ) -> None:
            if elapsed_s is None:
                return
            try:
                log_default_telemetry(
                    phase=phase,
                    elapsed_s=elapsed_s,
                    dataset_hash=dataset_hash,
                    extra=extra,
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar telemetría para snapshot.persist_async",
                    exc_info=True,
                )

        def _on_complete(result: snapshot_async.SnapshotResult) -> None:
            if result.error is not None:
                self._record_snapshot_event(
                    action="save",
                    status="error",
                    detail=str(result.error),
                )
                return
            if result.skipped:
                return

            saved = result.saved or {}
            storage_id = saved.get("id") if isinstance(saved, Mapping) else None
            self._record_snapshot_event(
                action="save",
                status="saved",
                storage_id=str(storage_id) if storage_id else None,
            )

            persisted_history: pd.DataFrame | None = None
            if result.history:
                try:
                    persisted_history = _history_df_from_snapshot_records(result.history)
                    if isinstance(persisted_history, pd.DataFrame) and persisted_history.empty:
                        persisted_history = None
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "No se pudo construir la historia persistida del portafolio",
                        exc_info=True,
                    )

            if storage_id or persisted_history is not None:
                with self._snapshot_lock:
                    current = self._snapshot
                    if current is snapshot:
                        history = persisted_history if persisted_history is not None else snapshot.historical_total
                        self._snapshot = PortfolioViewSnapshot(
                            df_view=snapshot.df_view,
                            totals=snapshot.totals,
                            apply_elapsed=snapshot.apply_elapsed,
                            totals_elapsed=snapshot.totals_elapsed,
                            generated_at=snapshot.generated_at,
                            historical_total=history,
                            contribution_metrics=snapshot.contribution_metrics,
                            storage_id=storage_id or snapshot.storage_id,
                            pending_metrics=snapshot.pending_metrics,
                            dataset_hash=snapshot.dataset_hash,
                            metadata=snapshot.metadata,
                        )

        snapshot_defer.queue_snapshot_persistence(
            kind=self._snapshot_kind,
            payload=payload,
            metadata=metadata,
            persist_fn=backend,
            list_fn=history_fetcher,
            dataset_hash=dataset_key,
            on_complete=_on_complete,
            telemetry_fn=_telemetry_hook,
        )

    def invalidate_positions(self, dataset_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambia el dataset base."""

        current_filters = self._filters_key
        with self._snapshot_lock:
            self._snapshot = None
            self._dataset_key = dataset_key
            self._current_dataset_hash = dataset_key
            self._filters_key = None
            self._history_records = []
            self._incremental_cache = {}
            self._last_incremental_stats = None
        adapter = self._dataset_cache_adapter
        if adapter is not None:
            try:
                adapter.clear()
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo limpiar el dataset cache tras invalidate_positions",
                    exc_info=True,
                )
        _PORTFOLIO_CACHE_TELEMETRY.record_invalidation("dataset_changed", detail=dataset_key)
        _log_dataset_cache_event(
            "invalidate",
            dataset_key,
            scope="positions",
            filters_key=current_filters,
        )
        logger.info("portfolio_view cache invalidated (positions) dataset=%s", dataset_key)

    def invalidate_filters(self, filters_key: str | None = None) -> None:
        """Invalida el snapshot cuando cambian los filtros relevantes."""

        current_dataset = self._dataset_key
        with self._snapshot_lock:
            self._snapshot = None
            self._filters_key = filters_key
            self._incremental_cache = {}
            self._last_incremental_stats = None
        adapter = self._dataset_cache_adapter
        if adapter is not None:
            try:
                adapter.clear()
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo limpiar el dataset cache tras invalidate_filters",
                    exc_info=True,
                )
        _PORTFOLIO_CACHE_TELEMETRY.record_invalidation("filters_changed", detail=filters_key)
        _log_dataset_cache_event(
            "invalidate",
            current_dataset,
            scope="filters",
            filters_key=filters_key,
        )
        logger.info("portfolio_view cache invalidated (filters) filters=%s", filters_key)

    def _store_dataset_cache_entry(
        self,
        *,
        dataset_key: str,
        filters_key: str,
        fingerprints: Mapping[str, str],
        snapshot: PortfolioViewSnapshot,
        pending_metrics: tuple[str, ...],
        timestamp_bucket: str | None,
        include_extended: bool,
    ) -> None:
        adapter = self._dataset_cache_adapter
        if adapter is None or not dataset_key:
            return
        cache_key = _dataset_cache_key(dataset_key, filters_key, fingerprints, timestamp_bucket)
        try:
            history_payload = tuple(dict(row) for row in self._history_records)
            entry = PortfolioDatasetCacheEntry(
                snapshot=snapshot,
                incremental_cache=dict(self._incremental_cache),
                fingerprints=dict(fingerprints),
                stats=dict(self._last_incremental_stats or {}),
                dataset_key=dataset_key,
                filters_key=filters_key,
                created_at=float(snapshot.generated_at),
                pending_metrics=tuple(pending_metrics),
                history_records=history_payload,
                timestamp_bucket=timestamp_bucket,
            )
            adapter.set(cache_key, entry, ttl=_DATASET_CACHE_TTL_SECONDS)
            _log_dataset_cache_event(
                "store",
                dataset_key,
                filters_key=filters_key,
                pending=";".join(pending_metrics) if pending_metrics else "none",
                include_extended=include_extended,
                timestamp_bucket=timestamp_bucket,
            )
        except Exception:  # pragma: no cover - cache errors should not break flow
            logger.debug("No se pudo almacenar la entrada del dataset cache", exc_info=True)

    def _compute_viewmodel_phase(
        self,
        *,
        df_pos,
        controls,
        cli,
        psvc,
        include_extended: bool,
        telemetry_phase: str,
        allow_pending_reuse: bool,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ) -> PortfolioViewSnapshot:
        dataset_base = str(dataset_hash) if dataset_hash else self._hash_dataset(df_pos)
        if not dataset_base:
            dataset_base = "empty"
        dataset_key = f"{dataset_base}|totals_v{PORTFOLIO_TOTALS_VERSION}"
        filters_key = self._filters_key_from(controls)

        timestamp_bucket = _extract_quotes_timestamp(cli, psvc)

        previous_dataset_key = self._dataset_key
        previous_version = _extract_totals_version(previous_dataset_key)
        current_version = _extract_totals_version(dataset_key)
        totals_version_changed = bool(
            previous_version and current_version and previous_version != current_version
        )
        version_only_change = False
        if totals_version_changed:
            previous_base = _extract_dataset_base(previous_dataset_key)
            version_only_change = bool(previous_base and previous_base == dataset_base)
            logger.info(
                "[PortfolioViewModelService] Forcing totals recomputation due to version mismatch "
                "(previous=v%s, current=v%s)",
                previous_version,
                current_version,
            )
            with self._snapshot_lock:
                self._incremental_cache = {}
                self._last_incremental_stats = None
            prev_label = f"v{previous_version}" if previous_version else "unknown"
            curr_label = f"v{current_version}" if current_version else "unknown"
            _PORTFOLIO_CACHE_TELEMETRY.record_invalidation(
                "totals_version_changed",
                detail=f"{prev_label}->{curr_label}",
            )

        skip_invalidation = bool(skip_invalidation)
        current_quotes_hash = _extract_quotes_hash(dataset_key)
        previous_quotes_hash = _extract_quotes_hash(self._dataset_key)
        quotes_changed = bool(current_quotes_hash != previous_quotes_hash)

        effective_skip_invalidation = skip_invalidation and not totals_version_changed and not quotes_changed
        self._soft_refresh_guard_active = False
        raw_dataset_changed = dataset_key != self._dataset_key or quotes_changed
        dataset_changed = (raw_dataset_changed and not effective_skip_invalidation) or totals_version_changed
        filters_changed = filters_key != self._filters_key

        should_invalidate_cache = self._should_invalidate_cache(dataset_key, effective_skip_invalidation)
        if totals_version_changed:
            should_invalidate_cache = True
        elif not should_invalidate_cache:
            dataset_changed = False

        if quotes_changed:
            logger.info(
                "[PortfolioView] Quotes hash changed; forcing dataset refresh",
                extra={
                    "previous_quotes_hash": previous_quotes_hash or "none",
                    "current_quotes_hash": current_quotes_hash or "none",
                },
            )

        if effective_skip_invalidation and dataset_key:
            logger.info(
                'portfolio_view.skip_invalidation_applied event="skip_invalidation" dataset=%s filters=%s',
                dataset_key,
                filters_key,
            )
            _log_dataset_cache_event(
                "skip_invalidation_applied",
                dataset_key,
                filters_key=filters_key,
            )

        render_start = time.perf_counter()

        with self._snapshot_lock:
            current_snapshot = self._snapshot

        fingerprints = _build_incremental_fingerprints(dataset_key, controls, filters_key)

        cache_adapter = self._dataset_cache_adapter
        cached_entry: PortfolioDatasetCacheEntry | None = None
        warm_entry: PortfolioDatasetCacheEntry | None = None
        if cache_adapter is not None and dataset_key:
            cache_key = _dataset_cache_key(dataset_key, filters_key, fingerprints, timestamp_bucket)
            candidate = cache_adapter.get(cache_key)
            if isinstance(candidate, PortfolioDatasetCacheEntry):
                cached_entry = candidate
            elif cache_adapter is not None:
                _log_dataset_cache_event(
                    "miss",
                    dataset_key,
                    filters_key=filters_key,
                    include_extended=include_extended,
                    timestamp_bucket=timestamp_bucket,
                )

        if cached_entry is not None:
            pending = tuple(cached_entry.pending_metrics)
            entry_fingerprints = dict(cached_entry.fingerprints)
            entry_timestamp_match = cached_entry.timestamp_bucket == timestamp_bucket
            entry_filters_match = cached_entry.filters_key == filters_key
            can_reuse_entry = entry_filters_match and entry_fingerprints == dict(fingerprints) and entry_timestamp_match
            reason = ""
            if include_extended and pending:
                can_reuse_entry = False
                reason = "pending_metrics"
            elif not include_extended and not allow_pending_reuse and pending:
                can_reuse_entry = False
                reason = "pending_metrics"

            if can_reuse_entry:
                snapshot = replace(
                    cached_entry.snapshot,
                    soft_refresh_guard=self._soft_refresh_guard_active,
                )
                with self._snapshot_lock:
                    self._snapshot = snapshot
                    self._dataset_key = dataset_key
                    self._current_dataset_hash = dataset_key
                    self._filters_key = filters_key
                    self._incremental_cache = dict(cached_entry.incremental_cache)
                    self._incremental_cache.setdefault("fingerprints", entry_fingerprints)
                    self._last_incremental_stats = dict(cached_entry.stats)
                    self._history_records = [dict(row) for row in cached_entry.history_records]
                elapsed = time.perf_counter() - render_start
                _PORTFOLIO_CACHE_TELEMETRY.record_hit(
                    elapsed_s=elapsed,
                    dataset_changed=dataset_changed,
                    filters_changed=filters_changed,
                )
                _log_dataset_cache_event(
                    "hit",
                    dataset_key,
                    filters_key=filters_key,
                    pending=";".join(pending) if pending else "none",
                    include_extended=include_extended,
                    timestamp_bucket=timestamp_bucket,
                )
                self._record_snapshot_event(
                    action="load",
                    status="reused",
                    storage_id=cached_entry.snapshot.storage_id,
                )
                try:
                    log_default_telemetry(
                        phase=telemetry_phase,
                        elapsed_s=cached_entry.snapshot.apply_elapsed,
                        dataset_hash=dataset_key,
                        memo_hit_ratio=1.0,
                        pipeline_cache_hit_ratio=1.0,
                    )
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "No se pudo registrar telemetría para %s (dataset hit)",
                        telemetry_phase,
                        exc_info=True,
                    )
                if current_snapshot is None:
                    stats = cached_entry.stats if isinstance(cached_entry.stats, Mapping) else {}
                    apply_stat = float(stats.get("apply_elapsed", 0.0) or 0.0)
                    totals_stat = float(stats.get("totals_elapsed", 0.0) or 0.0)
                    _PORTFOLIO_CACHE_TELEMETRY.record_miss(
                        elapsed_s=elapsed,
                        dataset_changed=dataset_changed,
                        filters_changed=filters_changed,
                        apply_elapsed=apply_stat,
                        totals_elapsed=totals_stat,
                    )
                _PORTFOLIO_CACHE_TELEMETRY.record_pipeline_event(hit=True)
                return snapshot

            warm_entry = cached_entry
            _log_dataset_cache_event(
                "bypass",
                dataset_key,
                filters_key=filters_key,
                pending=";".join(pending) if pending else "none",
                include_extended=include_extended,
                reason=reason or "mismatch",
                timestamp_bucket=timestamp_bucket,
            )

        if dataset_changed and should_invalidate_cache:
            if not version_only_change:
                self.invalidate_positions(dataset_key)
        elif filters_changed and should_invalidate_cache:
            self.invalidate_filters(filters_key)
        with self._snapshot_lock:
            previous_snapshot = self._snapshot

        cached_blocks: Mapping[str, Any] = {}
        cached_fingerprints: Mapping[str, str] = {}
        if dataset_changed and not version_only_change:
            if warm_entry is not None:
                cached_blocks = dict(warm_entry.incremental_cache)
                cached_fingerprints = dict(warm_entry.fingerprints)
                _log_dataset_cache_event(
                    "warm",
                    dataset_key,
                    filters_key=filters_key,
                    pending=";".join(warm_entry.pending_metrics) if warm_entry.pending_metrics else "none",
                    include_extended=include_extended,
                    timestamp_bucket=timestamp_bucket,
                )
        else:
            cached_blocks = dict(self._incremental_cache)
            cached_fp = cached_blocks.get("fingerprints")
            if isinstance(cached_fp, Mapping):
                cached_fingerprints = dict(cached_fp)
                if version_only_change:
                    cached_fingerprints["dataset"] = dataset_key
                    cached_blocks["fingerprints"] = dict(cached_fingerprints)

        if quotes_changed:
            cached_blocks = {}
            cached_fingerprints = {}

        incremental_changed = any(
            fingerprints.get(key) != cached_fingerprints.get(key)
            for key in ("filters.time", "filters.fx", "filters.misc")
        )
        effective_filters_changed = filters_changed or incremental_changed

        can_reuse_snapshot = previous_snapshot is not None and not dataset_changed and not effective_filters_changed

        if can_reuse_snapshot:
            pending = tuple(getattr(previous_snapshot, "pending_metrics", ()))
            if include_extended and pending:
                can_reuse_snapshot = False
            elif not include_extended and not allow_pending_reuse and pending:
                can_reuse_snapshot = False

        if can_reuse_snapshot:
            elapsed = time.perf_counter() - render_start
            if include_extended:
                _PORTFOLIO_CACHE_TELEMETRY.record_hit(
                    elapsed_s=elapsed,
                    dataset_changed=dataset_changed,
                    filters_changed=effective_filters_changed,
                )
                try:
                    log_default_telemetry(
                        phase=telemetry_phase,
                        elapsed_s=previous_snapshot.apply_elapsed,
                        dataset_hash=dataset_key,
                        memo_hit_ratio=1.0,
                        pipeline_cache_hit_ratio=1.0,
                    )
                except Exception:  # pragma: no cover - defensive safeguard
                    logger.debug(
                        "No se pudo registrar telemetría para %s (hit)",
                        telemetry_phase,
                        exc_info=True,
                    )
            _PORTFOLIO_CACHE_TELEMETRY.record_pipeline_event(hit=True)
            snapshot = replace(
                previous_snapshot,
                soft_refresh_guard=self._soft_refresh_guard_active,
            )
            with self._snapshot_lock:
                self._snapshot = snapshot
                self._dataset_key = dataset_key
                self._current_dataset_hash = dataset_key
                self._filters_key = filters_key
            self._record_snapshot_event(
                action="load",
                status="reused",
                storage_id=previous_snapshot.storage_id,
            )
            return snapshot

        positions_state: dict[str, Any] = {}

        def _load_positions() -> tuple[pd.DataFrame, float]:
            if "df" not in positions_state:
                start = time.perf_counter()
                positions_hash = _extract_positions_fingerprint(dataset_key) or dataset_key
                df = _apply_filters(
                    df_pos,
                    controls,
                    cli,
                    psvc,
                    dataset_hash=positions_hash,
                    skip_invalidation=skip_invalidation,
                )
                positions_state["df"] = df
                positions_state["elapsed"] = time.perf_counter() - start
            return (
                positions_state.get("df", pd.DataFrame()),
                float(positions_state.get("elapsed", 0.0)),
            )

        incremental = compute_incremental_view(
            dataset_changed=dataset_changed,
            fingerprints=fingerprints,
            previous_snapshot=previous_snapshot,
            previous_blocks=cached_blocks,
            load_positions=_load_positions,
            compute_totals_fn=calculate_totals,
            compute_contributions_fn=_compute_contribution_metrics,
            update_history_fn=self._update_history,
            include_extended=include_extended,
            force_totals_recompute=totals_version_changed,
            allow_dataset_reuse=version_only_change,
        )

        patch_applied = _apply_bopreal_postmerge_patch(incremental.df_view)
        if patch_applied:
            updated_totals = calculate_totals(incremental.df_view)
            updated_contributions = incremental.contribution_metrics
            if incremental.extended_computed:
                updated_contributions = _compute_contribution_metrics(incremental.df_view)
            incremental = replace(
                incremental,
                totals=updated_totals,
                contribution_metrics=updated_contributions,
            )

        generated_ts = time.time()
        pending_metrics: tuple[str, ...] = ()
        if not incremental.extended_computed:
            pending_metrics = ("extended_metrics",)

        storage_id = None
        if previous_snapshot is not None and not dataset_changed and not effective_filters_changed:
            storage_id = previous_snapshot.storage_id

        snapshot_metadata: dict[str, Any]
        prev_metadata = getattr(previous_snapshot, "metadata", None)
        if isinstance(prev_metadata, Mapping):
            snapshot_metadata = dict(prev_metadata)
        else:
            snapshot_metadata = {}
        snapshot_metadata.update(
            {
                "totals_version": f"v{PORTFOLIO_TOTALS_VERSION}",
                "dataset_key": dataset_key,
            }
        )
        if current_quotes_hash:
            snapshot_metadata["quotes_hash"] = current_quotes_hash

        snapshot = PortfolioViewSnapshot(
            df_view=incremental.df_view,
            totals=incremental.totals,
            apply_elapsed=incremental.apply_elapsed,
            totals_elapsed=incremental.totals_elapsed,
            generated_at=generated_ts,
            historical_total=incremental.historical_total,
            contribution_metrics=incremental.contribution_metrics,
            storage_id=storage_id,
            pending_metrics=pending_metrics,
            dataset_hash=dataset_key,
            soft_refresh_guard=self._soft_refresh_guard_active,
            metadata=snapshot_metadata,
        )

        total_blocks = len(incremental.reused_blocks) + len(incremental.recomputed_blocks)
        hit_ratio = (len(incremental.reused_blocks) / total_blocks) if total_blocks else 0.0
        pipeline_hit = "positions_df" in incremental.reused_blocks

        with self._snapshot_lock:
            self._snapshot = snapshot
            self._dataset_key = dataset_key
            self._current_dataset_hash = dataset_key
            self._filters_key = filters_key
            self._incremental_cache = {
                "positions_df": incremental.df_view,
                "returns_df": incremental.returns_df,
                "fingerprints": dict(fingerprints),
                "dataset_key": dataset_key,
                "filters_key": filters_key,
                "generated_at": generated_ts,
                "pending_metrics": pending_metrics,
            }
            self._last_incremental_stats = {
                "reused_blocks": incremental.reused_blocks,
                "recomputed_blocks": incremental.recomputed_blocks,
                "memoization_hit_ratio": hit_ratio,
                "dataset_changed": dataset_changed,
                "filters_changed": effective_filters_changed,
                "compute_duration_s": incremental.duration,
                "apply_elapsed": incremental.apply_elapsed,
                "totals_elapsed": incremental.totals_elapsed,
                "phase": telemetry_phase,
            }

        self._store_dataset_cache_entry(
            dataset_key=dataset_key,
            filters_key=filters_key,
            fingerprints=fingerprints,
            snapshot=snapshot,
            pending_metrics=pending_metrics,
            timestamp_bucket=timestamp_bucket,
            include_extended=include_extended,
        )

        _PORTFOLIO_CACHE_TELEMETRY.record_pipeline_event(hit=pipeline_hit)

        logger.info(
            "portfolio_view phase=%s dataset_changed=%s filters_changed=%s pending=%s "
            "reused=%s recomputed=%s apply=%.4fs totals=%.4fs",
            telemetry_phase,
            dataset_changed,
            effective_filters_changed,
            ",".join(pending_metrics) or "none",
            ",".join(incremental.reused_blocks) or "none",
            ",".join(incremental.recomputed_blocks) or "none",
            incremental.apply_elapsed,
            incremental.totals_elapsed,
        )

        total_elapsed = time.perf_counter() - render_start
        previous_pending = tuple(getattr(previous_snapshot, "pending_metrics", ()) or ())
        follow_up_extended = include_extended and previous_pending == ("extended_metrics",)

        if not follow_up_extended:
            _PORTFOLIO_CACHE_TELEMETRY.record_miss(
                elapsed_s=total_elapsed,
                dataset_changed=dataset_changed,
                filters_changed=effective_filters_changed,
                apply_elapsed=incremental.apply_elapsed,
                totals_elapsed=incremental.totals_elapsed,
            )

        if not follow_up_extended:
            _append_incremental_metric(
                dataset_hash=dataset_key,
                filters_changed=effective_filters_changed,
                reused_blocks=incremental.reused_blocks,
                recomputed_blocks=incremental.recomputed_blocks,
                total_duration=total_elapsed,
                memoization_hit_ratio=hit_ratio,
            )

        try:
            log_default_telemetry(
                phase=telemetry_phase,
                elapsed_s=incremental.duration,
                dataset_hash=dataset_key,
                memo_hit_ratio=hit_ratio,
                pipeline_cache_hit_ratio=1.0 if pipeline_hit else 0.0,
            )
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "No se pudo registrar telemetría para %s",
                telemetry_phase,
                exc_info=True,
            )

        if (
            include_extended
            and not pending_metrics
            and (
                dataset_changed
                or effective_filters_changed
                or previous_snapshot is None
                or getattr(previous_snapshot, "pending_metrics", ())
            )
        ):
            self._schedule_snapshot_persistence(
                snapshot=snapshot,
                controls=controls,
                dataset_key=dataset_key,
                filters_key=filters_key,
            )

        return snapshot

    def apply_dataset_pipeline(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash: str | None = None,
        mode: str = "full",
        skip_invalidation: bool = False,
    ) -> PortfolioViewSnapshot:
        """Ejecuta la canalización del dataset en el modo solicitado."""

        normalized = str(mode or "full").strip().lower()
        if normalized in {"basic", "minimal"}:
            include_extended = False
            telemetry_phase = "portfolio_view.apply_basic"
            allow_pending_reuse = True
        elif normalized in {"extended", "full"}:
            include_extended = True
            telemetry_phase = "portfolio_view.apply_extended"
            allow_pending_reuse = False
        else:
            raise ValueError(f"Invalid pipeline mode: {mode!r}")

        return self._compute_viewmodel_phase(
            df_pos=df_pos,
            controls=controls,
            cli=cli,
            psvc=psvc,
            include_extended=include_extended,
            telemetry_phase=telemetry_phase,
            allow_pending_reuse=allow_pending_reuse,
            dataset_hash=dataset_hash,
            skip_invalidation=skip_invalidation,
        )

    def build_minimal_viewmodel(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ) -> PortfolioViewSnapshot:
        """Construye el snapshot mínimo requerido para renderizar el portafolio."""

        return self.apply_dataset_pipeline(
            df_pos,
            controls,
            cli,
            psvc,
            dataset_hash=dataset_hash,
            mode="basic",
            skip_invalidation=skip_invalidation,
        )

    def compute_extended_metrics(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ) -> PortfolioViewSnapshot:
        """Completa las métricas extendidas del snapshot cuando son necesarias."""

        return self.apply_dataset_pipeline(
            df_pos,
            controls,
            cli,
            psvc,
            dataset_hash=dataset_hash,
            mode="extended",
            skip_invalidation=skip_invalidation,
        )

    def get_portfolio_view(
        self,
        df_pos,
        controls,
        cli,
        psvc,
        *,
        lazy_metrics: bool = False,
        dataset_hash: str | None = None,
        skip_invalidation: bool = False,
    ) -> PortfolioViewSnapshot:
        """Devuelve el snapshot del portafolio aplicando métricas diferidas si es necesario."""

        snapshot = self.build_minimal_viewmodel(
            df_pos,
            controls,
            cli,
            psvc,
            dataset_hash=dataset_hash,
            skip_invalidation=skip_invalidation,
        )
        if lazy_metrics and snapshot.pending_metrics:
            return snapshot
        if snapshot.pending_metrics:
            snapshot = self.compute_extended_metrics(
                df_pos,
                controls,
                cli,
                psvc,
                dataset_hash=dataset_hash,
                skip_invalidation=skip_invalidation,
            )
        return snapshot


def _coerce_json(value: Any) -> Any:
    if isinstance(value, (pd.Timestamp, pd.Timedelta)):
        return value.isoformat()
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    return str(value)


def _apply_filters(
    df_pos,
    controls,
    cli,
    psvc,
    *,
    dataset_hash: str | None = None,
    skip_invalidation: bool = False,
):
    from controllers.portfolio.filters import apply_filters as _apply

    return _apply(
        df_pos,
        controls,
        cli,
        psvc,
        dataset_hash=dataset_hash,
        skip_invalidation=skip_invalidation,
    )


def _apply_bopreal_postmerge_patch(df_view: pd.DataFrame) -> bool:
    """Ensure BOPREAL ARS rows keep the forced valuation after dataset merges."""

    if not isinstance(df_view, pd.DataFrame) or df_view.empty:
        return False

    required = {"simbolo", "moneda_origen", "cantidad", "ultimoPrecio"}
    if not required.issubset(df_view.columns):
        return False

    symbols = df_view["simbolo"].astype("string").fillna("").str.upper()
    currency = df_view["moneda_origen"].astype("string").fillna("").str.upper()
    pricing_series = (
        df_view.get("pricing_source", pd.Series(index=df_view.index, dtype="object"))
        .astype("string")
        .str.lower()
        .fillna("")
    )
    bopreal_mask = symbols.isin(_BOPREAL_ARS_SYMBOLS) & ~symbols.str.endswith("D")
    mask = bopreal_mask & currency.eq("ARS")
    guard_mask = pricing_series.eq("override_bopreal_forced")
    mask = mask & ~guard_mask
    if not bool(mask.any()):
        return False

    quantity_series = pd.to_numeric(df_view["cantidad"], errors="coerce")
    if "scale" in df_view.columns:
        scale_series = pd.to_numeric(df_view["scale"], errors="coerce").fillna(1.0)
    else:
        scale_series = pd.Series(1.0, index=df_view.index, dtype=float)

    effective_qty = quantity_series.mul(scale_series)

    last_series = pd.Series(np.nan, index=df_view.index, dtype=float)
    if "ultimo" in df_view.columns:
        last_series = pd.to_numeric(df_view["ultimo"], errors="coerce")
    if "ultimoPrecio" in df_view.columns:
        ultimo_series = pd.to_numeric(df_view["ultimoPrecio"], errors="coerce")
        last_series = last_series.where(last_series.notna(), ultimo_series)

    recalculated = last_series.mul(effective_qty)

    if "valor_actual" not in df_view.columns:
        df_view["valor_actual"] = np.nan
    current_values = pd.to_numeric(df_view["valor_actual"], errors="coerce")
    effective_values = recalculated.where(recalculated.notna(), current_values)
    df_view.loc[mask, "valor_actual"] = effective_values.loc[mask]
    if "valorizado" in df_view.columns:
        df_view.loc[mask, "valorizado"] = effective_values.loc[mask]
    if "ultimo" in df_view.columns:
        df_view.loc[mask, "ultimo"] = last_series.loc[mask]

    if "pl" not in df_view.columns:
        df_view["pl"] = np.nan
    if "pl_%" not in df_view.columns:
        df_view["pl_%"] = np.nan

    costo_series = pd.to_numeric(
        df_view.get("costo", pd.Series(index=df_view.index, dtype=float)),
        errors="coerce",
    )
    pl_values = effective_values.subtract(costo_series, fill_value=np.nan)
    df_view.loc[mask, "pl"] = pl_values.loc[mask]
    with np.errstate(divide="ignore", invalid="ignore"):
        pl_pct = (pl_values.divide(costo_series)).multiply(100.0)
    pl_pct = pl_pct.replace([np.inf, -np.inf], np.nan)
    df_view.loc[mask, "pl_%"] = pl_pct.loc[mask]

    df_view.loc[mask, "pricing_source"] = "override_bopreal_postmerge"

    if "audit" in df_view.columns:
        decision_tag = "override_bopreal_postmerge"

        def _append_decision(audit_value: Any) -> dict[str, Any]:
            if isinstance(audit_value, Mapping):
                audit_dict = dict(audit_value)
            else:
                audit_dict = {}
            existing = audit_dict.get("scale_decisions")
            if isinstance(existing, list):
                updated = list(existing)
            elif existing is None:
                updated = []
            else:
                updated = [existing]
            if decision_tag not in updated:
                updated.append(decision_tag)
            audit_dict["scale_decisions"] = updated
            return audit_dict

        df_view.loc[mask, "audit"] = df_view.loc[mask, "audit"].apply(_append_decision)

    return True


def _compute_contribution_metrics(
    df_view: pd.DataFrame,
) -> PortfolioContributionMetrics:
    if df_view is None or df_view.empty:
        return PortfolioContributionMetrics.empty()

    cols = {
        "valor_actual": "valor_actual",
        "costo": "costo",
        "pl": "pl",
        "pl_d": "pl_d",
    }
    df = df_view.copy()
    for src, dest in cols.items():
        if src in df.columns:
            df[dest] = pd.to_numeric(df[src], errors="coerce")
        else:
            df[dest] = np.nan

    required = ["tipo", "simbolo"]
    for col in required:
        if col not in df.columns:
            df[col] = ""

    numeric_cols = list(cols.values())

    by_symbol = df.groupby(["tipo", "simbolo"], dropna=False)[numeric_cols].sum(min_count=1).reset_index()

    total_val = by_symbol["valor_actual"].sum(min_count=1)
    total_pl = by_symbol["pl"].sum(min_count=1)

    if not np.isfinite(total_val) or np.isclose(total_val, 0.0):
        by_symbol["valor_actual_pct"] = np.nan
    else:
        by_symbol["valor_actual_pct"] = (by_symbol["valor_actual"] / total_val) * 100.0

    if not np.isfinite(total_pl) or np.isclose(total_pl, 0.0):
        by_symbol["pl_pct"] = np.nan
    else:
        by_symbol["pl_pct"] = (by_symbol["pl"] / total_pl) * 100.0

    by_type = df.groupby("tipo", dropna=False)[numeric_cols].sum(min_count=1).reset_index()

    if not np.isfinite(total_val) or np.isclose(total_val, 0.0):
        by_type["valor_actual_pct"] = np.nan
    else:
        by_type["valor_actual_pct"] = (by_type["valor_actual"] / total_val) * 100.0

    if not np.isfinite(total_pl) or np.isclose(total_pl, 0.0):
        by_type["pl_pct"] = np.nan
    else:
        by_type["pl_pct"] = (by_type["pl"] / total_pl) * 100.0

    return PortfolioContributionMetrics(by_symbol=by_symbol, by_type=by_type)


def _history_row(ts: float, totals: PortfolioTotals) -> dict[str, float]:
    return {
        "timestamp": ts,
        "total_value": float(totals.total_value),
        "total_cost": float(totals.total_cost),
        "total_pl": float(totals.total_pl),
    }


def _normalize_history_df(records: list[dict[str, float]]) -> pd.DataFrame:
    if not records:
        return pd.DataFrame(columns=["timestamp", "total_value", "total_cost", "total_pl"])
    df = pd.DataFrame.from_records(records)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    return df


def _append_history(records: list[dict[str, float]], entry: dict[str, float], maxlen: int) -> list[dict[str, float]]:
    records.append(entry)
    if maxlen and maxlen > 0:
        records = records[-maxlen:]
    return records


def _serialize_snapshot_payload(
    *,
    df_view: pd.DataFrame,
    totals: PortfolioTotals,
    generated_at: float,
    history: pd.DataFrame,
    contribution_metrics: PortfolioContributionMetrics,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "generated_at": float(generated_at),
        "totals": {
            "total_value": float(totals.total_value),
            "total_cost": float(totals.total_cost),
            "total_pl": float(totals.total_pl),
            "total_pl_pct": float(totals.total_pl_pct),
            "total_cash": float(totals.total_cash),
            "total_cash_ars": float(totals.total_cash_ars),
            "total_cash_usd": float(totals.total_cash_usd),
            "total_cash_combined": float(totals.total_cash_combined),
        },
    }

    breakdown = getattr(totals, "valuation_breakdown", None)
    breakdown_payload: dict[str, Any] = {}
    if breakdown is not None:
        try:
            breakdown_payload = asdict(breakdown)
        except TypeError:
            if isinstance(breakdown, Mapping):
                breakdown_payload = dict(breakdown)
    if breakdown_payload:
        impact = getattr(breakdown, "estimated_impact_pct", float("nan"))
        try:
            impact_value = float(impact)
        except (TypeError, ValueError):
            impact_value = float("nan")
        if not np.isfinite(impact_value):
            impact_value = float("nan")
        breakdown_payload.setdefault("estimated_impact_pct", impact_value)
        payload["totals"]["valuation_breakdown"] = breakdown_payload

    payload["df_view"] = _df_to_records(df_view)
    payload["history"] = _df_to_records(history)

    contrib = {}
    if isinstance(contribution_metrics, PortfolioContributionMetrics):
        contrib["by_symbol"] = _df_to_records(contribution_metrics.by_symbol)
        contrib["by_type"] = _df_to_records(contribution_metrics.by_type)
    payload["contribution_metrics"] = contrib
    return payload


def _df_to_records(df: pd.DataFrame | None) -> list[dict[str, Any]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []
    try:
        return df.replace({np.nan: None}).to_dict(orient="records")
    except Exception:
        return []


def _safe_asdict(obj: Any) -> dict[str, Any]:
    if obj is None:
        return {}
    try:
        return asdict(obj)
    except TypeError:
        result: dict[str, Any] = {}
        for key in dir(obj):
            if key.startswith("_"):
                continue
            try:
                value = getattr(obj, key)
            except Exception:
                continue
            if callable(value):
                continue
            result[key] = value
        return result


def _as_mapping(value: Any) -> dict[str, float]:
    if isinstance(value, Mapping):
        result: dict[str, float] = {}
        for key, val in value.items():
            try:
                result[str(key)] = float(val)
            except (TypeError, ValueError):
                continue
        return result
    return {}


def _history_df_from_snapshot_records(
    records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    history_rows: list[dict[str, float]] = []
    for row in records:
        payload = row.get("payload") if isinstance(row, Mapping) else {}
        totals = {}
        if isinstance(payload, Mapping):
            totals = payload.get("totals") or {}
        ts = row.get("created_at") if isinstance(row, Mapping) else None
        if not ts and isinstance(payload, Mapping):
            ts = payload.get("generated_at")
        entry = {
            "timestamp": float(ts or 0.0),
            "total_value": float(_get_numeric(totals, "total_value")),
            "total_cost": float(_get_numeric(totals, "total_cost")),
            "total_pl": float(_get_numeric(totals, "total_pl")),
        }
        history_rows.append(entry)
    return _normalize_history_df(history_rows)


def _get_numeric(payload: Mapping[str, Any], key: str) -> float:
    try:
        return float(payload.get(key, 0.0))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _ensure_container(placeholder: Any, references: MutableMapping[str, Any]) -> Any:
    """Return a persistent container bound to ``placeholder``."""

    container = references.get("container")
    if container is None or not hasattr(container, "__enter__"):
        try:
            container = placeholder.container()
        except AttributeError:
            container = placeholder
        references["container"] = container
    return container


def update_summary_section(
    placeholder: Any,
    *,
    render_fn: Callable[..., Any],
    df_view: Any,
    controls: Any,
    ccl_rate: Any,
    totals: Any = None,
    favorites: Any = None,
    historical_total: Any = None,
    contribution_metrics: Any = None,
    snapshot: Any = None,
    references: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Update the summary placeholder without rebuilding the container."""

    refs: MutableMapping[str, Any]
    if isinstance(references, MutableMapping):
        refs = references
    else:
        refs = {}
    container = _ensure_container(placeholder, refs)
    with container:
        result = render_fn(
            df_view,
            controls,
            ccl_rate,
            totals=totals,
            favorites=favorites,
            historical_total=historical_total,
            contribution_metrics=contribution_metrics,
            snapshot=snapshot,
        )
    refs["has_positions"] = bool(result)
    return refs


def update_table_data(
    placeholder: Any,
    *,
    render_fn: Callable[..., Any],
    df_view: Any,
    controls: Any,
    ccl_rate: Any,
    favorites: Any = None,
    references: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Refresh the table placeholder in-place using the provided renderer."""

    refs: MutableMapping[str, Any]
    if isinstance(references, MutableMapping):
        refs = references
    else:
        refs = {}
    container = _ensure_container(placeholder, refs)
    with container:
        render_fn(
            df_view,
            controls,
            ccl_rate,
            favorites=favorites,
        )
    return refs


def update_charts(
    placeholder: Any,
    *,
    render_fn: Callable[..., Any],
    df_view: Any,
    controls: Any,
    ccl_rate: Any,
    totals: Any = None,
    contribution_metrics: Any = None,
    snapshot: Any = None,
    references: MutableMapping[str, Any] | None = None,
) -> MutableMapping[str, Any]:
    """Update the charts placeholder with fresh figures."""

    refs: MutableMapping[str, Any]
    if isinstance(references, MutableMapping):
        refs = references
    else:
        refs = {}
    container = _ensure_container(placeholder, refs)
    with container:
        render_fn(
            df_view,
            controls,
            ccl_rate,
            totals=totals,
            contribution_metrics=contribution_metrics,
            snapshot=snapshot,
        )
    return refs
