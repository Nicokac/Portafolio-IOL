"""Quote caching and persistence helpers extracted from the legacy cache module."""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Mapping, Tuple

import requests

from infrastructure.iol.auth import InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider, IOLClient
from services.health import record_quote_load, record_quote_provider_usage
from services.performance_timer import performance_timer
from services.quote_rate_limit import quote_rate_limiter
from shared.cache import cache
from shared.errors import NetworkError
from shared.settings import cache_ttl_quotes, max_quote_workers, quotes_ttl_seconds
from shared.telemetry import log_default_telemetry

from .ui_adapter import _trigger_logout

logger = logging.getLogger(__name__)


# In-memory quote cache
_QUOTE_CACHE: Dict[Tuple[str, str, str | None], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()
_QUOTE_PERSIST_LOCK = Lock()
_QUOTE_PERSIST_CACHE: Dict[str, Any] | None = None
_QUOTE_WARM_START_APPLIED = False

QUOTE_STALE_TTL_SECONDS = float(quotes_ttl_seconds or 0)
_QUOTE_PERSIST_PATH = Path("data/quotes_cache.json")

_WARM_START_TTL_SECONDS = 5.0
_ACTIVE_DATASET_HASH: str | None = None


def set_active_dataset_hash(dataset_hash: str | None) -> None:
    """Expose the dataset hash associated with the current refresh cycle."""

    global _ACTIVE_DATASET_HASH
    _ACTIVE_DATASET_HASH = dataset_hash


def get_active_dataset_hash() -> str | None:
    """Return the dataset hash associated with the current refresh cycle."""

    return _ACTIVE_DATASET_HASH


_MAX_RATE_LIMIT_RETRIES = 2

_THREADPOOL_SUBLOT_TARGET = 6
_THREADPOOL_SUBLOT_MIN = 4
_THREADPOOL_SUBLOT_MAX = 10


class AdaptiveBatchController:
    """Simple heuristic to adjust quote refresh batch sizes based on latency."""

    def __init__(
        self,
        *,
        default_size: int,
        min_size: int,
        max_size: int,
        slow_threshold_ms: float = 700.0,
        fast_threshold_ms: float = 400.0,
        slow_target: int = 5,
        fast_target: int = 9,
    ) -> None:
        self._lock = Lock()
        self._size = int(default_size)
        self._avg_ms: float | None = None
        self._min = int(min_size)
        self._max = int(max_size)
        self._slow_threshold = float(slow_threshold_ms)
        self._fast_threshold = float(fast_threshold_ms)
        self._slow_target = int(slow_target)
        self._fast_target = int(fast_target)

    def current(self, population: int) -> int:
        """Return the batch size that should be used for the current refresh."""

        with self._lock:
            size = max(self._min, min(self._max, self._size))
        return max(1, min(size, population if population > 0 else size))

    def observe(self, avg_batch_time_ms: float | None, population: int) -> int:
        """Record the average duration for the executed refresh and adapt."""

        with self._lock:
            self._avg_ms = avg_batch_time_ms if avg_batch_time_ms is not None else None
            if avg_batch_time_ms is None:
                return max(self._min, min(self._max, self._size))

            if avg_batch_time_ms > self._slow_threshold:
                target = min(self._slow_target, self._max)
                self._size = max(self._min, target)
            elif avg_batch_time_ms < self._fast_threshold:
                target = max(self._fast_target, self._min)
                self._size = min(self._max, target)
            else:
                self._size = max(self._min, min(self._max, self._size))

            return max(1, min(self._size, population if population > 0 else self._size))

    def last_observed_avg(self) -> float | None:
        with self._lock:
            return self._avg_ms


_ADAPTIVE_BATCH_CONTROLLER = AdaptiveBatchController(
    default_size=_THREADPOOL_SUBLOT_TARGET,
    min_size=_THREADPOOL_SUBLOT_MIN,
    max_size=_THREADPOOL_SUBLOT_MAX,
)


class QuoteBatchStats:
    """Aggregate information about a batch of quotes."""

    def __init__(self, total_expected: int = 0) -> None:
        self.total_expected = int(total_expected)
        self.started_at = time.time()
        self._lock = Lock()
        self.count = 0
        self.fresh = 0
        self.stale = 0
        self.errors = 0
        self.fallbacks = 0
        self.rate_limited = 0
        self.elapsed_ms_total = 0.0

    def record_rate_limited(self) -> None:
        with self._lock:
            self.rate_limited += 1

    def record_result(
        self,
        *,
        provider: str | None,
        stale: bool,
        error: bool,
        fallback: bool,
        elapsed_ms: float | None,
    ) -> None:
        provider_key = (provider or "").strip().lower()
        fallback_flag = fallback or (provider_key not in ("", "iol", "cache")) or stale
        elapsed_value = float(elapsed_ms) if elapsed_ms else 0.0
        with self._lock:
            self.count += 1
            if error:
                self.errors += 1
            elif stale:
                self.stale += 1
            else:
                self.fresh += 1
            if fallback_flag:
                self.fallbacks += 1
            self.elapsed_ms_total += elapsed_value

    def record_payload(self, payload: Any) -> None:
        provider: str | None = None
        stale = False
        error = False
        fallback = False
        if isinstance(payload, Mapping):
            provider = payload.get("provider")
            stale = bool(payload.get("stale")) or payload.get("last") is None
            error = (provider or "").strip().lower() == "error"
            fallback = stale or (provider not in (None, "iol", "cache"))
        else:
            error = True
            fallback = True
        self.record_result(
            provider=provider,
            stale=stale,
            error=error,
            fallback=fallback,
            elapsed_ms=None,
        )

    def apply_provider_stats(self, stats: Mapping[str, Any]) -> None:
        with self._lock:
            rate_limited = _as_optional_int(stats.get("rate_limited"))
            if rate_limited is not None:
                self.rate_limited = max(self.rate_limited, rate_limited)

            fallback_count = _as_optional_int(stats.get("fallbacks"))
            if fallback_count is not None:
                self.fallbacks = max(self.fallbacks, fallback_count)

            count_value = _as_optional_int(stats.get("count"))
            if count_value is not None and count_value > 0:
                self.count = max(self.count, count_value)

            fresh_value = _as_optional_int(stats.get("fresh"))
            if fresh_value is not None:
                self.fresh = max(self.fresh, fresh_value)

            stale_value = _as_optional_int(stats.get("stale"))
            if stale_value is not None:
                self.stale = max(self.stale, stale_value)

            error_value = _as_optional_int(stats.get("errors"))
            if error_value is not None:
                self.errors = max(self.errors, error_value)

            elapsed_value = stats.get("elapsed_ms_total")
            if isinstance(elapsed_value, (int, float)):
                self.elapsed_ms_total = max(self.elapsed_ms_total, float(elapsed_value))

    def summary(self, elapsed: float) -> Dict[str, Any]:
        with self._lock:
            count = self.count
            total_elapsed = max(float(elapsed), 0.0)
            avg = total_elapsed / count if count else 0.0
            qps = count / total_elapsed if total_elapsed > 0 and count else 0.0
            return {
                "count": count,
                "fresh": self.fresh,
                "stale": self.stale,
                "errors": self.errors,
                "fallbacks": self.fallbacks,
                "rate_limited": self.rate_limited,
                "elapsed": total_elapsed,
                "avg": avg,
                "qps": qps,
            }


def _quote_cache_key(market: str, symbol: str, panel: str | None) -> str:
    panel_token = "" if panel in (None, "") else str(panel)
    return "|".join([market, symbol, panel_token])


def _normalize_bulk_key_components(key: Any) -> tuple[str, str, str | None] | None:
    """Try to extract (market, symbol, panel) information from bulk keys."""

    market: str | None = None
    symbol: str | None = None
    panel_value: str | None = None

    if isinstance(key, (list, tuple)):
        if len(key) >= 1 and key[0] is not None:
            market = str(key[0]).lower()
        if len(key) >= 2 and key[1] is not None:
            symbol = str(key[1]).upper()
        if len(key) >= 3:
            panel_raw = key[2]
            if panel_raw not in (None, ""):
                panel_value = str(panel_raw)
    elif isinstance(key, str):
        parts = re.split(r"[|:/]", key)
        if len(parts) >= 1 and parts[0]:
            market = parts[0].lower()
        if len(parts) >= 2 and parts[1]:
            symbol = parts[1].upper()
        if len(parts) >= 3 and parts[2]:
            panel_value = parts[2]
    else:
        market = getattr(key, "market", getattr(key, "mercado", None))
        symbol = getattr(key, "symbol", getattr(key, "simbolo", None))
        panel_raw = getattr(key, "panel", None)
        if market is not None:
            market = str(market).lower()
        if symbol is not None:
            symbol = str(symbol).upper()
        if panel_raw not in (None, ""):
            panel_value = str(panel_raw)

    if market is None or symbol is None:
        return None
    return market, symbol, panel_value


def _default_provider_for_client(cli: Any) -> str | None:
    module = getattr(getattr(cli, "__class__", None), "__module__", "")
    if not isinstance(module, str):
        return None
    module_lower = module.lower()
    if module_lower.startswith("infrastructure.iol.legacy"):
        return "legacy"
    if module_lower.startswith("infrastructure.iol"):
        return "iol"
    if "alphavantage" in module_lower or module_lower.endswith("av_client"):
        return "av"
    return None


def _resolve_rate_limit_provider(cli: Any) -> str:
    module = getattr(getattr(cli, "__class__", None), "__module__", "")
    if isinstance(module, str) and module.startswith("infrastructure.iol.legacy"):
        return "legacy"
    return "iol"


def _extract_provider_batch_stats(cli: Any) -> Mapping[str, Any] | None:
    visited = set()
    current = cli
    while current is not None and current not in visited:
        visited.add(current)
        stats = getattr(current, "_last_bulk_stats", None)
        if isinstance(stats, Mapping):
            return stats
        current = getattr(current, "_cli", None)
    return None


def _parse_retry_after_seconds(response) -> float | None:
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = (dt - now).total_seconds()
        return max(0.0, delta)


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> int | None:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_persisted_quotes() -> Dict[str, Any]:
    global _QUOTE_PERSIST_CACHE
    with _QUOTE_PERSIST_LOCK:
        if _QUOTE_PERSIST_CACHE is not None:
            return dict(_QUOTE_PERSIST_CACHE)
        try:
            text = _QUOTE_PERSIST_PATH.read_text(encoding="utf-8")
            raw = json.loads(text) or {}
            if not isinstance(raw, dict):
                raw = {}
        except (OSError, json.JSONDecodeError):
            raw = {}
        _QUOTE_PERSIST_CACHE = raw
        return dict(raw)


def _store_persisted_quotes(cache: Dict[str, Any]) -> None:
    global _QUOTE_PERSIST_CACHE
    with _QUOTE_PERSIST_LOCK:
        _QUOTE_PERSIST_CACHE = dict(cache)
        try:
            _QUOTE_PERSIST_PATH.parent.mkdir(parents=True, exist_ok=True)
            _QUOTE_PERSIST_PATH.write_text(
                json.dumps(_QUOTE_PERSIST_CACHE, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - defensive guard
            logger.warning("No se pudo persistir cache de cotizaciones: %s", exc)


def _warm_start_from_disk(now: float | None = None) -> tuple[int, float]:
    """Pre-hydrate the in-memory quote cache from persisted storage."""

    snapshot = _load_persisted_quotes()
    if not snapshot:
        return (0, 0.0)

    current_ts = float(now or time.time())
    ttl = max(
        min(float(cache_ttl_quotes or _WARM_START_TTL_SECONDS), _WARM_START_TTL_SECONDS),
        0.0,
    )
    if ttl == 0:
        ttl = _WARM_START_TTL_SECONDS

    loaded = 0
    total_age = 0.0

    with _QUOTE_LOCK:
        for cache_key, entry in snapshot.items():
            if not isinstance(entry, Mapping):
                continue
            data = entry.get("data")
            ts_value = _as_optional_float(entry.get("ts"))
            if not isinstance(data, dict) or ts_value is None:
                continue
            components = _normalize_bulk_key_components(cache_key)
            if components is None:
                continue
            market, symbol, panel_value = components
            normalized = _normalize_quote(data)
            record = {
                "ts": current_ts,
                "ttl": ttl,
                "data": normalized,
            }
            _QUOTE_CACHE[(market, symbol, panel_value)] = record
            loaded += 1
            total_age += max(current_ts - ts_value, 0.0)

    avg_age = total_age / loaded if loaded else 0.0
    return (loaded, avg_age)


def _load_persisted_entry(cache_key: str) -> tuple[dict[str, Any], float] | None:
    cache = _load_persisted_quotes()
    entry = cache.get(cache_key)
    if not isinstance(entry, dict):
        return None
    data = entry.get("data")
    ts_value = entry.get("ts")
    if not isinstance(data, dict):
        return None
    ts = _as_optional_float(ts_value)
    if ts is None:
        return None
    return dict(data), ts


def _recover_persisted_quote(cache_key: str, now: float) -> dict[str, Any] | None:
    persisted = _load_persisted_entry(cache_key)
    if persisted is None:
        return None
    cached_data, cached_ts = persisted
    ttl = QUOTE_STALE_TTL_SECONDS
    if ttl > 0:
        if now - cached_ts > ttl:
            return None
    elif ttl == 0:
        pass
    else:
        return None

    fallback_data = dict(_normalize_quote(cached_data))
    fallback_data["stale"] = True
    fallback_data.setdefault("provider", cached_data.get("provider") or "stale")
    if fallback_data.get("proveedor_original") is None:
        fallback_data["proveedor_original"] = fallback_data.get("provider")
    record_quote_provider_usage(
        fallback_data.get("provider") or "stale",
        elapsed_ms=None,
        stale=True,
        source="persistent",
    )
    return fallback_data


def _persist_quote(cache_key: str, payload: dict[str, Any], ts: float) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("last") is None:
        return
    normalized = _normalize_quote(payload)
    provider = None
    asof_value = None
    if isinstance(payload, dict):
        provider = payload.get("provider")
        asof_value = payload.get("asof")
        if normalized.get("moneda_origen") is None:
            currency_candidate = payload.get("moneda_origen")
            if currency_candidate is None:
                currency_candidate = payload.get("currency")
            if currency_candidate is None:
                currency_candidate = payload.get("moneda")
            if isinstance(currency_candidate, str):
                normalized["moneda_origen"] = currency_candidate.strip() or None
            elif currency_candidate is not None:
                normalized["moneda_origen"] = str(currency_candidate)
        if normalized.get("fx_aplicado") is None:
            fx_candidate = payload.get("fx_aplicado")
            if fx_candidate is None:
                fx_candidate = payload.get("fx_applied")
            if isinstance(fx_candidate, (int, float)):
                normalized["fx_aplicado"] = float(fx_candidate)
            elif isinstance(fx_candidate, str):
                normalized["fx_aplicado"] = fx_candidate.strip() or None
        if normalized.get("proveedor_original") is None:
            provider_original = payload.get("proveedor_original")
            if isinstance(provider_original, str):
                normalized["proveedor_original"] = provider_original.strip() or None
    if normalized.get("provider") is None and isinstance(provider, str):
        normalized["provider"] = provider or None
    if normalized.get("proveedor_original") is None and normalized.get("provider") is not None:
        normalized["proveedor_original"] = normalized.get("provider")
    if normalized.get("asof") is None and isinstance(asof_value, str):
        text = asof_value.strip()
        normalized["asof"] = text or None
    if normalized.get("last") is None:
        return
    entry = {"data": normalized, "ts": float(ts)}
    cache = _load_persisted_quotes()
    cache[cache_key] = entry
    _store_persisted_quotes(cache)


def _purge_expired_quotes(now: float, fallback_ttl: float) -> None:
    """Remove quote cache entries whose TTL has expired."""

    fallback = max(float(fallback_ttl), 0.0)
    if fallback == 0:
        _QUOTE_CACHE.clear()
        return

    expired_keys = []
    for cache_key, record in list(_QUOTE_CACHE.items()):
        record_ttl = record.get("ttl")
        if record_ttl is None:
            record_ttl = fallback
        try:
            record_ttl = float(record_ttl)
        except (TypeError, ValueError):
            record_ttl = fallback
        record["ttl"] = record_ttl
        ts = record.get("ts")
        if ts is None:
            ts_value = now
        else:
            try:
                ts_value = float(ts)
            except (TypeError, ValueError):
                ts_value = now
        record["ts"] = ts_value
        if record_ttl <= 0 or now - ts_value >= record_ttl:
            expired_keys.append(cache_key)

    for cache_key in expired_keys:
        _QUOTE_CACHE.pop(cache_key, None)


def _normalize_quote(raw: dict | None) -> dict:
    """Extract and compute basic quote information."""

    base = {
        "last": None,
        "chg_pct": None,
        "asof": None,
        "provider": None,
        "moneda_origen": None,
        "proveedor_original": None,
        "fx_aplicado": None,
    }
    if not isinstance(raw, dict) or not raw:
        return dict(base)

    data: dict[str, Any] = dict(base)

    provider_raw = raw.get("provider")
    if isinstance(provider_raw, str):
        provider = provider_raw.strip() or None
    elif provider_raw is None and raw.get("stale"):
        provider = "stale"
    elif provider_raw is None:
        provider = None
    else:
        provider = str(provider_raw)
    data["provider"] = provider

    provider_original_raw = raw.get("proveedor_original")
    if isinstance(provider_original_raw, str):
        provider_original = provider_original_raw.strip() or None
    elif provider_original_raw is None and isinstance(provider, str):
        provider_original = provider
    elif provider_original_raw is None:
        provider_original = None
    else:
        provider_original = str(provider_original_raw)
    data["proveedor_original"] = provider_original

    currency_raw = (
        raw.get("moneda_origen")
        if raw.get("moneda_origen") is not None
        else raw.get("currency")
        if raw.get("currency") is not None
        else raw.get("currency_base")
        if raw.get("currency_base") is not None
        else raw.get("moneda")
    )
    if isinstance(currency_raw, str):
        currency_value = currency_raw.strip() or None
    elif currency_raw is None:
        currency_value = None
    else:
        currency_value = str(currency_raw)
    data["moneda_origen"] = currency_value

    fx_raw = raw.get("fx_aplicado")
    if fx_raw is None:
        fx_raw = raw.get("fx_applied")
    if isinstance(fx_raw, (int, float)):
        fx_value: Any = float(fx_raw)
    elif isinstance(fx_raw, str):
        fx_value = fx_raw.strip() or None
    else:
        fx_value = None
    data["fx_aplicado"] = fx_value

    asof_value = raw.get("asof")
    if asof_value is None:
        for key in ("fecha", "fechaHora", "timestamp", "time", "date", "ts"):
            candidate = raw.get(key)
            if candidate is not None:
                asof_value = candidate
                break
    if hasattr(asof_value, "isoformat"):
        data["asof"] = asof_value.isoformat()
    elif isinstance(asof_value, (int, float)):
        data["asof"] = str(float(asof_value))
    elif isinstance(asof_value, str):
        text = asof_value.strip()
        data["asof"] = text or None

    raw_last = raw.get("last")
    if raw_last is None:
        last_value = IOLClient._parse_price_fields(raw)
    else:
        last_value = IOLClient._parse_price_fields({"last": raw_last})
        if last_value is None:
            try:
                last_value = float(raw_last)
            except (TypeError, ValueError):
                last_value = None
    data["last"] = last_value

    raw_chg = raw.get("chg_pct")
    if isinstance(raw_chg, (int, float)):
        chg_pct = float(raw_chg)
    elif isinstance(raw_chg, str):
        try:
            chg_pct = float(raw_chg.replace("%", "").strip())
        except (TypeError, ValueError):
            chg_pct = None
    else:
        chg_pct = None
    if chg_pct is None:
        chg_pct = IOLClient._parse_chg_pct_fields(raw, last_value)
    data["chg_pct"] = chg_pct

    return data


def _resolve_auth_ref(cli: Any):
    auth = getattr(cli, "auth", None)
    if auth is not None:
        return auth
    inner = getattr(cli, "_cli", None)
    if inner is not None:
        return getattr(inner, "auth", None)
    return None


def _get_quote_cached(
    cli: IIOLProvider,
    market: str,
    symbol: str,
    panel: str | None = None,
    ttl: int = cache_ttl_quotes,
    batch_stats: QuoteBatchStats | None = None,
) -> dict:
    norm_market = str(market).lower()
    norm_symbol = str(symbol).upper()
    cache_key = (norm_market, norm_symbol, panel)
    persist_key = _quote_cache_key(norm_market, norm_symbol, panel)
    try:
        ttl_seconds = float(ttl)
    except (TypeError, ValueError):
        ttl_seconds = float(cache_ttl_quotes or 0)
    if ttl_seconds < 0:
        ttl_seconds = 0.0
    now = time.time()
    fetch_start = now

    def _record_batch(payload: Any, *, elapsed_ms: float, fallback: bool | None = None) -> None:
        if batch_stats is None:
            return
        provider_name: str | None = None
        stale_flag = True
        if isinstance(payload, Mapping):
            provider_name = payload.get("provider")
            stale_flag = bool(payload.get("stale")) or payload.get("last") is None
        error_flag = (provider_name or "").strip().lower() == "error"
        if fallback is None:
            fallback_flag = stale_flag or (provider_name not in (None, "iol", "cache"))
        else:
            fallback_flag = fallback
        batch_stats.record_result(
            provider=provider_name,
            stale=stale_flag,
            error=error_flag,
            fallback=fallback_flag,
            elapsed_ms=elapsed_ms,
        )

    if ttl_seconds <= 0:
        with _QUOTE_LOCK:
            _QUOTE_CACHE.clear()
    else:
        with _QUOTE_LOCK:
            _purge_expired_quotes(now, ttl_seconds)
            rec = _QUOTE_CACHE.get(cache_key)
            if rec:
                try:
                    rec_ttl = float(rec.get("ttl", ttl_seconds))
                except (TypeError, ValueError):
                    rec_ttl = ttl_seconds
                    rec["ttl"] = rec_ttl
                ts = rec.get("ts", now)
                try:
                    ts_value = float(ts)
                except (TypeError, ValueError):
                    ts_value = now
                    rec["ts"] = ts_value
                if rec_ttl > 0 and now - ts_value < rec_ttl:
                    data = dict(rec.get("data", {}))
                    provider = data.get("provider") or "cache"
                    record_quote_provider_usage(
                        provider,
                        elapsed_ms=0.0,
                        stale=bool(data.get("stale")),
                        source="memory",
                    )
                    _record_batch(data, elapsed_ms=0.0, fallback=False)
                    return data
    provider_key = _resolve_rate_limit_provider(cli)
    q: dict[str, Any] | None = None
    data: dict[str, Any] | None = None
    last_error: Exception | None = None

    for attempt in range(_MAX_RATE_LIMIT_RETRIES + 1):
        quote_rate_limiter.wait_for_slot(provider_key)
        try:
            q = cli.get_quote(norm_market, norm_symbol, panel=panel) or {}
            data = _normalize_quote(q)
            break
        except InvalidCredentialsError:
            auth = _resolve_auth_ref(cli)
            if auth is not None:
                try:
                    auth.clear_tokens()
                except Exception:
                    pass
            _trigger_logout()
            q = {"provider": "error"}
            data = _normalize_quote(q)
            break
        except requests.HTTPError as http_exc:
            last_error = http_exc
            status_code = http_exc.response.status_code if http_exc.response is not None else None
            if status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                retry_after = _parse_retry_after_seconds(http_exc.response)
                wait_time = quote_rate_limiter.penalize(provider_key, minimum_wait=retry_after)
                if batch_stats is not None:
                    batch_stats.record_rate_limited()
                logger.info(
                    "Rate limited (429) %s:%s, waiting %.3fs before retry",
                    norm_market,
                    norm_symbol,
                    wait_time,
                )
                continue
            logger.warning(
                "get_quote HTTP error %s:%s -> %s",
                norm_market,
                norm_symbol,
                status_code or http_exc,
            )
            q = {"provider": "error"}
            data = _normalize_quote(q)
            break
        except Exception as e:
            last_error = e
            logger.warning("get_quote falló para %s:%s -> %s", norm_market, norm_symbol, e)
            q = {"provider": "error"}
            data = _normalize_quote(q)
            break

    if data is None:
        logger.warning(
            "get_quote no pudo recuperar datos para %s:%s (error=%s)",
            norm_market,
            norm_symbol,
            last_error,
        )
        q = {"provider": "error"}
        data = _normalize_quote(q)
    store_time = time.time()
    elapsed_ms = (store_time - fetch_start) * 1000.0

    stale = bool(q.get("stale")) if isinstance(q, dict) else False

    if data.get("provider") is None and not stale:
        data["provider"] = "iol"
    if data.get("proveedor_original") is None and data.get("provider") is not None:
        data["proveedor_original"] = data.get("provider")

    if ttl_seconds <= 0:
        provider_name = data.get("provider") or "unknown"
        _record_batch(data, elapsed_ms=elapsed_ms)
        record_quote_provider_usage(
            provider_name,
            elapsed_ms=elapsed_ms if data.get("last") is not None else None,
            stale=stale or data.get("last") is None,
            source="live" if elapsed_ms else "memory",
        )
        return data

    if data.get("last") is None:
        fallback_data = _recover_persisted_quote(persist_key, store_time)
        if fallback_data is not None:
            _record_batch(fallback_data, elapsed_ms=elapsed_ms, fallback=True)
            return fallback_data

    with _QUOTE_LOCK:
        _purge_expired_quotes(store_time, ttl_seconds)
        _QUOTE_CACHE[cache_key] = {"ts": store_time, "ttl": ttl_seconds, "data": data}
    if data.get("last") is not None and not stale:
        _persist_quote(persist_key, data, store_time)
    provider_name = data.get("provider") or "unknown"
    record_quote_provider_usage(
        provider_name,
        elapsed_ms=elapsed_ms if data.get("last") is not None else None,
        stale=stale or data.get("last") is None,
        source="live",
    )
    _record_batch(data, elapsed_ms=elapsed_ms)
    return data


@cache.cache_data(ttl=cache_ttl_quotes)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    items = list(items or [])
    telemetry: dict[str, object] = {
        "status": "success",
        "items": int(len(items)),
    }
    start = time.time()
    get_bulk = getattr(_cli, "get_quotes_bulk", None)
    fallback_mode = not callable(get_bulk)
    telemetry["fallback_mode"] = fallback_mode

    normalized: list[tuple[str, str, str | None]] = []

    global _QUOTE_WARM_START_APPLIED
    if not _QUOTE_WARM_START_APPLIED:
        warmed, avg_age = _warm_start_from_disk(start)
        _QUOTE_WARM_START_APPLIED = True
        if warmed:
            logger.info(
                "Warm-start applied from persisted cache (%s symbols, avg_age=%.1fs)",
                warmed,
                avg_age,
            )
            telemetry["warm_start_loaded"] = warmed
            telemetry["warm_start_avg_age_s"] = avg_age

    def _log_and_return(value, *, elapsed: float | None = None, subbatch_avg: float | None = None):
        duration = elapsed if elapsed is not None else (time.time() - start)
        try:
            log_default_telemetry(
                phase="quotes_refresh",
                elapsed_s=duration,
                dataset_hash=get_active_dataset_hash(),
                subbatch_avg_s=subbatch_avg,
            )
        except Exception:  # pragma: no cover - defensive safeguard
            logger.debug(
                "No se pudo registrar telemetría de quotes_refresh",
                exc_info=True,
            )
        return value

    with performance_timer("quotes_refresh", extra=telemetry):
        for raw in items:
            market: str | None = None
            symbol: str | None = None
            panel: str | None = None
            if isinstance(raw, dict):
                market = raw.get("market", raw.get("mercado"))
                symbol = raw.get("symbol", raw.get("simbolo"))
                panel = raw.get("panel")
            elif isinstance(raw, (list, tuple)):
                if len(raw) >= 2:
                    market = raw[0]
                    symbol = raw[1]
                if len(raw) >= 3:
                    panel = raw[2]
            else:
                market = getattr(raw, "market", getattr(raw, "mercado", None))
                symbol = getattr(raw, "symbol", getattr(raw, "simbolo", None))
                panel = getattr(raw, "panel", None)

            panel_value = None if panel is None else str(panel)
            norm_market = str(market or "bcba").lower()
            norm_symbol = str(symbol or "").upper()
            normalized.append((norm_market, norm_symbol, panel_value))

        batch_stats = QuoteBatchStats(total_expected=len(normalized))

        try:
            if callable(get_bulk):
                data = get_bulk(items)
                telemetry["mode"] = "bulk"
                if isinstance(data, dict):
                    normalized_bulk = {}
                    store_time = time.time()
                    try:
                        ttl_seconds = float(cache_ttl_quotes or 0)
                    except (TypeError, ValueError):
                        ttl_seconds = 0.0
                    if ttl_seconds < 0:
                        ttl_seconds = 0.0
                    if ttl_seconds > 0:
                        with _QUOTE_LOCK:
                            _purge_expired_quotes(store_time, ttl_seconds)
                    elapsed_ms = (store_time - start) * 1000.0
                    for k, v in data.items():
                        if v is None:
                            logger.warning(
                                "get_quotes_bulk returned empty entry for %s:%s",
                                k[0],
                                k[1],
                            )
                            continue
                        quote = _normalize_quote(v)
                        if isinstance(quote, dict):
                            stale_flag = bool(v.get("stale")) if isinstance(v, dict) else False
                            if stale_flag:
                                quote["stale"] = True
                            if quote.get("provider") is None and not stale_flag:
                                inferred_provider = _default_provider_for_client(_cli)
                                if inferred_provider is not None:
                                    quote["provider"] = inferred_provider
                                    if quote.get("proveedor_original") is None:
                                        quote["proveedor_original"] = inferred_provider

                            key_components = _normalize_bulk_key_components(k)
                            if key_components is None and len(normalized) == 1:
                                key_components = normalized[0]
                            if key_components is None:
                                normalized_bulk[k] = quote
                                provider_name = quote.get("provider") or "unknown"
                                record_quote_provider_usage(
                                    provider_name,
                                    elapsed_ms=elapsed_ms if quote.get("last") is not None else None,
                                    stale=stale_flag or quote.get("last") is None,
                                    source="bulk" if ttl_seconds > 0 else "live",
                                )
                                logger.debug("quote %s -> %s", k, quote)
                                continue

                            norm_market, norm_symbol, panel_value = key_components
                            persist_key = _quote_cache_key(norm_market, norm_symbol, panel_value)

                            if quote.get("last") is None:
                                fallback_quote = _recover_persisted_quote(persist_key, store_time)
                                if fallback_quote is not None:
                                    normalized_bulk[k] = fallback_quote
                                    continue

                            normalized_bulk[k] = quote
                            cache_key = (norm_market, norm_symbol, panel_value)
                            if ttl_seconds > 0:
                                with _QUOTE_LOCK:
                                    _QUOTE_CACHE[cache_key] = {
                                        "ts": store_time,
                                        "ttl": ttl_seconds,
                                        "data": quote,
                                    }
                                if quote.get("last") is not None and not stale_flag:
                                    _persist_quote(persist_key, quote, store_time)
                            provider_name = quote.get("provider") or "unknown"
                            record_quote_provider_usage(
                                provider_name,
                                elapsed_ms=elapsed_ms if quote.get("last") is not None else None,
                                stale=stale_flag or quote.get("last") is None,
                                source="bulk" if ttl_seconds > 0 else "live",
                            )
                            logger.debug("quote %s:%s -> %s", norm_market, norm_symbol, quote)
                    data = normalized_bulk
                if isinstance(data, Mapping):
                    for payload in data.values():
                        batch_stats.record_payload(payload)
                provider_stats = _extract_provider_batch_stats(_cli)
                if isinstance(provider_stats, Mapping):
                    batch_stats.apply_provider_stats(provider_stats)
                elapsed_seconds = time.time() - start
                elapsed_ms = elapsed_seconds * 1000.0
                summary = batch_stats.summary(elapsed_seconds)
                telemetry["processed"] = int(summary.get("count", 0))
                message = (
                    "✅ {count} quotes processed "
                    "(fresh={fresh}, stale={stale}, errors={errors}, "
                    "fallbacks={fallbacks}, rate_limited={rate_limited}) "
                    "in {elapsed:.3f}s (avg {avg:.3f}s, {qps:.2f} qps)"
                ).format(**summary)
                logger.info(message)
                record_quote_load(
                    elapsed_ms,
                    source="bulk" if not fallback_mode else "per-symbol",
                    count=len(items),
                )
                subbatch_avg = summary.get("avg") if isinstance(summary, Mapping) else None
                try:
                    subbatch_avg_value = float(subbatch_avg) if subbatch_avg is not None else None
                except (TypeError, ValueError):
                    subbatch_avg_value = None
                return _log_and_return(data, elapsed=elapsed_seconds, subbatch_avg=subbatch_avg_value)
        except InvalidCredentialsError:
            telemetry["status"] = "error"
            telemetry["detail"] = "auth"
            try:
                _cli._cli.auth.clear_tokens()
            except Exception:
                pass
            _trigger_logout()
            record_quote_load(None, source="auth-error", count=len(items))
            return _log_and_return({})
        except requests.RequestException as e:
            telemetry["status"] = "error"
            telemetry["detail"] = "network"
            logger.exception("get_quotes_bulk falló: %s", e)
            record_quote_load(None, source="error", count=len(items))
            try:
                log_default_telemetry(
                    phase="quotes_refresh",
                    elapsed_s=time.time() - start,
                    dataset_hash=get_active_dataset_hash(),
                )
            except Exception:  # pragma: no cover - defensive safeguard
                logger.debug(
                    "No se pudo registrar telemetría de quotes_refresh tras error",
                    exc_info=True,
                )
            raise NetworkError("Error de red al obtener cotizaciones") from e

        telemetry["mode"] = "threadpool"
        ttl = cache_ttl_quotes
        max_workers = max_quote_workers

        def _resolve_chunk_size() -> int:
            if not normalized:
                return _THREADPOOL_SUBLOT_MIN
            adaptive_size = _ADAPTIVE_BATCH_CONTROLLER.current(len(normalized))
            return max(1, min(adaptive_size, len(normalized)))

        chunk_size = _resolve_chunk_size()
        telemetry["sublot_size"] = chunk_size
        telemetry["adaptive_batch_size"] = chunk_size
        sublots = (
            [normalized[idx : idx + chunk_size] for idx in range(0, len(normalized), chunk_size)] if normalized else []
        )

        def _fetch_sublot(entries: list[tuple[str, str, str | None]]):
            started = time.perf_counter()
            results: dict[tuple[str, str], dict] = {}
            errors: list[tuple[str, str, Exception]] = []
            for market, symbol, panel in entries:
                try:
                    quote = _get_quote_cached(_cli, market, symbol, panel, ttl, batch_stats)
                except Exception as exc:  # pragma: no cover - network dependent
                    errors.append((market, symbol, exc))
                    quote = _normalize_quote({"provider": "error"})
                    batch_stats.record_payload(quote)
                else:
                    if isinstance(quote, dict):
                        logger.debug("quote %s:%s -> %s", market, symbol, quote)
            results[(market, symbol)] = quote
            duration = time.perf_counter() - started
            return results, errors, duration

        out: dict[tuple[str, str], dict] = {}
        worker_count = min(max_workers, len(sublots) or 1)
        batch_durations: list[float] = []
        with ThreadPoolExecutor(max_workers=worker_count) as ex:
            futures = {ex.submit(_fetch_sublot, sublot): sublot for sublot in sublots}
            for fut in as_completed(futures):
                sublot = futures[fut]
                try:
                    chunk_result, chunk_errors, duration = fut.result()
                except Exception as exc:  # pragma: no cover - defensive safeguard
                    telemetry.setdefault("errors", 0)
                    telemetry["errors"] = int(telemetry["errors"]) + len(sublot)
                    logger.exception(
                        "get_quote failed for sublote %s -> %s",
                        [(m, s) for m, s, _ in sublot],
                        exc,
                    )
                    for market, symbol, _panel in sublot:
                        quote = _normalize_quote({"provider": "error"})
                        out[(market, symbol)] = quote
                        batch_stats.record_payload(quote)
                    continue
                batch_durations.append(duration)
                out.update(chunk_result)
                for market, symbol, err in chunk_errors:
                    telemetry.setdefault("errors", 0)
                    telemetry["errors"] = int(telemetry["errors"]) + 1
                    logger.exception("get_quote failed for %s:%s -> %s", market, symbol, err)
        elapsed_seconds = time.time() - start
        elapsed_ms = elapsed_seconds * 1000.0
        summary = batch_stats.summary(elapsed_seconds)
        telemetry["processed"] = int(summary.get("count", 0))
        if batch_durations:
            avg_batch_time_ms = sum(batch_durations) / len(batch_durations) * 1000.0
            telemetry["avg_batch_time_ms"] = avg_batch_time_ms
            next_size = _ADAPTIVE_BATCH_CONTROLLER.observe(avg_batch_time_ms, len(normalized))
            telemetry["next_adaptive_batch_size"] = next_size
            logger.info(
                "quotes_refresh adaptive batch -> current=%s next=%s avg=%.2fms",
                chunk_size,
                next_size,
                avg_batch_time_ms,
            )
        else:
            _ADAPTIVE_BATCH_CONTROLLER.observe(None, len(normalized))
        message = (
            "✅ {count} quotes processed "
            "(fresh={fresh}, stale={stale}, errors={errors}, "
            "fallbacks={fallbacks}, rate_limited={rate_limited}) "
            "in {elapsed:.3f}s (avg {avg:.3f}s, {qps:.2f} qps)"
        ).format(**summary)
        logger.info(message)
        record_quote_load(
            elapsed_ms,
            source="fallback" if fallback_mode else "per-symbol",
            count=len(items),
        )
        subbatch_avg = None
        if batch_durations:
            try:
                subbatch_avg = sum(batch_durations) / len(batch_durations)
            except ZeroDivisionError:
                subbatch_avg = None
        elif isinstance(summary, Mapping):
            try:
                subbatch_avg = float(summary.get("avg"))
            except (TypeError, ValueError):
                subbatch_avg = None
        return _log_and_return(out, elapsed=elapsed_seconds, subbatch_avg=subbatch_avg)


__all__ = [
    "QuoteBatchStats",
    "_QUOTE_CACHE",
    "_QUOTE_LOCK",
    "_QUOTE_PERSIST_LOCK",
    "_QUOTE_PERSIST_CACHE",
    "QUOTE_STALE_TTL_SECONDS",
    "_QUOTE_PERSIST_PATH",
    "_MAX_RATE_LIMIT_RETRIES",
    "_quote_cache_key",
    "_normalize_bulk_key_components",
    "_default_provider_for_client",
    "_resolve_rate_limit_provider",
    "_extract_provider_batch_stats",
    "_parse_retry_after_seconds",
    "_as_optional_float",
    "_as_optional_int",
    "_load_persisted_quotes",
    "_store_persisted_quotes",
    "_warm_start_from_disk",
    "_load_persisted_entry",
    "_recover_persisted_quote",
    "_persist_quote",
    "_purge_expired_quotes",
    "_normalize_quote",
    "_resolve_auth_ref",
    "_get_quote_cached",
    "set_active_dataset_hash",
    "get_active_dataset_hash",
    "fetch_quotes_bulk",
]
