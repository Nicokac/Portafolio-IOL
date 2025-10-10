"""Caching, rate limiting and quote orchestration helpers."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from threading import Lock
from typing import Any, Callable, Dict, Mapping, Tuple
from uuid import uuid4

import requests
import streamlit as st

from shared.cache import cache
from shared.errors import ExternalAPIError, NetworkError, TimeoutError

from services.health import (
    record_fx_api_response,
    record_fx_cache_usage,
    record_iol_refresh,
    record_portfolio_load,
    record_quote_load,
    record_quote_provider_usage,
)

from infrastructure.iol.client import (
    IIOLProvider,
    IOLClient,
    build_iol_client as _build_iol_client,
)
from infrastructure.iol.auth import IOLAuth, InvalidCredentialsError
from infrastructure.fx.provider import FXProviderAdapter
from shared.settings import (
    cache_ttl_fx,
    cache_ttl_portfolio,
    cache_ttl_quotes,
    max_quote_workers,
    quotes_ttl_seconds,
    settings,
)
from services.quote_rate_limit import quote_rate_limiter


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float | None


class CacheService:
    """Thread-safe TTL cache used for offline fixtures and adapters."""

    def __init__(
        self,
        *,
        namespace: str | None = None,
        monotonic: Callable[[], float] | None = None,
        ttl_override: float | None = None,
    ) -> None:
        self._namespace = (namespace or "").strip()
        self._monotonic = monotonic or time.monotonic
        self._lock = Lock()
        self._store: Dict[str, _CacheEntry] = {}
        self._hits = 0
        self._misses = 0
        self._last_updated: datetime | None = None
        self._ttl_override = float(ttl_override) if ttl_override is not None else None
        self._last_ttl: float | None = None

    def _full_key(self, key: str) -> str:
        base_key = str(key)
        return f"{self._namespace}:{base_key}" if self._namespace else base_key

    def _is_expired(self, entry: _CacheEntry) -> bool:
        return entry.expires_at is not None and entry.expires_at <= self._monotonic()

    def get(self, key: str, default: Any = None) -> Any:
        full_key = self._full_key(key)
        with self._lock:
            entry = self._store.get(full_key)
            if entry is None:
                self._misses += 1
                return default
            if self._is_expired(entry):
                self._store.pop(full_key, None)
                self._misses += 1
                return default
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, *, ttl: float | None = None) -> Any:
        full_key = self._full_key(key)
        effective_ttl = self.get_effective_ttl(ttl)
        self._last_ttl = effective_ttl
        if effective_ttl is not None:
            effective_ttl = float(effective_ttl)
            if effective_ttl <= 0:
                with self._lock:
                    self._store.pop(full_key, None)
                return value
            expires_at = self._monotonic() + effective_ttl
        else:
            expires_at = None
        with self._lock:
            self._store[full_key] = _CacheEntry(value=value, expires_at=expires_at)
            self._last_updated = datetime.now(timezone.utc)
        return value

    def set_ttl_override(self, ttl_override: float | None) -> None:
        with self._lock:
            self._ttl_override = float(ttl_override) if ttl_override is not None else None

    def get_effective_ttl(self, ttl: float | None = None) -> float | None:
        if ttl is not None:
            ttl = float(ttl)
        if self._ttl_override is not None:
            return self._ttl_override
        if ttl is not None:
            return ttl
        return self._last_ttl

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    def get_or_set(
        self,
        key: str,
        loader: Callable[[], Any],
        *,
        ttl: float | None = None,
    ) -> Any:
        sentinel = object()
        cached_value = self.get(key, sentinel)
        if cached_value is not sentinel:
            return cached_value
        value = loader()
        self.set(key, value, ttl=ttl)
        return value

    def invalidate(self, key: str) -> None:
        full_key = self._full_key(key)
        with self._lock:
            self._store.pop(full_key, None)

    def clear(self) -> None:
        with self._lock:
            self._store.clear()
            self._hits = 0
            self._misses = 0
            self._last_updated = None
            self._last_ttl = None

    def hit_ratio(self) -> float:
        total = self._hits + self._misses
        if total <= 0:
            return 0.0
        return float(self._hits) / float(total) * 100.0

    @property
    def last_updated(self) -> datetime | None:
        return self._last_updated

    @property
    def last_updated_human(self) -> str:
        if self._last_updated is None:
            return "-"
        return self._last_updated.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

    def stats(self) -> Dict[str, Any]:
        """Expose cache statistics for reporting and diagnostics."""

        with self._lock:
            return {
                "namespace": self._namespace,
                "hits": self._hits,
                "misses": self._misses,
                "hit_ratio": self.hit_ratio(),
                "last_updated": self.last_updated_human,
            }



logger = logging.getLogger(__name__)


class RateLimiter:
    """Token bucket rate limiter used to throttle expensive operations."""

    def __init__(
        self,
        *,
        capacity: int,
        refill_rate: float,
        monotonic: Callable[[], float] | None = None,
        sleeper: Callable[[float], None] | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be greater than zero")
        if refill_rate <= 0:
            raise ValueError("refill_rate must be greater than zero")
        self._capacity = float(capacity)
        self._tokens = float(capacity)
        self._refill_rate = float(refill_rate)
        self._monotonic = monotonic or time.monotonic
        self._sleep = sleeper or time.sleep
        self._lock = Lock()
        self._last_refill = self._monotonic()

    def acquire(self, tokens: float = 1.0) -> None:
        """Block until ``tokens`` are available in the bucket."""

        if tokens <= 0:
            return

        request = float(tokens)
        while True:
            with self._lock:
                now = self._monotonic()
                elapsed = now - self._last_refill
                if elapsed > 0:
                    self._tokens = min(
                        self._capacity,
                        self._tokens + elapsed * self._refill_rate,
                    )
                    self._last_refill = now
                if self._tokens >= request:
                    self._tokens -= request
                    return
                needed = request - self._tokens
                wait_time = needed / self._refill_rate

            self._sleep(wait_time)


# In-memory quote cache
_QUOTE_CACHE: Dict[Tuple[str, str, str | None], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()
_QUOTE_PERSIST_LOCK = Lock()
_QUOTE_PERSIST_CACHE: Dict[str, Any] | None = None

QUOTE_STALE_TTL_SECONDS = float(quotes_ttl_seconds or 0)
_QUOTE_PERSIST_PATH = Path("data/quotes_cache.json")

_MAX_RATE_LIMIT_RETRIES = 2


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
        fallback_flag = fallback or (
            provider_key not in ("", "iol", "cache")
        ) or stale
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
    if normalized.get("provider") is None and isinstance(provider, str):
        normalized["provider"] = provider or None
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


def _trigger_logout() -> None:
    """Clear session and tokens triggering a fresh login."""
    try:
        from application import auth_service

        u = st.session_state.get("IOL_USERNAME", "")
        auth_service.logout(u)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("auto logout failed: %s", e)
        raise


def _normalize_quote(raw: dict | None) -> dict:
    """Extract and compute basic quote information."""

    base = {"last": None, "chg_pct": None, "asof": None, "provider": None}
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
    cli,
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
            data = {
                "last": None,
                "chg_pct": None,
                "asof": None,
                "provider": "error",
            }
            break
        except requests.HTTPError as http_exc:
            last_error = http_exc
            status_code = (
                http_exc.response.status_code if http_exc.response is not None else None
            )
            if status_code == 429 and attempt < _MAX_RATE_LIMIT_RETRIES:
                retry_after = _parse_retry_after_seconds(http_exc.response)
                wait_time = quote_rate_limiter.penalize(
                    provider_key, minimum_wait=retry_after
                )
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
            data = {
                "last": None,
                "chg_pct": None,
                "asof": None,
                "provider": "error",
            }
            break
        except Exception as e:
            last_error = e
            logger.warning("get_quote falló para %s:%s -> %s", norm_market, norm_symbol, e)
            q = {"provider": "error"}
            data = {
                "last": None,
                "chg_pct": None,
                "asof": None,
                "provider": "error",
            }
            break

    if data is None:
        logger.warning(
            "get_quote no pudo recuperar datos para %s:%s (error=%s)",
            norm_market,
            norm_symbol,
            last_error,
        )
        q = {"provider": "error"}
        data = {"last": None, "chg_pct": None, "asof": None, "provider": "error"}
    store_time = time.time()
    elapsed_ms = (store_time - fetch_start) * 1000.0

    stale = bool(q.get("stale")) if isinstance(q, dict) else False

    if data.get("provider") is None and not stale:
        data["provider"] = "iol"

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


@cache.cache_resource
def get_client_cached(
    cache_key: str, user: str, tokens_file: Path | str | None
) -> IIOLProvider:
    auth = IOLAuth(
        user,
        "",
        tokens_file=tokens_file,
        allow_plain_tokens=settings.allow_plain_tokens,
    )
    try:
        auth.refresh()
        record_iol_refresh(True)
    except InvalidCredentialsError as e:
        auth.clear_tokens()
        st.session_state["force_login"] = True
        record_iol_refresh(False, detail="Credenciales inválidas")
        raise e
    except Exception as e:
        record_iol_refresh(False, detail=e)
        raise
    return _build_iol_client(user, "", tokens_file=tokens_file, auth=auth)


@cache.cache_data(ttl=cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    tokens_path = getattr(getattr(_cli, "auth", None), "tokens_path", None)
    try:
        data = _cli.get_portfolio()
    except InvalidCredentialsError:
        auth = _resolve_auth_ref(_cli)
        if auth is not None:
            try:
                auth.clear_tokens()
            except Exception:
                pass
        _trigger_logout()
        logger.info(
            "fetch_portfolio using cache due to invalid credentials",
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="cache", detail="invalid-credentials")
        return {"_cached": True}
    except requests.Timeout as e:
        logger.info(
            "fetch_portfolio failed due to network timeout: %s",
            e,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="timeout")
        raise TimeoutError("Error de red al consultar el portafolio") from e
    except requests.RequestException as e:
        logger.info(
            "fetch_portfolio failed due to network error: %s",
            e,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="network-error")
        raise NetworkError("Error de red al consultar el portafolio") from e
    elapsed = (time.time() - start) * 1000
    record_portfolio_load(elapsed, source="api")
    log = logger.warning if elapsed > 600 else logger.info
    log(
        "fetch_portfolio done in %.0fms",
        elapsed,
        extra={"tokens_file": tokens_path},
    )
    return data


@cache.cache_data(ttl=cache_ttl_quotes)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    items = list(items or [])
    start = time.time()
    get_bulk = getattr(_cli, "get_quotes_bulk", None)
    fallback_mode = not callable(get_bulk)

    normalized: list[tuple[str, str, str | None]] = []

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
                            "get_quotes_bulk returned empty entry for %s:%s", k[0], k[1]
                        )
                        continue
                    quote = _normalize_quote(v)
                    if isinstance(quote, dict):
                        stale_flag = (
                            bool(v.get("stale")) if isinstance(v, dict) else False
                        )
                        if stale_flag:
                            quote["stale"] = True
                        if quote.get("provider") is None and not stale_flag:
                            inferred_provider = _default_provider_for_client(_cli)
                            if inferred_provider is not None:
                                quote["provider"] = inferred_provider

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
                            fallback_quote = _recover_persisted_quote(
                                persist_key, store_time
                            )
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
                        logger.debug(
                            "quote %s:%s -> %s", norm_market, norm_symbol, quote
                        )
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
            message = (
                "✅ {count} quotes processed "
                "(fresh={fresh}, stale={stale}, errors={errors}, "
                "fallbacks={fallbacks}, rate_limited={rate_limited}) "
                "in {elapsed:.3f}s (avg {avg:.3f}s, {qps:.2f} qps)"
            ).format(**summary)
            logger.info(message)
            record_quote_load(elapsed_ms, source="bulk", count=len(items))
            return data
    except InvalidCredentialsError:
        try:
            _cli._cli.auth.clear_tokens()
        except Exception:
            pass
        _trigger_logout()
        record_quote_load(None, source="auth-error", count=len(items))
        return {}
    except requests.RequestException as e:
        logger.exception("get_quotes_bulk falló: %s", e)
        record_quote_load(None, source="error", count=len(items))
        raise NetworkError("Error de red al obtener cotizaciones") from e

    out = {}
    ttl = cache_ttl_quotes
    max_workers = max_quote_workers
    with ThreadPoolExecutor(max_workers=min(max_workers, len(normalized) or 1)) as ex:
        futs = {
            ex.submit(
                _get_quote_cached, _cli, market, symbol, panel, ttl, batch_stats
            ): (market, symbol)
            for market, symbol, panel in normalized
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                quote = fut.result()
            except Exception as e:
                logger.exception(
                    "get_quote failed for %s:%s -> %s", key[0], key[1], e
                )
                quote = {"last": None, "chg_pct": None, "asof": None, "provider": "error"}
                batch_stats.record_payload(quote)
            if isinstance(quote, dict):
                logger.debug("quote %s:%s -> %s", key[0], key[1], quote)
            out[key] = quote
    elapsed_seconds = time.time() - start
    elapsed_ms = elapsed_seconds * 1000.0
    summary = batch_stats.summary(elapsed_seconds)
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
    return out


@cache.cache_resource
def get_fx_provider() -> FXProviderAdapter:
    return FXProviderAdapter()


@cache.cache_data(ttl=cache_ttl_fx)
def fetch_fx_rates():
    data: dict = {}
    error: str | None = None
    start = time.time()
    provider: FXProviderAdapter | None = None
    try:
        provider = get_fx_provider()
        data, error = provider.get_rates()
    except requests.RequestException as e:
        error = f"FX provider failed: {e}"
        logger.exception(error)
        raise ExternalAPIError(error) from e
    except RuntimeError as e:
        error = f"FX provider failed: {e}"
        logger.exception(error)
    finally:
        if provider is not None:
            provider.close()
        record_fx_api_response(
            error=error,
            elapsed_ms=(time.time() - start) * 1000,
        )
    return data, error


def get_fx_rates_cached():
    now = time.time()
    ttl = cache_ttl_fx
    last = st.session_state.get("fx_rates_ts", 0)
    if "fx_rates" not in st.session_state or now - last > ttl:
        data, error = fetch_fx_rates()
        st.session_state["fx_rates"] = data
        st.session_state["fx_rates_error"] = error
        st.session_state["fx_rates_ts"] = now
        record_fx_cache_usage("refresh", age=0.0)
    else:
        age = now - last if last else None
        record_fx_cache_usage("hit", age=age)
    return (
        st.session_state.get("fx_rates", {}),
        st.session_state.get("fx_rates_error"),
    )


def build_iol_client(
    user: str | None = None,
) -> tuple[IIOLProvider | None, Exception | None]:
    user = user or st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
    if not user:
        return None, RuntimeError("missing user")
    if "client_salt" not in st.session_state:
        st.session_state["client_salt"] = uuid4().hex
    salt = str(st.session_state.get("client_salt", ""))
    tokens_file = cache.get("tokens_file")
    if not tokens_file:
        sanitized = re.sub(r"[^A-Za-z0-9_-]", "_", user)
        user_hash = hashlib.sha256(user.encode()).hexdigest()[:12]
        tokens_file = Path("tokens") / f"{sanitized}-{user_hash}.json"
        cache.set("tokens_file", str(tokens_file))
    cache_key = hashlib.sha256(
        f"{tokens_file}:{salt}".encode()
    ).hexdigest()
    st.session_state["cache_key"] = cache_key
    try:
        cli = get_client_cached(cache_key, user, tokens_file)
        return cli, None
    except InvalidCredentialsError as e:
        _trigger_logout()
        return None, e
    except Exception as e:
        logger.exception("build_iol_client failed: %s", e)
        return None, e


