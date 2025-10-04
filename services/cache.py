import json
import logging, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Tuple, Any
import re
from uuid import uuid4

from shared.cache import cache
from shared.errors import ExternalAPIError, NetworkError, TimeoutError
import requests
import streamlit as st

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
    settings,
)


logger = logging.getLogger(__name__)


# In-memory quote cache
_QUOTE_CACHE: Dict[Tuple[str, str, str | None], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()
_QUOTE_PERSIST_LOCK = Lock()
_QUOTE_PERSIST_CACHE: Dict[str, Any] | None = None

QUOTE_STALE_TTL_SECONDS = 300.0
_QUOTE_PERSIST_PATH = Path("data/quotes_cache.json")


def _quote_cache_key(market: str, symbol: str, panel: str | None) -> str:
    panel_token = "" if panel in (None, "") else str(panel)
    return "|".join([market, symbol, panel_token])


def _as_optional_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
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


def _persist_quote(cache_key: str, payload: dict[str, Any], ts: float) -> None:
    if not isinstance(payload, dict):
        return
    if payload.get("last") is None:
        return
    normalized = _normalize_quote(payload)
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
                    return data
    q: dict[str, Any] | None = None
    try:
        q = cli.get_quote(norm_market, norm_symbol, panel=panel) or {}
        data = _normalize_quote(q)
    except InvalidCredentialsError:
        auth = _resolve_auth_ref(cli)
        if auth is not None:
            try:
                auth.clear_tokens()
            except Exception:
                pass
        _trigger_logout()
        q = {"provider": "error"}
        data = {"last": None, "chg_pct": None, "asof": None, "provider": "error"}
    except Exception as e:
        logger.warning("get_quote falló para %s:%s -> %s", norm_market, norm_symbol, e)
        q = {"provider": "error"}
        data = {"last": None, "chg_pct": None, "asof": None, "provider": "error"}
    store_time = time.time()
    elapsed_ms = (store_time - fetch_start) * 1000.0

    stale = bool(q.get("stale")) if isinstance(q, dict) else False

    if data.get("provider") is None and not stale:
        data["provider"] = "iol"

    if ttl_seconds <= 0:
        provider_name = data.get("provider") or "unknown"
        record_quote_provider_usage(
            provider_name,
            elapsed_ms=elapsed_ms if data.get("last") is not None else None,
            stale=stale or data.get("last") is None,
            source="live" if elapsed_ms else "memory",
        )
        return data

    if data.get("last") is None:
        persisted = _load_persisted_entry(persist_key)
        if persisted is not None:
            cached_data, cached_ts = persisted
            if QUOTE_STALE_TTL_SECONDS > 0:
                if store_time - cached_ts <= QUOTE_STALE_TTL_SECONDS:
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
            elif QUOTE_STALE_TTL_SECONDS == 0:
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

    try:
        if callable(get_bulk):
            data = get_bulk(items)
            if isinstance(data, dict):
                normalized_bulk = {}
                for k, v in data.items():
                    if v is None:
                        logger.warning(
                            "get_quotes_bulk returned empty entry for %s:%s", k[0], k[1]
                        )
                        continue
                    quote = _normalize_quote(v)
                    if isinstance(quote, dict):
                        normalized_bulk[k] = quote
                        logger.debug("quote %s:%s -> %s", k[0], k[1], quote)
                data = normalized_bulk
            elapsed = (time.time() - start) * 1000
            log = logger.warning if elapsed > 1000 else logger.info
            log("fetch_quotes_bulk done in %.0fms (%d items)", elapsed, len(items))
            record_quote_load(elapsed, source="bulk", count=len(items))
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

    with ThreadPoolExecutor(max_workers=min(max_workers, len(normalized) or 1)) as ex:
        futs = {
            ex.submit(_get_quote_cached, _cli, market, symbol, panel, ttl): (market, symbol)
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
            if isinstance(quote, dict):
                logger.debug("quote %s:%s -> %s", key[0], key[1], quote)
            out[key] = quote
    elapsed = (time.time() - start) * 1000
    log = logger.warning if elapsed > 1000 else logger.info
    log("fetch_quotes_bulk done in %.0fms (%d items)", elapsed, len(items))
    record_quote_load(
        elapsed,
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


