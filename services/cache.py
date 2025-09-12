import logging, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Tuple, Any

from shared.cache import cache
import requests
import streamlit as st

from infrastructure.iol.client import (
    IIOLProvider,
    build_iol_client as _build_iol_client,
)
from infrastructure.fx.provider import FXProviderAdapter
from shared.config import settings


logger = logging.getLogger(__name__)


# In-memory quote cache
_QUOTE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()


def _normalize_quote(raw: dict) -> dict:
    """Extract and compute basic quote information."""
    data = {"last": raw.get("last"), "chg_pct": raw.get("chg_pct")}
    if data.get("chg_pct") is None:
        try:
            u = float(raw.get("ultimo"))
            c = float(raw.get("cierreAnterior"))
            if c:
                data["chg_pct"] = (u - c) / c * 100.0
        except (TypeError, ValueError):
            pass
    return data


def _get_quote_cached(cli, mercado: str, simbolo: str, ttl: int = 8) -> dict:
    key = (str(mercado).lower(), str(simbolo).upper())
    now = time.time()
    with _QUOTE_LOCK:
        rec = _QUOTE_CACHE.get(key)
        if rec and now - rec["ts"] < ttl:
            return rec["data"]
    try:
        q = cli.get_quote(mercado=key[0], simbolo=key[1]) or {}
        data = _normalize_quote(q)
    except Exception as e:
        logger.warning("get_quote falló para %s:%s -> %s", mercado, simbolo, e)
        data = {"last": None, "chg_pct": None}
    with _QUOTE_LOCK:
        _QUOTE_CACHE[key] = {"ts": now, "data": data}
    return data


@cache.cache_resource
def get_client_cached(
    cache_key: str, user: str, password: str, tokens_file: Path | str | None
) -> IIOLProvider:
    _ = cache_key
    return _build_iol_client(user, password, tokens_file=tokens_file)


@cache.cache_data(ttl=settings.cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    data = _cli.get_portfolio()
    logger.info("fetch_portfolio done in %.0fms", (time.time() - start) * 1000)
    return data


@cache.cache_data(ttl=settings.cache_ttl_quotes)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    get_bulk = getattr(_cli, "get_quotes_bulk", None)

    try:
        if callable(get_bulk):
            data = get_bulk(items)
            if isinstance(data, dict):
                for k, v in data.items():
                    data[k] = _normalize_quote(v)
                    logger.debug("quote %s:%s -> %s", k[0], k[1], data[k])
            return data
    except requests.RequestException as e:
        logger.exception("get_quotes_bulk falló: %s", e)

    out = {}
    ttl = settings.cache_ttl_quotes
    max_workers = getattr(settings, "max_quote_workers", 12)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items) or 1)) as ex:
        futs = {
            ex.submit(_get_quote_cached, _cli, m, s, ttl): (
                str(m).lower(),
                str(s).upper(),
            )
            for m, s in items
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                quote = fut.result()
            except Exception as e:
                logger.exception(
                    "get_quote failed for %s:%s -> %s", key[0], key[1], e
                )
                quote = {"last": None, "chg_pct": None}
            logger.debug("quote %s:%s -> %s", key[0], key[1], quote)
            out[key] = quote
    return out


@cache.cache_resource
def get_fx_provider() -> FXProviderAdapter:
    return FXProviderAdapter()


@cache.cache_data(ttl=settings.cache_ttl_fx)
def fetch_fx_rates():
    data: dict = {}
    error: str | None = None
    try:
        data, error = get_fx_provider().get_rates()
    except (requests.RequestException, RuntimeError) as e:
        error = f"FX provider failed: {e}"
        logger.exception(error)
    return data, error


def get_fx_rates_cached():
    now = time.time()
    ttl = getattr(settings, "cache_ttl_fx", 0)
    last = st.session_state.get("fx_rates_ts", 0)
    if "fx_rates" not in st.session_state or now - last > ttl:
        data, error = fetch_fx_rates()
        st.session_state["fx_rates"] = data
        st.session_state["fx_rates_error"] = error
        st.session_state["fx_rates_ts"] = now
    return (
        st.session_state.get("fx_rates", {}),
        st.session_state.get("fx_rates_error"),
    )


def build_iol_client() -> tuple[IIOLProvider | None, str | None]:
    if st.session_state.get("force_login"):
        user = st.session_state.get("IOL_USERNAME")
        password = st.session_state.get("IOL_PASSWORD")
    else:
        user = st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
        password = st.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD
    salt = str(st.session_state.get("client_salt", ""))
    tokens_file = cache.get("tokens_file")
    if not tokens_file:
        tokens_file = Path("tokens") / f"{user}.json"
        cache.set("tokens_file", str(tokens_file))
    cache_key = hashlib.sha256(
        f"{user}:{password}:{salt}:{tokens_file}".encode()
    ).hexdigest()
    try:
        cli = get_client_cached(cache_key, user, password, tokens_file)
        return cli, None
    except (requests.RequestException, RuntimeError, ValueError) as e:
        logger.exception("build_iol_client failed: %s", e)
        return None, str(e)


