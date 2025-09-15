import logging, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import Dict, Tuple, Any
import re
from uuid import uuid4

from shared.cache import cache
import requests
import streamlit as st

from infrastructure.iol.client import (
    IIOLProvider,
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
_QUOTE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()

def _trigger_logout() -> None:
    """Clear session and tokens triggering a fresh login."""
    try:
        from application import auth_service

        u = st.session_state.get("IOL_USERNAME", "")
        auth_service.logout(u)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("auto logout failed: %s", e)
        raise


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


def _get_quote_cached(
    cli, mercado: str, simbolo: str, ttl: int = cache_ttl_quotes
) -> dict:
    key = (str(mercado).lower(), str(simbolo).upper())
    now = time.time()
    with _QUOTE_LOCK:
        rec = _QUOTE_CACHE.get(key)
        if rec and now - rec["ts"] < ttl:
            return rec["data"]
    try:
        q = cli.get_quote(mercado=key[0], simbolo=key[1]) or {}
        data = _normalize_quote(q)
    except InvalidCredentialsError:
        try:
            cli._cli.auth.clear_tokens()
        except Exception:
            pass
        _trigger_logout()
        data = {"last": None, "chg_pct": None}
    except Exception as e:
        logger.warning("get_quote falló para %s:%s -> %s", mercado, simbolo, e)
        data = {"last": None, "chg_pct": None}
    with _QUOTE_LOCK:
        _QUOTE_CACHE[key] = {"ts": now, "data": data}
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
    except InvalidCredentialsError as e:
        auth.clear_tokens()
        st.session_state["force_login"] = True
        raise e
    return _build_iol_client(user, "", tokens_file=tokens_file, auth=auth)


@cache.cache_data(ttl=cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    tokens_path = getattr(getattr(_cli, "auth", None), "tokens_path", None)
    try:
        data = _cli.get_portfolio()
    except InvalidCredentialsError:
        try:
            _cli._cli.auth.clear_tokens()
        except Exception:
            pass
        _trigger_logout()
        logger.info(
            "fetch_portfolio using cache due to invalid credentials",
            extra={"tokens_file": tokens_path},
        )
        return {"_cached": True}
    except requests.RequestException as e:
        logger.info(
            "fetch_portfolio using cache due to network error: %s",
            e,
            extra={"tokens_file": tokens_path},
        )
        return {"_cached": True}
    elapsed = (time.time() - start) * 1000
    log = logger.warning if elapsed > 600 else logger.info
    log(
        "fetch_portfolio done in %.0fms",
        elapsed,
        extra={"tokens_file": tokens_path},
    )
    return data


@cache.cache_data(ttl=cache_ttl_quotes)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    start = time.time()
    get_bulk = getattr(_cli, "get_quotes_bulk", None)

    try:
        if callable(get_bulk):
            data = get_bulk(items)
            if isinstance(data, dict):
                for k, v in data.items():
                    data[k] = _normalize_quote(v)
                    logger.debug("quote %s:%s -> %s", k[0], k[1], data[k])
            elapsed = (time.time() - start) * 1000
            log = logger.warning if elapsed > 1000 else logger.info
            log("fetch_quotes_bulk done in %.0fms (%d items)", elapsed, len(items))
            return data
    except InvalidCredentialsError:
        try:
            _cli._cli.auth.clear_tokens()
        except Exception:
            pass
        _trigger_logout()
        return {}
    except requests.RequestException as e:
        logger.exception("get_quotes_bulk falló: %s", e)

    out = {}
    ttl = cache_ttl_quotes
    max_workers = max_quote_workers
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
    elapsed = (time.time() - start) * 1000
    log = logger.warning if elapsed > 1000 else logger.info
    log("fetch_quotes_bulk done in %.0fms (%d items)", elapsed, len(items))
    return out


@cache.cache_resource
def get_fx_provider() -> FXProviderAdapter:
    return FXProviderAdapter()


@cache.cache_data(ttl=cache_ttl_fx)
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
    ttl = cache_ttl_fx
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


