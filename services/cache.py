import logging, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st

from infrastructure.iol.client import (
    IIOLProvider,
    build_iol_client as _build_iol_client,
)
from infrastructure.fx.provider import FXProviderAdapter
from shared.config import settings


logger = logging.getLogger(__name__)


@st.cache_resource
def get_client_cached(
    cache_key: str, user: str, password: str, tokens_file: Path | str | None
) -> IIOLProvider:
    _ = cache_key
    return _build_iol_client(user, password, tokens_file=tokens_file)


@st.cache_data(ttl=settings.cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    data = _cli.get_portfolio()
    logger.info("fetch_portfolio done in %.0fms", (time.time() - start) * 1000)
    return data


@st.cache_data(ttl=settings.cache_ttl_quotes)
# def fetch_quotes_bulk(cli: IIOLProvider, items):
#     get_bulk = getattr(cli, "get_quotes_bulk", None)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    get_bulk = getattr(_cli, "get_quotes_bulk", None)
    try:
        if callable(get_bulk):
            return get_bulk(items)
    except Exception as e:
        logger.warning("get_quotes_bulk falló: %s", e)
    out = {}
    max_workers = getattr(settings, "max_quote_workers", 12)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(items) or 1)) as ex:
        futs = {
            # ex.submit(cli.get_quote, mercado=m, simbolo=s): (str(m).lower(), str(s).upper())
            ex.submit(_cli.get_quote, mercado=m, simbolo=s): (str(m).lower(), str(s).upper())
            for m, s in items
        }
        for fut in as_completed(futs):
            key = futs[fut]
            try:
                out[key] = fut.result()
            except Exception as e:
                logger.warning("get_quote failed for %s:%s -> %s", key[0], key[1], e)
                out[key] = {"last": None, "chg_pct": None}
    return out


@st.cache_resource
def get_fx_provider() -> FXProviderAdapter:
    return FXProviderAdapter()


@st.cache_data(ttl=settings.cache_ttl_fx)
def fetch_fx_rates():
    data: dict = {}
    error: str | None = None
    try:
        data, error = get_fx_provider().get_rates()
    except Exception as e:
        error = f"FX provider failed: {e}"
        logger.warning(error)
    if error:
        st.warning(error)
    return data


def get_fx_rates_cached():
    now = time.time()
    ttl = getattr(settings, "cache_ttl_fx", 0)
    last = st.session_state.get("fx_rates_ts", 0)
    if "fx_rates" not in st.session_state or now - last > ttl:
        st.session_state["fx_rates"] = fetch_fx_rates()
        st.session_state["fx_rates_ts"] = now
    return st.session_state.get("fx_rates", {})


def build_iol_client() -> IIOLProvider:
    if st.session_state.get("force_login"):
        user = st.session_state.get("IOL_USERNAME")
        password = st.session_state.get("IOL_PASSWORD")
    else:
        user = st.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
        password = st.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD
    salt = str(st.session_state.get("client_salt", ""))
    tokens_file = settings.tokens_file
    cache_key = hashlib.sha256(
        f"{user}:{password}:{salt}:{tokens_file}".encode()
    ).hexdigest()
    try:
        return get_client_cached(cache_key, user, password, tokens_file)
    except Exception as e:
        logger.exception("build_iol_client failed: %s", e)
        st.session_state["login_error"] = str(e)
        st.session_state["force_login"] = True
        st.session_state["IOL_PASSWORD"] = ""
        st.rerun()
        return None


