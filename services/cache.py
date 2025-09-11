import logging, time, hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from infrastructure.cache import cache
import requests

from infrastructure.iol.client import (
    IIOLProvider,
    build_iol_client as _build_iol_client,
)
from infrastructure.fx.provider import FXProviderAdapter
from shared.config import settings


logger = logging.getLogger(__name__)


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
# def fetch_quotes_bulk(cli: IIOLProvider, items):
#     get_bulk = getattr(cli, "get_quotes_bulk", None)
def fetch_quotes_bulk(_cli: IIOLProvider, items):
    get_bulk = getattr(_cli, "get_quotes_bulk", None)
    try:
        if callable(get_bulk):
            return get_bulk(items)
    except requests.RequestException as e:
        logger.exception("get_quotes_bulk falló: %s", e)
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
            except requests.RequestException as e:
                logger.exception("get_quote failed for %s:%s -> %s", key[0], key[1], e)
                out[key] = {"last": None, "chg_pct": None}
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
    last = cache.session_state.get("fx_rates_ts", 0)
    if "fx_rates" not in cache.session_state or now - last > ttl:
        data, error = fetch_fx_rates()
        cache.session_state["fx_rates"] = data
        cache.session_state["fx_rates_error"] = error
        cache.session_state["fx_rates_ts"] = now
    return (
        cache.session_state.get("fx_rates", {}),
        cache.session_state.get("fx_rates_error"),
    )


def build_iol_client() -> tuple[IIOLProvider | None, str | None]:
    if cache.session_state.get("force_login"):
        user = cache.session_state.get("IOL_USERNAME")
        password = cache.session_state.get("IOL_PASSWORD")
    else:
        user = cache.session_state.get("IOL_USERNAME") or settings.IOL_USERNAME
        password = cache.session_state.get("IOL_PASSWORD") or settings.IOL_PASSWORD
    salt = str(cache.session_state.get("client_salt", ""))
    tokens_file = settings.tokens_file
    cache_key = hashlib.sha256(
        f"{user}:{password}:{salt}:{tokens_file}".encode()
    ).hexdigest()
    try:
        cli = get_client_cached(cache_key, user, password, tokens_file)
        return cli, None
    except (requests.RequestException, RuntimeError, ValueError) as e:
        logger.exception("build_iol_client failed: %s", e)
        return None, str(e)


