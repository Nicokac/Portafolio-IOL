"""Portfolio caching helpers extracted from the legacy cache module."""

from __future__ import annotations

import logging
import time

import requests

from infrastructure.iol.auth import InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider
from services.cache.quotes import _resolve_auth_ref
from services.cache.ui_adapter import _trigger_logout
from services.health import record_portfolio_load
from shared.cache import cache
from shared.errors import NetworkError, TimeoutError
from shared.settings import cache_ttl_portfolio

__all__ = ["fetch_portfolio"]

logger = logging.getLogger(__name__)


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
    except requests.Timeout as exc:
        logger.info(
            "fetch_portfolio failed due to network timeout: %s",
            exc,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="timeout")
        raise TimeoutError("Error de red al consultar el portafolio") from exc
    except requests.RequestException as exc:
        logger.info(
            "fetch_portfolio failed due to network error: %s",
            exc,
            extra={"tokens_file": tokens_path},
        )
        record_portfolio_load(None, source="error", detail="network-error")
        raise NetworkError("Error de red al consultar el portafolio") from exc

    elapsed = (time.time() - start) * 1000
    record_portfolio_load(elapsed, source="api")
    log = logger.warning if elapsed > 600 else logger.info
    log(
        "fetch_portfolio done in %.0fms",
        elapsed,
        extra={"tokens_file": tokens_path},
    )
    return data
