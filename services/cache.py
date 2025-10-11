"""Caching, rate limiting and quote orchestration helpers."""

from __future__ import annotations

import logging
import time
from threading import Lock
from typing import Callable

import requests

from infrastructure.fx.provider import FXProviderAdapter
from infrastructure.iol.auth import InvalidCredentialsError
from infrastructure.iol.client import IIOLProvider
from services.cache.core import CacheService, PredictiveCacheState
from services.cache.quotes import *  # noqa: F401,F403
from services.cache.ui_adapter import *  # noqa: F401,F403
from services.health import record_fx_api_response, record_portfolio_load
from shared.cache import cache
from shared.errors import ExternalAPIError, NetworkError, TimeoutError
from shared.settings import cache_ttl_fx, cache_ttl_portfolio


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
    except requests.RequestException as exc:
        error = f"FX provider failed: {exc}"
        logger.exception(error)
        raise ExternalAPIError(error) from exc
    except RuntimeError as exc:
        error = f"FX provider failed: {exc}"
        logger.exception(error)
    finally:
        if provider is not None:
            provider.close()
        record_fx_api_response(
            error=error,
            elapsed_ms=(time.time() - start) * 1000,
        )
    return data, error


