"""FX provider helpers extracted from the legacy cache module."""

from __future__ import annotations

import logging
import time

import requests

from infrastructure.fx.provider import FXProviderAdapter
from services.health import record_fx_api_response
from shared.cache import cache
from shared.errors import ExternalAPIError
from shared.settings import cache_ttl_fx

__all__ = ["get_fx_provider", "fetch_fx_rates"]

logger = logging.getLogger(__name__)


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
