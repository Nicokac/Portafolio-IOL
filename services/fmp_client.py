"""Client helpers for Financial Modeling Prep (FMP) API access."""
from __future__ import annotations

import logging
from functools import lru_cache
from typing import Any, Mapping

import requests

from shared.settings import fmp_api_key, fmp_base_url, fmp_timeout

logger = logging.getLogger(__name__)


class FinancialModelingPrepClient:
    """Thin HTTP client to interact with Financial Modeling Prep endpoints."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout: float | None = None,
        session: requests.Session | None = None,
    ) -> None:
        self.api_key = api_key if api_key is not None else fmp_api_key
        self.base_url = (base_url if base_url is not None else fmp_base_url).rstrip("/")
        self.timeout = timeout if timeout is not None else fmp_timeout
        self._session = session or requests.Session()

    def _request(self, endpoint: str) -> Any:
        if not self.api_key:
            logger.debug("FMP API key is not configured; skipping request to %s", endpoint)
            return {}
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        try:
            response = self._session.get(
                url,
                params={"apikey": self.api_key},
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:  # pragma: no cover - logging only
            logger.warning("FMP request failed (%s): %s", endpoint, exc)
            return {}
        try:
            return response.json()
        except ValueError:
            logger.warning("FMP response for %s is not valid JSON", endpoint)
            return {}

    @staticmethod
    def _first_entry(payload: Any) -> Mapping[str, Any]:
        if isinstance(payload, list):
            return payload[0] if payload else {}
        if isinstance(payload, Mapping):
            return payload
        return {}

    def get_ratios_ttm(self, symbol: str) -> Mapping[str, Any]:
        """Fetch trailing twelve month ratios for the given symbol."""
        payload = self._request(f"ratios-ttm/{symbol}")
        return self._first_entry(payload)

    def get_key_metrics_ttm(self, symbol: str) -> Mapping[str, Any]:
        """Fetch trailing twelve month key metrics for the given symbol."""
        payload = self._request(f"key-metrics-ttm/{symbol}")
        return self._first_entry(payload)


@lru_cache(maxsize=1)
def get_fmp_client() -> FinancialModelingPrepClient:
    """Return a cached instance of the FMP client."""
    return FinancialModelingPrepClient()


__all__ = ["FinancialModelingPrepClient", "get_fmp_client"]
