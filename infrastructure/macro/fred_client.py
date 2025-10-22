"""HTTP client for the Federal Reserve Economic Data (FRED) API.

The client centralizes authentication, error handling and rate limiting so the
rest of the application can focus on business logic. It only exposes the
minimal surface required for the opportunities screener enrichment but is
extensible for future indicators.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import requests

from infrastructure.http.session import build_session

from .rate_limiter import RateLimiter


class MacroAPIError(RuntimeError):
    """Base class for FRED related failures."""


class MacroAuthenticationError(MacroAPIError):
    """Raised when the provided credentials are rejected by the API."""


class MacroRateLimitError(MacroAPIError):
    """Raised when the upstream service signals that the rate limit was hit."""


@dataclass(frozen=True)
class MacroSeriesObservation:
    """Represents the most recent observation available for a series."""

    series_id: str
    value: float
    as_of: str


# Backwards compatibility for legacy imports expecting ``FredSeriesObservation``
FredSeriesObservation = MacroSeriesObservation


class FredClient:
    """Dedicated HTTP client for the FRED API."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = "https://api.stlouisfed.org/fred",
        session: Optional[requests.Session] = None,
        calls_per_minute: int = 120,
        user_agent: Optional[str] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required to talk with FRED")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = session or build_session(user_agent or "Portafolio-IOL/1.0")
        self._rate_limiter = RateLimiter(calls_per_minute=calls_per_minute, monotonic=monotonic, sleeper=sleeper)

    # Public API -----------------------------------------------------------
    def get_latest_observation(
        self, series_id: str, *, params: Optional[Mapping[str, Any]] = None
    ) -> Optional[MacroSeriesObservation]:
        """Return the most recent valid observation for ``series_id``."""

        payload = self._request_json(
            "series/observations",
            {
                "series_id": series_id,
                "sort_order": "desc",
                "limit": 5,
                "observation_start": params.get("observation_start") if params else None,
                "observation_end": params.get("observation_end") if params else None,
            },
        )
        observations = payload.get("observations")
        if not isinstance(observations, Iterable):
            return None
        for item in observations:
            if not isinstance(item, Mapping):
                continue
            raw_value = item.get("value")
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                continue
            as_of = str(item.get("observation_date") or item.get("date") or "").strip()
            if not as_of:
                continue
            return MacroSeriesObservation(series_id=series_id, value=value, as_of=as_of)
        return None

    def get_latest_observations(self, series_map: Mapping[str, str]) -> Dict[str, MacroSeriesObservation]:
        """Return observations for the provided mapping of label -> series ID."""

        results: Dict[str, MacroSeriesObservation] = {}
        for label, series_id in series_map.items():
            if not series_id:
                continue
            observation = self.get_latest_observation(series_id)
            if observation is None:
                continue
            results[label] = observation
        return results

    # Internal helpers ----------------------------------------------------
    def _request_json(self, endpoint: str, params: Optional[MutableMapping[str, Any]] = None) -> Mapping[str, Any]:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        query: Dict[str, Any] = {"file_type": "json", "api_key": self._api_key}
        if params:
            for key, value in params.items():
                if value is None:
                    continue
                query[key] = value
        self._rate_limiter.acquire()
        response = self._session.get(url, params=query)
        status = response.status_code
        if status == 401 or status == 403:
            raise MacroAuthenticationError("FRED API rejected the provided credentials")
        if status == 429:
            raise MacroRateLimitError("FRED API rate limit exceeded")
        if status >= 500:
            raise MacroAPIError(f"FRED API returned {status}")
        if status >= 400:
            detail = self._extract_error_detail(response)
            raise MacroAPIError(f"FRED API error {status}: {detail}")
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise MacroAPIError("Invalid JSON response from FRED") from exc
        if not isinstance(data, Mapping):
            raise MacroAPIError("Unexpected payload type from FRED")
        return data

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        try:
            data = response.json()
        except Exception:  # pragma: no cover - best effort logging
            return response.text or "unknown error"
        if isinstance(data, Mapping):
            message = data.get("error_message") or data.get("message")
            if message:
                return str(message)
        return response.text or "unknown error"
