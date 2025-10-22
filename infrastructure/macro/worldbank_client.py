"""HTTP client for the World Bank open data API."""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional

import requests

from infrastructure.http.session import build_session

from .fred_client import (
    MacroAPIError,
    MacroAuthenticationError,
    MacroRateLimitError,
    MacroSeriesObservation,
)
from .rate_limiter import RateLimiter


class WorldBankClient:
    """Dedicated HTTP client for the World Bank macro indicators API."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: str = "https://api.worldbank.org/v2",
        session: Optional[requests.Session] = None,
        calls_per_minute: int = 60,
        user_agent: Optional[str] = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._session = session or build_session(user_agent or "Portafolio-IOL/1.0")
        self._rate_limiter = RateLimiter(calls_per_minute=calls_per_minute, monotonic=monotonic, sleeper=sleeper)

    # Public API -----------------------------------------------------------
    def get_latest_observation(
        self, indicator: str, *, params: Optional[Mapping[str, Any]] = None
    ) -> Optional[MacroSeriesObservation]:
        """Return the most recent valid observation for ``indicator``."""

        payload = self._request_json(
            f"country/all/indicator/{indicator}",
            {
                "format": "json",
                "per_page": 5,
                "date": params.get("date") if params else None,
            },
        )

        if not isinstance(payload, Iterable):
            return None

        # The World Bank API typically returns a list: [metadata, [observations...]]
        observations = None
        for part in payload:
            if isinstance(part, Iterable) and not isinstance(part, (str, bytes, bytearray)):
                observations = part
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
            as_of = str(item.get("date") or item.get("observation_date") or "").strip()
            if not as_of:
                continue
            return MacroSeriesObservation(series_id=indicator, value=value, as_of=as_of)
        return None

    def get_latest_observations(self, indicators_map: Mapping[str, str]) -> Dict[str, MacroSeriesObservation]:
        """Return observations for the provided mapping of label -> indicator."""

        results: Dict[str, MacroSeriesObservation] = {}
        for label, indicator in indicators_map.items():
            if not indicator:
                continue
            observation = self.get_latest_observation(indicator)
            if observation is None:
                continue
            results[label] = observation
        return results

    # Internal helpers ----------------------------------------------------
    def _request_json(self, endpoint: str, params: Optional[MutableMapping[str, Any]] = None) -> Any:
        url = f"{self._base_url}/{endpoint.lstrip('/')}"
        query: Dict[str, Any] = {}
        headers: Dict[str, str] = {}
        if params:
            for key, value in params.items():
                if value is None:
                    continue
                query[key] = value
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        self._rate_limiter.acquire()
        response = self._session.get(url, params=query, headers=headers)
        status = response.status_code
        if status in {401, 403}:
            raise MacroAuthenticationError("World Bank API rejected the provided credentials")
        if status == 429:
            raise MacroRateLimitError("World Bank API rate limit exceeded")
        if status >= 500:
            raise MacroAPIError(f"World Bank API returned {status}")
        if status >= 400:
            detail = self._extract_error_detail(response)
            raise MacroAPIError(f"World Bank API error {status}: {detail}")
        try:
            data = response.json()
        except (ValueError, json.JSONDecodeError) as exc:
            raise MacroAPIError("Invalid JSON response from World Bank") from exc
        return data

    @staticmethod
    def _extract_error_detail(response: requests.Response) -> str:
        try:
            data = response.json()
        except Exception:  # pragma: no cover - best effort logging
            return response.text or "unknown error"
        if isinstance(data, Iterable):
            for part in data:
                if isinstance(part, Mapping):
                    message = part.get("message") or part.get("error")
                    if message:
                        if isinstance(message, Mapping):
                            text = message.get("value")
                            if text:
                                return str(text)
                        return str(message)
        if isinstance(data, Mapping):
            message = data.get("message")
            if isinstance(message, Mapping):
                text = message.get("value")
                if text:
                    return str(text)
            if message:
                return str(message)
        return response.text or "unknown error"


__all__ = ["WorldBankClient"]
