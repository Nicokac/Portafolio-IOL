"""Unified OHLC data adapter with Alpha Vantage and Polygon fallbacks."""
from __future__ import annotations

import datetime as dt
import logging
from typing import Any, Dict, Mapping, Sequence

import pandas as pd
import requests

from services.adapters.base_adapter import AdapterProvider, BaseMarketDataAdapter
from shared.settings import settings


logger = logging.getLogger(__name__)


_DEFAULT_ALPHA_URL = "https://www.alphavantage.co/query"
_DEFAULT_POLYGON_URL = "https://api.polygon.io"
_REQUEST_TIMEOUT = 30


def _parse_period_days(period: str | None) -> int:
    mapping = {
        "1mo": 31,
        "3mo": 93,
        "6mo": 186,
        "1y": 372,
        "2y": 744,
        "5y": 1860,
        "10y": 3720,
    }
    if not period:
        return 372
    normalized = period.lower().strip()
    if normalized in mapping:
        return mapping[normalized]
    if normalized.endswith("d"):
        try:
            return max(int(normalized[:-1]), 1)
        except (TypeError, ValueError):
            return 372
    if normalized.endswith("mo"):
        try:
            months = int(normalized[:-2])
            return max(months * 31, 31)
        except (TypeError, ValueError):
            return 372
    if normalized.endswith("y"):
        try:
            years = int(normalized[:-1])
            return max(years * 372, 372)
        except (TypeError, ValueError):
            return 372
    return mapping.get(normalized, 372)


def _parse_interval(interval: str | None) -> tuple[int, str]:
    if not interval:
        return 1, "day"
    normalized = interval.lower().strip()
    lookup: Dict[str, tuple[int, str]] = {
        "1m": (1, "minute"),
        "5m": (5, "minute"),
        "15m": (15, "minute"),
        "30m": (30, "minute"),
        "60m": (60, "minute"),
        "1h": (60, "minute"),
        "1d": (1, "day"),
        "1wk": (1, "week"),
        "1mo": (1, "month"),
    }
    return lookup.get(normalized, (1, "day"))


class OHLCAdapter(BaseMarketDataAdapter):
    """Adapter that requests OHLC data from configured providers."""

    output_columns = ("Open", "High", "Low", "Close", "Volume")

    def __init__(
        self,
        *,
        settings_module=settings,
        session: requests.Session | None = None,
        cache_ttl: float | None = None,
    ) -> None:
        self._settings = settings_module
        self._session = session or requests.Session()
        cache_seconds = (
            cache_ttl
            if cache_ttl is not None
            else getattr(settings_module, "cache_ttl_yf_history", None)
        )
        providers = self._build_providers()
        super().__init__(
            providers=providers,
            cache_ttl=cache_seconds,
            incident_source="ohlc",
        )

    # ------------------------------------------------------------------
    def empty_payload(self) -> pd.DataFrame:  # pragma: no cover - trivial
        return pd.DataFrame(columns=self.output_columns)

    def _normalize_payload(self, payload: Any, provider: str) -> pd.DataFrame:
        frame = super()._normalize_payload(payload, provider)
        missing = [col for col in self.output_columns if col not in frame.columns]
        if missing:
            raise ValueError(f"Provider {provider} missing columns: {missing}")
        return frame[list(self.output_columns)].sort_index()

    # ------------------------------------------------------------------
    def _build_providers(self) -> Sequence[AdapterProvider]:
        order: list[str] = []
        primary = getattr(self._settings, "OHLC_PRIMARY_PROVIDER", "alpha_vantage")
        if primary:
            order.append(str(primary).strip().lower())
        for name in getattr(self._settings, "OHLC_SECONDARY_PROVIDERS", []) or []:
            provider_name = str(name or "").strip().lower()
            if provider_name and provider_name not in order:
                order.append(provider_name)
        if not order:
            order = ["alpha_vantage", "polygon"]
        providers: list[AdapterProvider] = []
        for name in order:
            if name == "alpha_vantage":
                providers.append(
                    AdapterProvider(name="alpha_vantage", fetcher=self._alpha_vantage_fetch)
                )
            elif name == "polygon":
                providers.append(
                    AdapterProvider(name="polygon", fetcher=self._polygon_fetch)
                )
        return tuple(providers)

    # ------------------------------------------------------------------
    def _alpha_vantage_fetch(
        self, symbol: str, params: Mapping[str, Any]
    ) -> pd.DataFrame:
        api_key = getattr(self._settings, "ALPHA_VANTAGE_API_KEY", None)
        if not api_key:
            raise RuntimeError("Alpha Vantage API key is not configured")
        base_url = getattr(self._settings, "ALPHA_VANTAGE_BASE_URL", _DEFAULT_ALPHA_URL)
        interval = str(params.get("interval") or "1d")
        period = str(params.get("period") or "6mo")
        outputsize = "full" if _parse_period_days(period) > 100 else "compact"
        function = "TIME_SERIES_DAILY_ADJUSTED"
        if interval.endswith("m"):
            step = interval.rstrip("m")
            function = "TIME_SERIES_INTRADAY"
        query = {
            "function": function,
            "symbol": symbol,
            "apikey": api_key,
            "outputsize": outputsize,
        }
        if function == "TIME_SERIES_INTRADAY":
            query["interval"] = f"{step}min"
        response = self._session.get(base_url, params=query, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if "Error Message" in payload:
            raise RuntimeError(payload["Error Message"])
        if "Note" in payload:
            raise RuntimeError(payload["Note"])
        series_key = next((k for k in payload if "Time Series" in k), None)
        if not series_key:
            raise ValueError("Alpha Vantage response missing time series")
        series = payload.get(series_key) or {}
        if not isinstance(series, Mapping) or not series:
            raise ValueError("Alpha Vantage time series empty")
        records: list[dict[str, Any]] = []
        for date_str, values in series.items():
            if not isinstance(values, Mapping):
                continue
            ts = pd.to_datetime(date_str)
            row = {
                "Open": pd.to_numeric(values.get("1. open"), errors="coerce"),
                "High": pd.to_numeric(values.get("2. high"), errors="coerce"),
                "Low": pd.to_numeric(values.get("3. low"), errors="coerce"),
                "Close": pd.to_numeric(values.get("4. close"), errors="coerce"),
                "Volume": pd.to_numeric(values.get("6. volume"), errors="coerce"),
            }
            if any(pd.isna(v) for v in row.values()):
                continue
            records.append({"timestamp": ts, **row})
        if not records:
            raise ValueError("Alpha Vantage response contains no valid entries")
        frame = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
        return frame

    # ------------------------------------------------------------------
    def _polygon_fetch(self, symbol: str, params: Mapping[str, Any]) -> pd.DataFrame:
        api_key = getattr(self._settings, "POLYGON_API_KEY", None)
        if not api_key:
            raise RuntimeError("Polygon API key is not configured")
        base_url = getattr(self._settings, "POLYGON_BASE_URL", _DEFAULT_POLYGON_URL)
        interval = params.get("interval")
        period = params.get("period")
        multiplier, timespan = _parse_interval(interval)
        end = dt.datetime.now(dt.timezone.utc).date()
        start = end - dt.timedelta(days=_parse_period_days(str(period)))
        path = f"/v2/aggs/ticker/{symbol}/range/{multiplier}/{timespan}/{start}/{end}"
        url = base_url.rstrip("/") + path
        query = {
            "adjusted": "true",
            "sort": "asc",
            "limit": 5000,
            "apiKey": api_key,
        }
        response = self._session.get(url, params=query, timeout=_REQUEST_TIMEOUT)
        response.raise_for_status()
        payload = response.json()
        if payload.get("status") == "ERROR":
            raise RuntimeError(payload.get("error", "Polygon error"))
        results = payload.get("results") or []
        if not results:
            raise ValueError("Polygon response empty")
        records: list[dict[str, Any]] = []
        for item in results:
            ts_raw = item.get("t")
            if ts_raw is None:
                continue
            ts = pd.to_datetime(ts_raw, unit="ms", utc=True).tz_convert(None)
            row = {
                "Open": pd.to_numeric(item.get("o"), errors="coerce"),
                "High": pd.to_numeric(item.get("h"), errors="coerce"),
                "Low": pd.to_numeric(item.get("l"), errors="coerce"),
                "Close": pd.to_numeric(item.get("c"), errors="coerce"),
                "Volume": pd.to_numeric(item.get("v"), errors="coerce"),
            }
            if any(pd.isna(v) for v in row.values()):
                continue
            records.append({"timestamp": ts, **row})
        if not records:
            raise ValueError("Polygon response contains no valid entries")
        frame = pd.DataFrame.from_records(records).set_index("timestamp").sort_index()
        return frame


_default_adapter: OHLCAdapter | None = None


def get_ohlc_adapter() -> OHLCAdapter:
    """Return a shared :class:`OHLCAdapter` instance."""

    global _default_adapter
    if _default_adapter is None:
        _default_adapter = OHLCAdapter()
    return _default_adapter


def set_ohlc_adapter(adapter: OHLCAdapter | None) -> None:
    """Override the shared adapter (used in tests)."""

    global _default_adapter
    _default_adapter = adapter


__all__ = ["OHLCAdapter", "get_ohlc_adapter", "set_ohlc_adapter"]
