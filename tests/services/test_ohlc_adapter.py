from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
from requests.exceptions import HTTPError

from services.ohlc_adapter import OHLCAdapter


class DummyResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise HTTPError(f"HTTP {self.status_code}")


class DummySession:
    def __init__(self, alpha_payload: object, polygon_payload: object) -> None:
        self.alpha_payload = alpha_payload
        self.polygon_payload = polygon_payload
        self.calls: list[str] = []

    def get(self, url: str, params=None, timeout: float | None = None):
        self.calls.append(url)
        payload = self.alpha_payload if "alpha" in url else self.polygon_payload
        if isinstance(payload, Exception):
            raise payload
        if isinstance(payload, dict):
            return DummyResponse(payload)
        raise RuntimeError("Unsupported payload type")


def _build_settings(**overrides) -> SimpleNamespace:
    base = {
        "OHLC_PRIMARY_PROVIDER": "alpha_vantage",
        "OHLC_SECONDARY_PROVIDERS": ["polygon"],
        "ALPHA_VANTAGE_API_KEY": "alpha-key",
        "POLYGON_API_KEY": "poly-key",
        "ALPHA_VANTAGE_BASE_URL": "https://alpha.test",
        "POLYGON_BASE_URL": "https://poly.test",
        "cache_ttl_yf_history": 60,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _polygon_payload() -> dict:
    return {
        "status": "OK",
        "results": [
            {
                "t": pd.Timestamp("2024-01-02", tz="UTC").value // 1_000_000,
                "o": 101.0,
                "h": 105.0,
                "l": 99.0,
                "c": 103.0,
                "v": 1_000_000,
            }
        ],
    }


def _alpha_payload() -> dict:
    return {
        "Time Series (Daily)": {
            "2024-01-02": {
                "1. open": "100.0",
                "2. high": "104.0",
                "3. low": "98.0",
                "4. close": "102.0",
                "5. adjusted close": "102.0",
                "6. volume": "1000000",
            }
        }
    }


def test_adapter_uses_secondary_when_primary_fails(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    settings = _build_settings()
    session = DummySession(alpha_payload={"Error Message": "throttled"}, polygon_payload=_polygon_payload())
    adapter = OHLCAdapter(settings_module=settings, session=session)

    frame = adapter.fetch("AAPL", period="1mo", interval="1d")
    assert not frame.empty
    assert frame.iloc[-1]["Close"] == pytest.approx(103.0)

    incidents = streamlit_stub.session_state.get("health_metrics", {}).get("market_data_incidents")
    assert incidents
    assert incidents[-1]["provider"] == "polygon"
    assert incidents[-2]["status"] == "error"


def test_adapter_serves_cache_when_all_providers_fail(monkeypatch: pytest.MonkeyPatch, streamlit_stub):
    settings = _build_settings()
    session = DummySession(alpha_payload=_alpha_payload(), polygon_payload={"status": "ERROR", "error": "bad"})
    adapter = OHLCAdapter(settings_module=settings, session=session, cache_ttl=120)

    initial = adapter.fetch("MSFT", period="1mo", interval="1d")
    assert not initial.empty

    session.alpha_payload = HTTPError("boom")
    session.polygon_payload = HTTPError("kaboom")

    cached = adapter.fetch("MSFT", period="1mo", interval="1d")
    pd.testing.assert_frame_equal(initial, cached)

    incidents = streamlit_stub.session_state.get("health_metrics", {}).get("market_data_incidents")
    assert incidents
    assert incidents[-1]["detail"] == "cache-hit"
    assert incidents[-1]["fallback"]
