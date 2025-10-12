"""Integration tests for the cache management endpoints."""

from __future__ import annotations

from collections.abc import Generator

import pytest
pytest.importorskip("httpx")
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from api.main import app
from services.auth import generate_token
from services.cache.core import CacheService
from services.cache.market_data_cache import MarketDataCache


class FakeClock:
    def __init__(self) -> None:
        self._now = 0.0

    def __call__(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)


@pytest.fixture()
def auth_headers(monkeypatch: pytest.MonkeyPatch) -> dict[str, str]:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key)
    token = generate_token("tester", expiry=600)
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def client() -> Generator[TestClient, None, None]:
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def market_cache(monkeypatch: pytest.MonkeyPatch) -> MarketDataCache:
    clock = FakeClock()
    history = CacheService(namespace="cache_history", monotonic=clock)
    fundamentals = CacheService(namespace="cache_fund", monotonic=clock)
    predictions = CacheService(namespace="cache_pred", monotonic=clock)
    cache = MarketDataCache(
        history_cache=history,
        fundamentals_cache=fundamentals,
        prediction_cache=predictions,
        default_ttl=120.0,
    )
    cache._clock = clock  # type: ignore[attr-defined]
    monkeypatch.setattr("api.routers.cache.get_market_data_cache", lambda: cache)
    return cache


def test_cache_status_requires_auth(client: TestClient) -> None:
    response = client.get("/cache/status")
    assert response.status_code == 401


def test_cache_status_returns_summary(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
) -> None:
    market_cache.history_cache.set("AAPL", {"close": 10.5}, ttl=120.0)
    _ = market_cache.history_cache.get("AAPL")
    _ = market_cache.fundamentals_cache.get("MISSING")

    response = client.get("/cache/status", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["total_entries"] == 1
    assert pytest.approx(payload["hit_ratio"], rel=1e-3) == 50.0
    assert pytest.approx(payload["avg_ttl_seconds"], rel=1e-3) == 120.0
    assert payload["size_mb"] > 0


def test_cache_invalidate_by_keys(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
) -> None:
    market_cache.history_cache.set("AAPL", {"close": 11.0}, ttl=60.0)
    market_cache.history_cache.set("MSFT", {"close": 9.0}, ttl=60.0)
    market_cache.fundamentals_cache.set("AAPL", {"metric": 1}, ttl=60.0)

    response = client.post(
        "/cache/invalidate",
        json={"keys": ["AAPL"]},
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed"] == 2
    assert payload["elapsed_s"] >= 0
    assert market_cache.history_cache.get("AAPL") is None
    assert market_cache.fundamentals_cache.get("AAPL") is None


def test_cache_invalidate_by_pattern(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
) -> None:
    market_cache.history_cache.set("TECH_AAPL", 1, ttl=60.0)
    market_cache.history_cache.set("TECH_MSFT", 1, ttl=60.0)
    market_cache.prediction_cache.set("OTHER_SYMBOL", 1, ttl=60.0)

    response = client.post(
        "/cache/invalidate",
        json={"pattern": "TECH_*"},
        headers=auth_headers,
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["removed"] == 2
    assert market_cache.history_cache.get("TECH_AAPL") is None
    assert market_cache.history_cache.get("TECH_MSFT") is None
    assert market_cache.prediction_cache.get("OTHER_SYMBOL") == 1


def test_cache_cleanup_removes_expired_and_orphans(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
) -> None:
    market_cache.history_cache.set("SHORT", 1, ttl=10.0)
    clock = getattr(market_cache, "_clock")
    assert clock is not None
    clock.advance(20.0)

    store = getattr(market_cache.history_cache, "_store")
    lock = getattr(market_cache.history_cache, "_lock")
    full_key = market_cache.history_cache._full_key("ORPHAN")
    if lock is not None:
        with lock:
            store[full_key] = object()
    else:
        store[full_key] = object()

    response = client.post("/cache/cleanup", headers=auth_headers)

    assert response.status_code == 200
    payload = response.json()
    assert payload["expired_removed"] == 1
    assert payload["orphans_removed"] == 1
    assert payload["elapsed_s"] >= 0

