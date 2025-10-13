"""Integration tests for the cache management endpoints."""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Generator

import pytest
pytest.importorskip("httpx")
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from api.main import app
from services.auth import generate_token
from services.cache.core import CacheService
from services.cache.market_data_cache import MarketDataCache
from shared.errors import CacheUnavailableError


_STRUCTURED_LOGGER_NAMES = {"performance", "performance_test"}


def _structured_logs(caplog: pytest.LogCaptureFixture) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for record in caplog.records:
        if record.name not in _STRUCTURED_LOGGER_NAMES:
            continue
        try:
            payload = json.loads(record.message)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            entries.append(payload)
    return entries


def _prepare_performance_logger(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    logger = logging.getLogger("performance_test")
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = True
    monkeypatch.setattr("api.routers.cache.PERF_LOGGER", logger)
    caplog.set_level(logging.INFO, logger="performance_test")
    caplog.set_level(logging.INFO, logger="performance")


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


def test_cache_invalidate_rejects_empty_pattern(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/cache/invalidate",
        json={"pattern": "   "},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid pattern"


def test_cache_invalidate_rejects_empty_keys_list(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    response = client.post(
        "/cache/invalidate",
        json={"keys": []},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "invalid pattern"


def test_cache_invalidate_enforces_max_keys(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    keys = [f"SYMBOL_{index}" for index in range(0, 501)]

    response = client.post(
        "/cache/invalidate",
        json={"keys": keys},
        headers=auth_headers,
    )

    assert response.status_code == 400
    assert "max keys exceeded" in response.json()["detail"]


def test_cache_metrics_exposed_after_operations(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
) -> None:
    prometheus_client = pytest.importorskip("prometheus_client")
    CollectorRegistry = prometheus_client.CollectorRegistry

    from api.routers import cache as cache_router
    from services import performance_timer as perf_timer

    perf_timer.PROMETHEUS_REGISTRY = CollectorRegistry(auto_describe=True)
    perf_timer.PROMETHEUS_ENABLED = True
    cache_router._CACHE_STATUS_COUNTER = None
    cache_router._CACHE_INVALIDATE_COUNTER = None
    cache_router._CACHE_CLEANUP_COUNTER = None
    cache_router._CACHE_OPERATION_DURATION = None

    response_status = client.get("/cache/status", headers=auth_headers)
    assert response_status.status_code == 200

    response_invalidate = client.post(
        "/cache/invalidate",
        json={"keys": ["MISSING"]},
        headers=auth_headers,
    )
    assert response_invalidate.status_code == 200

    response_cleanup = client.post("/cache/cleanup", headers=auth_headers)
    assert response_cleanup.status_code == 200

    metrics_response = client.get("/metrics")

    assert metrics_response.status_code == 200
    body = metrics_response.text
    assert "cache_status_requests_total{result=\"success\"}" in body
    assert "cache_invalidate_total{result=\"success\"}" in body
    assert "cache_cleanup_total{result=\"success\"}" in body
    assert "cache_operation_duration_seconds_sum" in body


def test_cache_cleanup_emits_structured_log(
    client: TestClient,
    auth_headers: dict[str, str],
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _prepare_performance_logger(caplog, monkeypatch)
    caplog.clear()

    response = client.post("/cache/cleanup", headers=auth_headers)

    assert response.status_code == 200
    entries = _structured_logs(caplog)
    assert entries, "Debe registrarse un log estructurado de performance"
    payload = entries[-1]
    assert payload.get("operation") == "cleanup"
    assert payload.get("success") is True
    assert payload.get("elapsed_s") is not None


def test_cache_status_handles_cache_unavailable(
    client: TestClient,
    auth_headers: dict[str, str],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _prepare_performance_logger(caplog, monkeypatch)
    caplog.clear()

    def _raise_cache_unavailable() -> MarketDataCache:
        raise CacheUnavailableError("backend down")

    monkeypatch.setattr("api.routers.cache._market_cache", _raise_cache_unavailable)

    response = client.get("/cache/status", headers=auth_headers)

    assert response.status_code == 500
    assert response.json()["detail"] == "El servicio de caché no está disponible"
    entries = _structured_logs(caplog)
    assert entries, "Debe registrarse un log estructurado en fallos"
    payload = entries[-1]
    assert payload.get("operation") == "status"
    assert payload.get("success") is False


def test_cache_invalidate_handles_sqlite_locked(
    client: TestClient,
    auth_headers: dict[str, str],
    market_cache: MarketDataCache,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _prepare_performance_logger(caplog, monkeypatch)
    caplog.clear()

    def _raise_locked(*args: object, **kwargs: object) -> int:
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(market_cache, "invalidate_matching", _raise_locked)

    response = client.post(
        "/cache/invalidate",
        json={"keys": ["GGAL"]},
        headers=auth_headers,
    )

    assert response.status_code == 500
    assert response.json()["detail"] == "Error inesperado invalidando la caché"
    entries = _structured_logs(caplog)
    assert entries, "El fallo debe registrar un log estructurado"
    payload = entries[-1]
    assert payload.get("operation") == "invalidate"
    assert payload.get("success") is False

