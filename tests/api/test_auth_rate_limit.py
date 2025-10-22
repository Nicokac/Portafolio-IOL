"""Tests for refresh endpoint rate limiting and anti-replay guards."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from api.middleware.refresh_rate_limit import REFRESH_RATE_LIMITER
from services import auth as auth_module
from services.auth import (
    ACTIVE_TOKENS,
    MAX_TOKEN_TTL_SECONDS,
    USED_REFRESH_NONCES,
    generate_token,
    verify_token,
)
from tests.fixtures.time import FakeTime


@pytest.fixture(autouse=True)
def _configure_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", key)
    monkeypatch.setenv("IOL_TOKENS_KEY", Fernet.generate_key().decode())
    ACTIVE_TOKENS.clear()
    USED_REFRESH_NONCES.clear()
    original_time_source = REFRESH_RATE_LIMITER.time_source
    REFRESH_RATE_LIMITER.reset()
    yield
    ACTIVE_TOKENS.clear()
    USED_REFRESH_NONCES.clear()
    REFRESH_RATE_LIMITER.reset()
    REFRESH_RATE_LIMITER.set_time_source(original_time_source)


@pytest.fixture()
def client() -> TestClient:
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def fake_time(monkeypatch: pytest.MonkeyPatch) -> FakeTime:
    controller = FakeTime(2_000_000_000)
    monkeypatch.setattr(auth_module.time, "time", controller)
    REFRESH_RATE_LIMITER.set_time_source(controller)
    return controller


def _advance_into_window(fake_time: FakeTime, claims: dict[str, int]) -> None:
    fake_time.set(claims["exp"] - 120)


def test_refresh_rate_limiter_blocks_after_threshold(fake_time: FakeTime, client: TestClient) -> None:
    headers = {"Authorization": "Bearer invalid-token"}

    for _ in range(10):
        response = client.post("/auth/refresh", headers=headers)
        assert response.status_code == 401

    throttled = client.post("/auth/refresh", headers=headers)
    assert throttled.status_code == 429
    assert throttled.json()["detail"].startswith("Too many refresh attempts")
    assert REFRESH_RATE_LIMITER.total_attempts == 11


def test_replay_is_rejected_with_nonce_error(fake_time: FakeTime, client: TestClient) -> None:
    token = generate_token("nonce-user", expiry=MAX_TOKEN_TTL_SECONDS)
    claims = verify_token(token)
    _advance_into_window(fake_time, claims)

    headers = {"Authorization": f"Bearer {token}"}
    response = client.post("/auth/refresh", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    replay_response = client.post("/auth/refresh", headers=headers)
    assert replay_response.status_code == 401
    assert "replay" in replay_response.json()["detail"].lower()

    fake_time.advance(MAX_TOKEN_TTL_SECONDS - 120)
    headers = {"Authorization": f"Bearer {payload['access_token']}"}
    second = client.post("/auth/refresh", headers=headers)
    assert second.status_code == 200
