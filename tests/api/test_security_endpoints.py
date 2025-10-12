"""Integration tests for secured API endpoints."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from services import auth as auth_module
from services.auth import ACTIVE_TOKENS, MAX_TOKEN_TTL_SECONDS, generate_token


@pytest.fixture(autouse=True)
def _configure_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", key)
    monkeypatch.delenv("IOL_TOKENS_KEY", raising=False)
    ACTIVE_TOKENS.clear()
    yield
    ACTIVE_TOKENS.clear()


@pytest.fixture()
def client() -> TestClient:
    from api.main import app

    with TestClient(app) as test_client:
        yield test_client


def test_profile_requires_authentication(client: TestClient) -> None:
    response = client.get("/profile/summary")
    assert response.status_code == 401


def test_cache_endpoint_disabled(client: TestClient) -> None:
    response = client.get("/cache/")
    assert response.status_code == 404


def test_refresh_endpoint_enforces_window(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    base_time = 1_700_100_000
    monkeypatch.setattr(auth_module.time, "time", lambda: base_time)
    token = generate_token("tester", expiry=MAX_TOKEN_TTL_SECONDS)

    # With more than five minutes remaining the refresh is rejected.
    monkeypatch.setattr(
        auth_module.time, "time", lambda: base_time + MAX_TOKEN_TTL_SECONDS - 360
    )
    response = client.post("/auth/refresh", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401


def test_refresh_endpoint_rotates_token(monkeypatch: pytest.MonkeyPatch, client: TestClient) -> None:
    base_time = 1_700_200_000
    monkeypatch.setattr(auth_module.time, "time", lambda: base_time)
    token = generate_token("tester", expiry=MAX_TOKEN_TTL_SECONDS)
    headers = {"Authorization": f"Bearer {token}"}

    # Move into the refresh window and request a new token.
    monkeypatch.setattr(
        auth_module.time, "time", lambda: base_time + MAX_TOKEN_TTL_SECONDS - 120
    )
    response = client.post("/auth/refresh", headers=headers)
    assert response.status_code == 200
    payload = response.json()

    assert payload["session_id"]
    assert payload["expires_in"] == MAX_TOKEN_TTL_SECONDS

    new_token = payload["access_token"]
    assert new_token != token

    # Old token should now be invalid while the new one grants access.
    old_response = client.get(
        "/profile/summary", headers={"Authorization": f"Bearer {token}"}
    )
    assert old_response.status_code == 401

    profile_response = client.get(
        "/profile/summary", headers={"Authorization": f"Bearer {new_token}"}
    )
    assert profile_response.status_code == 200
