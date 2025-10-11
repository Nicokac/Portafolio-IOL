"""Integration tests covering FastAPI authentication requirements."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet
from fastapi.testclient import TestClient

from api.main import app
from services.auth import generate_token


@pytest.fixture()
def auth_env(monkeypatch: pytest.MonkeyPatch) -> str:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key)
    return key


@pytest.fixture()
def client(auth_env: str) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_predict_requires_authorization(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"opportunities": []},
    )

    assert response.status_code == 401
    assert response.json()["detail"]


def test_predict_accepts_valid_token(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    auth_env: str,
) -> None:
    monkeypatch.setattr(
        "api.routers.predictive.predict_sector_performance",
        lambda *_args, **_kwargs: [],
    )
    headers = {"Authorization": f"Bearer {generate_token('tester', expiry=120)}"}

    response = client.post(
        "/predict",
        json={"opportunities": []},
        headers=headers,
    )

    assert response.status_code == 200
    assert response.json()["predictions"] == []


def test_cache_status_rejects_invalid_token(
    monkeypatch: pytest.MonkeyPatch,
    client: TestClient,
    auth_env: str,
) -> None:
    other_key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", other_key)
    invalid_token = generate_token("tester", expiry=120)
    monkeypatch.setenv("IOL_TOKENS_KEY", auth_env)

    response = client.get(
        "/cache/status",
        headers={"Authorization": f"Bearer {invalid_token}"},
    )

    assert response.status_code == 401
    assert "token" in response.json()["detail"].lower()
