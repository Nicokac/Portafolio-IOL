"""Unit tests for the shared authentication helpers."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from services import auth as auth_module
from services.auth import AuthTokenError, generate_token, verify_token


def test_generate_and_verify_token_returns_claims(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key)

    token = generate_token("alice", expiry=60)
    claims = verify_token(token)

    assert claims["sub"] == "alice"
    assert claims["exp"] > claims["iat"]


def test_generate_token_requires_positive_expiry(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key)

    with pytest.raises(AuthTokenError):
        generate_token("bob", expiry=0)


def test_verify_token_expired(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key)

    base_time = 1_700_000_000
    monkeypatch.setattr(auth_module.time, "time", lambda: base_time)
    token = generate_token("carol", expiry=1)

    monkeypatch.setattr(auth_module.time, "time", lambda: base_time + 5)
    with pytest.raises(AuthTokenError):
        verify_token(token)


def test_verify_token_with_wrong_key(monkeypatch: pytest.MonkeyPatch) -> None:
    key_one = Fernet.generate_key().decode()
    key_two = Fernet.generate_key().decode()
    monkeypatch.setenv("IOL_TOKENS_KEY", key_one)

    token = generate_token("dave", expiry=60)
    monkeypatch.setenv("IOL_TOKENS_KEY", key_two)

    with pytest.raises(AuthTokenError):
        verify_token(token)
