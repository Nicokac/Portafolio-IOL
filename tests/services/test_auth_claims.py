"""Security-focused tests for authentication token claims."""

from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from services import auth as auth_module
from services.auth import (
    ACTIVE_TOKENS,
    AuthTokenError,
    MAX_TOKEN_TTL_SECONDS,
    TOKEN_AUDIENCE,
    TOKEN_ISSUER,
    TOKEN_VERSION,
    generate_token,
    refresh_active_token,
    revoke_token,
    verify_token,
)


@pytest.fixture(autouse=True)
def _configure_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    key = Fernet.generate_key().decode()
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", key)
    monkeypatch.delenv("IOL_TOKENS_KEY", raising=False)
    ACTIVE_TOKENS.clear()
    yield
    ACTIVE_TOKENS.clear()


def test_tokens_include_enriched_claims_and_limited_ttl() -> None:
    token = generate_token("alice", expiry=3_600)
    claims = verify_token(token)

    assert claims["iss"] == TOKEN_ISSUER
    assert claims["aud"] == TOKEN_AUDIENCE
    assert claims["version"] == TOKEN_VERSION
    assert claims["exp"] - claims["iat"] == MAX_TOKEN_TTL_SECONDS
    assert claims["session_id"] in ACTIVE_TOKENS


def test_refresh_requires_window_and_preserves_session(monkeypatch: pytest.MonkeyPatch) -> None:
    base_time = 1_700_000_000
    monkeypatch.setattr(auth_module.time, "time", lambda: base_time)
    token = generate_token("bob", expiry=MAX_TOKEN_TTL_SECONDS)

    # Still outside the refresh window (> 5 minutes remaining).
    monkeypatch.setattr(
        auth_module.time, "time", lambda: base_time + MAX_TOKEN_TTL_SECONDS - 400
    )
    claims = verify_token(token)
    with pytest.raises(AuthTokenError):
        refresh_active_token(claims)

    # Move to the last five minutes and refresh.
    monkeypatch.setattr(
        auth_module.time, "time", lambda: base_time + MAX_TOKEN_TTL_SECONDS - 200
    )
    refreshed = refresh_active_token(verify_token(token))
    new_token = refreshed["token"]
    new_claims = refreshed["claims"]

    assert new_token != token
    assert new_claims["session_id"] == claims["session_id"]
    assert new_claims["exp"] - new_claims["iat"] == MAX_TOKEN_TTL_SECONDS

    with pytest.raises(AuthTokenError):
        verify_token(token)


def test_revoked_token_cannot_be_verified() -> None:
    token = generate_token("carol", expiry=MAX_TOKEN_TTL_SECONDS)
    revoke_token(token)
    with pytest.raises(AuthTokenError):
        verify_token(token)
