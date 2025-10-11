"""Shared authentication helpers for Streamlit and FastAPI."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken


class AuthTokenError(ValueError):
    """Raised when there is an issue generating or verifying auth tokens."""


def _get_tokens_key() -> bytes:
    """Return the configured Fernet key used to protect authentication tokens."""

    key = os.getenv("IOL_TOKENS_KEY")
    if not key:
        raise RuntimeError(
            "IOL_TOKENS_KEY environment variable is not configured for authentication tokens."
        )
    normalized = key.strip()
    if not normalized:
        raise RuntimeError("IOL_TOKENS_KEY cannot be empty.")
    return normalized.encode()


def _get_fernet() -> Fernet:
    """Instantiate a Fernet cipher with the configured key."""

    return Fernet(_get_tokens_key())


def generate_token(username: str, expiry: int) -> str:
    """Generate a signed token for the given ``username`` valid for ``expiry`` seconds."""

    if not username:
        raise AuthTokenError("Username is required to generate a token.")
    if expiry <= 0:
        raise AuthTokenError("Token expiry must be a positive integer.")

    issued_at = int(time.time())
    payload: Dict[str, Any] = {
        "sub": username,
        "iat": issued_at,
        "exp": issued_at + int(expiry),
    }
    token_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    cipher = _get_fernet()
    return cipher.encrypt(token_bytes).decode("utf-8")


def verify_token(token: str) -> Dict[str, Any]:
    """Validate ``token`` and return its payload if it is still valid."""

    if not token:
        raise AuthTokenError("Token is required for verification.")

    cipher = _get_fernet()
    try:
        data = cipher.decrypt(token.encode("utf-8"))
    except InvalidToken as exc:  # pragma: no cover - handled via tests
        raise AuthTokenError("Invalid authentication token.") from exc

    try:
        payload: Dict[str, Any] = json.loads(data.decode("utf-8"))
    except ValueError as exc:  # pragma: no cover - malformed payload
        raise AuthTokenError("Malformed authentication token payload.") from exc

    expires_at = int(payload.get("exp", 0))
    if expires_at <= int(time.time()):
        raise AuthTokenError("Authentication token has expired.")

    return payload


__all__ = ["generate_token", "verify_token", "AuthTokenError"]
