"""Shared authentication helpers for Streamlit and FastAPI."""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised indirectly via integration tests
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    FASTAPI_AVAILABLE = True
    logger.info("FastAPI active ✅")
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
    FASTAPI_AVAILABLE = False

    class _StatusStub:
        HTTP_401_UNAUTHORIZED = 401

    status = _StatusStub()

    class HTTPException(Exception):
        """Fallback HTTPException compatible with FastAPI signature."""

        def __init__(self, status_code: int, detail: str | None = None):
            super().__init__(detail)
            self.status_code = status_code
            self.detail = detail

    def Depends(dependency: Any = None) -> Any:  # type: ignore[override]
        """Return the provided dependency unchanged for Streamlit usage."""

        return dependency

    class HTTPAuthorizationCredentials:  # pragma: no cover - compatibility shim
        """Minimal stub emulating FastAPI's credentials container."""

        def __init__(self, scheme: str | None = None, credentials: str | None = None):
            self.scheme = scheme or ""
            self.credentials = credentials or ""

    class HTTPBearer:  # pragma: no cover - compatibility shim
        """Stub security dependency raising when used without FastAPI."""

        def __init__(self, auto_error: bool = False):
            self.auto_error = auto_error

        async def __call__(self, *args: Any, **kwargs: Any) -> HTTPAuthorizationCredentials:
            raise RuntimeError("HTTPBearer unavailable without FastAPI installed.")

    logger.warning("FastAPI not installed — running in Streamlit-only mode.")
    logger.warning("Streamlit-only mode ⚠️")


class AuthTokenError(ValueError):
    """Raised when there is an issue generating or verifying auth tokens."""


def _get_tokens_key() -> bytes:
    """Return the configured Fernet key used to protect FastAPI auth tokens."""

    key = os.getenv("FASTAPI_TOKENS_KEY")
    if not key:
        raise RuntimeError(
            "FASTAPI_TOKENS_KEY environment variable is not configured for authentication tokens."
        )
    normalized = key.strip()
    if not normalized:
        raise RuntimeError("FASTAPI_TOKENS_KEY cannot be empty.")

    iol_key = os.getenv("IOL_TOKENS_KEY", "").strip()
    if iol_key and iol_key == normalized:
        raise RuntimeError(
            "FASTAPI_TOKENS_KEY must be different from IOL_TOKENS_KEY to avoid key reuse."
        )
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


security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
) -> Dict[str, Any]:
    """Validate bearer tokens and expose the associated claims."""

    if credentials is None or credentials.scheme.lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or invalid bearer token.",
        )
    try:
        return verify_token(credentials.credentials)
    except AuthTokenError as exc:  # pragma: no cover - exercised via tests
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc


__all__ = [
    "generate_token",
    "verify_token",
    "AuthTokenError",
    "get_current_user",
]
