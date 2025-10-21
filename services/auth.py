"""Shared authentication helpers for Streamlit and FastAPI."""

from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, Dict

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)
audit_logger = logging.getLogger("audit")

try:  # pragma: no cover - exercised indirectly via integration tests
    from fastapi import Depends, HTTPException, status
    from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

    logger.info("FastAPI active ✅")
except ModuleNotFoundError:  # pragma: no cover - depends on optional dependency
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
    """Raised when there is an issue generating, refreshing or verifying auth tokens."""


TOKEN_ISSUER = "portafolio-iol"
TOKEN_AUDIENCE = "frontend"
TOKEN_VERSION = "1.0"
MAX_TOKEN_TTL_SECONDS = 15 * 60

# Active token registry used to track issued sessions and support revocation.
ACTIVE_TOKENS: Dict[str, Dict[str, Any]] = {}

# Registry tracking nonce usage for refresh anti-replay protections.
USED_REFRESH_NONCES: Dict[str, Dict[str, int]] = {}

# Allow nonce history to linger slightly longer than the token lifetime to catch
# delayed replays while still preventing unbounded growth.
NONCE_HISTORY_TTL_SECONDS = MAX_TOKEN_TTL_SECONDS + 300


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


def _normalize_expiry(expiry: int) -> int:
    """Clamp the requested ``expiry`` to the maximum supported TTL."""

    return max(1, min(int(expiry), MAX_TOKEN_TTL_SECONDS))


def _cleanup_nonce_history(session_id: str | None = None, now: int | None = None) -> None:
    """Remove nonce entries older than ``NONCE_HISTORY_TTL_SECONDS``."""

    if session_id is not None:
        entries = USED_REFRESH_NONCES.get(session_id)
        if not entries:
            return
        reference = int(now if now is not None else time.time()) - NONCE_HISTORY_TTL_SECONDS
        for value, ts in list(entries.items()):
            if ts < reference:
                del entries[value]
        if not entries:
            USED_REFRESH_NONCES.pop(session_id, None)
        return

    reference = int(now if now is not None else time.time()) - NONCE_HISTORY_TTL_SECONDS
    for session, entries in list(USED_REFRESH_NONCES.items()):
        for value, ts in list(entries.items()):
            if ts < reference:
                del entries[value]
        if not entries:
            USED_REFRESH_NONCES.pop(session, None)


def _register_token(session_id: str, token: str, payload: Dict[str, Any], ttl: int) -> None:
    """Register ``token`` as active for the given ``session_id``."""

    nonce = str(payload.get("nonce", ""))
    if not nonce:
        raise AuthTokenError("Token is missing nonce.")

    now = int(time.time())
    _cleanup_nonce_history(session_id, now)

    ACTIVE_TOKENS[session_id] = {
        "token": token,
        "payload": payload,
        "ttl": int(ttl),
        "issued_at": int(payload.get("iat", 0)),
        "expires_at": int(payload.get("exp", 0)),
        "username": payload.get("sub", ""),
        "nonce": nonce,
    }


def _remove_active_session(session_id: str) -> None:
    """Remove the active session with ``session_id`` if present."""

    ACTIVE_TOKENS.pop(session_id, None)
    USED_REFRESH_NONCES.pop(session_id, None)


def describe_active_token(token: str | None) -> Dict[str, Any] | None:
    """Return metadata for the active ``token`` if it exists."""

    if not token:
        return None
    for session_id, entry in ACTIVE_TOKENS.items():
        if entry.get("token") != token:
            continue
        snapshot: Dict[str, Any] = {
            "session_id": session_id,
            "ttl": entry.get("ttl"),
            "issued_at": entry.get("issued_at"),
            "expires_at": entry.get("expires_at"),
            "username": entry.get("username"),
            "nonce": entry.get("nonce"),
        }
        payload = entry.get("payload")
        if isinstance(payload, Dict):
            snapshot["claims"] = dict(payload)
        return snapshot
    return None


def generate_token(username: str, expiry: int) -> str:
    """Generate a signed token for the given ``username`` valid for ``expiry`` seconds."""

    if not username:
        raise AuthTokenError("Username is required to generate a token.")
    if expiry <= 0:
        raise AuthTokenError("Token expiry must be a positive integer.")

    issued_at = int(time.time())
    ttl = _normalize_expiry(expiry)
    session_id = uuid.uuid4().hex
    nonce = uuid.uuid4().hex
    payload: Dict[str, Any] = {
        "sub": username,
        "iat": issued_at,
        "exp": issued_at + ttl,
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "session_id": session_id,
        "version": TOKEN_VERSION,
        "nonce": nonce,
    }
    token_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    cipher = _get_fernet()
    token = cipher.encrypt(token_bytes).decode("utf-8")
    _register_token(session_id, token, payload, ttl)
    return token


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

    session_id = str(payload.get("session_id", ""))
    if not session_id:
        raise AuthTokenError("Authentication token missing session identifier.")

    entry = ACTIVE_TOKENS.get(session_id)
    nonce = str(payload.get("nonce", ""))
    used_nonces = USED_REFRESH_NONCES.get(session_id, {})
    if entry is None:
        if nonce and nonce in used_nonces:
            raise AuthTokenError("Token refresh request has been replayed.")
        raise AuthTokenError("Authentication token has been revoked.")

    if entry.get("token") != token:
        if nonce and nonce in used_nonces:
            raise AuthTokenError("Token refresh request has been replayed.")
        raise AuthTokenError("Authentication token has been revoked.")

    expires_at = int(payload.get("exp", 0))
    now = int(time.time())
    if expires_at <= now:
        _remove_active_session(session_id)
        raise AuthTokenError("Authentication token has expired.")

    _cleanup_nonce_history(session_id, now)

    return payload


def revoke_token(token: str | None) -> None:
    """Revoke ``token`` from the active registry."""

    if not token:
        return
    for session_id, entry in list(ACTIVE_TOKENS.items()):
        if entry.get("token") == token:
            _remove_active_session(session_id)
            break


def revoke_session(session_id: str | None) -> None:
    """Revoke the active session identified by ``session_id``."""

    if not session_id:
        return
    _remove_active_session(str(session_id))


def refresh_active_token(claims: Dict[str, Any]) -> Dict[str, Any]:
    """Refresh the active token described by ``claims`` and return the new payload."""

    session_id = str(claims.get("session_id", ""))
    if not session_id:
        raise AuthTokenError("Token is missing session context.")

    entry = ACTIVE_TOKENS.get(session_id)
    if entry is None:
        raise AuthTokenError("Session is not active.")

    now = int(time.time())
    expires_at = int(entry.get("expires_at") or claims.get("exp", 0))
    if expires_at <= now:
        _remove_active_session(session_id)
        raise AuthTokenError("Authentication token has expired.")

    if expires_at - now > 300:
        raise AuthTokenError("Token is not eligible for refresh yet.")

    nonce = str(claims.get("nonce", ""))
    if not nonce:
        raise AuthTokenError("Token is missing nonce.")

    expected_nonce = str(entry.get("nonce", ""))
    if expected_nonce != nonce:
        raise AuthTokenError("Token refresh request has been replayed.")

    used_nonces = USED_REFRESH_NONCES.setdefault(session_id, {})
    _cleanup_nonce_history(session_id, now)
    if nonce in used_nonces:
        raise AuthTokenError("Token refresh request has been replayed.")
    used_nonces[nonce] = now

    ttl = int(entry.get("ttl") or max(1, int(claims.get("exp", 0)) - int(claims.get("iat", 0))))
    ttl = _normalize_expiry(ttl)
    issued_at = now
    new_nonce = uuid.uuid4().hex
    payload: Dict[str, Any] = {
        "sub": claims.get("sub"),
        "iat": issued_at,
        "exp": issued_at + ttl,
        "iss": TOKEN_ISSUER,
        "aud": TOKEN_AUDIENCE,
        "session_id": session_id,
        "version": TOKEN_VERSION,
        "nonce": new_nonce,
    }
    token_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    cipher = _get_fernet()
    token = cipher.encrypt(token_bytes).decode("utf-8")
    _register_token(session_id, token, payload, ttl)
    payload["ttl"] = ttl
    audit_logger.info(
        "token_refreshed",
        extra={"session_id": session_id, "user": claims.get("sub"), "result": "ok"},
    )
    return {"token": token, "claims": payload}


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
        claims = verify_token(credentials.credentials)
    except AuthTokenError as exc:  # pragma: no cover - exercised via tests
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
        ) from exc
    if claims.get("iss") != TOKEN_ISSUER:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token issuer.",
        )
    if claims.get("aud") != TOKEN_AUDIENCE:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token audience.",
        )
    if claims.get("version") != TOKEN_VERSION:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Unsupported token version.",
        )

    return claims


__all__ = [
    "generate_token",
    "verify_token",
    "AuthTokenError",
    "get_current_user",
    "refresh_active_token",
    "revoke_token",
    "revoke_session",
    "describe_active_token",
    "ACTIVE_TOKENS",
    "USED_REFRESH_NONCES",
    "TOKEN_ISSUER",
    "TOKEN_AUDIENCE",
    "TOKEN_VERSION",
    "MAX_TOKEN_TTL_SECONDS",
]
