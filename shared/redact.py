"""Utilities to redact sensitive information from nested payloads."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

_REDACTED = "***REDACTED***"
_SENSITIVE_KEYS = {
    "authorization",
    "authorizationheader",
    "set-cookie",
    "cookie",
    "cookies",
    "refresh",
    "refresh_token",
    "access_token",
    "bearer",
    "x-refresh-token",
    "x-api-key",
    "api-key",
}


def _redact_mapping(mapping: Mapping[Any, Any]) -> dict[Any, Any]:
    redacted: dict[Any, Any] = {}
    for key, value in mapping.items():
        key_text = str(key).lower()
        if key_text in _SENSITIVE_KEYS:
            redacted[key] = _REDACTED
        else:
            redacted[key] = redact_secrets(value)
    return redacted


def _redact_sequence(seq: Sequence[Any]) -> Sequence[Any]:
    if isinstance(seq, tuple):
        return tuple(redact_secrets(item) for item in seq)
    if isinstance(seq, list):
        return [redact_secrets(item) for item in seq]
    if isinstance(seq, set):
        return {redact_secrets(item) for item in seq}
    try:
        return type(seq)(redact_secrets(item) for item in seq)
    except TypeError:
        return [redact_secrets(item) for item in seq]


def redact_secrets(obj: Any) -> Any:
    """Return ``obj`` with sensitive keys anonymised.

    The function performs a deep traversal over dictionaries and common sequences,
    replacing well known secret-bearing keys with a ``***REDACTED***`` marker.
    Non container types are returned unchanged to preserve the original values.
    """

    if isinstance(obj, Mapping):
        return _redact_mapping(obj)
    if isinstance(obj, (list, tuple, set)):
        return _redact_sequence(obj)
    return obj


__all__ = ["redact_secrets"]
