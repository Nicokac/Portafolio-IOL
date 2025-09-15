"""Shared exception types exposed for cross-layer use."""
from __future__ import annotations

from infrastructure.iol.auth import InvalidCredentialsError, NetworkError

__all__ = ["InvalidCredentialsError", "NetworkError"]
