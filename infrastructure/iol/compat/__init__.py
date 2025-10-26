"""Compatibility helpers replacing the previous `legacy` package."""

from __future__ import annotations

from . import iol_client as iol_client
from . import session as session

__all__ = [
    "iol_client",
    "session",
]
