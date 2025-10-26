"""Compatibility wrapper exposing the modern :class:`IOLClient` implementation.

This module used to house a legacy client with duplicated networking and
rate-limiting logic.  The implementation now lives in
``infrastructure.iol.client`` and this compatibility shim simply re-exports the
modern class to avoid code drift while keeping import paths stable.
"""

from __future__ import annotations

from ..client import IOLClient

__all__ = ["IOLClient"]
