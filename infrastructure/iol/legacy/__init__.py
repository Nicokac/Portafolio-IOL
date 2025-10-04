"""Deprecated IOL legacy package kept for backwards compatibility."""

from __future__ import annotations

import warnings

from . import iol_client as iol_client

warnings.warn(
    "`infrastructure.iol.legacy` está deprecado; utiliza `infrastructure.iol.client.IOLClient`.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["iol_client"]

