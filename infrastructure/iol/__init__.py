"""Infrastructure entrypoint exposing compatibility aliases."""

from __future__ import annotations

import sys
from types import ModuleType

from . import compat


def _register_legacy_alias() -> None:
    """Expose the removed ``legacy`` package as a runtime alias."""

    legacy_module = ModuleType("infrastructure.iol.legacy")
    legacy_module.__dict__.update(
        {
            "__package__": "infrastructure.iol.legacy",
            "iol_client": compat.iol_client,
            "session": compat.session,
        }
    )
    sys.modules.setdefault("infrastructure.iol.legacy", legacy_module)
    sys.modules.setdefault("infrastructure.iol.legacy.iol_client", compat.iol_client)
    sys.modules.setdefault("infrastructure.iol.legacy.session", compat.session)


_register_legacy_alias()

__all__ = ["compat"]
