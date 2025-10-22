# infrastructure\fx\ports.py
from __future__ import annotations

from typing import Mapping, Optional, Protocol, Tuple


class IFXProvider(Protocol):
    """Puerto de tipos de cambio (FX)."""

    def get_rates(self) -> Tuple[Mapping[str, float], Optional[str]]:
        """Devuelve un mapping con claves como 'ccl', 'mep', 'blue', etc. + mensaje de error opcional."""
        ...
