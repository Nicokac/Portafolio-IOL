# infrastructure\fx\ports.py
from __future__ import annotations
from typing import Protocol, Mapping

class IFXProvider(Protocol):
    """Puerto de tipos de cambio (FX)."""
    def get_rates(self) -> Mapping[str, float]:
        """Devuelve un mapping con claves como 'ccl', 'mep', 'blue', etc."""
        ...
