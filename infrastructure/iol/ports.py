# infrastructure\iol\ports.py
from __future__ import annotations
from typing import Protocol, Any, Dict, Optional, runtime_checkable

@runtime_checkable
class IIOLProvider(Protocol):
    """Puerto (interfaz) para cualquier proveedor IOL-like."""
    def get_portfolio(self) -> Any: ...
    def get_last_price(self, *, mercado: str, simbolo: str) -> Any: ...
    def get_quote(self, market: str, symbol: str, panel: str | None = None) -> Dict[str, Any]: ...
