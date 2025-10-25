# infrastructure\iol\ports.py
from __future__ import annotations

from typing import Any, Dict, Protocol, runtime_checkable


@runtime_checkable
class IIOLProvider(Protocol):
    """Puerto (interfaz) para cualquier proveedor IOL-like."""

    def get_portfolio(self, country: str = "argentina") -> Any: ...
    def get_last_price(self, *, mercado: str, simbolo: str) -> Any: ...
    def get_quote(self, market: str, symbol: str, panel: str | None = None) -> Dict[str, Any]: ...
    def fetch_market_price(self, symbol: str, *, market: str = "BCBA") -> tuple[float | None, str | None]: ...
