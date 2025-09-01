# infrastructure\iol\client.py
from __future__ import annotations
from .ports import IIOLProvider
from .legacy.iol_client import IOLClient as _LegacyIOLClient  # <- ahora desde legacy

class IOLClientAdapter(IIOLProvider):
    def __init__(self, user: str, password: str):
        self._cli = _LegacyIOLClient(user, password)

    def get_portfolio(self) -> dict:
        return self._cli.get_portfolio()

    def get_last_price(self, mercado: str, simbolo: str):
        return self._cli.get_last_price(mercado=mercado, simbolo=simbolo)

    def get_quote(self, mercado: str, simbolo: str):
        return self._cli.get_quote(mercado=mercado, simbolo=simbolo)

def build_iol_client(user: str, password: str) -> IOLClientAdapter:
    return IOLClientAdapter(user, password)
