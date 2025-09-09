# infrastructure\iol\client.py
from __future__ import annotations
import json
import logging
from pathlib import Path

from .ports import IIOLProvider
from .legacy.iol_client import IOLClient as _LegacyIOLClient  # <- ahora desde legacy

logger = logging.getLogger(__name__)
PORTFOLIO_CACHE = Path(".cache/last_portfolio.json")

class IOLClientAdapter(IIOLProvider):
    def __init__(self, user: str, password: str, tokens_file: Path | str | None = None):
        self._cli = _LegacyIOLClient(user, password, tokens_file=tokens_file)

    def get_portfolio(self) -> dict:
        try:
            data = self._cli.get_portfolio() or {}
            try:
                PORTFOLIO_CACHE.parent.mkdir(parents=True, exist_ok=True)
                PORTFOLIO_CACHE.write_text(
                    json.dumps(data, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
            except Exception as e:
                logger.debug("No se pudo guardar cache portafolio: %s", e)
            return data
        except Exception as e:
            logger.warning("get_portfolio falló: %s", e)
            try:
                data = json.loads(PORTFOLIO_CACHE.read_text(encoding="utf-8"))
                return data
            except Exception:
                return {"activos": []}

    def get_last_price(self, mercado: str, simbolo: str):
        return self._cli.get_last_price(mercado=mercado, simbolo=simbolo)

    def get_quote(self, mercado: str, simbolo: str):
        return self._cli.get_quote(mercado=mercado, simbolo=simbolo)

def build_iol_client(user: str, password: str, tokens_file: Path | str | None = None) -> IOLClientAdapter:
    return IOLClientAdapter(user, password, tokens_file=tokens_file)
