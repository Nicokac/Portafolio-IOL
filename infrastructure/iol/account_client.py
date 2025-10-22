# infrastructure/iol/account_client.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Mapping

import requests

from shared.errors import InvalidCredentialsError
from shared.utils import _to_float

logger = logging.getLogger(__name__)


REQ_TIMEOUT = 30

_AMOUNT_KEYS = (
    "disponibleParaOperar",
    "disponibleOrdenes",
    "disponible",
    "saldoDisponible",
    "efectivoDisponible",
    "efectivo",
    "saldo",
    "cash",
    "monto",
)


def _detect_currency(label: Any) -> str | None:
    text = str(label or "").strip().upper()
    if not text:
        return None
    if "USD" in text or "DOLAR" in text or "DÓLAR" in text:
        return "USD"
    if "ARS" in text or "PESO" in text:
        return "ARS"
    return None


def _as_float(value: Any) -> float | None:
    parsed = _to_float(value)
    if parsed is None:
        return None
    return float(parsed)


@dataclass(frozen=True)
class AccountCashSummary:
    """Normalized representation of the cash balances returned by IOL."""

    cash_ars: float = 0.0
    cash_usd: float = 0.0
    usd_rate: float | None = None

    def usd_ars_equivalent(self) -> float:
        if self.usd_rate is None or not self.cash_usd:
            return 0.0
        return float(self.cash_usd) * float(self.usd_rate)

    def to_payload(self) -> dict[str, float]:
        payload: dict[str, float] = {
            "cash_ars": float(self.cash_ars or 0.0),
            "cash_usd": float(self.cash_usd or 0.0),
        }
        if self.usd_rate is not None:
            payload["usd_rate"] = float(self.usd_rate)
            payload["cash_usd_ars_equivalent"] = self.usd_ars_equivalent()
        return payload


class IOLAccountClient:
    """HTTP client that fetches account balances from ``/estadocuenta``."""

    def __init__(
        self,
        *,
        auth,
        session: requests.Session | None = None,
        api_base: str,
    ) -> None:
        self._auth = auth
        self._session = session or requests.Session()
        self._api_base = api_base.rstrip("/")

    def _request(self, method: str, url: str, **kwargs: Any) -> requests.Response:
        self._auth.ensure_token(silent=True)
        headers = kwargs.pop("headers", {})
        headers.update(self._auth.auth_header())
        response = self._session.request(
            method,
            url,
            headers=headers,
            timeout=REQ_TIMEOUT,
            **kwargs,
        )
        if response.status_code == 401:
            self._auth.refresh(silent=True)
            headers = kwargs.pop("headers", {})
            headers.update(self._auth.auth_header())
            response = self._session.request(
                method,
                url,
                headers=headers,
                timeout=REQ_TIMEOUT,
                **kwargs,
            )
            if response.status_code == 401:
                raise InvalidCredentialsError("Credenciales inválidas")
        response.raise_for_status()
        return response

    def fetch_balances(self) -> AccountCashSummary:
        url = f"{self._api_base}/estadocuenta"
        response = self._request("GET", url)
        try:
            payload = response.json() or {}
        except ValueError as exc:  # pragma: no cover - unexpected payload
            logger.warning("Respuesta inválida de /estadocuenta: %s", exc)
            return AccountCashSummary()
        return self._parse_payload(payload)

    def _parse_payload(self, payload: Mapping[str, Any]) -> AccountCashSummary:
        cash_ars = 0.0
        cash_usd = 0.0
        usd_rate = _as_float(payload.get("cotizacionDolar"))

        top_ars = _as_float(payload.get("disponibleEnPesos"))
        if top_ars is not None:
            cash_ars += top_ars

        top_usd = _as_float(payload.get("disponibleEnDolares"))
        if top_usd is not None:
            cash_usd += top_usd

        accounts = []
        if isinstance(payload.get("cuentas"), list):
            accounts.extend(payload["cuentas"])
        if isinstance(payload.get("cuentasActivas"), list):
            accounts.extend(payload["cuentasActivas"])

        for entry in accounts:
            if not isinstance(entry, Mapping):
                continue
            currency = _detect_currency(entry.get("moneda"))
            if currency is None:
                currency = _detect_currency(entry.get("descripcion"))
            amount = None
            for key in _AMOUNT_KEYS:
                amount = _as_float(entry.get(key))
                if amount is not None:
                    break
            if amount is None:
                continue
            if currency == "USD":
                cash_usd += amount
                if usd_rate is None:
                    usd_rate = _as_float(entry.get("cotizacion")) or _as_float(entry.get("cotizacionCartera"))
            elif currency == "ARS":
                cash_ars += amount

        return AccountCashSummary(cash_ars=cash_ars, cash_usd=cash_usd, usd_rate=usd_rate)

