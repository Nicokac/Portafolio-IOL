# infrastructure/iol/account_client.py
from __future__ import annotations

import logging
import math
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

    def fetch_account_status(self) -> Mapping[str, Any]:
        url = f"{self._api_base}/estadocuenta"
        response = self._request("GET", url)
        try:
            payload = response.json() or {}
        except ValueError as exc:  # pragma: no cover - unexpected payload
            logger.warning("Respuesta inválida de /estadocuenta: %s", exc)
            return {}
        if not isinstance(payload, Mapping):
            return {}
        return payload

    def fetch_balances(self) -> AccountCashSummary:
        payload = self.fetch_account_status()
        if not isinstance(payload, Mapping):
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
        usd_accounts_total = 0.0

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
                usd_accounts_total += amount
                if usd_rate is None:
                    usd_rate = _as_float(entry.get("cotizacion")) or _as_float(entry.get("cotizacionCartera"))
            elif currency == "ARS":
                cash_ars += amount

        cash_usd_top = 0.0
        matches_accounts = False
        if top_usd is not None:
            normalized_top_usd = top_usd
            if usd_rate and usd_rate > 0:
                should_normalize = False
                if usd_accounts_total:
                    expected_ars = usd_accounts_total * usd_rate
                    tolerance = max(10.0, abs(expected_ars) * 0.01)
                    should_normalize = math.isclose(normalized_top_usd, expected_ars, rel_tol=0.05, abs_tol=tolerance)
                elif top_ars and top_ars > 0:
                    should_normalize = normalized_top_usd * usd_rate > (top_ars * 1.5)

                if should_normalize:
                    # El endpoint puede devolver ``disponibleEnDolares`` ya convertido a ARS
                    # (caso cuentas locales) o realmente en USD (cuentas billete). Cuando
                    # detectamos que coincide con el total USD detallado en ``cuentas`` o
                    # desborda contra el saldo en pesos, lo reescalamos para conservar USD.
                    normalized_top_usd = normalized_top_usd / usd_rate

            cash_usd_top = float(normalized_top_usd)
            if usd_accounts_total:
                matches_accounts = math.isclose(
                    cash_usd_top,
                    usd_accounts_total,
                    rel_tol=0.01,
                    abs_tol=max(0.01, abs(usd_accounts_total) * 0.01),
                )

        if matches_accounts:
            cash_usd = usd_accounts_total
        else:
            cash_usd = cash_usd_top + usd_accounts_total

        return AccountCashSummary(cash_ars=cash_ars, cash_usd=cash_usd, usd_rate=usd_rate)

