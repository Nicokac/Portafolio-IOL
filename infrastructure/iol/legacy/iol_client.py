# infrastructure\iol\legacy\iol_client.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple
import time
import logging
import threading
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

from shared.config import settings
from shared.settings import legacy_login_backoff_base, legacy_login_max_retries
from shared.utils import _to_float
from infrastructure.iol.auth import IOLAuth, InvalidCredentialsError
from services.quote_rate_limit import quote_rate_limiter
from infrastructure.iol.legacy.session import LegacySession
PORTFOLIO_URL = "https://api.invertironline.com/api/v2/portafolio"

REQ_TIMEOUT = 30
RETRIES = max(int(legacy_login_max_retries), 0)  # reintentos adicionales
BACKOFF_SEC = max(float(legacy_login_backoff_base), 0.0)
USER_AGENT = "IOL-Portfolio/1.0 (+iol_client)"

logger = logging.getLogger(__name__)


# =========================
# Utilidades
# =========================


def _elapsed_ms(start_ts: float) -> int:
    return int((time.time() - start_ts) * 1000)


# =========================
# Cliente de API
# =========================

class IOLClient:
    """
    - Endpoints de cuenta (portafolio) vía requests + Bearer
    - Datos de mercado vía iolConn (sesión persistente y reautenticación)
    """

    def __init__(
        self,
        user: str,
        password: str,
        tokens_file: Path | str | None = None,
        auth: IOLAuth | None = None,
    ):
        self.user = (user or "").strip()
        self.password = (password or "").strip()
        if auth is not None:
            self.auth = auth
        else:
            self.auth = IOLAuth(
                self.user,
                self.password,
                tokens_file=tokens_file,
                allow_plain_tokens=settings.allow_plain_tokens,
            )
        # Sesión HTTP para endpoints de cuenta
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self._legacy_session = LegacySession.get()

    # -------- Base request con retry/refresh --------
    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        """
        Request con:
        - Header Authorization dinámico
        - Reintento simple ante errores transitorios
        - Refresh de token ante 401 y reintento
        """
        last_exc: Optional[Exception] = None
        for attempt in range(RETRIES + 1):
            headers = kwargs.pop("headers", {})
            headers.update(self.auth.auth_header())
            try:
                quote_rate_limiter.wait_for_slot("legacy")
                r = self.session.request(
                    method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs
                )
                if r.status_code == 429:
                    wait_time = quote_rate_limiter.penalize(
                        "legacy", minimum_wait=_parse_retry_after_seconds(r)
                    )
                    logger.info(
                        "Legacy IOL 429 %s %s, esperando %.3fs antes de reintentar",
                        method,
                        url,
                        wait_time,
                    )
                    if attempt < RETRIES:
                        continue
                if r.status_code == 401:
                    # Token expirado o inválido -> refresh y reintento 1 vez
                    try:
                        self.auth.refresh()
                    except InvalidCredentialsError:
                        raise
                    headers = kwargs.pop("headers", {})
                    headers.update(self.auth.auth_header())
                    quote_rate_limiter.wait_for_slot("legacy")
                    r = self.session.request(
                        method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs
                    )
                    if r.status_code == 401:
                        raise InvalidCredentialsError("Credenciales inválidas")
                    if r.status_code == 429:
                        wait_time = quote_rate_limiter.penalize(
                            "legacy", minimum_wait=_parse_retry_after_seconds(r)
                        )
                        logger.info(
                            "Legacy IOL 429 tras refresh %s, espera %.3fs",
                            url,
                            wait_time,
                        )
                        if attempt < RETRIES:
                            continue
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                last_exc = e
                if e.response is not None and e.response.status_code == 404:
                    logger.warning("%s %s devolvió 404", method, url)
                    return None
                if (
                    e.response is not None
                    and e.response.status_code == 429
                    and attempt < RETRIES
                ):
                    wait_time = quote_rate_limiter.penalize(
                        "legacy",
                        minimum_wait=_parse_retry_after_seconds(e.response),
                    )
                    logger.info(
                        "Legacy IOL HTTP 429 %s %s, espera %.3fs",
                        method,
                        url,
                        wait_time,
                    )
                    continue
            except requests.RequestException as e:
                last_exc = e
            if attempt < RETRIES:
                time.sleep(BACKOFF_SEC * (attempt + 1))
        if last_exc:
            logger.warning("Request %s %s falló: %s", method, url, last_exc)
        return None

    # -------- Cuenta --------
    def get_portfolio(self) -> Dict[str, Any]:
        start = time.time()
        r = self._request("GET", PORTFOLIO_URL)
        logger.info("get_portfolio %s ms", _elapsed_ms(start))
        return r.json() if r is not None else {}

    # -------- Mercado (iolConn) --------
    def _fetch_market_payload(
        self,
        mercado: str,
        simbolo: str,
        panel: str | None = None,
    ) -> tuple[Optional[Dict[str, Any]], bool]:
        data, auth_failed = self._legacy_session.fetch_with_backoff(
            mercado,
            simbolo,
            panel=panel,
            auth_user=self.user,
            auth_password=self.password,
            auth=self.auth,
        )
        if data is not None and not isinstance(data, dict):
            return None, auth_failed
        return data, auth_failed

    def _inject_flags(self, payload: Dict[str, Any], auth_failed: bool) -> Dict[str, Any]:
        if auth_failed or self._legacy_session.is_auth_unavailable():
            payload = dict(payload)
            payload["legacy_auth_unavailable"] = True
        return payload

    @staticmethod
    def _parse_price_fields(d: Dict[str, Any]) -> Optional[float]:
        """
        Extrae un precio de un dict con claves variables y/o anidadas.
        """
        if not isinstance(d, dict):
            return None
        for k in ("ultimoPrecio", "ultimo", "last", "lastPrice", "precio", "cierre", "close"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                f = _to_float(v)
                if f is not None:
                    return f
            if isinstance(v, dict):
                for kk in ("value", "amount", "precio"):
                    vv = v.get(kk)
                    f = _to_float(vv)
                    if f is not None:
                        return f
        return None

    @staticmethod
    def _parse_chg_pct_fields(d: Dict[str, Any], last: Optional[float] = None) -> Optional[float]:
        """
        Extrae variación porcentual en % (no fracción), tolerando strings con coma/porcentual.
        Si no se encuentra expresada explícitamente, intenta calcularla a partir de
        "ultimo" y "cierreAnterior" (o aliases similares).
        """
        if not isinstance(d, dict):
            return None

        # Campos que pueden traer el porcentaje explícito
        for k in ("variacion", "variacionPorcentual", "cambioPorcentual", "changePercent"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                v = v.replace("%", "").strip()
                f = _to_float(v)
                if f is not None:
                    return f

        # Intento de cálculo a partir de cierre anterior / puntos variación
        prev_close: Optional[float] = None
        for k in ("cierreAnterior", "previousClose"):
            v = d.get(k)
            prev_close = _to_float(v)
            if prev_close is not None:
                break

        if prev_close is None:
            return None

        delta: Optional[float] = _to_float(d.get("puntosVariacion"))
        if delta is not None:
            return delta / prev_close * 100 if prev_close else None

        if last is None:
            last = IOLClient._parse_price_fields(d)

        if last is None:
            return None

        return (last - prev_close) / prev_close * 100

    def get_last_price(self, mercado: str, simbolo: str) -> Optional[float]:
        """
        Último precio simple.
        """
        mercado = (mercado or "bcba").lower()
        simbolo = (simbolo or "").upper()
        try:
            data, auth_failed = self._fetch_market_payload(mercado, simbolo)
        except Exception as e:
            logger.warning("get_last_price error %s:%s -> %s", mercado, simbolo, e)
            return None

        if data is None or auth_failed:
            return None
        return self._parse_price_fields(data)

    def get_quote(
        self,
        market: str = "bcba",
        symbol: str = "",
        panel: str | None = None,
        *,
        mercado: str | None = None,
        simbolo: str | None = None,
    ) -> Dict[str, Any]:
        """
        Devuelve {"last": float|None, "chg_pct": float|None}
        chg_pct es la variación diaria en %, si está disponible.
        """
        resolved_market = (mercado if mercado is not None else market or "bcba").lower()
        resolved_symbol = (simbolo if simbolo is not None else symbol or "").upper()
        try:
            data, auth_failed = self._fetch_market_payload(resolved_market, resolved_symbol, panel)
        except Exception as e:
            logger.warning("get_quote error %s:%s -> %s", resolved_market, resolved_symbol, e)
            empty = {"last": None, "chg_pct": None}
            return self._inject_flags(empty, False)

        if data is None:
            empty = {"last": None, "chg_pct": None}
            return self._inject_flags(empty, auth_failed)

        last = self._parse_price_fields(data)
        chg_pct = self._parse_chg_pct_fields(data, last)
        if chg_pct is None:
            logger.warning("chg_pct indeterminado para %s:%s", resolved_market, resolved_symbol)
        payload = {"last": last, "chg_pct": chg_pct}
        return self._inject_flags(payload, auth_failed)

    # -------- Opcional: cotizaciones en batch --------
    def get_quotes_bulk(
        self,
        items: Iterable[Tuple[str, str] | Tuple[str, str, str | None]],
        max_workers: int = 8
    ) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
        """
        Descarga cotizaciones para múltiples (mercado, símbolo) en paralelo.
        Retorna: { (mercado, símbolo): {"last": float|None, "chg_pct": float|None} }
        """
        requests: list[tuple[str, str, str | None]] = []
        for raw in items:
            market: str | None = None
            symbol: str | None = None
            panel: str | None = None
            if isinstance(raw, dict):
                market = raw.get("market", raw.get("mercado"))
                symbol = raw.get("symbol", raw.get("simbolo"))
                panel = raw.get("panel")
            elif isinstance(raw, (list, tuple)):
                if len(raw) >= 2:
                    market = raw[0]
                    symbol = raw[1]
                if len(raw) >= 3:
                    panel = raw[2]
            else:
                market = getattr(raw, "market", getattr(raw, "mercado", None))
                symbol = getattr(raw, "symbol", getattr(raw, "simbolo", None))
                panel = getattr(raw, "panel", None)

            norm_market = (market or "bcba").lower()
            norm_symbol = (symbol or "").upper()
            requests.append((norm_market, norm_symbol, panel))

        if not requests:
            return {}
        out: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(requests)))) as ex:
            fut_map = {
                ex.submit(self.get_quote, m, s, panel): (m, s)
                for (m, s, panel) in requests
            }
            for fut in as_completed(fut_map):
                key = fut_map[fut]
                try:
                    out[key] = fut.result()
                except Exception as e:
                    logger.warning("get_quotes_bulk %s:%s error -> %s", key[0], key[1], e)
                    out[key] = {"last": None, "chg_pct": None}
        return out
def _parse_retry_after_seconds(response) -> Optional[float]:
    if response is None:
        return None
    headers = getattr(response, "headers", {}) or {}
    value = headers.get("Retry-After")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            dt = parsedate_to_datetime(value)
        except (TypeError, ValueError, OverflowError):
            return None
        if dt is None:
            return None
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        return max(0.0, (dt - now).total_seconds())

