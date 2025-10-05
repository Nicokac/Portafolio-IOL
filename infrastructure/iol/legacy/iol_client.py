# infrastructure\iol\legacy\iol_client.py
from __future__ import annotations

from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, Iterable, Tuple
import time
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from iolConn import Iol
from iolConn.common.exceptions import NoAuthException  # <- importante
import streamlit as st

from shared.config import settings
from shared.utils import _to_float
from shared.time_provider import TimeProvider
from infrastructure.iol.auth import IOLAuth, InvalidCredentialsError
PORTFOLIO_URL = "https://api.invertironline.com/api/v2/portafolio"

REQ_TIMEOUT = 30
RETRIES = 3                # reintentos además del primer intento
BACKOFF_SEC = 0.5
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
        # iolConn para mercado
        self.iol_market: Optional[Iol] = None
        self._market_ready = False
        self._market_lock = threading.RLock()
        self._ensure_market_auth()

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
            delay = BACKOFF_SEC * (attempt + 1)
            try:
                r = self.session.request(method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs)
                if r.status_code == 401:
                    # Token expirado o inválido -> refresh y reintento 1 vez
                    try:
                        self.auth.refresh()
                    except InvalidCredentialsError:
                        raise
                    headers = kwargs.pop("headers", {})
                    headers.update(self.auth.auth_header())
                    r = self.session.request(method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs)
                    if r.status_code == 401:
                        raise InvalidCredentialsError("Credenciales inválidas")
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                last_exc = e
                status_code = e.response.status_code if e.response is not None else None
                if status_code == 404:
                    logger.warning("%s %s devolvió 404", method, url)
                    return None
                if status_code == 429:
                    delay = BACKOFF_SEC * (2**attempt)
            except requests.RequestException as e:
                last_exc = e
            if attempt < RETRIES:
                time.sleep(delay)
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
    def _ensure_market_auth(self) -> None:
        """
        Inicializa/Reautentica iolConn una vez (thread-safe).
        """
        if self._market_ready and self.iol_market is not None:
            return
        with self._market_lock:
            if self._market_ready and self.iol_market is not None:
                return
            if self.password:
                self.iol_market = Iol(self.user, self.password)
            else:
                self.iol_market = Iol(self.user, self.password)
                bearer = self.auth.tokens.get("access_token")
                refresh = self.auth.tokens.get("refresh_token")
                if bearer and refresh:
                    self.iol_market.bearer = bearer
                    self.iol_market.refresh_token = refresh
                    # <== LÍNEA CORREGIDA: Usa la nueva API de TimeProvider
                    # iolConn requiere un datetime naive para compatibilidad
                    bearer_time = TimeProvider.now_datetime().replace(tzinfo=None)
                    self.iol_market.bearer_time = bearer_time
                else:
                    st.session_state["force_login"] = True
                    raise InvalidCredentialsError("Token inválido")
            try:
                # algunas versiones requieren gestionar() para renovar bearer interno
                logger.debug("Autenticando mercado con bearer")
                self.iol_market.gestionar()
                logger.info("Autenticación mercado con bearer ok")
            except NoAuthException:
                logger.info("Bearer inválido; autenticando con contraseña")
                if self.password:
                    # recrea sesión y reintenta una vez
                    self.iol_market = Iol(self.user, self.password)
                try:
                    self.iol_market.gestionar()
                    logger.info("Autenticación mercado con contraseña ok")
                except NoAuthException as e:
                    logger.error("Autenticación mercado con contraseña falló", exc_info=True)
                    raise e
                else:
                    st.session_state["force_login"] = True
                    raise InvalidCredentialsError("Token inválido")
            self._market_ready = True

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
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=mercado, simbolo=simbolo)
        except NoAuthException:
            self._market_ready = False
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=mercado, simbolo=simbolo)
        except Exception as e:
            logger.warning("get_last_price error %s:%s -> %s", mercado, simbolo, e)
            return None

        return self._parse_price_fields(data) if isinstance(data, dict) else None

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
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=resolved_market, simbolo=resolved_symbol)
        except NoAuthException:
            self._market_ready = False
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=resolved_market, simbolo=resolved_symbol)
        except Exception as e:
            logger.warning("get_quote error %s:%s -> %s", resolved_market, resolved_symbol, e)
            return {"last": None, "chg_pct": None}

        if not isinstance(data, dict):
            return {"last": None, "chg_pct": None}

        last = self._parse_price_fields(data)
        chg_pct = self._parse_chg_pct_fields(data, last)
        if chg_pct is None:
            logger.warning("chg_pct indeterminado para %s:%s", resolved_market, resolved_symbol)
        return {"last": last, "chg_pct": chg_pct}

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
        self._ensure_market_auth()

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
