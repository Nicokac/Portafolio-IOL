# infrastructure\iol\legacy\iol_client.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, Iterable, Tuple
import json
import time
import logging
import threading
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from cryptography.fernet import InvalidToken
from iolConn import Iol
from iolConn.common.exceptions import NoAuthException  # <- importante

from shared.config import settings
from shared.utils import _to_float
from infrastructure.iol.auth import FERNET

TOKEN_URL = "https://api.invertironline.com/token"
PORTFOLIO_URL = "https://api.invertironline.com/api/v2/portafolio"

REQ_TIMEOUT = 30
RETRIES = 1                # reintento simple (además del primer intento)
BACKOFF_SEC = 0.5
USER_AGENT = "IOL-Portfolio/1.0 (+iol_client)"

logger = logging.getLogger(__name__)


# =========================
# Utilidades
# =========================


def _elapsed_ms(start_ts: float) -> int:
    return int((time.time() - start_ts) * 1000)


# =========================
# Autenticación
# =========================

#class IOLAuth:
class _LegacyIOLAuth:
    """
    Manejo de tokens OAuth de IOL:
    - login() obtiene access_token y refresh_token
    - refresh() renueva; si falla, re-login
    - persiste en JSON (ruta configurable)
    """

    def __init__(self, user: str, password: str, tokens_file: Path | str | None = None):
        self.user = (user or "").strip()
        self.password = (password or "").strip()
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.tokens_file = Path(tokens_file or settings.tokens_file)
        # asegurar carpeta
        try:
            self.tokens_file.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("No se pudo crear directorio de tokens: %s", e)
        self._lock = threading.RLock()
        self.tokens: Dict[str, Any] = self._load_tokens() or {}

    def _load_tokens(self) -> Dict[str, Any]:
        if self.tokens_file.exists():
            try:
                raw = self.tokens_file.read_bytes()
                if FERNET:
                    raw = FERNET.decrypt(raw)
                return json.loads(raw.decode("utf-8"))
            except (InvalidToken, json.JSONDecodeError) as e:
                logger.warning("No se pudieron leer tokens: %s", e)
                return {}
            except OSError as e:
                logger.warning("No se pudieron leer tokens: %s", e)
                return {}
        return {}

    def _save_tokens(self, data: Dict[str, Any]) -> None:
        try:
            tmp = self.tokens_file.with_suffix(self.tokens_file.suffix + ".tmp")
            content = json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
            if FERNET:
                content = FERNET.encrypt(content)
            tmp.write_bytes(content)
            tmp.replace(self.tokens_file)
            try:
                os.chmod(self.tokens_file, 0o600)
            except OSError:
                pass
        except Exception as e:
            logger.error("No se pudo guardar el archivo de tokens: %s", e)

    def login(self) -> Dict[str, Any]:
        with self._lock:
            start = time.time()
            payload = {"username": self.user, "password": self.password, "grant_type": "password"}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
                r.raise_for_status()
            except requests.RequestException as e:
                logger.warning("Login IOL falló: %s", e)
                return {}
            self.tokens = r.json() or {}
            self._save_tokens(self.tokens)
            logger.info("IOL login ok en %d ms", _elapsed_ms(start))
            return self.tokens

    def refresh(self) -> Dict[str, Any]:
        with self._lock:
            start = time.time()
            if not self.tokens.get("refresh_token"):
                logger.info("Sin refresh_token; ejecutando login()")
                return self.login()
            payload = {"refresh_token": self.tokens["refresh_token"], "grant_type": "refresh_token"}
            headers = {"Content-Type": "application/x-www-form-urlencoded"}
            try:
                r = self.session.post(TOKEN_URL, data=payload, headers=headers, timeout=REQ_TIMEOUT)
                r.raise_for_status()
            except requests.RequestException as e:
                # si el refresh falla, re-login completo
                logger.warning("Refresh falló; intentando login(): %s", e)
                return self.login()
            self.tokens = r.json() or {}
            self._save_tokens(self.tokens)
            logger.info("IOL refresh ok en %d ms", _elapsed_ms(start))
            return self.tokens

    def clear_tokens(self):
        with self._lock:
            self.tokens = {}
            try:
                if self.tokens_file.exists():
                    self.tokens_file.unlink()
            except Exception as e:
                logger.error("No se pudo eliminar el archivo de tokens: %s", e)

    def auth_header(self) -> Dict[str, str]:
        # Nota: no chequeamos expiración aquí; _request se ocupa ante 401
        if not self.tokens.get("access_token"):
            self.login()
        token = self.tokens.get("access_token")
        if not token:
            return {}
        return {"Authorization": f"Bearer {token}"}


# =========================
# Cliente de API
# =========================

class IOLClient:
    """
    - Endpoints de cuenta (portafolio) vía requests + Bearer
    - Datos de mercado vía iolConn (sesión persistente y reautenticación)
    """

    def __init__(self, user: str, password: str, tokens_file: Path | str | None = None):
        self.user = (user or "").strip()
        self.password = (password or "").strip()
        #self.auth = IOLAuth(self.user, self.password)
        self.auth = _LegacyIOLAuth(self.user, self.password, tokens_file=tokens_file)
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
            try:
                r = self.session.request(method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs)
                if r.status_code == 401:
                    # Token expirado o inválido -> refresh y reintento 1 vez
                    self.auth.refresh()
                    headers = kwargs.pop("headers", {})
                    headers.update(self.auth.auth_header())
                    r = self.session.request(method, url, headers=headers, timeout=REQ_TIMEOUT, **kwargs)
                r.raise_for_status()
                return r
            except requests.HTTPError as e:
                last_exc = e
                if e.response is not None and e.response.status_code == 404:
                    logger.warning("%s %s devolvió 404", method, url)
                    return None
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
    def _ensure_market_auth(self) -> None:
        """
        Inicializa/Reautentica iolConn una vez (thread-safe).
        """
        if self._market_ready and self.iol_market is not None:
            return
        with self._market_lock:
            if self._market_ready and self.iol_market is not None:
                return
            self.iol_market = Iol(self.user, self.password)
            try:
                # algunas versiones requieren gestionar() para renovar bearer interno
                self.iol_market.gestionar()
            except NoAuthException:
                # recrea sesión y reintenta una vez
                self.iol_market = Iol(self.user, self.password)
                self.iol_market.gestionar()
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
    def _parse_chg_pct_fields(d: Dict[str, Any]) -> Optional[float]:
        """
        Extrae variación porcentual en % (no fracción), tolerando strings con coma/porcentual.
        """
        if not isinstance(d, dict):
            return None
        for k in ("variacion", "variacionPorcentual", "cambioPorcentual", "changePercent"):
            v = d.get(k)
            if isinstance(v, (int, float)):
                return float(v)
            if isinstance(v, str):
                v = v.replace("%", "").strip()
                f = _to_float(v)
                if f is not None:
                    return f
        return None

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

    def get_quote(self, mercado: str, simbolo: str) -> Dict[str, Any]:
        """
        Devuelve {"last": float|None, "chg_pct": float|None}
        chg_pct es la variación diaria en %, si está disponible.
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
            logger.warning("get_quote error %s:%s -> %s", mercado, simbolo, e)
            return {"last": None, "chg_pct": None}

        if not isinstance(data, dict):
            return {"last": None, "chg_pct": None}

        return {
            "last": self._parse_price_fields(data),
            "chg_pct": self._parse_chg_pct_fields(data),
        }

    # -------- Opcional: cotizaciones en batch --------
    def get_quotes_bulk(
        self,
        items: Iterable[Tuple[str, str]],
        max_workers: int = 8
    ) -> Dict[Tuple[str, str], Dict[str, Optional[float]]]:
        """
        Descarga cotizaciones para múltiples (mercado, símbolo) en paralelo.
        Retorna: { (mercado, símbolo): {"last": float|None, "chg_pct": float|None} }
        """
        pairs = [( (m or "bcba").lower(), (s or "").upper() ) for (m, s) in items]
        if not pairs:
            return {}
        self._ensure_market_auth()

        out: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
        with ThreadPoolExecutor(max_workers=min(max_workers, max(1, len(pairs)))) as ex:
            fut_map = {
                ex.submit(self.get_quote, m, s): (m, s)
                for (m, s) in pairs
            }
            for fut in as_completed(fut_map):
                key = fut_map[fut]
                try:
                    out[key] = fut.result()
                except Exception as e:
                    logger.warning("get_quotes_bulk %s:%s error -> %s", key[0], key[1], e)
                    out[key] = {"last": None, "chg_pct": None}
        return out
