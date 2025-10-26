# infrastructure/iol/client.py
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple

import requests
import streamlit as st
from iolConn import Iol
from iolConn.common.exceptions import NoAuthException

from infrastructure.iol.compat.session import LegacySession
from services.health import record_quote_provider_usage
from shared.config import settings
from shared.errors import InvalidCredentialsError
from shared.time_provider import TimeProvider
from shared.utils import _to_float

from .account_client import IOLAccountClient
from .auth import IOLAuth
from .ports import IIOLProvider

warnings.filterwarnings(
    "ignore",
    message="`infrastructure.iol.legacy` está deprecado",
    category=DeprecationWarning,
)


logger = logging.getLogger(__name__)

API_BASE_URL = "https://api.invertironline.com/api/v2"
PORTFOLIO_CACHE = Path(".cache/last_portfolio.json")
DEFAULT_COUNTRY = "argentina"
PORTFOLIO_URL = f"{API_BASE_URL}/portafolio/{{pais}}"
PROFILE_URL = f"{API_BASE_URL}/datos-perfil"

REQ_TIMEOUT = 30
RETRIES = 3
BACKOFF_SEC = 0.5
USER_AGENT = "IOL-Portfolio/1.0 (+iol_client)"

PROFILE_PREFERENCES_MAP = {
    "conservador": ["Liquidez", "Bonos Cortos", "FCI Cobertura"],
    "moderado": ["CEDEARs Diversificados", "FCI Balanceados"],
    "agresivo": ["Acciones", "CEDEARs", "ETFs"],
}


def _elapsed_ms(start_ts: float) -> int:
    return int((time.time() - start_ts) * 1000)


class IOLClient(IIOLProvider):
    """Native IOL client implementing the IIOLProvider contract."""

    def __init__(
        self,
        user: str,
        password: str,
        tokens_file: Path | str | None = None,
        auth: IOLAuth | None = None,
    ) -> None:
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

        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        self.api_base = API_BASE_URL

        self.iol_market: Optional[Iol] = None
        self._market_ready = False
        self._market_lock = threading.RLock()
        self._quotes_lock = threading.RLock()
        self._legacy_session = LegacySession.get()
        self._ensure_market_auth()
        self.account_client = IOLAccountClient(
            auth=self.auth,
            session=self.session,
            api_base=self.api_base,
        )

        safe_user = f"{self.user[:3]}***" if self.user else ""
        tokens_path = getattr(self.auth, "tokens_path", str(tokens_file))
        has_refresh = bool(getattr(self.auth, "refresh", None))
        logger.info(
            "IOLClient init",
            extra={
                "user": safe_user,
                "tokens_file": tokens_path,
                "has_refresh": has_refresh,
            },
        )

        self._legacy_last_http_label: Optional[str] = None
        self._active_batch_stats: Optional[dict[str, Any]] = None
        self._last_bulk_stats: Optional[dict[str, Any]] = None
        self._notification_cache: Optional[dict[str, Any]] = None
        self._notification_cache_marker: Optional[str] = None

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------
    def _record_batch_rate_limit(self) -> None:
        stats = self._active_batch_stats
        if stats is None:
            return
        stats["rate_limited"] = int(stats.get("rate_limited", 0) or 0) + 1

    def _record_batch_result(
        self,
        market: str,
        symbol: str,
        *,
        provider: str | None,
        elapsed_ms: float | None,
        stale: bool,
        fallback: bool,
        error: bool,
    ) -> None:
        stats = self._active_batch_stats
        if stats is None:
            return

        provider_key = (provider or "").strip().lower() or None
        stats["count"] = int(stats.get("count", 0) or 0) + 1
        if error:
            stats["errors"] = int(stats.get("errors", 0) or 0) + 1
        elif stale:
            stats["stale"] = int(stats.get("stale", 0) or 0) + 1
        else:
            stats["fresh"] = int(stats.get("fresh", 0) or 0) + 1

        if fallback or (provider_key not in (None, "iol", "cache")):
            stats["fallbacks"] = int(stats.get("fallbacks", 0) or 0) + 1

        total_elapsed = float(elapsed_ms) if elapsed_ms else 0.0
        stats["elapsed_ms_total"] = float(stats.get("elapsed_ms_total", 0.0)) + total_elapsed

        details = stats.setdefault("details", [])
        if isinstance(details, list):
            details.append(
                {
                    "market": market,
                    "symbol": symbol,
                    "provider": provider_key,
                    "stale": stale,
                    "error": error,
                    "elapsed_ms": total_elapsed,
                    "fallback": fallback or (provider_key not in (None, "iol", "cache")),
                }
            )

    def _request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        last_exc: Optional[Exception] = None
        for attempt in range(RETRIES + 1):
            try:
                self.auth.ensure_token(silent=True)
            except InvalidCredentialsError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt >= RETRIES:
                    raise
                time.sleep(BACKOFF_SEC * (attempt + 1))
                continue
            headers = kwargs.pop("headers", {})
            headers.update(self.auth.auth_header())
            delay = BACKOFF_SEC * (attempt + 1)
            try:
                response = self.session.request(
                    method,
                    url,
                    headers=headers,
                    timeout=REQ_TIMEOUT,
                    **kwargs,
                )
                if response.status_code == 401:
                    try:
                        self.auth.refresh()
                    except InvalidCredentialsError:
                        raise
                    headers = kwargs.pop("headers", {})
                    headers.update(self.auth.auth_header())
                    response = self.session.request(
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
            except requests.HTTPError as exc:
                last_exc = exc
                status_code = exc.response.status_code if exc.response is not None else None
                if status_code == 404:
                    logger.warning("%s %s devolvió 404", method, url)
                    return None
                if status_code == 429:
                    self._record_batch_rate_limit()
                    delay = BACKOFF_SEC * (2**attempt)
            except requests.RequestException as exc:
                last_exc = exc

            if attempt < RETRIES:
                time.sleep(delay)

        if last_exc:
            if (
                isinstance(last_exc, requests.HTTPError)
                and last_exc.response is not None
                and last_exc.response.status_code >= 500
            ):
                raise last_exc
            logger.warning("Request %s %s falló: %s", method, url, last_exc)
        return None

    @staticmethod
    def _notification_token_marker(tokens: Mapping[str, Any] | None) -> Optional[str]:
        if not isinstance(tokens, Mapping):
            return None
        raw = tokens.get("access_token") or tokens.get("refresh_token")
        if raw is None:
            raw = tokens.get("timestamp")
        if raw is None:
            return None
        try:
            value = str(raw).encode("utf-8")
        except Exception:
            return None
        return hashlib.sha256(value).hexdigest()

    @staticmethod
    def _parse_notification_datetime(value: Any) -> Optional[datetime]:
        if value is None:
            return None
        if isinstance(value, datetime):
            dt = value
        elif isinstance(value, (int, float)):
            snapshot = TimeProvider.from_timestamp(value)
            return snapshot.moment if snapshot else None
        elif isinstance(value, str):
            text = value.strip()
            if not text:
                return None
            candidates = (text, text.replace("Z", "+00:00"))
            for candidate in candidates:
                try:
                    dt = datetime.fromisoformat(candidate)
                except ValueError:
                    continue
                else:
                    break
            else:
                snapshot = TimeProvider.from_timestamp(text)
                return snapshot.moment if snapshot else None
        else:
            return None

        tz = TimeProvider.timezone()
        if dt.tzinfo is None:
            try:
                dt = dt.replace(tzinfo=tz)
            except Exception:
                return dt
        else:
            try:
                dt = dt.astimezone(tz)
            except Exception:
                return dt
        return dt

    def get_profile(self) -> dict[str, Any] | None:
        """Fetch user profile and investor test info from /api/v2/datos-perfil."""

        try:
            response = self._request("GET", PROFILE_URL)
        except InvalidCredentialsError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("get_profile request error: %s", exc, exc_info=True)
            return None

        if response is None or not response.content:
            return None

        try:
            payload = response.json() or {}
        except ValueError as exc:
            logger.warning("get_profile JSON inválido: %s", exc)
            return None

        if not isinstance(payload, Mapping):
            logger.info(
                "get_profile payload inesperado",
                extra={"payload_type": type(payload).__name__},
            )
            return None

        def _sanitize_text(value: Any) -> Optional[str]:
            if value is None:
                return None
            text = str(value).strip()
            return text or None

        nombre = _sanitize_text(payload.get("nombre"))

        test_info = payload.get("testInversor")
        perfil_inversor: Optional[str] = None
        vigencia: Optional[datetime] = None
        preferencias: Optional[list[str]] = None

        if isinstance(test_info, Mapping):
            activo = test_info.get("activo")
            if isinstance(activo, bool) and not activo:
                return None
            if isinstance(activo, str) and activo.strip().lower() in {"false", "0", "no", "n"}:
                return None
            vigente = test_info.get("vigente")
            if isinstance(vigente, bool) and not vigente:
                return None
            if isinstance(vigente, str) and vigente.strip().lower() in {"false", "0", "no", "n"}:
                return None

            perfil_inversor = _sanitize_text(test_info.get("perfil"))
            vigencia = self._parse_notification_datetime(test_info.get("fechaVencimiento"))

            if isinstance(vigencia, datetime):
                now = TimeProvider.now_datetime()
                try:
                    candidate = (
                        vigencia.astimezone(now.tzinfo) if vigencia.tzinfo is not None else vigencia.replace(tzinfo=now.tzinfo)
                    )
                except Exception:  # pragma: no cover - defensive guard
                    candidate = vigencia
                if candidate < now:
                    return None

            if perfil_inversor:
                preferencias_base = PROFILE_PREFERENCES_MAP.get(perfil_inversor.casefold())
                if preferencias_base:
                    preferencias = list(preferencias_base)

        normalized = {
            "nombre": nombre,
            "perfil_inversor": perfil_inversor,
            "vigencia": vigencia,
            "preferencias": preferencias,
        }

        return normalized

    def get_notification(self) -> Optional[dict[str, Any]]:
        marker = self._notification_token_marker(getattr(self.auth, "tokens", None))
        if marker == self._notification_cache_marker and self._notification_cache is not None:
            return dict(self._notification_cache)

        url = f"{self.api_base}/Notificacion"
        try:
            response = self._request("GET", url)
        except InvalidCredentialsError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            logger.warning("get_notification request error: %s", exc, exc_info=True)
            self._notification_cache = None
            self._notification_cache_marker = marker
            return None

        if response is None:
            self._notification_cache = None
            self._notification_cache_marker = marker
            return None

        status_code = response.status_code
        if status_code == 204 or not response.content:
            logger.info(
                "IOL notification fetch vacío",
                extra={"notification_status": status_code},
            )
            self._notification_cache = None
            self._notification_cache_marker = marker
            return None

        try:
            payload = response.json() or {}
        except ValueError as exc:
            logger.warning("get_notification JSON inválido: %s", exc)
            self._notification_cache = None
            self._notification_cache_marker = marker
            return None

        if not isinstance(payload, Mapping):
            logger.info(
                "IOL notification payload inesperado",
                extra={"payload_type": type(payload).__name__},
            )
            self._notification_cache = None
            self._notification_cache_marker = marker
            return None

        raw_message = payload.get("mensaje") or payload.get("message")
        message = raw_message.strip() if isinstance(raw_message, str) else None

        activo_raw = payload.get("activo")
        if isinstance(activo_raw, bool):
            activo = activo_raw
        elif isinstance(activo_raw, str):
            activo = activo_raw.strip().lower() in {"1", "true", "t", "yes", "si", "sí"}
        elif activo_raw is None:
            activo = True
        else:
            activo = bool(activo_raw)

        vigencia = self._parse_notification_datetime(payload.get("vigencia"))
        vigencia_hasta = self._parse_notification_datetime(
            payload.get("vigenciaHasta")
            or payload.get("vigenciaFin")
            or payload.get("vigenciaHastaUtc")
            or payload.get("vigenciaHastaUTC")
        )

        log_extra = {
            "notification_status": status_code,
            "notification_active": bool(activo),
            "notification_has_message": bool(message),
            "notification_vigencia": vigencia.isoformat() if isinstance(vigencia, datetime) else None,
            "notification_vigencia_hasta": (
                vigencia_hasta.isoformat() if isinstance(vigencia_hasta, datetime) else None
            ),
        }
        logger.info("IOL notification fetch", extra=log_extra)

        now = TimeProvider.now_datetime()
        expiry = vigencia_hasta or vigencia
        if not activo or not message:
            normalized = None
        elif expiry and expiry < now:
            normalized = None
        else:
            normalized = {
                "mensaje": message,
                "activo": bool(activo),
                "vigencia": vigencia,
                "vigencia_hasta": vigencia_hasta,
            }

        self._notification_cache = normalized
        self._notification_cache_marker = marker
        return dict(normalized) if normalized is not None else None

    # ------------------------------------------------------------------
    # Portfolio
    # ------------------------------------------------------------------
    def _fetch_portfolio_live(self, country: str) -> Dict[str, Any]:
        start = time.time()
        country_slug = (country or DEFAULT_COUNTRY).strip().lower() or DEFAULT_COUNTRY
        response = self._request("GET", PORTFOLIO_URL.format(pais=country_slug))
        elapsed = _elapsed_ms(start)
        logger.info("get_portfolio %s ms", elapsed)
        if response is None:
            raise requests.RequestException("Respuesta vacía del endpoint de portafolio")
        try:
            data = response.json() or {}
        except ValueError as exc:
            raise requests.RequestException("Respuesta inválida de portafolio") from exc
        return data

    def _write_portfolio_cache(self, data: Dict[str, Any]) -> None:
        try:
            PORTFOLIO_CACHE.parent.mkdir(parents=True, exist_ok=True)
            PORTFOLIO_CACHE.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:  # pragma: no cover - warning path
            logger.warning("No se pudo guardar cache portafolio: %s", exc, exc_info=True)

    def _load_portfolio_cache(self) -> Dict[str, Any]:
        try:
            raw = PORTFOLIO_CACHE.read_text(encoding="utf-8")
            return json.loads(raw)
        except (OSError, json.JSONDecodeError) as exc:
            logger.error("No se pudo leer cache portafolio: %s", exc, exc_info=True)
            return {"activos": []}

    def get_portfolio(self, country: str = DEFAULT_COUNTRY) -> Dict[str, Any]:
        try:
            data = self._fetch_portfolio_live(country)
        except InvalidCredentialsError:
            raise
        except requests.RequestException as exc:
            logger.warning("get_portfolio falló: %s", exc, exc_info=True)
            return self._load_portfolio_cache()
        except Exception:
            logger.exception("get_portfolio falló inesperadamente")
            raise
        else:
            try:
                cash_summary = self.account_client.fetch_balances()
            except InvalidCredentialsError:
                raise
            except requests.RequestException as exc:
                logger.info("No se pudo obtener /estadocuenta: %s", exc)
            except Exception:
                logger.debug("Error inesperado consultando /estadocuenta", exc_info=True)
            else:
                data.setdefault("_cash_balances", cash_summary.to_payload())
            self._write_portfolio_cache(data)
            return data

    # ------------------------------------------------------------------
    # Market helpers
    # ------------------------------------------------------------------
    def _ensure_market_auth(self) -> None:
        if self._market_ready and self.iol_market is not None:
            return
        with self._market_lock:
            if self._market_ready and self.iol_market is not None:
                return

            self.iol_market = Iol(self.user, self.password)
            bearer = self.auth.tokens.get("access_token")
            refresh = self.auth.tokens.get("refresh_token")
            if bearer and refresh:
                self.iol_market.bearer = bearer
                self.iol_market.refresh_token = refresh
                bearer_time = TimeProvider.now_datetime().replace(tzinfo=None)
                self.iol_market.bearer_time = bearer_time
            elif not self.password:
                st.session_state["force_login"] = True
                raise InvalidCredentialsError("Token inválido")

            try:
                logger.debug("Autenticando mercado con bearer")
                self.iol_market.gestionar()
                logger.info("Autenticación mercado con bearer ok")
            except NoAuthException:
                logger.info("Bearer inválido; autenticando con contraseña")
                self.iol_market = Iol(self.user, self.password)
                try:
                    self.iol_market.gestionar()
                    logger.info("Autenticación mercado con contraseña ok")
                except NoAuthException as exc:
                    logger.error("Autenticación mercado con contraseña falló", exc_info=True)
                    raise exc
            self._market_ready = True

    # ------------------------------------------------------------------
    # Quote parsing helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _parse_price_fields(data: Dict[str, Any]) -> Optional[float]:
        if not isinstance(data, dict):
            return None
        for key in (
            "ultimoPrecio",
            "ultimo",
            "last",
            "lastPrice",
            "precio",
            "cierre",
            "close",
        ):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                parsed = _to_float(value)
                if parsed is not None:
                    return parsed
            if isinstance(value, dict):
                for nested_key in ("value", "amount", "precio"):
                    nested_value = value.get(nested_key)
                    parsed = _to_float(nested_value)
                    if parsed is not None:
                        return parsed
        return None

    @staticmethod
    def _parse_chg_pct_fields(data: Dict[str, Any], last: Optional[float] = None) -> Optional[float]:
        if not isinstance(data, dict):
            return None

        for key in (
            "variacion",
            "variacionPorcentual",
            "cambioPorcentual",
            "changePercent",
        ):
            value = data.get(key)
            if isinstance(value, (int, float)):
                return float(value)
            if isinstance(value, str):
                value = value.replace("%", "").strip()
                parsed = _to_float(value)
                if parsed is not None:
                    return parsed

        prev_close: Optional[float] = None
        for key in ("cierreAnterior", "previousClose"):
            prev_close = _to_float(data.get(key))
            if prev_close is not None:
                break

        if prev_close is None:
            return None

        delta = _to_float(data.get("puntosVariacion"))
        if delta is not None:
            return delta / prev_close * 100 if prev_close else None

        if last is None:
            last = IOLClient._parse_price_fields(data)
        if last is None:
            return None

        return (last - prev_close) / prev_close * 100

    @classmethod
    def _normalize_quote_payload(cls, data: Any) -> Dict[str, Any]:
        if not isinstance(data, dict):
            return {
                "last": None,
                "chg_pct": None,
                "asof": None,
                "provider": None,
                "moneda_origen": None,
                "proveedor_original": None,
                "fx_aplicado": None,
            }

        last = cls._parse_price_fields(data)
        chg_pct = cls._parse_chg_pct_fields(data, last)

        asof: Optional[str]
        asof_value = data.get("asof")
        if asof_value is None:
            for candidate in (
                "fechaHora",  # legacy IOL
                "fecha",  # v2 payloads
                "timestamp",
                "time",
                "date",
                "ts",
            ):
                asof_value = data.get(candidate)
                if asof_value:
                    break
        if hasattr(asof_value, "isoformat"):
            asof = asof_value.isoformat()
        elif isinstance(asof_value, (int, float)):
            asof = str(float(asof_value))
        elif isinstance(asof_value, str):
            asof = asof_value.strip() or None
        else:
            asof = None

        provider_raw = data.get("provider")
        if isinstance(provider_raw, str):
            provider = provider_raw.strip() or None
        elif provider_raw is None:
            provider = "iol"
        else:
            provider = str(provider_raw)
        currency_raw = data.get("moneda")
        if currency_raw is None:
            currency_raw = data.get("currency")
        if currency_raw is None:
            currency_raw = data.get("currency_base")
        if isinstance(currency_raw, str):
            currency_value: Optional[str] = currency_raw.strip() or None
        elif currency_raw is None:
            currency_value = None
        else:
            currency_value = str(currency_raw)

        fx_raw = data.get("fx_aplicado")
        if fx_raw is None:
            fx_raw = data.get("fxAplicado")
        if fx_raw is None:
            fx_raw = data.get("fx_applied")
        if isinstance(fx_raw, (int, float)):
            fx_value: Optional[float | str] = float(fx_raw)
        elif isinstance(fx_raw, str):
            fx_value = fx_raw.strip() or None
        else:
            fx_value = None

        payload: Dict[str, Any] = {
            "last": last,
            "chg_pct": chg_pct,
            "asof": asof,
            "provider": provider,
            "moneda_origen": currency_value,
            "proveedor_original": provider,
            "fx_aplicado": fx_value,
        }
        if currency_value is not None:
            payload["currency"] = currency_value
        return payload

    # ------------------------------------------------------------------
    # Quotes helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _map_symbol_for_adapter(market: str, symbol: str) -> str:
        market_key = (market or "").strip().lower()
        ticker = (symbol or "").strip().upper()
        if not ticker:
            return ticker
        if market_key not in {"bcba", "merv"}:
            return ticker
        try:
            from shared.config import get_config

            cfg = get_config()
        except Exception:  # pragma: no cover - defensive guard
            return ticker

        mapping = cfg.get("cedear_to_us") if isinstance(cfg, dict) else None
        if isinstance(mapping, dict):
            mapped = mapping.get(ticker)
            if isinstance(mapped, str) and mapped.strip():
                return mapped.strip().upper()
        return ticker

    @staticmethod
    def _normalize_ohlc_payload(frame: Any, provider: str) -> Dict[str, Optional[float]]:
        try:
            import pandas as pd  # type: ignore
        except Exception:  # pragma: no cover - defensive import
            pd = None  # type: ignore

        if pd is None or frame is None:
            return {"last": None, "chg_pct": None, "asof": None, "provider": None}

        try:
            df = frame if isinstance(frame, pd.DataFrame) else None
        except Exception:  # pragma: no cover - defensive guard
            df = None
        if df is None or df.empty:
            return {"last": None, "chg_pct": None, "asof": None, "provider": None}

        df_sorted = df.sort_index()
        last_row = df_sorted.iloc[-1]
        last_close = _to_float(last_row.get("Close"))
        if last_close is None:
            return {"last": None, "chg_pct": None, "asof": None, "provider": None}

        prev_close = None
        if len(df_sorted.index) >= 2:
            prev_close = _to_float(df_sorted.iloc[-2].get("Close"))
        chg_pct = None
        if prev_close not in (None, 0):
            chg_pct = (last_close - prev_close) / prev_close * 100.0

        timestamp = df_sorted.index[-1]
        if hasattr(timestamp, "isoformat"):
            asof = timestamp.isoformat()
        else:
            asof = str(timestamp)

        provider_key = str(provider or "").strip().lower()
        if provider_key == "alpha_vantage":
            provider_key = "av"

        return {
            "last": last_close,
            "chg_pct": chg_pct,
            "asof": asof,
            "provider": provider_key or None,
        }

    def _log_quote_event(
        self,
        market: str,
        symbol: str,
        *,
        phase: str,
        result: str,
        provider: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "quote_market": market,
            "quote_symbol": symbol,
            "quote_phase": phase,
            "quote_result": result,
        }
        if provider:
            payload["quote_provider"] = provider
        if detail:
            payload["quote_detail"] = detail
        logger.info(
            "quote_event %s:%s phase=%s result=%s",
            market,
            symbol,
            phase,
            result,
            extra=payload,
        )

    def _fallback_quote_via_ohlc(
        self,
        market: str,
        symbol: str,
        *,
        panel: str | None = None,
    ) -> Dict[str, Optional[float]] | None:
        mapped_symbol = self._map_symbol_for_adapter(market, symbol)
        params = {"period": "1mo", "interval": "1d"}
        try:
            from services.ohlc_adapter import OHLCAdapter

            adapter = OHLCAdapter()
            frame = adapter.fetch(mapped_symbol, **params)
            provider_name: Optional[str] = None
            cache_key = adapter._make_cache_key(mapped_symbol, params)  # type: ignore[attr-defined]
            cache_store = getattr(adapter, "_cache", {})
            entry = cache_store.get(cache_key) if isinstance(cache_store, dict) else None
            if entry is not None:
                provider_name = getattr(entry, "provider", None)
            payload = self._normalize_ohlc_payload(frame, provider_name or "")
            if payload["last"] is not None:
                logger.info(
                    "IOLClient fallback OHLCAdapter %s:%s -> %s",
                    market,
                    symbol,
                    payload.get("provider"),
                )
                self._log_quote_event(
                    market,
                    symbol,
                    phase="ohlc",
                    result="success",
                    provider=str(payload.get("provider") or "ohlc"),
                )
                return payload
        except Exception as exc:
            logger.warning(
                "IOLClient OHLCAdapter fallback failed %s:%s -> %s",
                market,
                symbol,
                exc,
            )
            self._log_quote_event(
                market,
                symbol,
                phase="ohlc",
                result="error",
                detail=str(exc),
        )
        return None

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------
    def fetch_market_price(
        self,
        symbol: str,
        *,
        market: str = "BCBA",
    ) -> tuple[float | None, str | None]:
        """Fetch a real-time price for ``symbol`` directly from the market endpoints."""

        resolved_symbol = str(symbol or "").strip().upper()
        if not resolved_symbol:
            return None, None

        resolved_market = str(market or "BCBA").strip().upper() or "BCBA"
        base_url = getattr(self, "_base", self.api_base).rstrip("/")
        endpoints = (
            (f"{base_url}/{resolved_market}/Titulos/{resolved_symbol}/Cotizacion", "cotizacion"),
            (
                f"{base_url}/{resolved_market}/Titulos/{resolved_symbol}/CotizacionDetalle",
                "cotizacion_detalle",
            ),
        )

        def _coerce(raw: Any) -> float | None:
            value = _to_float(raw)
            if value is None:
                return None
            try:
                if not math.isfinite(value):
                    return None
            except (TypeError, ValueError):
                return None
            if value <= 0:
                return None
            return float(value)

        for url, label in endpoints:
            try:
                response = self._request("GET", url)
            except InvalidCredentialsError:
                raise
            except Exception as exc:  # pragma: no cover - defensive guard
                logger.debug("fetch_market_price request failed", exc_info=True)
                logger.warning("fetch_market_price %s error: %s", url, exc)
                continue

            if response is None:
                continue

            try:
                payload = response.json() or {}
            except ValueError as exc:  # pragma: no cover - unexpected payload
                logger.warning("fetch_market_price JSON inválido desde %s: %s", url, exc)
                continue

            last_price = _coerce(payload.get("ultimoPrecio"))
            if last_price is not None:
                logger.info(
                    "fetch_market_price resolved",
                    extra={
                        "symbol": resolved_symbol,
                        "market": resolved_market,
                        "source": label,
                        "last": last_price,
                    },
                )
                return last_price, url

            bid_price: float | None = None
            ask_price: float | None = None
            for key in ("precioCompra", "precioCompraPuntas", "mejorPrecioCompra"):
                bid_price = _coerce(payload.get(key))
                if bid_price is not None:
                    break
            for key in ("precioVenta", "precioVentaPuntas", "mejorPrecioVenta"):
                ask_price = _coerce(payload.get(key))
                if ask_price is not None:
                    break

            candidates = [price for price in (bid_price, ask_price) if price is not None]
            if not candidates:
                continue

            if len(candidates) == 2:
                last_price = sum(candidates) / 2.0
            else:
                last_price = candidates[0]

            logger.info(
                "fetch_market_price resolved via book",
                extra={
                    "symbol": resolved_symbol,
                    "market": resolved_market,
                    "source": label,
                    "bid": bid_price,
                    "ask": ask_price,
                    "last": last_price,
                },
            )
            return float(last_price), url

        return None, None

    def get_last_price(self, *, mercado: str, simbolo: str) -> Optional[float]:
        mercado = (mercado or "bcba").lower()
        simbolo = (simbolo or "").upper()
        try:
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=mercado, simbolo=simbolo)
        except NoAuthException:
            self._market_ready = False
            self._ensure_market_auth()
            data = self.iol_market.price_to_json(mercado=mercado, simbolo=simbolo)
        except Exception as exc:
            logger.warning("get_last_price error %s:%s -> %s", mercado, simbolo, exc)
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
    ) -> Optional[Dict[str, Any]]:
        resolved_market = (mercado if mercado is not None else market or "bcba").lower()
        resolved_symbol = (simbolo if simbolo is not None else symbol or "").upper()
        base_url = getattr(self, "_base", self.api_base).rstrip("/")
        url = f"{base_url}/{resolved_market}/Titulos/{resolved_symbol}/Cotizacion"
        params: dict[str, str] | None = None
        if panel:
            params = {"panel": panel}

        transitions = ["v2"]
        start = time.time()
        response: Optional[requests.Response] = None
        v2_error: Exception | None = None

        try:
            request_kwargs = {"params": params} if params else {}
            response = self._request("GET", url, **request_kwargs)
        except InvalidCredentialsError:
            self._log_quote_event(
                resolved_market,
                resolved_symbol,
                phase="iol_v2",
                result="auth_error",
            )
            raise
        except Exception as exc:  # pragma: no cover - defensive guard
            v2_error = exc

        if response is not None:
            status_code = getattr(response, "status_code", "n/a")
            logger.debug("IOLClient.get_quote -> %s [%s]", url, status_code)
            try:
                data = response.json() or {}
            except ValueError as exc:
                v2_error = exc
            else:
                payload = self._normalize_quote_payload(data)
                payload["provider"] = "iol"
                if payload.get("proveedor_original") is None:
                    payload["proveedor_original"] = "iol"
                elapsed_ms = (time.time() - start) * 1000.0
                provider_name = payload.get("provider") or "iol"
                last_value = payload.get("last")
                if last_value is not None:
                    record_quote_provider_usage(
                        provider_name,
                        elapsed_ms=elapsed_ms,
                        stale=False,
                        source="v2",
                        ok=True,
                    )
                    self._record_batch_result(
                        resolved_market,
                        resolved_symbol,
                        provider=provider_name,
                        elapsed_ms=elapsed_ms,
                        stale=False,
                        fallback=False,
                        error=False,
                    )
                    return payload

                logger.warning(
                    "Quote payload missing last price from /Titulos/Cotizacion",
                    extra={
                        "market": resolved_market,
                        "symbol": resolved_symbol,
                        "panel": panel,
                    },
                )
                record_quote_provider_usage(
                    provider_name,
                    elapsed_ms=None,
                    stale=True,
                    source="v2-missing-last",
                    ok=False,
                )
                v2_error = RuntimeError("missing-last-price")

        if v2_error is None:
            v2_error = RuntimeError("empty-response")

        record_quote_provider_usage(
            "iol",
            elapsed_ms=None,
            stale=True,
            source="v2-error",
        )
        transitions.append("legacy")
        logger.info(
            "Quote fallback transition: %s for %s:%s -> %s",
            " -> ".join(transitions[-2:]),
            resolved_market,
            resolved_symbol,
            v2_error,
        )

        legacy_payload, legacy_flags = self._legacy_quote_fallback(resolved_market, resolved_symbol, panel)
        legacy_auth_unavailable = bool(legacy_flags.get("legacy_auth_unavailable"))
        legacy_provider = (legacy_payload.get("provider") if isinstance(legacy_payload, dict) else "legacy") or "legacy"

        if legacy_payload is not None and not legacy_auth_unavailable and legacy_payload.get("last") is not None:
            record_quote_provider_usage(
                legacy_provider,
                elapsed_ms=None,
                stale=False,
                source="v2->legacy",
            )
            logger.info(
                "Quote fallback chain resolved: %s for %s:%s",
                " -> ".join(transitions),
                resolved_market,
                resolved_symbol,
            )
            self._record_batch_result(
                resolved_market,
                resolved_symbol,
                provider=legacy_provider,
                elapsed_ms=(time.time() - start) * 1000.0,
                stale=False,
                fallback=True,
                error=False,
            )
            return legacy_payload

        record_quote_provider_usage(
            legacy_provider,
            elapsed_ms=None,
            stale=True,
            source="v2->legacy",
        )

        transitions.append("ohlc")
        logger.info(
            "Quote fallback transition: %s for %s:%s",
            " -> ".join(transitions[-2:]),
            resolved_market,
            resolved_symbol,
        )
        fallback = self._fallback_quote_via_ohlc(resolved_market, resolved_symbol, panel=panel)
        if fallback is not None:
            if legacy_auth_unavailable:
                fallback = dict(fallback)
                fallback["legacy_auth_unavailable"] = True
            provider = fallback.get("provider") or "ohlc"
            record_quote_provider_usage(
                provider,
                elapsed_ms=None,
                stale=fallback.get("last") is None,
                source="legacy->ohlc",
            )
            logger.info(
                "Quote fallback chain resolved: %s for %s:%s",
                " -> ".join(transitions),
                resolved_market,
                resolved_symbol,
            )
            self._record_batch_result(
                resolved_market,
                resolved_symbol,
                provider=provider,
                elapsed_ms=(time.time() - start) * 1000.0,
                stale=fallback.get("last") is None,
                fallback=True,
                error=False,
            )
            return fallback

        transitions.append("stale")
        stale_payload: Dict[str, Any] = {
            "last": None,
            "chg_pct": None,
            "asof": None,
            "provider": "stale",
            "proveedor_original": "stale",
            "moneda_origen": None,
            "fx_aplicado": None,
        }
        if legacy_auth_unavailable:
            stale_payload["legacy_auth_unavailable"] = True
        record_quote_provider_usage(
            "stale",
            elapsed_ms=None,
            stale=True,
            source="ohlc->stale",
        )
        logger.info(
            "Quote fallback chain resolved: %s for %s:%s",
            " -> ".join(transitions),
            resolved_market,
            resolved_symbol,
        )
        self._record_batch_result(
            resolved_market,
            resolved_symbol,
            provider="stale",
            elapsed_ms=(time.time() - start) * 1000.0,
            stale=True,
            fallback=True,
            error=False,
        )
        return stale_payload

    def _legacy_quote_fallback(
        self,
        resolved_market: str,
        resolved_symbol: str,
        panel: str | None,
    ) -> tuple[Optional[Dict[str, Optional[float]]], Dict[str, bool]]:
        flags: Dict[str, bool] = {}
        try:
            from infrastructure.iol.compat.iol_client import (
                IOLClient as LegacyIOLClient,
            )

            legacy_client = LegacyIOLClient(
                self.user,
                self.password,
                tokens_file=getattr(self.auth, "tokens_path", None),
                auth=self.auth,
            )
            payload = legacy_client.get_quote(
                market=resolved_market,
                symbol=resolved_symbol,
                panel=panel,
            )
        except requests.HTTPError as http_exc:
            status = None
            if getattr(http_exc, "response", None) is not None:
                status = http_exc.response.status_code
            if status == 429:
                self._legacy_last_http_label = "legacy_429"
            elif status in (401, 403):
                self._legacy_last_http_label = "legacy_auth_fail"
            else:
                logger.warning(
                    "Legacy IOLClient.get_quote HTTP error %s:%s -> %s",
                    resolved_market,
                    resolved_symbol,
                    status or http_exc,
                )
            return None, flags
        except requests.RequestException as req_exc:
            logger.warning(
                "Legacy IOLClient.get_quote request error %s:%s -> %s",
                resolved_market,
                resolved_symbol,
                req_exc,
            )
            return None, flags
        except Exception as fallback_exc:  # pragma: no cover - defensive guard
            logger.error(
                "Fallback legacy IOLClient.get_quote falló %s:%s -> %s",
                resolved_market,
                resolved_symbol,
                fallback_exc,
                exc_info=True,
            )
            return None, flags

        if not isinstance(payload, dict):
            return None, flags

        normalized = dict(payload)
        if normalized.get("legacy_auth_unavailable"):
            flags["legacy_auth_unavailable"] = True
        normalized.setdefault("provider", "legacy")
        normalized.setdefault("proveedor_original", normalized.get("provider"))
        return normalized, flags

    def get_quotes_bulk(
        self,
        items: Iterable[Tuple[str, str] | Tuple[str, str, str | None]],
        max_workers: int = 8,
    ) -> Dict[Tuple[str, str], Dict[str, Any]]:
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

        result: Dict[Tuple[str, str], Dict[str, Optional[float]]] = {}
        batch_start = time.time()
        stats: dict[str, Any] = {
            "count": 0,
            "fresh": 0,
            "stale": 0,
            "errors": 0,
            "fallbacks": 0,
            "rate_limited": 0,
            "elapsed_ms_total": 0.0,
            "details": [],
        }
        previous_stats = self._active_batch_stats
        self._active_batch_stats = stats
        try:
            with self._quotes_lock:
                for market, symbol, panel in requests:
                    pre_count = int(stats.get("count", 0) or 0)
                    try:
                        payload = self.get_quote(market, symbol, panel)
                    except Exception as exc:  # pragma: no cover - defensive guard
                        logger.warning("get_quotes_bulk %s:%s error -> %s", market, symbol, exc)
                        self._record_batch_result(
                            market,
                            symbol,
                            provider="error",
                            elapsed_ms=None,
                            stale=True,
                            fallback=True,
                            error=True,
                        )
                        payload = self._normalize_quote_payload({"provider": "error"})
                    else:
                        if payload is None and int(stats.get("count", 0) or 0) == pre_count:
                            self._record_batch_result(
                                market,
                                symbol,
                                provider="error",
                                elapsed_ms=None,
                                stale=True,
                                fallback=True,
                                error=True,
                            )
                    if payload is None:
                        payload = self._normalize_quote_payload({"provider": "error"})
                    result[(market, symbol)] = payload
        finally:
            stats["duration_ms"] = (time.time() - batch_start) * 1000.0
            details = stats.get("details")
            if isinstance(details, list):
                copied_details = [dict(item) for item in details if isinstance(item, dict)]
            else:
                copied_details = []
            summary = dict(stats)
            summary["details"] = copied_details
            self._last_bulk_stats = summary
            self._active_batch_stats = previous_stats
        return result


class IOLClientAdapter(IOLClient):
    """Backward compatible alias for the old adapter name."""


def build_iol_client(
    user: str,
    password: str,
    tokens_file: Path | str | None = None,
    auth: IOLAuth | None = None,
) -> IOLClient:
    return IOLClient(user, password, tokens_file=tokens_file, auth=auth)


__all__ = ["IOLClient", "IOLClientAdapter", "build_iol_client"]
