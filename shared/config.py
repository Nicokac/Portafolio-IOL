# shared/config.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional
from dotenv import load_dotenv
from functools import lru_cache
import logging
import sys
import streamlit as st

try:  # pragma: no cover - import may fail in tests
    from streamlit.runtime.secrets import StreamlitSecretNotFoundError
except Exception:  # pragma: no cover - streamlit may not expose runtime module
    class StreamlitSecretNotFoundError(Exception):
        """Fallback cuando streamlit no expone StreamlitSecretNotFoundError."""
        pass

logger = logging.getLogger(__name__)

# Raíz del proyecto (donde están app.py, .env, config.json, etc.)
BASE_DIR = Path(__file__).resolve().parents[1]

# Cargar variables del .env en la raíz (y fallback al cwd por si acaso)
load_dotenv(BASE_DIR / ".env")
load_dotenv()

def _load_cfg() -> Dict[str, Any]:
    """
    Carga (opcional) config.json desde la raíz del proyecto (o cwd). Si no existe, {}.
    """
    candidates = [BASE_DIR / "config.json", Path.cwd() / "config.json"]
    for p in candidates:
        try:
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            logger.exception("No se pudo cargar configuración %s: %s", p, e)
    return {}

class Settings:
    def __init__(self) -> None:
        cfg = _load_cfg()

        # --- Identidad / headers ---
        self.USER_AGENT: str = os.getenv("USER_AGENT", cfg.get("USER_AGENT", "IOL-Portfolio/1.0 (+app)"))

        # --- Servicios auxiliares ---
        self.NOTIFICATIONS_URL: str | None = self.secret_or_env(
            "NOTIFICATIONS_URL", cfg.get("NOTIFICATIONS_URL")
        )
        self.NOTIFICATIONS_TIMEOUT: float = float(
            os.getenv("NOTIFICATIONS_TIMEOUT", cfg.get("NOTIFICATIONS_TIMEOUT", 3.0))
        )

        # --- Macro data providers ---
        self.MACRO_API_PROVIDER: str = os.getenv(
            "MACRO_API_PROVIDER", cfg.get("MACRO_API_PROVIDER", "fred")
        )
        raw_series = self.secret_or_env("FRED_SECTOR_SERIES", cfg.get("FRED_SECTOR_SERIES"))
        self.FRED_SECTOR_SERIES: Dict[str, str] = self._parse_sector_series(raw_series)
        raw_fallback = self.secret_or_env(
            "MACRO_SECTOR_FALLBACK", cfg.get("MACRO_SECTOR_FALLBACK")
        )
        self.MACRO_SECTOR_FALLBACK: Dict[str, Dict[str, Any]] = self._parse_macro_fallback(
            raw_fallback
        )
        self.FRED_API_KEY: str | None = self.secret_or_env(
            "FRED_API_KEY", cfg.get("FRED_API_KEY")
        )
        self.FRED_API_BASE_URL: str = os.getenv(
            "FRED_API_BASE_URL",
            cfg.get("FRED_API_BASE_URL", "https://api.stlouisfed.org/fred"),
        )
        self.FRED_API_RATE_LIMIT_PER_MINUTE: int = int(
            os.getenv(
                "FRED_API_RATE_LIMIT_PER_MINUTE",
                cfg.get("FRED_API_RATE_LIMIT_PER_MINUTE", 120),
            )
        )

        default_markets = ["NASDAQ", "NYSE", "AMEX"]
        sentinel = object()
        raw_markets: Any = os.getenv("OPPORTUNITIES_TARGET_MARKETS", sentinel)
        if raw_markets is sentinel:
            raw_markets = cfg.get("OPPORTUNITIES_TARGET_MARKETS", default_markets)

        parsed_candidates: Any = raw_markets
        if isinstance(parsed_candidates, str):
            try:
                parsed_candidates = json.loads(parsed_candidates)
            except json.JSONDecodeError:
                parsed_candidates = [part.strip() for part in parsed_candidates.split(",")]

        markets: list[str] = []
        if isinstance(parsed_candidates, (list, tuple, set)):
            for candidate in parsed_candidates:
                text = str(candidate or "").strip()
                if text:
                    markets.append(text.upper())
        elif isinstance(parsed_candidates, str):
            text = parsed_candidates.strip()
            if text:
                markets.append(text.upper())

        self.OPPORTUNITIES_TARGET_MARKETS: list[str] = markets or [item.upper() for item in default_markets]


        # --- Credenciales IOL ---
        self.IOL_USERNAME: str | None = self.secret_or_env("IOL_USERNAME", cfg.get("IOL_USERNAME"))
        self.IOL_PASSWORD: str | None = self.secret_or_env("IOL_PASSWORD", cfg.get("IOL_PASSWORD"))

        # --- Cache/TTLs usados en app.py ---
        self.cache_ttl_portfolio: int = int(os.getenv("CACHE_TTL_PORTFOLIO", cfg.get("CACHE_TTL_PORTFOLIO", 20)))
        self.cache_ttl_last_price: int = int(os.getenv("CACHE_TTL_LAST_PRICE", cfg.get("CACHE_TTL_LAST_PRICE", 10)))
        self.cache_ttl_fx: int = int(os.getenv("CACHE_TTL_FX", cfg.get("CACHE_TTL_FX", 60)))
        self.cache_ttl_quotes: int = int(os.getenv("CACHE_TTL_QUOTES", cfg.get("CACHE_TTL_QUOTES", 8)))
        self.cache_ttl_yf_indicators: int = int(
            os.getenv("CACHE_TTL_YF_INDICATORS", cfg.get("CACHE_TTL_YF_INDICATORS", 900))
        )
        self.cache_ttl_yf_history: int = int(
            os.getenv("CACHE_TTL_YF_HISTORY", cfg.get("CACHE_TTL_YF_HISTORY", 3600))
        )
        self.cache_ttl_yf_fundamentals: int = int(
            os.getenv("CACHE_TTL_YF_FUNDAMENTALS", cfg.get("CACHE_TTL_YF_FUNDAMENTALS", 21600))
        )
        self.cache_ttl_yf_portfolio_fundamentals: int = int(
            os.getenv(
                "CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS",
                cfg.get("CACHE_TTL_YF_PORTFOLIO_FUNDAMENTALS", 14400),
            )
        )
        self.quotes_hist_maxlen: int = int(os.getenv("QUOTES_HIST_MAXLEN", cfg.get("QUOTES_HIST_MAXLEN", 500)))
        self.max_quote_workers: int = int(os.getenv("MAX_QUOTE_WORKERS", cfg.get("MAX_QUOTE_WORKERS", 12)))
        self.YAHOO_FUNDAMENTALS_TTL: int = int(
            os.getenv("YAHOO_FUNDAMENTALS_TTL", cfg.get("YAHOO_FUNDAMENTALS_TTL", 3600))
        )
        self.YAHOO_QUOTES_TTL: int = int(
            os.getenv("YAHOO_QUOTES_TTL", cfg.get("YAHOO_QUOTES_TTL", 300))
        )

        self.min_score_threshold: int = int(
            os.getenv("MIN_SCORE_THRESHOLD", cfg.get("MIN_SCORE_THRESHOLD", 80))
        )
        self.max_results: int = int(os.getenv("MAX_RESULTS", cfg.get("MAX_RESULTS", 20)))

        self.STUB_MAX_RUNTIME_WARN: float = float(
            os.getenv("STUB_MAX_RUNTIME_WARN", cfg.get("STUB_MAX_RUNTIME_WARN", 0.25))
        )

        flag_value = os.getenv(
            "FEATURE_OPPORTUNITIES_TAB",
            cfg.get("FEATURE_OPPORTUNITIES_TAB", "true"),
        )
        if isinstance(flag_value, bool):
            self.FEATURE_OPPORTUNITIES_TAB = flag_value
        else:
            self.FEATURE_OPPORTUNITIES_TAB = str(flag_value).lower() in {"1", "true", "yes", "on"}

        # --- Archivo de tokens (IOLAuth) ---
        # Por defecto lo guardamos en la raíz junto a app.py (compat con tu tokens_iol.json existente)
        self.tokens_file: str = self.secret_or_env(
            "IOL_TOKENS_FILE", cfg.get("IOL_TOKENS_FILE", str(BASE_DIR / "tokens_iol.json"))
        )
        # Clave opcional para cifrar/descifrar el archivo de tokens (Fernet)
        self.tokens_key: str | None = self.secret_or_env("IOL_TOKENS_KEY", cfg.get("IOL_TOKENS_KEY"))
        # Permite (opcionalmente) guardar tokens sin cifrar si falta tokens_key
        self.allow_plain_tokens: bool = (
            os.getenv("IOL_ALLOW_PLAIN_TOKENS", str(cfg.get("IOL_ALLOW_PLAIN_TOKENS", ""))).lower()
            in ("1", "true", "yes")
        )
        # TTL máximo para reutilizar tokens guardados (en días)
        self.tokens_ttl_days: int = int(
            os.getenv("IOL_TOKENS_TTL_DAYS", cfg.get("IOL_TOKENS_TTL_DAYS", 30))
        )

        # --- Derivados de dólar (Ahorro/Tarjeta a partir del oficial) ---
        self.fx_ahorro_multiplier: float = float(os.getenv("FX_AHORRO_MULTIPLIER", cfg.get("FX_AHORRO_MULTIPLIER", 1.30)))
        self.fx_tarjeta_multiplier: float = float(os.getenv("FX_TARJETA_MULTIPLIER", cfg.get("FX_TARJETA_MULTIPLIER", 1.35)))

        # --- Logging ---
        self.LOG_LEVEL: str = os.getenv("LOG_LEVEL", cfg.get("LOG_LEVEL", "INFO")).upper()
        self.LOG_FORMAT: str = os.getenv("LOG_FORMAT", cfg.get("LOG_FORMAT", "plain")).lower()

    def secret_or_env(self, key: str, default: Any | None = None) -> Any | None:
        try:
            return st.secrets[key]
        except (KeyError, StreamlitSecretNotFoundError, AttributeError):
            return os.getenv(key, default)

    @staticmethod
    def _parse_jsonish(raw: Any) -> Any:
        if raw is None:
            return None
        if isinstance(raw, str):
            try:
                return json.loads(raw)
            except json.JSONDecodeError:
                return raw
        return raw

    def _parse_sector_series(self, raw: Any) -> Dict[str, str]:
        parsed = self._parse_jsonish(raw)
        if not isinstance(parsed, Mapping):
            return {}
        mapping: Dict[str, str] = {}
        for key, value in parsed.items():
            label = str(key or "").strip()
            if not label:
                continue
            series_id: str = ""
            if isinstance(value, Mapping):
                series_id = str(
                    value.get("series_id")
                    or value.get("series")
                    or value.get("id")
                    or ""
                ).strip()
            else:
                series_id = str(value or "").strip()
            if not series_id:
                continue
            mapping[label] = series_id
        return mapping

    def _parse_macro_fallback(self, raw: Any) -> Dict[str, Dict[str, Any]]:
        parsed = self._parse_jsonish(raw)
        if not isinstance(parsed, Mapping):
            return {}
        fallback: Dict[str, Dict[str, Any]] = {}
        for key, value in parsed.items():
            label = str(key or "").strip()
            if not label:
                continue
            numeric_value: Optional[float] = None
            as_of: Optional[str] = None
            if isinstance(value, Mapping):
                raw_value = value.get("value")
                try:
                    numeric_value = float(raw_value)
                except (TypeError, ValueError):
                    continue
                raw_as_of = value.get("as_of")
                if raw_as_of is not None:
                    text = str(raw_as_of).strip()
                    if text:
                        as_of = text
            else:
                try:
                    numeric_value = float(value)
                except (TypeError, ValueError):
                    continue
            entry: Dict[str, Any] = {"value": numeric_value}
            if as_of:
                entry["as_of"] = as_of
            fallback[label] = entry
        return fallback

settings = Settings()


def ensure_tokens_key() -> None:
    """Verifica que exista una clave para cifrar tokens.

    Si falta ``IOL_TOKENS_KEY`` y no se habilitó ``IOL_ALLOW_PLAIN_TOKENS``,
    se registra un error y se aborta la ejecución con ``sys.exit(1)``.
    """

    if not settings.tokens_key and not settings.allow_plain_tokens:
        logger.error(
            "IOL_TOKENS_KEY no está configurada y IOL_ALLOW_PLAIN_TOKENS no está habilitado."
        )
        sys.exit(1)


class JsonFormatter(logging.Formatter):
    """Formato JSON simple para registros de log."""

    def format(self, record: logging.LogRecord) -> str:
        log_record = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        user = os.getenv("LOG_USER")
        if user:
            log_record["user"] = user
        return json.dumps(log_record)


def configure_logging(level: str | None = None, json_format: bool | None = None) -> None:
    """Configura el logging global.

    Por defecto usa nivel ``INFO`` y formato ``"plain"``. Los valores
    configurados se normalizan y, si son inválidos, se revierte a estos
    predeterminados. Los parámetros permiten sobrescribir el nivel y el
    formato configurados mediante variables de entorno.
    """

    level_name = (level or getattr(settings, "LOG_LEVEL", "INFO")).upper()
    level_value = getattr(logging, level_name, None)
    if not isinstance(level_value, int):
        level_name = "INFO"
        level_value = logging.INFO

    if json_format is None:
        fmt = os.getenv("LOG_FORMAT", getattr(settings, "LOG_FORMAT", "plain"))
        fmt = str(fmt).lower()
        if fmt not in {"json", "plain"}:
            fmt = "plain"
        json_format = fmt == "json"

    if json_format:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter(datefmt="%Y-%m-%d %H:%M:%S"))
        root = logging.getLogger()
        root.setLevel(level_value)
        root.handlers = [handler]
    else:
        logging.basicConfig(
            level=level_value,
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

@lru_cache(maxsize=1)
def get_config() -> dict:
    path = os.getenv("PORTFOLIO_CONFIG_PATH", "config.json")
    try:
        with open(path, "r", encoding="utf-8") as f:
            cfg = json.load(f) or {}
            if not isinstance(cfg.get("cedear_to_us", {}), dict):
                cfg["cedear_to_us"] = {}
            if not isinstance(cfg.get("etfs", []), list):
                cfg["etfs"] = []
            if not isinstance(cfg.get("acciones_ar", []), list):
                cfg["acciones_ar"] = []
            if not isinstance(cfg.get("fci_symbols", []), list):
                cfg["fci_symbols"] = []
            if not isinstance(cfg.get("scale_overrides", {}), dict):
                cfg["scale_overrides"] = {}
            if not isinstance(cfg.get("classification_patterns", {}), dict):
                cfg["classification_patterns"] = {}
            return cfg
    except FileNotFoundError:
        logger.warning("No se encontró archivo de configuración: %s", path)
        return {}
    except (OSError, json.JSONDecodeError) as e:
        logger.exception("Error cargando configuración %s: %s", path, e)
        return {}
