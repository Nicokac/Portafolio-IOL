# shared/config.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict
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

        def secret_or_env(key: str, default: Any | None = None) -> Any | None:
            val = None
            try:
                val = st.secrets.get(key)
            except (StreamlitSecretNotFoundError, AttributeError):
                val = None
            if val is None:
                return os.getenv(key, default)
            return val

        # --- Credenciales IOL ---
        self.IOL_USERNAME: str | None = secret_or_env("IOL_USERNAME", cfg.get("IOL_USERNAME"))
        self.IOL_PASSWORD: str | None = secret_or_env("IOL_PASSWORD", cfg.get("IOL_PASSWORD"))

        # --- Cache/TTLs usados en app.py ---
        self.cache_ttl_portfolio: int = int(os.getenv("CACHE_TTL_PORTFOLIO", cfg.get("CACHE_TTL_PORTFOLIO", 20)))
        self.cache_ttl_last_price: int = int(os.getenv("CACHE_TTL_LAST_PRICE", cfg.get("CACHE_TTL_LAST_PRICE", 10)))
        self.cache_ttl_fx: int = int(os.getenv("CACHE_TTL_FX", cfg.get("CACHE_TTL_FX", 60)))
        self.cache_ttl_quotes: int = int(os.getenv("CACHE_TTL_QUOTES", cfg.get("CACHE_TTL_QUOTES", 8)))
        self.quotes_hist_maxlen: int = int(os.getenv("QUOTES_HIST_MAXLEN", cfg.get("QUOTES_HIST_MAXLEN", 500)))
        self.max_quote_workers: int = int(os.getenv("MAX_QUOTE_WORKERS", cfg.get("MAX_QUOTE_WORKERS", 12)))

        # --- Archivo de tokens (IOLAuth) ---
        # Por defecto lo guardamos en la raíz junto a app.py (compat con tu tokens_iol.json existente)
        self.tokens_file: str = secret_or_env(
            "IOL_TOKENS_FILE", cfg.get("IOL_TOKENS_FILE", str(BASE_DIR / "tokens_iol.json"))
        )
        # Clave opcional para cifrar/descifrar el archivo de tokens (Fernet)
        self.tokens_key: str | None = secret_or_env("IOL_TOKENS_KEY", cfg.get("IOL_TOKENS_KEY"))
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
