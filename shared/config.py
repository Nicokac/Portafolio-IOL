# shared/config.py
from __future__ import annotations
import os, json
from pathlib import Path
from typing import Any, Dict
from dotenv import load_dotenv
from functools import lru_cache
import logging

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

        # --- Credenciales IOL ---
        self.IOL_USERNAME: str | None = os.getenv("IOL_USERNAME", cfg.get("IOL_USERNAME"))
        self.IOL_PASSWORD: str | None = os.getenv("IOL_PASSWORD", cfg.get("IOL_PASSWORD"))

        # --- Cache/TTLs usados en app.py ---
        self.cache_ttl_portfolio: int = int(os.getenv("CACHE_TTL_PORTFOLIO", cfg.get("CACHE_TTL_PORTFOLIO", 20)))
        self.cache_ttl_last_price: int = int(os.getenv("CACHE_TTL_LAST_PRICE", cfg.get("CACHE_TTL_LAST_PRICE", 10)))
        self.cache_ttl_fx: int = int(os.getenv("CACHE_TTL_FX", cfg.get("CACHE_TTL_FX", 60)))
        self.cache_ttl_quotes: int = int(os.getenv("CACHE_TTL_QUOTES", cfg.get("CACHE_TTL_QUOTES", 8)))
        self.quotes_hist_maxlen: int = int(os.getenv("QUOTES_HIST_MAXLEN", cfg.get("QUOTES_HIST_MAXLEN", 500)))
        self.max_quote_workers: int = int(os.getenv("MAX_QUOTE_WORKERS", cfg.get("MAX_QUOTE_WORKERS", 12)))

        # --- Archivo de tokens (IOLAuth) ---
        # Por defecto lo guardamos en la raíz junto a app.py (compat con tu tokens_iol.json existente)
        self.tokens_file: str = os.getenv("IOL_TOKENS_FILE", cfg.get("IOL_TOKENS_FILE", str(BASE_DIR / "tokens_iol.json")))

        # --- Derivados de dólar (Ahorro/Tarjeta a partir del oficial) ---
        self.fx_ahorro_multiplier: float = float(os.getenv("FX_AHORRO_MULTIPLIER", cfg.get("FX_AHORRO_MULTIPLIER", 1.30)))
        self.fx_tarjeta_multiplier: float = float(os.getenv("FX_TARJETA_MULTIPLIER", cfg.get("FX_TARJETA_MULTIPLIER", 1.35)))

settings = Settings()

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
