from __future__ import annotations
import time
import json
import logging
from pathlib import Path
from threading import Lock
from typing import Dict, Tuple, Any

logger = logging.getLogger(__name__)

# Caché thread-safe con TTL para cotizaciones
_QUOTE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()
QUOTE_CACHE_DIR = Path(".cache/quotes")

def get_quote_cached(cli, mercado: str, simbolo: str, ttl: int = 8) -> dict:
    """
    Devuelve {'last': float|None, 'chg_pct': float|None} con cache TTL (segundos).
    Mantiene la firma que venías usando: recibe 'cli' con método .get_quote().
    """
    key = (str(mercado).lower(), str(simbolo).upper())
    now = time.time()
    file_path = QUOTE_CACHE_DIR / f"{key[0]}_{key[1]}.json"

    # Intento de cache en memoria
    with _QUOTE_LOCK:
        rec = _QUOTE_CACHE.get(key)
        if rec and (now - rec["ts"] < ttl):
            return rec["data"]

    # Intento de cache en disco
    try:
        obj = json.loads(file_path.read_text(encoding="utf-8"))
        if now - obj.get("ts", 0) < ttl:
            data = obj.get("data", {})
            with _QUOTE_LOCK:
                _QUOTE_CACHE[key] = {"ts": obj.get("ts", now), "data": data}
            return data
    except Exception:
        pass

    # Llamada real fuera del lock
    try:
        q = cli.get_quote(mercado=key[0], simbolo=key[1]) or {}
        data = {"last": q.get("last"), "chg_pct": q.get("chg_pct")}
    except Exception as e:
        logger.warning("get_quote falló para %s:%s -> %s", mercado, simbolo, e)
        data = {"last": None, "chg_pct": None}

    # Guardar en cache en memoria
    with _QUOTE_LOCK:
        _QUOTE_CACHE[key] = {"ts": now, "data": data}

    # Persistir en disco
    try:
        QUOTE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        file_path.write_text(
            json.dumps({"ts": now, "data": data}, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception as e:
        logger.debug("No se pudo guardar cache de cotizacion: %s", e)

    return data
