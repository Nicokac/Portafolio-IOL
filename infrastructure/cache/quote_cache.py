from __future__ import annotations
import time
from threading import Lock
from typing import Dict, Tuple, Any

# Caché thread-safe con TTL para cotizaciones
_QUOTE_CACHE: Dict[Tuple[str, str], Dict[str, Any]] = {}
_QUOTE_LOCK = Lock()

def get_quote_cached(cli, mercado: str, simbolo: str, ttl: int = 8) -> dict:
    """
    Devuelve {'last': float|None, 'chg_pct': float|None} con cache TTL (segundos).
    Mantiene la firma que venías usando: recibe 'cli' con método .get_quote().
    """
    key = (str(mercado).lower(), str(simbolo).upper())
    now = time.time()

    # Intento de cache
    with _QUOTE_LOCK:
        rec = _QUOTE_CACHE.get(key)
        if rec and (now - rec["ts"] < ttl):
            return rec["data"]

    # Llamada real fuera del lock
    try:
        q = cli.get_quote(mercado=key[0], simbolo=key[1]) or {}
        data = {"last": q.get("last"), "chg_pct": q.get("chg_pct")}
    except Exception:
        data = {"last": None, "chg_pct": None}

    # Guardar en cache
    with _QUOTE_LOCK:
        _QUOTE_CACHE[key] = {"ts": now, "data": data}

    return data
