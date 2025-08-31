# fx_service.py
from __future__ import annotations
from typing import Dict, Any, Optional
import time
import logging
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# =========================
# Config
# =========================
REQ_TIMEOUT = 8  # segundos
USER_AGENT = "IOL-Portfolio/1.0 (+fx_service)"

logger = logging.getLogger(__name__)

__all__ = ["get_fx_rates"]

# =========================
# Utilidades
# =========================

def _to_float(x) -> Optional[float]:
    """
    Conversión robusta a float:
    - Acepta números, strings con coma decimal y/o puntos de miles.
    - Devuelve None si no puede convertir.
    """
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    # Caso "1.234,56" -> "1234.56"
    if "," in s and s.count(",") == 1 and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

def _mid(compra: Any, venta: Any) -> Optional[float]:
    """
    Promedio simple entre compra/venta cuando ambos existen; si falta uno, usa el disponible.
    """
    c = _to_float(compra)
    v = _to_float(venta)
    if c is not None and v is not None:
        return (c + v) / 2.0
    return v if v is not None else c

def _http_get(url: str, *, timeout: float = REQ_TIMEOUT) -> requests.Response:
    """
    GET con reintentos básicos y User-Agent consistente.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(2):  # 1 intento + 1 reintento simple
        try:
            r = requests.get(url, timeout=timeout, headers={"User-Agent": USER_AGENT})
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            # pequeño backoff
            time.sleep(0.4 * (attempt + 1))
    assert last_exc is not None
    raise last_exc

# =========================
# Fetchers por fuente
# =========================

def _fetch_dolarapi() -> Dict[str, Any]:
    """
    Fuente: https://dolarapi.com/v1/dolares  (sin token)
    Suele traer: oficial, blue, bolsa (MEP), ccl, tarjeta, mayorista, solidario
    Claves normalizadas de salida: oficial, blue, mep, ccl, tarjeta, mayorista, ahorro
    """
    out: Dict[str, Any] = {}
    try:
        r = _http_get("https://dolarapi.com/v1/dolares")
        data = r.json()
        for item in data:
            tipo = (item.get("casa") or item.get("nombre") or "").lower()
            compra = item.get("compra")
            venta = item.get("venta")
            mid = _mid(compra, venta)

            if "oficial" in tipo:
                out["oficial"] = mid
            elif "blue" in tipo:
                out["blue"] = mid
            elif "bolsa" in tipo or "mep" in tipo:
                out["mep"] = mid
            elif "liqui" in tipo or "ccl" in tipo:
                out["ccl"] = mid
            elif "tarjeta" in tipo:
                out["tarjeta"] = mid
            elif "mayorista" in tipo:
                out["mayorista"] = mid
            elif "solidario" in tipo or "ahorro" in tipo:
                out["ahorro"] = mid
    except Exception as e:
        logger.warning("Falló el fetch a DolarAPI: %s", e)
    return out

def _fetch_bluelytics() -> Dict[str, Any]:
    """
    Fuente: https://api.bluelytics.com.ar/v2/latest
    Devuelve 'oficial' y 'blue'
    """
    out: Dict[str, Any] = {}
    try:
        r = _http_get("https://api.bluelytics.com.ar/v2/latest")
        data = r.json() or {}
        # oficial
        ofc = data.get("oficial", {})
        out["oficial"] = _mid(ofc.get("value_buy"), ofc.get("value_sell"))
        # blue
        blu = data.get("blue", {})
        out["blue"] = _mid(blu.get("value_buy"), blu.get("value_sell"))
    except Exception as e:
        logger.warning("Falló el fetch a Bluelytics: %s", e)
    return out

def _fetch_criptoya() -> Dict[str, Any]:
    """
    Fuente: https://criptoya.com/api/dolar
    Suele traer: ccl, mep, blue, oficial, solidario, tarjeta, mayorista
    """
    out: Dict[str, Any] = {}
    try:
        r = _http_get("https://criptoya.com/api/dolar")
        j = r.json() or {}
        for k in ("oficial", "blue", "mep", "ccl", "solidario", "tarjeta", "mayorista"):
            if k in j:
                v = _to_float(j[k])
                out["ahorro" if k == "solidario" else k] = v
    except Exception as e:
        logger.warning("Falló el fetch a CriptoYa: %s", e)
    return out

# =========================
# Orquestador
# =========================

def get_fx_rates() -> Dict[str, Any]:
    """
    Devuelve un dict con las principales cotizaciones (ARS por USD).
    Prioridad por campo: DolarAPI > CriptoYa > Bluelytics (para oficial/blue).
    Si faltan campos, se intentan completar con fuentes secundarias.
    Claves esperadas: oficial, blue, mep, ccl, tarjeta, mayorista, ahorro, _ts
    """
    # 1) Ejecuta las 3 fuentes en paralelo para bajar latencia total
    fetchers = {
        "dolarapi": _fetch_dolarapi,
        "criptoya": _fetch_criptoya,
        "bluelytics": _fetch_bluelytics,
    }
    raw: Dict[str, Dict[str, Any]] = {}
    with ThreadPoolExecutor(max_workers=3) as ex:
        future_map = {ex.submit(fn): name for name, fn in fetchers.items()}
        for fut in as_completed(future_map):
            name = future_map[fut]
            try:
                raw[name] = fut.result() or {}
            except Exception as e:
                logger.warning("Fetcher %s falló: %s", name, e)
                raw[name] = {}

    # 2) Mezcla por prioridad de campo
    rates: Dict[str, Any] = {}

    def _merge_key(key: str, *sources: str) -> None:
        for src in sources:
            val = raw.get(src, {}).get(key)
            if val is not None:
                rates[key] = _to_float(val)
                return

    # oficial / blue toman 3 fuentes
    _merge_key("oficial", "dolarapi", "criptoya", "bluelytics")
    _merge_key("blue",    "dolarapi", "criptoya", "bluelytics")

    # resto: dos fuentes principales
    for k in ("mep", "ccl", "tarjeta", "mayorista", "ahorro"):
        _merge_key(k, "dolarapi", "criptoya")

    # 3) Alias/derivados: si no hay 'ahorro' y hay 'oficial', fallback
    if rates.get("ahorro") is None and rates.get("oficial") is not None:
        # NOTA: los impuestos pueden cambiar; multiplicador “placeholder” conservador
        rates["ahorro"] = rates["oficial"] * 1.65

    # 4) Marca de tiempo (epoch seconds)
    rates["_ts"] = int(time.time())
    return rates
