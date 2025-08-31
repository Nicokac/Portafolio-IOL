# infrastructure/fx/provider.py
from __future__ import annotations
import time
import logging
import json
from pathlib import Path
from typing import Optional, Dict

from infrastructure.http.session import build_session
from shared.config import settings

logger = logging.getLogger(__name__)

CACHE_FILE = Path(".cache/fx_rates.json")
CACHE_TTL  = 45  # segundos


# -------------------------
# Helpers de normalización
# -------------------------
def _to_float(x):
    try:
        s = str(x).replace("%", "").replace(",", ".").strip()
        return float(s)
    except Exception:
        return None

def _mul(x, m):
    if x is None:
        return None
    try:
        return float(x) * float(m)
    except Exception:
        return None

def _normalize_rates(raw: dict) -> dict:
    """
    Devuelve un dict con claves 'oficial','mayorista','ahorro','tarjeta',
    'blue','mep','ccl','cripto' (las que se puedan obtener/derivar) + _ts.
    """
    r = dict(raw or {})
    now = int(time.time())

    oficial    = _to_float(r.get("oficial")    or r.get("oficial_bna") or r.get("oficial_bcra"))
    mayorista  = _to_float(r.get("mayorista")) or oficial
    blue       = _to_float(r.get("blue"))
    mep        = _to_float(r.get("mep"))
    ccl        = _to_float(r.get("ccl"))
    cripto     = _to_float(r.get("cripto")) or ccl

    # Derivados configurables desde settings
    ahorro  = _mul(oficial, settings.fx_ahorro_multiplier)   if oficial else None
    tarjeta = _mul(oficial, settings.fx_tarjeta_multiplier)  if oficial else None

    out = {
        "oficial": oficial,
        "mayorista": mayorista,
        "ahorro": ahorro,
        "tarjeta": tarjeta,
        "blue": blue,
        "mep": mep,
        "ccl": ccl,
        "cripto": cripto,
        "_ts": int(r.get("_ts", now)),
    }
    # Sólo devolvemos numéricos + _ts
    return {k: v for k, v in out.items() if (k == "_ts" or v is not None)}


class FXProviderAdapter:
    def __init__(self):
        ua = settings.USER_AGENT
        self.session = build_session(ua, retries=2, backoff=0.3, timeout=12)

    # ---------------
    # Cache local
    # ---------------
    def _load_cache(self) -> Optional[dict]:
        try:
            if CACHE_FILE.exists():
                data = json.loads(CACHE_FILE.read_text(encoding="utf-8"))
                # tolerante: si el cache está viejo, ignoramos
                if time.time() - float(data.get("_ts", 0)) < CACHE_TTL:
                    # por si el cache viejo no estaba normalizado
                    return _normalize_rates(data)
        except Exception:
            pass
        return None

    def _save_cache(self, data: dict) -> None:
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_FILE.with_suffix(CACHE_FILE.suffix + ".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(CACHE_FILE)
        except Exception:
            pass

    # ---------------
    # Fetch + normalize
    # ---------------
    def get_rates(self) -> Dict[str, float]:
        # 1) Cache
        cached = self._load_cache()
        if cached:
            return cached

        # 2) Fuentes (mantén/ajusta las que venías usando)
        urls = {
            "blue":     "https://api.bluelytics.com.ar/v2/latest",
            "oficial":  "https://dolarapi.com/v1/dolares/oficial",
            "mep":      "https://dolarapi.com/v1/dolares/bolsa",
            "ccl":      "https://dolarapi.com/v1/dolares/contadoconliqui",
        }

        raw: Dict[str, float] = {}
        # blue (bluelytics)
        try:
            r = self.session.get(urls["blue"])
            if r.ok:
                j = r.json()
                # bluelytics tiene "blue": {"value_avg": ...}
                raw["blue"] = float(j["blue"]["value_avg"])
        except Exception:
            logger.info("No se pudo obtener blue")

        # oficial / mep / ccl (dolarapi)
        for k in ("oficial", "mep", "ccl"):
            try:
                r = self.session.get(urls[k])
                if r.ok:
                    j = r.json()
                    # dolarapi usa 'venta'
                    raw[k] = float(j["venta"])
            except Exception:
                logger.info("No se pudo obtener %s", k)

        raw["_ts"] = int(time.time())

        # 3) Normalización y cache
        normalized = _normalize_rates(raw)
        self._save_cache(normalized)
        return normalized
