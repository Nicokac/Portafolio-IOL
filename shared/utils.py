# shared\utils.py
from __future__ import annotations
import json
import numpy as np
from functools import lru_cache

def _as_float_or_none(x) -> float | None:
    try:
        f = float(x)
        if not np.isfinite(f):
            return None
        return f
    except Exception:
        return None

def _is_none_nan_inf(x) -> bool:
    return _as_float_or_none(x) is None

def format_money(value: float | int | None, currency: str = "ARS") -> str:
    v = _as_float_or_none(value)
    if v is None:
        return "—"
    sign = "-" if v < 0 else ""
    v = abs(v)
    s = f"{v:,.2f}".replace(",", "_").replace(".", ",").replace("_", ".")
    prefix = "US$ " if currency.upper() == "USD" else "$ "
    return f"{sign}{prefix}{s}"

def format_number(value: float | int | None) -> str:
    v = _as_float_or_none(value)
    if v is None:
        return "—"
    ent_str = f"{int(round(v)):,}".replace(",", ".")
    return ent_str

def format_price(value: float | int | None, currency: str = "ARS") -> str:
    v = _as_float_or_none(value)
    if v is None:
        return "—"
    s = f"{v:,.3f}".replace(",", "_").replace(".", ",").replace("_", ".")
    prefix = "US$ " if currency.upper() == "USD" else "$ "
    return prefix + s

def format_percent(value: float | None) -> str:
    v = _as_float_or_none(value)
    if v is None:
        return "—"
    return f"{v:.2f} %"

@lru_cache()
def get_config() -> dict:
    with open("config.json", encoding="utf-8") as f:
        return json.load(f)

def _to_float(x) -> float | None:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    # "1.234,56" -> "1234.56"
    if "," in s and s.count(",") == 1 and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None

