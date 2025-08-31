from __future__ import annotations
import numpy as np

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
