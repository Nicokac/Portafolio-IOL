# shared\utils.py
from __future__ import annotations

import numpy as np
import logging

logger = logging.getLogger(__name__)


def _as_float_or_none(x, log: bool = True) -> float | None:
    """Convert the input to ``float`` if possible, handling localized formats."""

    f = _to_float(x, log=log)
    if f is None:
        return None
    if not np.isfinite(f):
        return None
    return f

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

def format_percent(value: float | None, spaced: bool = False) -> str:
    """Format ``value`` as a percentage string with two decimals."""

    v = _as_float_or_none(value)
    if v is None:
        return "—"
    suffix = " %" if spaced else "%"
    return f"{v:.2f}{suffix}"

def _to_float(x, log: bool = True) -> float | None:
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
    except ValueError:
        if log:
            logger.warning("Valor inválido para conversión a float: %s", s)
        return None

