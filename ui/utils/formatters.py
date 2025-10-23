"""Utility helpers to keep UI formatting consistent across modules."""

from __future__ import annotations

from typing import Mapping

import numpy as np

from shared.settings import CACHE_HIT_THRESHOLDS
from shared.utils import format_percent  # legacy alias

__all__ = [
    "format_currency",
    "format_percent",
    "format_float",
    "format_currency_delta",
    "format_percent_delta",
    "format_float_delta",
    "normalise_hit_ratio",
    "resolve_badge_state",
    "format_asset_type",
]


_ASSET_TYPE_ALIASES: Mapping[str, str] = {
    "ACCIONES": "Acción",
    "ACCION": "Acción",
    "CEDEAR": "Cedear",
    "CEDEARS": "Cedear",
    "LETRA": "Letra",
    "LETRAS": "Letra",
    "BONO": "Bono",
    "TITULOSPUBLICOS": "Bono",
    "FONDOCOMUNDEINVERSION": "FCI",
    "FCI": "FCI",
}


def format_asset_type(raw: str | None) -> str:
    """Return a human-friendly label for asset types displayed in the UI."""

    if raw is None:
        return "N/D"

    if isinstance(raw, str):
        token = raw.strip()
    else:
        try:
            if np.isnan(raw):  # type: ignore[arg-type]
                return "N/D"
        except TypeError:
            pass
        token = str(raw).strip()

    if not token:
        return "N/D"

    lowered = token.lower()
    if lowered in {"nan", "none", "<na>"}:
        return "N/D"

    normalized = token.upper()
    return _ASSET_TYPE_ALIASES.get(normalized, token)


def format_currency(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"${value:,.0f}".replace(",", ".")


def format_float(value: float) -> str:
    if not np.isfinite(value):
        return "-"
    return f"{value:.2f}"


def format_currency_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}${abs(value):,.0f}".replace(",", ".")


def format_percent_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}%"


def format_float_delta(value: float) -> str:
    if not np.isfinite(value) or abs(value) < 1e-6:
        return "-"
    sign = "+" if value > 0 else "-"
    return f"{sign}{abs(value):.2f}"


def normalise_hit_ratio(value: object) -> float:
    try:
        ratio = float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return 0.0
    if not np.isfinite(ratio):
        return 0.0
    return float(min(max(ratio, 0.0), 1.0))


def resolve_badge_state(ratio: float, thresholds: Mapping[str, object] | None = None) -> str:
    mapping = thresholds or CACHE_HIT_THRESHOLDS
    try:
        green_threshold = float(mapping.get("green", 0.7))
    except Exception:  # pragma: no cover - defensive
        green_threshold = 0.7
    try:
        yellow_threshold = float(mapping.get("yellow", 0.4))
    except Exception:  # pragma: no cover - defensive
        yellow_threshold = 0.4

    if ratio >= green_threshold:
        return "green"
    if ratio >= yellow_threshold:
        return "yellow"
    return "red"
