# application\portfolio_service.py
# Lógica de negocio: normalización de posiciones y cálculo de métricas P/L

from __future__ import annotations

import logging
import re
from datetime import datetime
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Dict, Iterable, List, Mapping
from collections.abc import Mapping as ABCMapping, Sequence
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType

import numpy as np
import pandas as pd
import requests

from shared.config import get_config, settings
from shared.pandas_attrs import unwrap_callable_attr
from shared.utils import _to_float


# Increment this value whenever the valuation or totals aggregation logic changes.
# It is used to invalidate cached portfolio snapshots and UI summaries so that
# new deployments propagate updated totals without requiring manual cache clears.
PORTFOLIO_TOTALS_VERSION = 6.1


_BASIC_JSON_TYPES = (str, int, float, bool, type(None))

try:  # pragma: no cover - defensive import guard
    _RLOCK_TYPE = type(RLock())
except TypeError:  # pragma: no cover - alternative runtime implementations
    _RLOCK_TYPE = ()


def _sanitize_attrs_value(value: Any, seen: set[int]) -> Any:
    if isinstance(value, np.generic):
        return _sanitize_attrs_value(value.item(), seen)
    if isinstance(value, _BASIC_JSON_TYPES):
        return value
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, (MethodType, FunctionType, BuiltinFunctionType, ModuleType)):
        return str(value)
    if _RLOCK_TYPE and isinstance(value, _RLOCK_TYPE):
        return str(value)
    if isinstance(value, type):
        return value.__name__
    if isinstance(value, ABCMapping):
        obj_id = id(value)
        if obj_id in seen:
            return str(value)
        seen.add(obj_id)
        try:
            sanitized_dict: dict[str, Any] = {}
            for key, inner_value in value.items():
                sanitized_dict[str(key)] = _sanitize_attrs_value(inner_value, seen)
            return sanitized_dict
        finally:
            seen.remove(obj_id)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        obj_id = id(value)
        if obj_id in seen:
            return str(value)
        seen.add(obj_id)
        try:
            sanitized_list: list[Any] = []
            for item in value:
                sanitized_list.append(_sanitize_attrs_value(item, seen))
            return sanitized_list
        finally:
            seen.remove(obj_id)
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive
        return None


def sanitize_attrs(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure DataFrame attrs only contains JSON-serialisable structures."""

    attrs = getattr(df, "attrs", None)
    if not isinstance(attrs, ABCMapping):
        return df
    sanitized = _sanitize_attrs_value(dict(attrs), set())
    if isinstance(sanitized, dict):
        df.attrs.clear()
        df.attrs.update(sanitized)
    return df


@dataclass(frozen=True)
class ValuationBreakdown:
    """Summary of valuation quality grouped by conversion confidence."""

    confirmed_rows: int = 0
    confirmed_value: float = 0.0
    estimated_rows: int = 0
    estimated_value: float = 0.0
    unconverted_rows: int = 0
    unconverted_value: float = 0.0

    def __post_init__(self) -> None:
        confirmed_rows = _to_float(self.confirmed_rows)
        estimated_rows = _to_float(self.estimated_rows)
        unconverted_rows = _to_float(self.unconverted_rows)
        object.__setattr__(self, "confirmed_rows", int(confirmed_rows) if confirmed_rows is not None else 0)
        object.__setattr__(self, "estimated_rows", int(estimated_rows) if estimated_rows is not None else 0)
        object.__setattr__(self, "unconverted_rows", int(unconverted_rows) if unconverted_rows is not None else 0)

        confirmed_value = _to_float(self.confirmed_value)
        estimated_value = _to_float(self.estimated_value)
        unconverted_value = _to_float(self.unconverted_value)
        object.__setattr__(self, "confirmed_value", float(confirmed_value or 0.0))
        object.__setattr__(self, "estimated_value", float(estimated_value or 0.0))
        object.__setattr__(self, "unconverted_value", float(unconverted_value or 0.0))

    @property
    def estimated_impact_pct(self) -> float:
        """Return the impact of estimated rows relative to confirmed valuations."""

        base = float(self.confirmed_value)
        if not np.isfinite(base) or base <= 0.0:
            return float("nan")
        impact = float(self.estimated_value)
        if not np.isfinite(impact):
            return float("nan")
        return (impact / base) * 100.0


@dataclass(frozen=True)
class PortfolioTotals:
    """Totales básicos del portafolio con desglose de efectivo."""

    total_value: float
    total_cost: float
    total_pl: float
    total_pl_pct: float
    total_cash: float = 0.0
    total_cash_ars: float = 0.0
    total_cash_usd: float = 0.0
    total_cash_combined: float | None = None
    usd_rate: float | None = None
    valuation_breakdown: ValuationBreakdown = field(default_factory=ValuationBreakdown)

    def __post_init__(self) -> None:
        cash = _to_float(self.total_cash)
        cash_ars = _to_float(self.total_cash_ars)
        cash_usd = _to_float(self.total_cash_usd)
        combined = _to_float(self.total_cash_combined)

        object.__setattr__(self, "total_cash", float(cash or 0.0))
        object.__setattr__(self, "total_cash_ars", float(cash_ars or 0.0))
        object.__setattr__(self, "total_cash_usd", float(cash_usd or 0.0))
        if combined is None:
            combined = float(self.total_cash) + float(self.total_cash_ars) + float(self.total_cash_usd)
        object.__setattr__(self, "total_cash_combined", float(combined or 0.0))

        rate = _to_float(self.usd_rate)
        object.__setattr__(self, "usd_rate", None if rate is None else float(rate))

        breakdown = getattr(self, "valuation_breakdown", None)
        if breakdown is None:
            breakdown_obj = ValuationBreakdown()
        elif isinstance(breakdown, Mapping):
            breakdown_obj = ValuationBreakdown(**breakdown)
        elif isinstance(breakdown, ValuationBreakdown):
            breakdown_obj = breakdown
        else:  # pragma: no cover - defensive conversion
            breakdown_obj = ValuationBreakdown()
        object.__setattr__(self, "valuation_breakdown", breakdown_obj)


def calculate_totals(df_view: pd.DataFrame | None) -> PortfolioTotals:
    """Calcula totales agregados de valorizado, costo y P/L."""

    if df_view is None or df_view.empty:
        return PortfolioTotals(0.0, 0.0, 0.0, float("nan"), 0.0)

    index = df_view.index

    def _numeric_series(name: str) -> pd.Series:
        if name in df_view.columns:
            return pd.to_numeric(df_view[name], errors="coerce")
        return pd.Series(index=index, dtype=float)

    def _string_series(name: str) -> pd.Series:
        if name in df_view.columns:
            return df_view[name].astype("string")
        return pd.Series(index=index, dtype="string")

    valor_actual_series = _numeric_series("valor_actual").replace([np.inf, -np.inf], np.nan)
    costo_series = _numeric_series("costo").replace([np.inf, -np.inf], np.nan)

    currency_series = _string_series("moneda_origen").fillna("").str.upper()
    moneda_series = _string_series("moneda").fillna("").str.upper()
    provider_series = _string_series("pricing_source")
    if provider_series.isna().all():
        provider_series = _string_series("provider")
    if provider_series.isna().all():
        provider_series = _string_series("proveedor_original")
    provider_series = provider_series.fillna("").str.lower()

    simbolo_series = _string_series("simbolo").fillna("").str.upper()
    fx_series = _numeric_series("fx_aplicado").replace([np.inf, -np.inf], np.nan)

    has_value = valor_actual_series.notna()
    ars_quote_mask = currency_series.str.contains("ARS", case=False, regex=False)
    ars_position_mask = moneda_series.str.contains("ARS", case=False, regex=False)
    official_providers = {"valorizado", "ultimoprecio", "iol", "cache", "stale", "last"}
    cash_like_mask = simbolo_series.isin({"IOLPORA", "PARKING"})

    confirmed_mask = has_value & (
        ars_quote_mask
        | ars_position_mask
        | provider_series.isin(official_providers)
        | cash_like_mask
    )
    if not confirmed_mask.any() and has_value.any():
        confirmed_mask = has_value

    fx_positive = (fx_series > 0).fillna(False)
    estimated_mask = has_value & ~confirmed_mask & fx_positive
    unconverted_mask = ~confirmed_mask & ~fx_positive

    confirmed_value = float(np.nansum(valor_actual_series.where(confirmed_mask).to_numpy()))
    estimated_value = float(np.nansum(valor_actual_series.where(estimated_mask).to_numpy()))
    unconverted_value = float(np.nansum(valor_actual_series.where(unconverted_mask).to_numpy()))

    total_value = confirmed_value
    total_cost = float(np.nansum(costo_series.where(confirmed_mask).to_numpy()))
    total_pl = total_value - total_cost

    cash_mask = df_view.get("simbolo")
    cash_series = pd.Series(dtype=float)
    if cash_mask is not None and "valor_actual" in df_view.columns:
        try:
            mask = cash_mask.astype(str).str.upper().isin({"IOLPORA", "PARKING"})
            cash_series = valor_actual_series.loc[mask]
        except Exception:
            cash_series = pd.Series(dtype=float)
    total_cash = float(np.nansum(getattr(cash_series, "values", [])))

    if np.isfinite(total_cost) and not np.isclose(total_cost, 0.0):
        total_pl_pct = (total_pl / total_cost) * 100.0
    else:
        total_pl_pct = float("nan")

    cash_info: Mapping[str, Any] = getattr(df_view, "attrs", {}).get("cash_balances", {})
    cash_ars = _to_float(cash_info.get("cash_ars")) or 0.0
    cash_usd = _to_float(cash_info.get("cash_usd")) or 0.0
    usd_equiv = _to_float(cash_info.get("cash_usd_ars_equivalent"))
    usd_rate = _to_float(cash_info.get("usd_rate"))

    account_cash_total = cash_ars
    if usd_equiv is not None:
        account_cash_total += usd_equiv
    else:
        account_cash_total += cash_usd

    include_cash_rows = True
    if cash_series.size and account_cash_total:
        try:
            if np.isfinite(total_cash) and np.isfinite(account_cash_total):
                include_cash_rows = not np.isclose(
                    total_cash,
                    account_cash_total,
                    rtol=1e-3,
                    atol=max(1.0, account_cash_total * 0.005),
                )
        except Exception:
            include_cash_rows = True

    combined = account_cash_total
    if include_cash_rows:
        combined += total_cash

    breakdown = ValuationBreakdown(
        confirmed_rows=int(confirmed_mask.sum()),
        confirmed_value=confirmed_value,
        estimated_rows=int(estimated_mask.sum()),
        estimated_value=estimated_value,
        unconverted_rows=int(unconverted_mask.sum()),
        unconverted_value=unconverted_value,
    )

    return PortfolioTotals(
        total_value,
        total_cost,
        total_pl,
        total_pl_pct,
        total_cash,
        total_cash_ars=cash_ars,
        total_cash_usd=cash_usd,
        total_cash_combined=combined,
        usd_rate=usd_rate,
        valuation_breakdown=breakdown,
    )


def detect_currency(sym: str, tipo: str | None) -> str:
    """Determina la moneda en base al símbolo informado."""

    return "USD" if str(sym).upper() in {"PRPEDOB"} else "ARS"


logger = logging.getLogger(__name__)


# ---------- Helpers y configuración ----------
BOPREAL_SYMBOLS = {"BPOA7", "BPOB7", "BPOC7", "BPOD7"}
BOPREAL_HISTORICAL_SCALE = 0.01
BOPREAL_FORCE_FACTOR = 1 / BOPREAL_HISTORICAL_SCALE
BOPREAL_FORCED_REVALUATION_TAG = "override_bopreal_ars_forced_revaluation"
BOPREAL_TRUSTED_PROVIDERS = frozenset(
    {
        "",
        "IOL",
        "IOL-LIVE",
        "CACHE",
        "STALE",
        "VALORIZADO",
        "VALORIZADO_RESCALED",
        "OVERRIDE_BOPREAL_FORCED",
        "OVERRIDE_BOPREAL_POSTMERGE",
        BOPREAL_FORCED_REVALUATION_TAG.upper(),
    }
)


def _is_blank(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    return False


def _normalize_upper(value: Any) -> str:
    if _is_blank(value):
        return ""
    if isinstance(value, str):
        return value.strip().upper()
    return str(value).upper()


def _first_present(row: Mapping[str, Any] | pd.Series, keys: Iterable[str]) -> Any:
    for key in keys:
        value = row.get(key) if hasattr(row, "get") else None
        if not _is_blank(value):
            return value
    return None


def _bopreal_symbol(symbol: Any) -> str | None:
    cleaned = clean_symbol(symbol)
    if not cleaned or cleaned.endswith("D"):
        return None
    if cleaned not in BOPREAL_SYMBOLS:
        return None
    return cleaned


def is_bopreal_ars(row: Mapping[str, Any] | pd.Series) -> bool:
    """Return True when the row corresponds to a BOPREAL series in ARS."""

    if isinstance(row, pd.Series):
        data = row
    elif isinstance(row, Mapping):
        data = pd.Series(row)
    else:  # pragma: no cover - defensive guard for unexpected inputs
        raise TypeError(f"is_bopreal_ars expects Mapping or Series, got {type(row)!r}")

    bopreal_symbol = _bopreal_symbol(data.get("simbolo"))
    if not bopreal_symbol:
        return False

    currency_value = _first_present(data, ("moneda_origen", "moneda"))
    currency = _normalize_upper(currency_value)
    return currency == "ARS"


def _extract_scale_context(row: Mapping[str, Any] | pd.Series) -> dict[str, Any]:
    if isinstance(row, pd.Series):
        data = row
    elif isinstance(row, Mapping):
        data = pd.Series(row)
    else:
        raise TypeError(f"scale context expects Mapping or Series, got {type(row)!r}")

    raw_symbol = data.get("simbolo")
    symbol_clean = clean_symbol(raw_symbol)

    tipo_value = data.get("tipo_estandar")
    if _is_blank(tipo_value):
        tipo_value = data.get("tipo_activo")
    if _is_blank(tipo_value):
        tipo_value = data.get("tipo")
    tipo_norm = str(tipo_value or "").lower()

    currency_value = _first_present(data, ("moneda_origen", "moneda"))
    currency = _normalize_upper(currency_value)

    provider_value = _first_present(data, ("proveedor_original", "pricing_source", "provider"))
    provider = _normalize_upper(provider_value)

    bopreal_symbol = _bopreal_symbol(raw_symbol)
    bopreal_override = bool(
        bopreal_symbol and currency == "ARS" and provider in BOPREAL_TRUSTED_PROVIDERS
    )

    return {
        "symbol_raw": raw_symbol,
        "symbol_clean": symbol_clean,
        "tipo_norm": tipo_norm,
        "currency": currency,
        "provider": provider,
        "bopreal_symbol": bopreal_symbol,
        "bopreal_override": bopreal_override,
    }


def _scale_from_context(context: Mapping[str, Any]) -> float:
    if context.get("bopreal_override"):
        return 1.0

    symbol_clean = context.get("symbol_clean", "")
    cfg = get_config()
    scale_overrides = cfg.get("scale_overrides", {}) or {}
    if symbol_clean in scale_overrides:
        try:
            override_value = float(scale_overrides[symbol_clean])
            if override_value > 0:
                return override_value
        except (TypeError, ValueError) as exc:
            logger.exception("scale_overrides inválido para %s: %s", symbol_clean, exc)

    tipo_norm = context.get("tipo_norm", "")
    currency = context.get("currency", "")
    if ("bono" in tipo_norm or "letra" in tipo_norm) and currency == "USD":
        return 0.01

    return 1.0


def clean_symbol(s: str) -> str:
    """Normaliza el símbolo: mayúsculas, sin espacios raros, sólo chars permitidos."""
    s = str(s or "").upper().strip()
    s = s.replace("\u00a0", "").replace("\u200b", "")
    return re.sub(r"[^A-Z0-9._^-]", "", s)


def map_to_us_ticker(simbolo: str) -> str:
    """Map a local symbol to its corresponding US ticker.

    Levanta ``ValueError`` si no se encuentra un ticker válido.
    """
    s = clean_symbol(simbolo)
    cfg = get_config()
    cedear_map = cfg.get("cedear_to_us", {}) or {}

    if s in cedear_map:
        return clean_symbol(cedear_map[s])

    # ⛑️ Fallback si es una acción local
    if s in cfg.get("acciones_ar", []):
        return s + ".BA"  # yfinance usa sufijo .BA para acciones argentinas

    raise ValueError(f"Símbolo no válido: {simbolo}")


def scale_for(row: Mapping[str, Any] | pd.Series) -> float:
    """Determina el factor de escala según tipo de activo y moneda."""

    context = _extract_scale_context(row)
    return _scale_from_context(context)


def classify_asset(it: dict) -> dict[str, str]:
    """Expose the asset type exactly as reported by IOL."""

    raw_title = it.get("titulo")
    tipo_value: object | None = None
    if isinstance(raw_title, Mapping):
        tipo_value = raw_title.get("tipo")

    if tipo_value in (None, ""):
        tipo = "N/D"
    elif isinstance(tipo_value, str):
        tipo = tipo_value
    else:
        tipo = str(tipo_value)

    return {
        "tipo": tipo,
        "tipo_estandar": tipo,
        "tipo_iol": tipo,
    }


# ---------- Normalización de payload ----------


def _extract_cash_balances(payload: Mapping[str, Any] | None) -> dict[str, float]:
    if not isinstance(payload, Mapping):
        return {}
    raw = payload.get("_cash_balances")
    if not isinstance(raw, Mapping):
        return {}
    balances: dict[str, float] = {}
    for key in ("cash_ars", "cash_usd", "cash_usd_ars_equivalent", "usd_rate"):
        value = _to_float(raw.get(key))
        if value is not None:
            balances[key] = float(value)
    return balances


def normalize_positions(payload: Dict[str, Any]) -> pd.DataFrame:
    """
    Devuelve columnas: simbolo, mercado, cantidad, costo_unitario
    Compatible con /api/v2/portafolio {'pais','activos': [...]}
    Tolera variantes: 'titulos','valorizaciones','posiciones','portafolio' o listas crudas.
    """
    items: List[Dict[str, Any]] = []

    # Detectar fuente
    source = []
    if isinstance(payload, dict):
        if isinstance(payload.get("activos"), list):
            source = payload["activos"]
        else:
            for key in ("titulos", "valorizaciones", "posiciones", "portafolio"):
                if isinstance(payload.get(key), list):
                    source = payload[key]
                    break
    elif isinstance(payload, list):
        source = payload

    def _pick(*values: Any) -> Any:
        for value in values:
            if value is not None:
                return value
        return None

    for it in source:
        t = it.get("titulo") if isinstance(it, dict) else None

        simbolo = _pick(
            it.get("simbolo") if isinstance(it, dict) else None,
            t.get("simbolo") if isinstance(t, dict) else None,
            it.get("ticker") if isinstance(it, dict) else None,
            it.get("codigo") if isinstance(it, dict) else None,
            "",
        )
        mercado = _pick(
            it.get("mercado") if isinstance(it, dict) else None,
            t.get("mercado") if isinstance(t, dict) else None,
            it.get("market") if isinstance(it, dict) else None,
            "bcba",
        )
        cantidad = _pick(
            it.get("cantidad") if isinstance(it, dict) else None,
            it.get("cant") if isinstance(it, dict) else None,
            it.get("cantidadDisponible") if isinstance(it, dict) else None,
            it.get("cantidadNominal") if isinstance(it, dict) else None,
            it.get("tenencia") if isinstance(it, dict) else None,
            0,
        )

        costo_unit = _pick(
            it.get("costoUnitario") if isinstance(it, dict) else None,
            it.get("ppc") if isinstance(it, dict) else None,
            t.get("costoUnitario") if isinstance(t, dict) else None,
            t.get("ppc") if isinstance(t, dict) else None,
        )

        moneda = _pick(
            it.get("moneda") if isinstance(it, dict) else None,
            t.get("moneda") if isinstance(t, dict) else None,
        )
        plazo = _pick(
            it.get("plazo") if isinstance(it, dict) else None,
            t.get("plazo") if isinstance(t, dict) else None,
        )
        ultimo_precio = _pick(
            it.get("ultimoPrecio") if isinstance(it, dict) else None,
            it.get("ultimo") if isinstance(it, dict) else None,
            t.get("ultimoPrecio") if isinstance(t, dict) else None,
        )
        variacion_diaria = _pick(
            it.get("variacionDiaria") if isinstance(it, dict) else None,
            t.get("variacionDiaria") if isinstance(t, dict) else None,
        )
        tiene_panel_raw = _pick(
            it.get("tienePanel") if isinstance(it, dict) else None,
            t.get("tienePanel") if isinstance(t, dict) else None,
        )
        riesgo = _pick(
            it.get("riesgo") if isinstance(it, dict) else None,
            t.get("riesgo") if isinstance(t, dict) else None,
        )

        tipo_original = ""
        descripcion_original = ""
        tipo_valor = "N/D"
        if isinstance(t, dict):
            raw_tipo = t.get("tipo")
            raw_desc = t.get("descripcion")
            if raw_tipo not in (None, ""):
                tipo_original = str(raw_tipo).strip()
                tipo_valor = raw_tipo if isinstance(raw_tipo, str) else str(raw_tipo)
            if raw_desc not in (None, ""):
                descripcion_original = str(raw_desc).strip()
        if not tipo_original:
            tipo_original = tipo_valor if isinstance(tipo_valor, str) else str(tipo_valor)

        if costo_unit is None:
            total_costo = (
                (it.get("costoTotal") if isinstance(it, dict) else None)
                or (it.get("inversion") if isinstance(it, dict) else None)
                or (t.get("costoTotal") if isinstance(t, dict) else None)
                or (t.get("inversion") if isinstance(t, dict) else None)
            )
            cf = _to_float(total_costo)
            qf = _to_float(cantidad)
            if cf is not None and qf not in (None, 0):
                costo_unit = cf / qf

        cantidad_f = _to_float(cantidad) or 0.0
        costo_unit_f = _to_float(costo_unit) or 0.0

        moneda_str = str(moneda).strip() if moneda is not None else ""
        if not moneda_str:
            moneda_str = "s/d"
        plazo_str = str(plazo).strip() if plazo is not None else ""
        if not plazo_str:
            plazo_str = "s/d"
        riesgo_str = str(riesgo).strip() if riesgo is not None else ""
        if not riesgo_str:
            riesgo_str = "s/d"

        valorizado = _pick(
            it.get("valorizado") if isinstance(it, dict) else None,
            t.get("valorizado") if isinstance(t, dict) else None,
        )

        ultimo_precio_f = _to_float(ultimo_precio)
        variacion_diaria_f = _to_float(variacion_diaria)
        valorizado_f = _to_float(valorizado)

        tiene_panel: bool | None
        if isinstance(tiene_panel_raw, bool):
            tiene_panel = tiene_panel_raw
        elif isinstance(tiene_panel_raw, str):
            normalized = tiene_panel_raw.strip().lower()
            if normalized in {"true", "1", "si", "sí", "on"}:
                tiene_panel = True
            elif normalized in {"false", "0", "no", "off"}:
                tiene_panel = False
            else:
                tiene_panel = None
        elif isinstance(tiene_panel_raw, (int, float)):
            if isinstance(tiene_panel_raw, float) and np.isnan(tiene_panel_raw):
                tiene_panel = None
            else:
                tiene_panel = bool(tiene_panel_raw)
        else:
            tiene_panel = None

        if simbolo and cantidad_f:
            items.append(
                {
                    "simbolo": clean_symbol(simbolo),
                    "mercado": str(mercado).strip().lower(),
                    "cantidad": float(cantidad_f),
                    "costo_unitario": float(costo_unit_f),
                    "moneda": moneda_str,
                    "plazo": plazo_str,
                    "ultimoPrecio": ultimo_precio_f,
                    "variacionDiaria": variacion_diaria_f,
                    "tienePanel": tiene_panel,
                    "riesgo": riesgo_str,
                    "titulo_tipo_original": tipo_original,
                    "titulo_descripcion_original": descripcion_original,
                    "tipo": tipo_valor,
                    "tipo_iol": tipo_valor,
                    "tipo_estandar": tipo_valor,
                    "valorizado": float(valorizado_f) if valorizado_f is not None else np.nan,
                }
            )

    df = pd.DataFrame(
        items,
        columns=[
            "simbolo",
            "mercado",
            "cantidad",
            "costo_unitario",
            "moneda",
            "plazo",
            "ultimoPrecio",
            "variacionDiaria",
            "tienePanel",
            "riesgo",
            "titulo_tipo_original",
            "titulo_descripcion_original",
            "tipo",
            "tipo_iol",
            "tipo_estandar",
            "valorizado",
        ],
    )
    balances = _extract_cash_balances(payload)
    if balances:
        df.attrs.setdefault("cash_balances", {}).update(balances)
    return df


# ---------- Cálculo de métricas ----------


def calc_rows(get_quote_fn, df_pos: pd.DataFrame, exclude_syms: Iterable[str]) -> pd.DataFrame:
    """Calcula métricas de valuación y P/L para cada posición."""
    market_price_fetcher_raw = None
    if isinstance(df_pos, pd.DataFrame):
        attrs_ref = getattr(df_pos, "attrs", None)
        if isinstance(attrs_ref, ABCMapping):
            market_price_fetcher_raw = attrs_ref.get("market_price_fetcher")
        sanitize_attrs(df_pos)
    attrs: dict[str, Any] = dict(getattr(df_pos, "attrs", {}) or {})
    market_price_fetcher = unwrap_callable_attr(market_price_fetcher_raw)
    attrs.pop("market_price_fetcher", None)
    if not callable(market_price_fetcher):
        market_price_fetcher = getattr(get_quote_fn, "fetch_market_price", None)
        if not callable(market_price_fetcher):
            market_price_fetcher = None
    cols = [
        "simbolo",
        "mercado",
        "tipo",
        "tipo_iol",
        "tipo_estandar",
        "cantidad",
        "ppc",
        "scale",
        "moneda",
        "moneda_origen",
        "fx_aplicado",
        "pricing_source",
        "ultimo",
        "ultimoPrecio",
        "valor_actual",
        "costo",
        "pl",
        "pl_%",
        "pl_d",
        "pld_%",
    ]

    if df_pos is None or df_pos.empty:
        empty_df = pd.DataFrame(columns=cols)
        if attrs:
            empty_df.attrs.update(attrs)
        return sanitize_attrs(empty_df)

    # Normalización básica y exclusiones ---------------------------------
    df = df_pos.copy()
    if "moneda" not in df.columns:
        df["moneda"] = ""
    df["simbolo"] = df["simbolo"].map(clean_symbol)
    df["mercado"] = df["mercado"].astype(str).str.lower()
    ex = {clean_symbol(s) for s in (exclude_syms or [])}

    df = df[~df["simbolo"].isin(ex)]
    if df.empty:
        empty_df = pd.DataFrame(columns=cols)
        if attrs:
            empty_df.attrs.update(attrs)
        return sanitize_attrs(empty_df)

    df["cantidad"] = df["cantidad"].map(_to_float).fillna(0.0)
    df["ppc"] = df.get("costo_unitario", np.nan).map(_to_float).fillna(0.0)

    # ----- Cotizaciones -------------------------------------------------
    uniq = df[["mercado", "simbolo"]].drop_duplicates().reset_index(drop=True)

    # ----- cotización (acepta float o dict) -----
    def _fetch(row: pd.Series) -> pd.Series:
        last, chg_pct, prev_close = None, None, None
        provider = None
        currency = None
        fx_value: float | None = None
        ratio_value: float | None = None
        q = None
        try:
            q = get_quote_fn(row["mercado"], row["simbolo"])
        except requests.RequestException as e:
            logger.exception(
                "get_quote_fn lanzó excepción para %s:%s -> %s",
                row["mercado"],
                row["simbolo"],
                e,
            )

        if isinstance(q, dict):
            for k in ("last", "ultimo", "ultimoPrecio", "precio", "close", "cierre"):
                if k in q:
                    last = _to_float(q.get(k))
                    if last is not None:
                        break
            chg_pct = _to_float(q.get("chg_pct"))
            prev_close = _to_float(q.get("cierreAnterior"))
            provider_raw = q.get("provider")
            if isinstance(provider_raw, str):
                provider = provider_raw.strip() or None
            elif provider_raw is not None:
                provider = str(provider_raw)
            currency_raw = q.get("moneda_origen")
            if currency_raw is None:
                currency_raw = q.get("currency")
            if isinstance(currency_raw, str):
                currency = currency_raw.strip() or None
            elif currency_raw is not None:
                currency = str(currency_raw)
            fx_raw = q.get("fx_aplicado")
            if fx_raw is None:
                fx_raw = q.get("fxAplicado")
            if fx_raw is None:
                fx_raw = q.get("fx_applied")
            fx_value = _to_float(fx_raw)
            ratio_value = _to_float(q.get("ratioCEDEAR"), log=False)
        else:
            last = _to_float(q)

        return pd.Series(
            {
                "last": last,
                "chg_pct": chg_pct,
                "prev_close": prev_close,
                "provider": provider,
                "moneda_origen": currency,
                "fx_aplicado": fx_value,
                "ratioCEDEAR": ratio_value,
            }
        )

    quotes_df = uniq.apply(_fetch, axis=1)
    quotes_df = pd.concat([uniq, quotes_df], axis=1)

    df = df.merge(quotes_df, on=["mercado", "simbolo"], how="left")

    # ----- Clasificación y escala --------------------------------------
    if "tipo" not in df.columns:
        df["tipo"] = "N/D"
    else:
        df["tipo"] = df["tipo"].where(df["tipo"].notna(), "N/D")

    if "tipo_iol" in df.columns:
        df["tipo_iol"] = df["tipo_iol"].where(df["tipo_iol"].notna(), df["tipo"])
    else:
        df["tipo_iol"] = df["tipo"]

    if "tipo_estandar" in df.columns:
        df["tipo_estandar"] = df["tipo_estandar"].where(df["tipo_estandar"].notna(), df["tipo"])
    else:
        df["tipo_estandar"] = df["tipo"]

    scale_columns = [
        "simbolo",
        "tipo_estandar",
        "tipo_activo",
        "tipo",
        "moneda_origen",
        "moneda",
        "proveedor_original",
        "pricing_source",
        "provider",
    ]
    available_scale_columns = [col for col in scale_columns if col in df.columns]
    uniq = df.drop_duplicates("simbolo")[available_scale_columns]

    scale_map: dict[str, float] = {}
    bopreal_override_symbols: set[str] = set()
    for _, row in uniq.iterrows():
        context = _extract_scale_context(row)
        symbol_raw = context.get("symbol_raw")
        if _is_blank(symbol_raw):
            continue
        scale_value = _scale_from_context(context)
        scale_map[symbol_raw] = scale_value
        if context.get("bopreal_override") and context.get("bopreal_symbol"):
            bopreal_override_symbols.add(context["bopreal_symbol"])

    df["scale"] = pd.to_numeric(df["simbolo"].map(scale_map), errors="coerce").fillna(1.0)

    # Ajuste condicional de escala para bonos/letras ---------------------
    valorizado_series = pd.to_numeric(df.get("valorizado"), errors="coerce")
    ultimo_precio_series = pd.to_numeric(df.get("ultimoPrecio"), errors="coerce")
    qty_series = pd.to_numeric(df.get("cantidad"), errors="coerce")

    scale_audit: list[dict[str, Any]] = []
    final_scale = df["scale"].copy()

    for idx in df.index:
        configured = float(final_scale.at[idx]) if idx in final_scale.index else 1.0
        if not np.isfinite(configured) or configured <= 0:
            configured = 1.0

        detected = configured
        sym_value = df.at[idx, "simbolo"] if idx in df.index else ""
        sym_clean = clean_symbol(sym_value)
        valorizado_total = float(valorizado_series.at[idx]) if idx in valorizado_series.index else float("nan")
        ultimo_precio = float(ultimo_precio_series.at[idx]) if idx in ultimo_precio_series.index else float("nan")
        cantidad = float(qty_series.at[idx]) if idx in qty_series.index else float("nan")
        valorizado_unit = float("nan")
        absolute_diff = float("nan")
        relative_diff = float("nan")
        base_reason = "override_bopreal_ars" if sym_clean in bopreal_override_symbols else None
        reason = base_reason or "configured"

        if configured < 1.0 and np.isfinite(valorizado_total) and np.isfinite(cantidad) and cantidad != 0.0:
            valorizado_unit = valorizado_total / cantidad
            if np.isfinite(valorizado_unit) and valorizado_unit != 0.0 and np.isfinite(ultimo_precio):
                absolute_diff = abs(valorizado_unit - ultimo_precio)
                baseline = max(abs(valorizado_unit), abs(ultimo_precio), 1.0)
                relative_diff = absolute_diff / baseline if baseline != 0 else float("nan")
                if absolute_diff <= 1e-3:
                    detected = 1.0
                    reason = "payload_aligned_abs"
                elif relative_diff < 0.01:
                    detected = 1.0
                    reason = "payload_aligned"
                else:
                    reason = "scale_required"
            else:
                reason = "insufficient_data"
        elif configured < 1.0:
            reason = "insufficient_data"
        else:
            reason = base_reason or "passthrough"

        final_scale.at[idx] = detected
        scale_audit.append(
            {
                "simbolo": df.at[idx, "simbolo"],
                "tipo": df.at[idx, "tipo_estandar"],
                "scale_configured": configured,
                "scale_detected": detected,
                "valorizado_unit": None if not np.isfinite(valorizado_unit) else valorizado_unit,
                "ultimo_precio": None if not np.isfinite(ultimo_precio) else ultimo_precio,
                "cantidad": None if not np.isfinite(cantidad) else cantidad,
                "absolute_diff": None if not np.isfinite(absolute_diff) else absolute_diff,
                "relative_diff": None if not np.isfinite(relative_diff) else relative_diff,
                "reason": reason,
            }
        )

    df["scale"] = final_scale

    bopreal_ars_mask = (
        df.apply(is_bopreal_ars, axis=1).astype(bool)
        if not df.empty
        else pd.Series(False, index=df.index, dtype=bool)
    )

    # ----- Valoraciones -------------------------------------------------
    provider_series = (
        df.get("provider", pd.Series(index=df.index, dtype="object")).astype("string").str.lower().fillna("")
    )
    currency_series = (
        df.get("moneda_origen", pd.Series(index=df.index, dtype="object")).astype("string").str.upper().fillna("")
    )
    fx_series = pd.to_numeric(df.get("fx_aplicado"), errors="coerce")
    last_series = pd.to_numeric(df.get("last"), errors="coerce")

    invalid_quote_mask = (
        last_series.notna()
        & currency_series.ne("ARS")
        & currency_series.ne("")
        & fx_series.isna()
    )
    if invalid_quote_mask.any():
        last_series = last_series.mask(invalid_quote_mask)
        df.loc[invalid_quote_mask, "chg_pct"] = np.nan
        df.loc[invalid_quote_mask, "prev_close"] = np.nan
        df.loc[invalid_quote_mask, "last"] = np.nan

    safe_mode = _safe_valuation_mode_enabled()
    if safe_mode:
        external_provider_mask = ~provider_series.isin(_SAFE_PROVIDERS)
        if external_provider_mask.any():
            last_series = last_series.mask(external_provider_mask)
            fx_series = fx_series.mask(external_provider_mask)
            df.loc[external_provider_mask, ["last", "chg_pct", "prev_close"]] = np.nan

    df["ultimo"] = np.nan
    df["valor_actual"] = np.nan
    proveedor_utilizado: pd.Series = pd.Series(index=df.index, dtype="string")

    qty_scale = df["cantidad"] * df["scale"]

    valorizado_series = pd.Series(index=df.index, dtype=float)
    forced_valorizado_mask = pd.Series(False, index=df.index, dtype=bool)
    forced_revaluation_symbols: set[str] = set()
    bopreal_force_audit_entries: list[dict[str, Any]] = []
    bopreal_market_audit_entries: list[dict[str, Any]] = []
    fallback_market_source: str | None = None

    if "valorizado" in df.columns:
        valorizado_series = pd.to_numeric(df["valorizado"], errors="coerce")
        mask_valorizado = valorizado_series.notna()
        if mask_valorizado.any():
            forced_valorizado_mask = mask_valorizado & bopreal_ars_mask
            if forced_valorizado_mask.any():
                forced_symbols = df.loc[forced_valorizado_mask, "simbolo"].apply(clean_symbol)
                forced_revaluation_symbols.update(sym for sym in forced_symbols if sym)

            effective_valorizado_mask = mask_valorizado & ~bopreal_ars_mask
            if effective_valorizado_mask.any():
                denom = qty_scale.replace({0: np.nan})
                price = pd.Series(np.nan, index=df.index, dtype=float)
                valid_price_mask = effective_valorizado_mask & denom.notna()
                price.loc[valid_price_mask] = (
                    valorizado_series.loc[valid_price_mask] / denom.loc[valid_price_mask]
                )
                df.loc[
                    effective_valorizado_mask & price.notna(), "ultimo"
                ] = price.loc[effective_valorizado_mask & price.notna()]
                df.loc[effective_valorizado_mask, "valor_actual"] = valorizado_series.loc[
                    effective_valorizado_mask
                ]
                proveedor_utilizado.loc[effective_valorizado_mask] = "valorizado"

    if "ultimoPrecio" in df.columns:
        ultimo_precio = pd.to_numeric(df["ultimoPrecio"], errors="coerce")
    else:
        ultimo_precio = pd.Series(index=df.index, dtype=float)
    moneda_pos = df.get("moneda", pd.Series(index=df.index, dtype="object")).astype("string").str.upper().fillna("")
    mask_ultimo = (
        df["valor_actual"].isna()
        & ultimo_precio.notna()
        & (moneda_pos.eq("ARS") | bopreal_ars_mask)
    )
    if mask_ultimo.any():
        df.loc[mask_ultimo, "ultimo"] = ultimo_precio.loc[mask_ultimo]
        df.loc[mask_ultimo, "valor_actual"] = ultimo_precio.loc[mask_ultimo] * qty_scale.loc[mask_ultimo]
        proveedor_utilizado.loc[mask_ultimo] = "ultimoPrecio"
    df["ultimoPrecio"] = ultimo_precio

    if market_price_fetcher is not None and bopreal_ars_mask.any():
        fallback_threshold = 10_000.0
        with np.errstate(divide="ignore", invalid="ignore"):
            unit_valorizado = valorizado_series.divide(qty_scale)
            unit_valorizado = unit_valorizado.replace([np.inf, -np.inf], np.nan)
        low_price_mask = ultimo_precio.notna() & (ultimo_precio < fallback_threshold)
        low_valorizado_mask = unit_valorizado.notna() & (unit_valorizado < fallback_threshold)
        fallback_candidates = bopreal_ars_mask & (low_price_mask | low_valorizado_mask)
        if fallback_candidates.any():
            for idx in df.index[fallback_candidates]:
                symbol_value = df.at[idx, "simbolo"] if idx in df.index else ""
                symbol_clean = clean_symbol(symbol_value)
                if not symbol_clean:
                    continue
                payload_last = _to_float(ultimo_precio.loc[idx]) if idx in ultimo_precio.index else None
                payload_val = _to_float(valorizado_series.loc[idx]) if idx in valorizado_series.index else None
                try:
                    fetched_price, source_url = market_price_fetcher(symbol_clean)
                except Exception as exc:  # pragma: no cover - defensive safeguard
                    logger.warning(
                        "market_revaluation_fallback error for %s: %s",
                        symbol_clean,
                        exc,
                    )
                    continue
                if fetched_price is None:
                    continue
                try:
                    if not np.isfinite(fetched_price) or fetched_price <= 0.0:
                        continue
                except Exception:
                    continue
                price_value = float(fetched_price)
                qty_effective = qty_scale.loc[idx] if idx in qty_scale.index else np.nan
                if not np.isfinite(qty_effective):
                    qty_effective = df.loc[idx, "cantidad"] if idx in df.index else np.nan
                if not np.isfinite(qty_effective):
                    qty_effective = 0.0
                market_value = price_value * float(qty_effective)
                ultimo_precio.loc[idx] = price_value
                valorizado_series.loc[idx] = market_value
                last_series.loc[idx] = price_value
                df.at[idx, "ultimo"] = price_value
                df.at[idx, "ultimoPrecio"] = price_value
                df.at[idx, "valor_actual"] = market_value
                if "valorizado" in df.columns:
                    df.at[idx, "valorizado"] = market_value
                df.at[idx, "last"] = price_value
                proveedor_utilizado.loc[idx] = "market_revaluation_fallback"
                fallback_market_source = source_url or fallback_market_source
                factor = None
                if payload_last not in (None, 0):
                    try:
                        factor = price_value / float(payload_last)
                    except ZeroDivisionError:
                        factor = None
                bopreal_market_audit_entries.append(
                    {
                        "simbolo": symbol_value,
                        "cantidad": _to_float(df.at[idx, "cantidad"]),
                        "ultimoPrecio_payload": payload_last,
                        "valorizado_payload": payload_val,
                        "market_price": price_value,
                        "market_value": market_value,
                        "endpoint": source_url,
                        "factor_estimado": factor,
                    }
                )
                logger.info(
                    "[Audit] override_bopreal_market applied",
                    extra={
                        "symbol": symbol_clean,
                        "source": source_url,
                        "payload_last": payload_last,
                        "market_last": price_value,
                        "factor": factor,
                    },
                )

    forced_pending_mask = forced_valorizado_mask & df["valor_actual"].isna()
    if forced_pending_mask.any():
        rescale_factor = df.loc[forced_pending_mask, "scale"] / BOPREAL_HISTORICAL_SCALE
        rescale_factor = rescale_factor.replace([np.inf, -np.inf], np.nan)
        valor_rescaled = valorizado_series.loc[forced_pending_mask] * rescale_factor
        df.loc[forced_pending_mask, "valor_actual"] = valor_rescaled
        denom = qty_scale.replace({0: np.nan})
        valid_forced_mask = forced_pending_mask & denom.notna()
        if valid_forced_mask.any():
            price_rescaled = valor_rescaled.loc[valid_forced_mask] / denom.loc[valid_forced_mask]
            df.loc[valid_forced_mask, "ultimo"] = price_rescaled
        proveedor_utilizado.loc[forced_pending_mask] = "valorizado_rescaled"

    tipo_series = (
        df.get("tipo_estandar", pd.Series(index=df.index, dtype="object"))
        .astype("string")
        .str.lower()
        .fillna("")
    )
    provider_upper = provider_series.str.upper()
    if "proveedor_original" in df.columns:
        provider_upper = provider_upper.where(
            provider_upper.ne(""),
            df["proveedor_original"].astype("string").str.upper().fillna(""),
        )
    provider_upper = provider_upper.where(
        provider_upper.ne(""),
        df.get("pricing_source", pd.Series(index=df.index, dtype="object"))
        .astype("string")
        .str.upper()
        .fillna(""),
    )

    bopreal_force_candidates = (
        bopreal_ars_mask
        & tipo_series.str.contains("bono", regex=False, na=False)
        & provider_upper.isin({"IOL", "IOL-LIVE", ""})
    )

    if bopreal_force_candidates.any():
        ppc_series = pd.to_numeric(df.get("ppc"), errors="coerce")
        ratio_to_ppc = ultimo_precio.divide(ppc_series).replace([np.inf, -np.inf], np.nan)
        low_ratio_mask = ratio_to_ppc <= (BOPREAL_HISTORICAL_SCALE * 3.0)
        low_price_mask = ultimo_precio <= 5_000.0
        effective_mask = bopreal_force_candidates & ultimo_precio.notna() & (
            low_ratio_mask.fillna(False) | low_price_mask.fillna(False)
        )
        if effective_mask.any():
            forced_last = ultimo_precio.loc[effective_mask] * BOPREAL_FORCE_FACTOR
            forced_last = forced_last.replace([np.inf, -np.inf], np.nan)
            forced_value = forced_last * qty_scale.loc[effective_mask]
            forced_value = forced_value.replace([np.inf, -np.inf], np.nan)
            df.loc[effective_mask, "ultimo"] = forced_last
            df.loc[effective_mask, "valor_actual"] = forced_value
            df.loc[effective_mask, "valorizado"] = forced_value
            proveedor_utilizado.loc[effective_mask] = "override_bopreal_forced"
            forced_revaluation_symbols.update(
                clean_symbol(sym)
                for sym in df.loc[effective_mask, "simbolo"].tolist()
                if sym
            )
            for idx in df.index[effective_mask]:
                bopreal_force_audit_entries.append(
                    {
                        "simbolo": df.at[idx, "simbolo"],
                        "forced": True,
                        "factor_aplicado": BOPREAL_FORCE_FACTOR,
                        "ultimo_precio_original": _to_float(ultimo_precio.loc[idx]),
                        "ultimo_precio_forzado": _to_float(df.loc[idx, "ultimo"]),
                    }
                )

    mask_cotizacion = (
        df["valor_actual"].isna()
        & last_series.notna()
        & provider_series.isin({"iol", "cache", "stale"})
    )
    if mask_cotizacion.any():
        df.loc[mask_cotizacion, "ultimo"] = last_series.loc[mask_cotizacion]
        df.loc[mask_cotizacion, "valor_actual"] = last_series.loc[mask_cotizacion] * qty_scale.loc[mask_cotizacion]
        proveedor_utilizado.loc[mask_cotizacion] = provider_series.loc[mask_cotizacion]

    mask_externo = (
        df["valor_actual"].isna()
        & last_series.notna()
        & ~provider_series.isin({"iol", "cache", "stale", ""})
        & fx_series.notna()
    )
    if mask_externo.any():
        df.loc[mask_externo, "ultimo"] = last_series.loc[mask_externo]
        df.loc[mask_externo, "valor_actual"] = last_series.loc[mask_externo] * qty_scale.loc[mask_externo]
        proveedor_utilizado.loc[mask_externo] = provider_series.loc[mask_externo]

    df["ultimo"] = df["ultimo"].fillna(last_series)
    fallback_mask = df["valor_actual"].isna() & df["ultimo"].notna()
    if fallback_mask.any():
        df.loc[fallback_mask, "valor_actual"] = df.loc[fallback_mask, "ultimo"] * qty_scale.loc[fallback_mask]
        proveedor_utilizado.loc[fallback_mask] = proveedor_utilizado.loc[fallback_mask].where(
            proveedor_utilizado.loc[fallback_mask].notna(),
            provider_series.loc[fallback_mask].where(provider_series.loc[fallback_mask] != "", "last"),
        )
    df["costo"] = df["cantidad"] * df["ppc"] * df["scale"]
    df["pl"] = df["valor_actual"] - df["costo"]
    df["pl_%"] = np.where(df["costo"] != 0, df["pl"] / df["costo"] * 100.0, np.nan)

    # ----- P/L diario ---------------------------------------------------
    chg_series = pd.to_numeric(df["chg_pct"], errors="coerce")
    pct = chg_series / 100.0
    mask = chg_series.isna() & df["ultimo"].notna() & df["prev_close"].notna() & (df["prev_close"] != 0)
    pct = pct.where(~mask, (df["ultimo"] - df["prev_close"]) / df["prev_close"])
    denom = 1.0 + pct
    df["pl_d"] = np.where(
        df["valor_actual"].notna() & pct.notna() & (denom != 0),
        df["valor_actual"] * pct / denom,
        np.nan,
    )
    df["pld_%"] = pct * 100.0
    if "variacionDiaria" in df.columns:
        fallback_variacion = pd.to_numeric(df["variacionDiaria"], errors="coerce")
        df["pld_%"] = df["pld_%"].where(df["pld_%"].notna(), fallback_variacion)

    # return pd.DataFrame(rows)
    # Orden final --------------------------------------------------------
    df["moneda_origen"] = currency_series
    df["fx_aplicado"] = fx_series
    pricing_source = proveedor_utilizado.astype("string")
    df["pricing_source"] = pricing_source.fillna("")
    df["mercado"] = df["mercado"].str.upper()

    def _safe_float_log(value: Any) -> float | None:
        parsed = _to_float(value, log=False)
        if parsed is None or not np.isfinite(parsed):
            return None
        return float(parsed)

    ratio_column = None
    for candidate in ("ratioCEDEAR", "ratioCEDEAR_x", "ratioCEDEAR_y"):
        if candidate in df.columns:
            ratio_column = candidate
            break
    ratio_series = (
        pd.to_numeric(df[ratio_column], errors="coerce")
        if ratio_column is not None
        else pd.Series(index=df.index, dtype=float)
    )

    valuation_entries: list[dict[str, Any]] = []
    for idx in df.index:
        raw_provider = proveedor_utilizado.loc[idx] if idx in proveedor_utilizado.index else None
        provider_text = None
        if raw_provider is not None and pd.notna(raw_provider):
            provider_candidate = str(raw_provider).strip()
            provider_text = provider_candidate or None
        fx_value_log = _safe_float_log(fx_series.loc[idx] if idx in fx_series.index else None)
        ratio_value_log = _safe_float_log(ratio_series.loc[idx] if idx in ratio_series.index else None)
        valuation_entries.append(
            {
                "simbolo": df.at[idx, "simbolo"],
                "mercado": df.at[idx, "mercado"],
                "proveedor_utilizado": provider_text,
                "fx_aplicado": fx_value_log,
                "ratioCEDEAR": ratio_value_log,
            }
        )

    result = df[cols]
    if forced_revaluation_symbols and scale_audit:
        forced_clean = {clean_symbol(sym) for sym in forced_revaluation_symbols}
        for entry in scale_audit:
            sym = clean_symbol(entry.get("simbolo")) if isinstance(entry, Mapping) else ""
            if sym in forced_clean:
                entry["reason"] = BOPREAL_FORCED_REVALUATION_TAG
                tags = entry.get("tags") if isinstance(entry, dict) else None
                if isinstance(tags, list):
                    if BOPREAL_FORCED_REVALUATION_TAG not in tags:
                        tags.append(BOPREAL_FORCED_REVALUATION_TAG)
                else:
                    entry["tags"] = [BOPREAL_FORCED_REVALUATION_TAG]
                entry["forced_revaluation"] = True
    if scale_audit or bopreal_force_audit_entries or bopreal_market_audit_entries:
        audit_section = attrs.get("audit")
        if isinstance(audit_section, Mapping):
            audit_data: dict[str, Any] = dict(audit_section)
        elif audit_section:
            audit_data = {"legacy": audit_section}
        else:
            audit_data = {}
        if scale_audit:
            audit_data["scale_decisions"] = scale_audit
        if bopreal_force_audit_entries:
            existing_bopreal = audit_data.get("bopreal")
            if isinstance(existing_bopreal, list):
                audit_data["bopreal"] = [*existing_bopreal, *bopreal_force_audit_entries]
            elif existing_bopreal:
                audit_data["bopreal"] = [existing_bopreal, *bopreal_force_audit_entries]
            else:
                audit_data["bopreal"] = bopreal_force_audit_entries
        if bopreal_market_audit_entries:
            existing_market = audit_data.get("bopreal_market")
            if isinstance(existing_market, list):
                audit_data["bopreal_market"] = [
                    *existing_market,
                    *bopreal_market_audit_entries,
                ]
            elif existing_market:
                audit_data["bopreal_market"] = [existing_market, *bopreal_market_audit_entries]
            else:
                audit_data["bopreal_market"] = bopreal_market_audit_entries
            audit_data["override_bopreal_market"] = True
            if fallback_market_source:
                audit_data["market_price_source"] = fallback_market_source
            audit_data["timestamp_fallback"] = datetime.utcnow()
            quotes_hash_value = attrs.get("quotes_hash")
            if quotes_hash_value is not None:
                audit_data["quotes_hash"] = quotes_hash_value
        attrs["audit"] = audit_data
    source_counts = proveedor_utilizado.dropna().value_counts().to_dict()
    if source_counts:
        logger.info("calc_rows proveedor_utilizado=%s", source_counts)
    if valuation_entries:
        logger.info(
            "portfolio_valuation_sources",
            extra={
                "event": "portfolio_valuation_sources",
                "valuations": valuation_entries,
                "safe_mode": safe_mode,
            },
        )
    if attrs:
        result.attrs.update(attrs)
    return sanitize_attrs(result)


def detect_bond_scale_anomalies(df_view: pd.DataFrame | None) -> tuple[pd.DataFrame, float]:
    """Detecta posibles errores de escala en bonos y letras."""

    report_columns = [
        "simbolo",
        "tipo",
        "ultimoPrecio",
        "valorizado_div_cantidad",
        "scale",
        "proveedor_utilizado",
        "diagnostico",
        "motivo_scale",
        "impacto_estimado",
    ]

    empty_report = pd.DataFrame(columns=report_columns)
    if df_view is None or df_view.empty:
        return empty_report, 0.0

    df = df_view.copy()
    tipo_series = df.get("tipo_estandar")
    if tipo_series is None:
        tipo_series = df.get("tipo")
    tipo_series = tipo_series.astype("string") if tipo_series is not None else pd.Series(dtype="string")
    mask = tipo_series.str.lower().str.contains("bono|letra", regex=True, na=False)
    df_subset = df.loc[mask].copy()
    if df_subset.empty:
        return empty_report, 0.0

    def _ensure_series(values: Any) -> pd.Series:
        if isinstance(values, pd.Series):
            return values
        return pd.Series(values, index=df_subset.index)

    qty = pd.to_numeric(_ensure_series(df_subset.get("cantidad")), errors="coerce")
    valor_actual = pd.to_numeric(_ensure_series(df_subset.get("valor_actual")), errors="coerce")
    valorizado_total = pd.to_numeric(_ensure_series(df_subset.get("valorizado")), errors="coerce")
    ultimo_precio = pd.to_numeric(_ensure_series(df_subset.get("ultimoPrecio")), errors="coerce")
    scale = pd.to_numeric(_ensure_series(df_subset.get("scale")), errors="coerce").fillna(1.0)
    provider = (
        _ensure_series(df_subset.get("pricing_source", pd.Series(index=df_subset.index, dtype="object")))
        .astype("string")
        .fillna("")
    )
    ppc_series = pd.to_numeric(_ensure_series(df_subset.get("ppc")), errors="coerce")
    currency_subset = (
        _ensure_series(df_subset.get("moneda_origen", pd.Series(index=df_subset.index, dtype="object")))
        .astype("string")
        .str.upper()
        .fillna("")
    )
    provider_upper = provider.astype("string").str.upper().fillna("")
    symbol_clean_series = df_subset.get("simbolo", pd.Series(index=df_subset.index, dtype="object")).apply(clean_symbol)
    bopreal_candidates = (
        symbol_clean_series.isin(BOPREAL_SYMBOLS)
        & currency_subset.eq("ARS")
        & provider_upper.isin(BOPREAL_TRUSTED_PROVIDERS)
        & ppc_series.notna()
        & (ppc_series > 0)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        valor_unitario_payload = valorizado_total.divide(qty)
        valor_unitario_payload = valor_unitario_payload.replace([np.inf, -np.inf], np.nan)
        valor_unitario_actual = valor_actual.divide(qty)
        valor_unitario_actual = valor_unitario_actual.replace([np.inf, -np.inf], np.nan)

    valor_unitario_ref = valor_unitario_payload.combine_first(valor_unitario_actual)

    def _pct_diff(series_a: pd.Series, series_b: pd.Series) -> pd.Series:
        diff = (series_a - series_b).abs()
        base = pd.concat([series_a.abs(), series_b.abs()], axis=1).max(axis=1)
        base = base.replace({0: np.nan})
        return diff.divide(base)

    price_diff_pct = _pct_diff(ultimo_precio, valor_unitario_actual)
    ppc_diff_pct = _pct_diff(ultimo_precio, ppc_series)

    double_scale_mask = (
        (scale < 1.0)
        & ultimo_precio.notna()
        & valor_unitario_actual.notna()
        & (price_diff_pct > 0.10)
        & (ppc_diff_pct <= 0.10)
    )

    bopreal_low_price_mask = (
        bopreal_candidates
        & ultimo_precio.notna()
        & (ultimo_precio <= (ppc_series / 50.0))
    )

    diagnostico = np.where(double_scale_mask, "escala duplicada", "correcto")
    diagnostico = np.where(bopreal_low_price_mask, "bopreal_precio_truncado", diagnostico)

    impacto_double = (ultimo_precio * qty) - valor_actual
    impacto_bopreal = (ultimo_precio * BOPREAL_FORCE_FACTOR * qty) - valor_actual
    impacto_estimado_values = np.where(double_scale_mask, impacto_double, 0.0)
    impacto_estimado_values = np.where(bopreal_low_price_mask, impacto_bopreal, impacto_estimado_values)
    impacto_estimado = pd.Series(impacto_estimado_values, index=df_subset.index, dtype=float)

    attrs = getattr(df_view, "attrs", {}) if df_view is not None else {}
    audit_entries = attrs.get("audit") if isinstance(attrs, Mapping) else None
    scale_decisions = {}
    if isinstance(audit_entries, Mapping):
        raw_decisions = audit_entries.get("scale_decisions", [])
        if isinstance(raw_decisions, list):
            for entry in raw_decisions:
                if not isinstance(entry, Mapping):
                    continue
                sym = clean_symbol(entry.get("simbolo"))
                if sym and sym not in scale_decisions:
                    scale_decisions[sym] = entry

    motivo_series = []
    for idx in df_subset.index:
        sym = clean_symbol(df_subset.at[idx, "simbolo"])
        decision = scale_decisions.get(sym, {})
        motivo_series.append(decision.get("reason"))
    motivo_series = pd.Series(motivo_series, index=df_subset.index, dtype="object")

    report = pd.DataFrame(
        {
            "simbolo": df_subset["simbolo"],
            "tipo": tipo_series.loc[df_subset.index] if not tipo_series.empty else df_subset.get("tipo"),
            "ultimoPrecio": ultimo_precio,
            "valorizado_div_cantidad": valor_unitario_ref,
            "scale": scale,
            "proveedor_utilizado": provider,
            "diagnostico": diagnostico,
            "motivo_scale": motivo_series,
            "impacto_estimado": impacto_estimado,
        }
    )

    total_impact = float(np.nansum(impacto_estimado.to_numpy()))
    return report, total_impact


class PortfolioService:
    """Fachada del portafolio que envuelve tus funciones ya existentes."""

    def normalize_positions(self, payload: dict | list) -> pd.DataFrame:
        """Normalize raw IOL payload into a DataFrame of positions."""
        return normalize_positions(payload)

    def calc_rows(self, price_fn, df_pos, exclude_syms=None):
        """Calculate valuation and P/L metrics for each position."""
        return calc_rows(price_fn, df_pos, exclude_syms or [])

    def classify_asset(self, row):
        """Classify asset row without caching."""
        return classify_asset(row)

    def clean_symbol(self, sym: str) -> str:
        """Normalize a symbol string."""
        return clean_symbol(sym)

    def simulate_allocation(
        self,
        *,
        portfolio_positions: pd.DataFrame | None,
        totals: PortfolioTotals | None,
        recommendations: pd.DataFrame | None,
        expected_returns: Mapping[str, float] | None = None,
        betas: Mapping[str, float] | None = None,
    ) -> dict[str, dict[str, float]]:
        """Combina el portafolio actual con nuevas asignaciones y resume métricas."""

        portfolio_df = portfolio_positions.copy() if isinstance(portfolio_positions, pd.DataFrame) else pd.DataFrame()
        rec_df = recommendations.copy() if isinstance(recommendations, pd.DataFrame) else pd.DataFrame()

        base_totals = totals if isinstance(totals, PortfolioTotals) else calculate_totals(portfolio_df)
        before_value = float(getattr(base_totals, "total_value", 0.0) or 0.0)
        if not np.isfinite(before_value):
            before_value = 0.0
        base_pl_pct = getattr(base_totals, "total_pl_pct", float("nan"))
        if not np.isfinite(base_pl_pct):
            base_rate = 0.0
        else:
            base_rate = float(base_pl_pct) / 100.0
        before_expected_value = before_value * base_rate

        allocation = pd.to_numeric(rec_df.get("allocation_amount"), errors="coerce").fillna(0.0)
        rec_symbols = rec_df.get("symbol", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
        new_capital = float(allocation.sum()) if len(allocation) else 0.0

        expected_lookup: dict[str, float] = {}
        for key, value in (expected_returns or {}).items():
            symbol = str(key or "").strip().upper()
            if not symbol:
                continue
            try:
                rate = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(rate):
                expected_lookup[symbol] = rate

        beta_lookup: dict[str, float] = {}
        for key, value in (betas or {}).items():
            symbol = str(key or "").strip().upper()
            if not symbol:
                continue
            try:
                beta_val = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(beta_val):
                beta_lookup[symbol] = beta_val

        existing_values: dict[str, float] = {}
        if not portfolio_df.empty and "valor_actual" in portfolio_df.columns:
            values = pd.to_numeric(portfolio_df["valor_actual"], errors="coerce").fillna(0.0)
            symbols = portfolio_df.get("simbolo", pd.Series(dtype=str)).astype("string").fillna("").str.upper()
            for symbol, value in zip(symbols, values):
                if not symbol:
                    continue
                existing_values[symbol] = existing_values.get(symbol, 0.0) + float(value)

        combined_values = dict(existing_values)
        additional_expected = 0.0

        for symbol, amount in zip(rec_symbols, allocation):
            amount_value = float(amount)
            if amount_value <= 0.0 or not symbol:
                continue
            combined_values[symbol] = combined_values.get(symbol, 0.0) + amount_value
            rate = expected_lookup.get(symbol)
            if rate is None or not np.isfinite(rate):
                rate = base_rate * 100.0
            additional_expected += amount_value * (float(rate) / 100.0)

        after_value = before_value + new_capital
        after_expected_value = before_expected_value + additional_expected

        def _weighted_metric(weights: Mapping[str, float]) -> float:
            total_weight = 0.0
            weighted_sum = 0.0
            for sym, weight in weights.items():
                if weight <= 0.0:
                    continue
                total_weight += float(weight)
                metric = beta_lookup.get(sym)
                try:
                    metric_val = float(metric)
                except (TypeError, ValueError):  # pragma: no cover - defensive
                    metric_val = float("nan")
                if not np.isfinite(metric_val):
                    metric_val = 1.0
                weighted_sum += float(weight) * metric_val
            if total_weight <= 0.0:
                return float("nan")
            return weighted_sum / total_weight

        beta_before = _weighted_metric(existing_values)
        beta_after = _weighted_metric(combined_values)

        projected_before = base_rate * 100.0 if before_value > 0.0 else 0.0
        projected_after = after_expected_value / after_value * 100.0 if after_value > 0.0 else 0.0

        return {
            "before": {
                "total_value": before_value,
                "projected_return": projected_before,
                "beta": beta_before,
            },
            "after": {
                "total_value": after_value,
                "projected_return": projected_after,
                "beta": beta_after,
                "additional_investment": new_capital,
            },
        }

_SAFE_PROVIDERS = frozenset({"iol", "cache", "stale", ""})


def _safe_valuation_mode_enabled() -> bool:
    try:
        return bool(getattr(settings, "SAFE_VALUATION_MODE", False))
    except Exception:  # pragma: no cover - defensive guard when settings unavailable
        return False
