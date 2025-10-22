# application\portfolio_service.py
# Lógica de negocio: normalización de posiciones y cálculo de métricas P/L

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping

import numpy as np
import pandas as pd
import requests

from infrastructure.asset_catalog import get_asset_catalog
from shared.config import get_config
from shared.utils import _to_float


@dataclass(frozen=True)
class PortfolioTotals:
    """Totales básicos del portafolio."""

    total_value: float
    total_cost: float
    total_pl: float
    total_pl_pct: float
    total_cash: float = 0.0


def calculate_totals(df_view: pd.DataFrame | None) -> PortfolioTotals:
    """Calcula totales agregados de valorizado, costo y P/L."""

    if df_view is None or df_view.empty:
        return PortfolioTotals(0.0, 0.0, 0.0, float("nan"), 0.0)

    valor_actual = df_view.get("valor_actual", pd.Series(dtype=float))
    costo = df_view.get("costo", pd.Series(dtype=float))

    total_val = float(np.nansum(getattr(valor_actual, "values", [])))
    total_cost = float(np.nansum(getattr(costo, "values", [])))
    total_pl = total_val - total_cost

    cash_mask = df_view.get("simbolo")
    cash_series = pd.Series(dtype=float)
    if cash_mask is not None and "valor_actual" in df_view.columns:
        try:
            mask = cash_mask.astype(str).str.upper().isin({"IOLPORA", "PARKING"})
            cash_series = pd.to_numeric(df_view.loc[mask, "valor_actual"], errors="coerce")
        except Exception:
            cash_series = pd.Series(dtype=float)
    total_cash = float(np.nansum(getattr(cash_series, "values", [])))

    if np.isfinite(total_cost) and not np.isclose(total_cost, 0.0):
        total_pl_pct = (total_pl / total_cost) * 100.0
    else:
        total_pl_pct = float("nan")

    return PortfolioTotals(total_val, total_cost, total_pl, total_pl_pct, total_cash)


def detect_currency(sym: str, tipo: str | None) -> str:
    """Determina la moneda en base al símbolo informado."""

    return "USD" if str(sym).upper() in {"PRPEDOB"} else "ARS"


logger = logging.getLogger(__name__)


# ---------- Helpers y configuración ----------
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


# ---------- Clasificación y escala ----------


def classify_symbol(sym: str) -> str:
    """
    Clasifica por símbolo usando listas del config (cedears/etfs) y patrones
    configurables. Devuelve una de: 'CEDEAR','ETF','Bono','Letra','FCI','Acción','Otro'
    """
    s = clean_symbol(sym)
    cfg = get_config()
    catalog = get_asset_catalog()
    cedears_map = cfg.get("cedear_to_us", {}) or {}
    etf_set = set(map(clean_symbol, cfg.get("etfs", []) or []))
    acciones_ar = set(map(clean_symbol, cfg.get("acciones_ar", []) or []))
    fci_set = set(map(clean_symbol, cfg.get("fci_symbols", []) or []))
    pattern_map = cfg.get("classification_patterns", {}) or {}

    # Catálogo centralizado
    if s in catalog:
        return catalog[s]

    # Listas explícitas desde config
    if s in etf_set:
        return "ETF"
    if s in cedears_map:
        return "CEDEAR"
    if s in fci_set:
        return "FCI"

    # Heurísticas para Argentina (bonos/letras, etc.)
    if s.startswith(("AL", "GD", "AE")):
        return "Bono"
    if s.startswith("S") and any(ch.isdigit() for ch in s):
        return "Letra"
    # Clasificación basada en patrones configurables
    for tipo, patterns in pattern_map.items():
        for pat in patterns or []:
            try:
                if re.match(pat, s):
                    return tipo
            except re.error:
                continue

    if s in acciones_ar:
        return "Acción"
    if s.isalpha() and 3 <= len(s) <= 5:
        return "CEDEAR"
    return "Otro"


def scale_for(sym: str, tipo: str) -> float:
    """
    Factor de escala para convertir el precio/costo reportado a valuación.
    - Overrides por símbolo (desde config['scale_overrides'])
    - Bonos/Letra típicamente VN 100 -> factor 0.01
    - Resto -> 1.0
    """
    s = clean_symbol(sym)
    cfg = get_config()
    scale_overrides = cfg.get("scale_overrides", {}) or {}
    if s in scale_overrides:
        try:
            f = float(scale_overrides[s])
            if f > 0:
                return f
        except (TypeError, ValueError) as e:
            logger.exception("scale_overrides inválido para %s: %s", s, e)

    tipo_norm = (tipo or "").lower()
    if any(x in tipo_norm for x in ("bono", "letra")):
        return 0.01
    return 1.0


def _match_declared_type(text: str) -> str | None:
    """Return portfolio type based on labels declared by IOL."""

    label = (text or "").strip().lower()
    if not label:
        return None

    if "bono" in label or "oblig" in label or "negociable" in label:
        return "Bono"
    if "letra" in label:
        return "Letra"
    if "fci" in label or "fondo" in label or "money market" in label:
        return "FCI"
    if "cedear" in label:
        return "CEDEAR"
    if "etf" in label:
        return "ETF"
    if "acción" in label or "accion" in label or "equity" in label:
        return "Acción"

    return None


def classify_asset(it: dict) -> str:
    """
    Clasifica activo según 'titulo.tipo/descripcion' si existe, o heurística por símbolo.
    Devuelve una de: 'CEDEAR','ETF','Bono','Letra','FCI','Acción','Otro'
    """

    t = it.get("titulo") or {}
    sym = clean_symbol(it.get("simbolo", ""))

    candidates = []
    tipo_value = t.get("tipo")
    descripcion_value = t.get("descripcion")

    if tipo_value:
        candidates.append(tipo_value)
    if descripcion_value and descripcion_value not in candidates:
        candidates.append(descripcion_value)

    for candidate in candidates:
        matched = _match_declared_type(candidate)
        if matched:
            return matched

    # Fallback por símbolo
    return classify_symbol(sym)


# ---------- Normalización de payload ----------


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

    for it in source:
        t = it.get("titulo") if isinstance(it, dict) else None

        simbolo = (
            (it.get("simbolo") if isinstance(it, dict) else None)
            or (t.get("simbolo") if isinstance(t, dict) else None)
            or (it.get("ticker") if isinstance(it, dict) else None)
            or (it.get("codigo") if isinstance(it, dict) else None)
            or ""
        )
        mercado = (
            (it.get("mercado") if isinstance(it, dict) else None)
            or (t.get("mercado") if isinstance(t, dict) else None)
            or (it.get("market") if isinstance(it, dict) else None)
            or "bcba"
        )
        cantidad = (
            (it.get("cantidad") if isinstance(it, dict) else None)
            or (it.get("cant") if isinstance(it, dict) else None)
            or (it.get("cantidadDisponible") if isinstance(it, dict) else None)
            or (it.get("cantidadNominal") if isinstance(it, dict) else None)
            or (it.get("tenencia") if isinstance(it, dict) else None)
            or 0
        )

        costo_unit = (
            (it.get("costoUnitario") if isinstance(it, dict) else None)
            or (it.get("ppc") if isinstance(it, dict) else None)
            or (t.get("costoUnitario") if isinstance(t, dict) else None)
            or (t.get("ppc") if isinstance(t, dict) else None)
        )

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

        if simbolo and cantidad_f:
            items.append(
                {
                    "simbolo": clean_symbol(simbolo),
                    "mercado": str(mercado).strip().lower(),
                    "cantidad": float(cantidad_f),
                    "costo_unitario": float(costo_unit_f),
                }
            )

    return pd.DataFrame(items, columns=["simbolo", "mercado", "cantidad", "costo_unitario"])


# ---------- Cálculo de métricas ----------


def calc_rows(get_quote_fn, df_pos: pd.DataFrame, exclude_syms: Iterable[str]) -> pd.DataFrame:
    """Calcula métricas de valuación y P/L para cada posición.

    La lógica original iteraba fila por fila; aquí se vectoriza el cálculo:
      * Se construye un DataFrame con cotizaciones (``last`` y ``chg_pct``) y
        se une a ``df_pos`` por ``simbolo``/``mercado``.
      * ``classify_symbol`` y ``scale_for`` se aplican sobre columnas completas.
      * Las métricas ``valor_actual``, ``costo`` y P/L se derivan mediante
        operaciones vectorizadas de ``pandas``.
    """
    cols = [
        "simbolo",
        "mercado",
        "tipo",
        "cantidad",
        "ppc",
        "ultimo",
        "valor_actual",
        "costo",
        "pl",
        "pl_%",
        "pl_d",
        "pld_%",
    ]

    if df_pos is None or df_pos.empty:
        return pd.DataFrame(columns=cols)

    # Normalización básica y exclusiones ---------------------------------
    df = df_pos.copy()
    df["simbolo"] = df["simbolo"].map(clean_symbol)
    df["mercado"] = df["mercado"].astype(str).str.lower()
    ex = {clean_symbol(s) for s in (exclude_syms or [])}

    df = df[~df["simbolo"].isin(ex)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    df["cantidad"] = df["cantidad"].map(_to_float).fillna(0.0)
    df["ppc"] = df.get("costo_unitario", np.nan).map(_to_float).fillna(0.0)

    # ----- Cotizaciones -------------------------------------------------
    uniq = df[["mercado", "simbolo"]].drop_duplicates().reset_index(drop=True)

    # ----- cotización (acepta float o dict) -----
    def _fetch(row: pd.Series) -> pd.Series:
        last, chg_pct, prev_close = None, None, None
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
        else:
            last = _to_float(q)

        return pd.Series({"last": last, "chg_pct": chg_pct, "prev_close": prev_close})

    quotes_df = uniq.apply(_fetch, axis=1)
    quotes_df = pd.concat([uniq, quotes_df], axis=1)

    df = df.merge(quotes_df, on=["mercado", "simbolo"], how="left")

    # ----- Clasificación y escala --------------------------------------
    df["tipo"] = df["simbolo"].map(classify_symbol)
    uniq = df.drop_duplicates("simbolo")[["simbolo", "tipo"]]
    scale_map = {s: scale_for(s, t) for s, t in uniq.itertuples(index=False)}
    df["scale"] = df["simbolo"].map(scale_map)

    # ----- Valoraciones -------------------------------------------------
    df["ultimo"] = df["last"]
    df["costo"] = df["cantidad"] * df["ppc"] * df["scale"]
    df["valor_actual"] = df["cantidad"] * df["ultimo"] * df["scale"]
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

    # return pd.DataFrame(rows)
    # Orden final --------------------------------------------------------
    df["mercado"] = df["mercado"].str.upper()
    return df[cols]


@lru_cache(maxsize=1024)
def _classify_sym_cache(sym: str) -> str:
    """Cachea la clasificación por símbolo usando la función existente classify_asset."""
    try:
        return classify_asset({"simbolo": str(sym).upper(), "titulo": {}}) or ""
    except (ValueError, TypeError) as e:
        logger.exception("No se pudo clasificar símbolo %s: %s", sym, e)
        return ""


class PortfolioService:
    """Fachada del portafolio que envuelve tus funciones ya existentes."""

    def normalize_positions(self, payload: dict | list) -> pd.DataFrame:
        """Normalize raw IOL payload into a DataFrame of positions."""
        return normalize_positions(payload)

    def calc_rows(self, price_fn, df_pos, exclude_syms=None):
        """Calculate valuation and P/L metrics for each position."""
        return calc_rows(price_fn, df_pos, exclude_syms or [])

    def classify_asset_cached(self, sym: str) -> str:
        """Classify an asset symbol using a cached helper."""
        return _classify_sym_cache(sym)

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
