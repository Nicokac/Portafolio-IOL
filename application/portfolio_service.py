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
from shared.asset_type_aliases import normalize_asset_type
from shared.config import get_config
from shared.utils import _to_float


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

    return PortfolioTotals(
        total_val,
        total_cost,
        total_pl,
        total_pl_pct,
        total_cash,
        total_cash_ars=cash_ars,
        total_cash_usd=cash_usd,
        total_cash_combined=combined,
        usd_rate=usd_rate,
    )


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




def _canonical_type(label: str | None, *, default: str = "Otro") -> str:
    normalized = normalize_asset_type(label)
    if normalized:
        return normalized
    if label:
        return str(label)
    return default

def classify_symbol(sym: str) -> str:
    """
    Clasifica por símbolo usando listas del config (cedears/etfs) y patrones
    configurables. Devuelve una de las categorías estandarizadas como 'Acción',
    'CEDEAR', 'ETF', 'Bono / ON', 'FCI / Money Market', 'Plazo Fijo', 'Caución'
    u 'Otro'.
    """
    s = clean_symbol(sym)
    cfg = get_config()
    catalog = get_asset_catalog()
    cedears_map = cfg.get("cedear_to_us", {}) or {}
    etf_set = set(map(clean_symbol, cfg.get("etfs", []) or []))
    acciones_ar = set(map(clean_symbol, cfg.get("acciones_ar", []) or []))
    fci_set = set(map(clean_symbol, cfg.get("fci_symbols", []) or []))
    pattern_map = cfg.get("classification_patterns", {}) or {}

    entry = catalog.get(s)
    if entry:
        if isinstance(entry, Mapping):
            tipo_estandar = entry.get("tipo_estandar")
            if tipo_estandar:
                return tipo_estandar
            return _canonical_type(entry.get("tipo") or entry.get("descripcion"))
        return _canonical_type(str(entry))

    if s in etf_set:
        return _canonical_type("ETF")
    if s in cedears_map:
        return _canonical_type("CEDEAR")
    if s in fci_set:
        return _canonical_type("FCI")

    if s.startswith(("AL", "GD", "AE")):
        return _canonical_type("Bono")
    if s.startswith("S") and any(ch.isdigit() for ch in s):
        return _canonical_type("Letra")

    for tipo, patterns in pattern_map.items():
        for pat in patterns or []:
            try:
                if re.match(pat, s):
                    return _canonical_type(tipo)
            except re.error:
                continue

    if s in acciones_ar:
        return _canonical_type("Acción")
    if s.isalpha() and 3 <= len(s) <= 5:
        return _canonical_type("CEDEAR")
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

    normalized = normalize_asset_type(text)
    if normalized:
        return normalized

    label = (text or "").strip().lower()
    if not label:
        return None

    if "bono" in label or "oblig" in label or "negociable" in label:
        return _canonical_type("Bono")
    if "letra" in label:
        return _canonical_type("Letra")
    if "fci" in label or "fondo" in label or "money market" in label:
        return _canonical_type("FCI")
    if "cedear" in label:
        return _canonical_type("CEDEAR")
    if "etf" in label:
        return _canonical_type("ETF")
    if "acción" in label or "accion" in label or "equity" in label:
        return _canonical_type("Acción")

    return None





def classify_asset(it: dict) -> dict[str, str]:
    """Return both the original and normalized asset type for a payload item."""

    t = it.get("titulo") or {}
    sym = clean_symbol(it.get("simbolo", ""))

    candidates: list[str] = []
    tipo_value = t.get("tipo")
    descripcion_value = t.get("descripcion")

    if isinstance(tipo_value, str) and tipo_value.strip():
        candidates.append(tipo_value.strip())
    elif tipo_value not in (None, ""):
        candidates.append(str(tipo_value))

    if isinstance(descripcion_value, str) and descripcion_value.strip():
        candidate_desc = descripcion_value.strip()
        if candidate_desc not in candidates:
            candidates.append(candidate_desc)
    elif descripcion_value not in (None, ""):
        candidate_desc = str(descripcion_value)
        if candidate_desc not in candidates:
            candidates.append(candidate_desc)

    matched_label = ""
    normalized_type = ""
    for candidate in candidates:
        matched = _match_declared_type(candidate)
        if matched:
            normalized_type = matched
            matched_label = candidate
            break

    if not normalized_type:
        normalized_type = classify_symbol(sym)

    if not matched_label:
        matched_label = next((candidate for candidate in candidates if candidate), "")

    final_type = normalized_type or ""

    return {
        "tipo": final_type,
        "tipo_estandar": final_type,
        "tipo_iol": matched_label,
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
        if isinstance(t, dict):
            raw_tipo = t.get("tipo")
            raw_desc = t.get("descripcion")
            if raw_tipo not in (None, ""):
                tipo_original = str(raw_tipo).strip()
            if raw_desc not in (None, ""):
                descripcion_original = str(raw_desc).strip()

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
            "valorizado",
        ],
    )
    balances = _extract_cash_balances(payload)
    if balances:
        df.attrs.setdefault("cash_balances", {}).update(balances)
    return df


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
    attrs: dict[str, Any] = dict(getattr(df_pos, "attrs", {}) or {})
    cols = [
        "simbolo",
        "mercado",
        "tipo",
        "tipo_iol",
        "tipo_estandar",
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
        empty_df = pd.DataFrame(columns=cols)
        if attrs:
            empty_df.attrs.update(attrs)
        return empty_df

    # Normalización básica y exclusiones ---------------------------------
    df = df_pos.copy()
    df["simbolo"] = df["simbolo"].map(clean_symbol)
    df["mercado"] = df["mercado"].astype(str).str.lower()
    ex = {clean_symbol(s) for s in (exclude_syms or [])}

    df = df[~df["simbolo"].isin(ex)]
    if df.empty:
        empty_df = pd.DataFrame(columns=cols)
        if attrs:
            empty_df.attrs.update(attrs)
        return empty_df

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
    def _classify(row: pd.Series) -> pd.Series:
        titulo_payload: dict[str, Any] = {}
        tipo_orig = row.get("titulo_tipo_original")
        desc_orig = row.get("titulo_descripcion_original")
        if isinstance(tipo_orig, str) and tipo_orig.strip():
            titulo_payload["tipo"] = tipo_orig.strip()
        elif tipo_orig not in (None, ""):
            titulo_payload["tipo"] = str(tipo_orig)
        if isinstance(desc_orig, str) and desc_orig.strip():
            titulo_payload.setdefault("descripcion", desc_orig.strip())
        elif desc_orig not in (None, ""):
            titulo_payload.setdefault("descripcion", str(desc_orig))

        result = classify_asset({
            "simbolo": row.get("simbolo"),
            "titulo": titulo_payload,
        })

        if isinstance(result, Mapping):
            tipo_estandar = str(result.get("tipo_estandar") or result.get("tipo") or "")
            tipo_iol = str(result.get("tipo_iol") or "")
        else:
            tipo_estandar = str(result or "")
            tipo_iol = str(titulo_payload.get("tipo") or titulo_payload.get("descripcion") or "")

        if not tipo_estandar:
            try:
                tipo_estandar = classify_symbol(str(row.get("simbolo", "")))
            except Exception:  # pragma: no cover - defensive
                tipo_estandar = ""

        return pd.Series(
            {
                "tipo": tipo_estandar,
                "tipo_estandar": tipo_estandar,
                "tipo_iol": tipo_iol,
            }
        )

    classification = df.apply(_classify, axis=1)
    df = pd.concat([df, classification], axis=1)

    uniq = df.drop_duplicates("simbolo")[["simbolo", "tipo_estandar"]]
    scale_map = {s: scale_for(s, t) for s, t in uniq.itertuples(index=False)}
    df["scale"] = df["simbolo"].map(scale_map)

    # ----- Valoraciones -------------------------------------------------
    df["ultimo"] = df["last"]
    if "ultimoPrecio" in df.columns:
        fallback_price = pd.to_numeric(df["ultimoPrecio"], errors="coerce")
        df["ultimo"] = df["ultimo"].fillna(fallback_price)
    df["costo"] = df["cantidad"] * df["ppc"] * df["scale"]
    df["valor_actual"] = df["cantidad"] * df["ultimo"] * df["scale"]
    if "valorizado" in df.columns:
        fallback_valorizado = pd.to_numeric(df["valorizado"], errors="coerce")
        df["valor_actual"] = df["valor_actual"].where(df["valor_actual"].notna(), fallback_valorizado)
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
    df["mercado"] = df["mercado"].str.upper()
    result = df[cols]
    if attrs:
        result.attrs.update(attrs)
    return result


@lru_cache(maxsize=1024)
def _classify_sym_cache(sym: str) -> str:
    """Cachea la clasificación por símbolo usando la función existente classify_asset."""
    try:
        result = classify_asset({"simbolo": str(sym).upper(), "titulo": {}})
        if isinstance(result, Mapping):
            return str(result.get("tipo_estandar") or result.get("tipo") or "")
        return str(result or "")
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
