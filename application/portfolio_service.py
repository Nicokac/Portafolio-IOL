# application\portfolio_service.py
# Lógica de negocio: normalización de posiciones y cálculo de métricas P/L

from __future__ import annotations

from typing import Dict, Any, List, Iterable, Optional
import os
import re
import json
import logging
from functools import lru_cache
from shared.utils import _to_float
from shared.config import get_config

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------- Helpers y configuración ----------
def clean_symbol(s: str) -> str:
    """Normaliza el símbolo: mayúsculas, sin espacios raros, sólo chars permitidos."""
    s = str(s or "").upper().strip()
    s = s.replace("\u00A0", "").replace("\u200B", "")
    return re.sub(r"[^A-Z0-9._-]", "", s)

def map_to_us_ticker(simbolo: str) -> Optional[str]:
    s = clean_symbol(simbolo)
    cfg = get_config()
    cedear_map = cfg.get("cedear_to_us", {}) or {}

    if s in cedear_map:
        return clean_symbol(cedear_map[s])

    # ⛑️ Fallback si es una acción local
    if s in cfg.get("acciones_ar", []):
        return s + ".BA"  # yfinance usa sufijo .BA para acciones argentinas

    return None

# ---------- Clasificación y escala ----------

def classify_symbol(sym: str) -> str:
    """
    Clasifica por símbolo usando listas del config (cedears/etfs) y heurísticas.
    Devuelve una de: 'CEDEAR','ETF','Bono','Letra','FCI','Acción','Otro'
    """
    s = clean_symbol(sym)
    cfg = get_config()
    cedears_map = cfg.get("cedear_to_us", {}) or {}
    etf_set = set(map(clean_symbol, cfg.get("etfs", []) or []))
    acciones_ar = set(map(clean_symbol, cfg.get("acciones_ar", []) or []))
    fci_set = set(map(clean_symbol, cfg.get("fci_symbols", []) or []))

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
        except Exception:
            pass

    tipo_norm = (tipo or "").lower()
    if any(x in tipo_norm for x in ("bono", "letra")):
        return 0.01
    return 1.0


def classify_asset(it: dict) -> str:
    """
    Clasifica activo según 'titulo.tipo/descripcion' si existe, o heurística por símbolo.
    Devuelve una de: 'CEDEAR','ETF','Bono','Letra','FCI','Acción','Otro'
    """
    t = it.get("titulo") or {}
    tipo_txt = (t.get("tipo") or t.get("descripcion") or "").strip().lower()
    sym = clean_symbol(it.get("simbolo", ""))

    # Si IOL lo declara, priorizamos esa etiqueta:
    if "bono" in tipo_txt:
        return "Bono"
    if "letra" in tipo_txt:
        return "Letra"
    if "fci" in tipo_txt or "fondo" in tipo_txt:
        return "FCI"
    if "cedear" in tipo_txt:
        return "CEDEAR"
    if "etf" in tipo_txt:
        return "ETF"
    if "acción" in tipo_txt or "accion" in tipo_txt:
        return "Acción"

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
            it.get("cantidad") if isinstance(it, dict) else None
        ) or (it.get("cant") if isinstance(it, dict) else None) \
          or (it.get("cantidadDisponible") if isinstance(it, dict) else None) \
          or (it.get("cantidadNominal") if isinstance(it, dict) else None) \
          or (it.get("tenencia") if isinstance(it, dict) else None) \
          or 0

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
        # return pd.DataFrame(columns=[
        #     "simbolo", "mercado", "tipo", "cantidad", "ppc",
        #     "ultimo", "valor_actual", "costo", "pl", "pl_%",
        #     "pl_d", "pld_%"
        # ])
        return pd.DataFrame(columns=cols)

    # Normalización básica y exclusiones ---------------------------------
    df = df_pos.copy()
    df["simbolo"] = df["simbolo"].map(clean_symbol)
    df["mercado"] = df["mercado"].astype(str).str.lower()
    ex = {clean_symbol(s) for s in (exclude_syms or [])}
    # rows: List[Dict[str, Any]] = []

    df = df[~df["simbolo"].isin(ex)]
    if df.empty:
        return pd.DataFrame(columns=cols)

    # for _, p in df_pos.iterrows():
    #     simbolo = clean_symbol(p["simbolo"])
    #     if simbolo in ex:
    #         continue
    df["cantidad"] = df["cantidad"].map(_to_float).fillna(0.0)
    df["ppc"] = df.get("costo_unitario", np.nan).map(_to_float).fillna(0.0)

        # mercado = str(p["mercado"]).lower()
        # cantidad = _to_float(p["cantidad"]) or 0.0
        # ppc = _to_float(p.get("costo_unitario")) or 0.0  # precio prom. compra por unidad

    # ----- Cotizaciones -------------------------------------------------
    uniq = df[["mercado", "simbolo"]].drop_duplicates().reset_index(drop=True)

        # ----- cotización (acepta float o dict) -----
    def _fetch(row: pd.Series) -> pd.Series:
        last, chg_pct = None, None
        q = None
        try:
            # q = get_quote_fn(mercado, simbolo)
            q = get_quote_fn(row["mercado"], row["simbolo"])
        except Exception as e:
            # logger.debug("get_quote_fn lanzó excepción para %s:%s -> %s", mercado, simbolo, e)
            logger.debug(
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
        else:
            last = _to_float(q)

        # # Tipo y escala
        # tipo = classify_symbol(simbolo)
        # scale = scale_for(simbolo, tipo)

        # # Valoraciones (acumulado)
        # costo = cantidad * ppc * scale
        # valor = (cantidad * last * scale) if last is not None else np.nan
        # pl = (valor - costo) if not np.isnan(valor) else np.nan
        # pl_pct = (pl / costo * 100.0) if costo else np.nan

        # # ----- P/L diaria -----
        # # Si 'valor' es el valor ACTUAL y 'chg_pct' es el % vs cierre previo,
        # # el P/L diario en $ es: V_actual - V_previo = V_actual * (p/100) / (1 + p/100)
        # if (chg_pct is not None) and (valor == valor):  # no-NaN
        #     denom = 1.0 + (float(chg_pct) / 100.0)
        #     if denom != 0:
        #         pl_d = float(valor) * (float(chg_pct) / 100.0) / denom
        #         pld_pct = float(chg_pct)
        #     else:
        #         pl_d = np.nan
        #         pld_pct = np.nan
        # else:
        #     pl_d = np.nan
        #     pld_pct = np.nan

        # rows.append(
        #     {
        #         "simbolo": simbolo,
        #         "mercado": mercado.upper(),
        #         "tipo": tipo,
        #         "cantidad": cantidad,
        #         "ppc": ppc,
        #         "ultimo": last,
        #         "valor_actual": valor,
        #         "costo": costo,
        #         "pl": pl,
        #         "pl_%": pl_pct,
        #         "pl_d": pl_d,
        #         "pld_%": pld_pct,
        #     }
        # )
        return pd.Series({"last": last, "chg_pct": chg_pct})

    quotes_df = uniq.apply(_fetch, axis=1)
    quotes_df = pd.concat([uniq, quotes_df], axis=1)

    df = df.merge(quotes_df, on=["mercado", "simbolo"], how="left")

    # ----- Clasificación y escala --------------------------------------
    df["tipo"] = df["simbolo"].map(classify_symbol)
    df["scale"] = df.apply(lambda r: scale_for(r["simbolo"], r["tipo"]), axis=1)

    # ----- Valoraciones -------------------------------------------------
    df["ultimo"] = df["last"]
    df["costo"] = df["cantidad"] * df["ppc"] * df["scale"]
    df["valor_actual"] = df["cantidad"] * df["ultimo"] * df["scale"]
    df["pl"] = df["valor_actual"] - df["costo"]
    df["pl_%"] = np.where(df["costo"] != 0, df["pl"] / df["costo"] * 100.0, np.nan)

    # ----- P/L diario ---------------------------------------------------
    pct = df["chg_pct"].astype(float) / 100.0
    denom = 1.0 + pct
    df["pl_d"] = np.where(
        df["valor_actual"].notna() & df["chg_pct"].notna() & (denom != 0),
        df["valor_actual"] * pct / denom,
        np.nan,
    )
    df["pld_%"] = df["chg_pct"]

        # return pd.DataFrame(rows)
    # Orden final --------------------------------------------------------
    df["mercado"] = df["mercado"].str.upper()
    return df[cols]

# --- Agregar al final de application/portfolio_service.py ---
from functools import lru_cache

@lru_cache(maxsize=1024)
def _classify_sym_cache(sym: str) -> str:
    """Cachea la clasificación por símbolo usando la función existente classify_asset."""
    try:
        return classify_asset({"simbolo": str(sym).upper(), "titulo": {}}) or ""
    except Exception:
        return ""

class PortfolioService:
    """Fachada del portafolio que envuelve tus funciones ya existentes."""
    def normalize_positions(self, payload: dict | list) -> pd.DataFrame:
        return normalize_positions(payload)

    def calc_rows(self, price_fn, df_pos, exclude_syms=None):
        return calc_rows(price_fn, df_pos, exclude_syms or [])

    def classify_asset_cached(self, sym: str) -> str:
        return _classify_sym_cache(sym)

    # Accesos directos por si los necesitas
    def classify_asset(self, row):
        return classify_asset(row)

    def clean_symbol(self, sym: str) -> str:
        # return clean_symbol(sym)
        return clean_symbol(sym)