from __future__ import annotations

import logging
import math
from collections.abc import MutableMapping

import numpy as np
import pandas as pd
import streamlit as st

from application.portfolio_service import calculate_totals, detect_currency
from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from application.portfolio_service import calculate_totals, detect_currency, PortfolioTotals
from shared.utils import (
    _as_float_or_none,
    _is_none_nan_inf,
    format_money,
)
from ui.lazy.runtime import (
    emit_fragment_ready,
    ensure_fragment_ready_script,
    mark_fragment_ready,
    register_fragment_ready,
)

_SEARCH_WIDGET_KEY = "portfolio_table_search_input"
_SEARCH_DATASET_STATE_KEY = "__portfolio_table_search_dataset__"
_DATASET_HASH_STATE_KEY = "dataset_hash"
from .palette import get_active_palette
from .export import download_csv

def render_totals(
    df_view: pd.DataFrame,
    ccl_rate: float | None = None,
    totals: PortfolioTotals | None = None,
):
    totals = totals or calculate_totals(df_view)
    total_val = totals.total_value
    total_cost = totals.total_cost
    total_pl = totals.total_pl
    total_pl_pct = totals.total_pl_pct
    total_cash = totals.total_cash

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Valorizado", format_money(total_val))
    c2.metric("Cash", format_money(total_cash))
    c3.metric("Costo", format_money(total_cost))
    c4.metric(
        "P/L",
        format_money(total_pl),
        delta=None if not np.isfinite(total_pl_pct) else f"{total_pl_pct:.2f}%",
    )
    c5.metric(
        "P/L %",
        "—" if not np.isfinite(total_pl_pct) else f"{total_pl_pct:.2f}%",
    )

    if _as_float_or_none(ccl_rate):
        rate = float(ccl_rate)
        usd_val = total_val / rate
        usd_cost = total_cost / rate
        usd_pl = total_pl / rate
        c1b, c2b, c3b, c4b = st.columns(4)
        c1b.metric("Valorizado (USD CCL)", format_money(usd_val, currency="USD"))
        c2b.metric("Costo (USD CCL)", format_money(usd_cost, currency="USD"))
        c3b.metric("P/L (USD CCL)", format_money(usd_pl, currency="USD"))
        c4b.metric("CCL usado", format_money(rate))

def render_table(
    df_view: pd.DataFrame,
    order_by: str,
    desc: bool,
    ccl_rate: float | None = None,
    show_usd: bool = False,
    *,
    favorites: FavoriteSymbols | None = None,
):
    ensure_fragment_ready_script("portfolio_table")

    def _register_visibility(visible: bool) -> None:
        try:
            dataset_hash = st.session_state.get("dataset_hash")
        except Exception:  # pragma: no cover - session state may be missing in tests
            dataset_hash = None
        register_fragment_ready(
            "portfolio_table",
            dataset_hash=str(dataset_hash or "") or None,
            visible=visible,
        )

    if df_view is None or df_view.empty:
        st.info("Sin datos para mostrar.")
        _register_visibility(False)
        return

    favorites = favorites or get_persistent_favorites()
    palette = get_active_palette()

    cols_order = ["mercado", "simbolo", "tipo", "cantidad", "ultimo", "valor_actual", "costo", "pl", "pl_%", "pl_d", "chg_%"]
    for c in cols_order:
        if c not in df_view.columns:
            df_view[c] = np.nan

    try:
        df_sorted = df_view.sort_values(order_by, ascending=not desc).copy()
    except Exception as e:
        logging.getLogger(__name__).warning("Ordenamiento falló: %s", e)
        df_sorted = df_view.copy()

    dataset_token = ""
    try:  # pragma: no cover - defensive when session state not available
        state = getattr(st, "session_state", None)
    except Exception:
        state = None
    if isinstance(state, MutableMapping):
        dataset_token = str(state.get(_DATASET_HASH_STATE_KEY) or "")
        previous_dataset = state.get(_SEARCH_DATASET_STATE_KEY)
        if previous_dataset != dataset_token:
            try:
                state.pop(_SEARCH_WIDGET_KEY, None)
            except Exception:  # pragma: no cover - defensive when state immutable
                pass
            try:
                state[_SEARCH_DATASET_STATE_KEY] = dataset_token
            except Exception:  # pragma: no cover - defensive when state immutable
                pass

    search = st.text_input("Buscar", "", key=_SEARCH_WIDGET_KEY).strip().lower()
    if search:
        mask = (
            df_sorted["simbolo"].astype(str).str.lower().str.contains(search)
            | df_sorted["tipo"].astype(str).str.lower().str.contains(search)
        )
        df_sorted = df_sorted[mask]

    if df_sorted.empty:
        st.info("Sin datos para mostrar.")
        _register_visibility(False)
        return

    quotes_hist: dict = st.session_state.get("quotes_hist", {})
    SPARK_N = 30

    fmt_rows = []
    chg_list: list[float | None] = []
    all_spark_vals: list[float] = []

    for _, r in df_sorted.iterrows():
        sym = str(r["simbolo"])
        tipo = str(r.get("tipo") or "")
        cur = detect_currency(sym, tipo)

        is_favorite = favorites.is_favorite(sym)

        row = {
            "Símbolo": sym,
            "Tipo": tipo,
            "Favorito": "⭐" if is_favorite else "",
            "es_favorito": is_favorite,
            "cantidad_num": _as_float_or_none(r["cantidad"]),
            "ultimo_num": _as_float_or_none(r["ultimo"]),
            "valor_actual_num": _as_float_or_none(r["valor_actual"]),
            "costo_num": _as_float_or_none(r["costo"]),
        }

        pl_val = r.get("pl")
        pl_pct_val = r.get("pl_%")
        row["pl_num"] = _as_float_or_none(pl_val)
        row["pl_pct_num"] = _as_float_or_none(pl_pct_val)

        pl_d_val = r.get("pl_d")
        chg_pct = r.get("chg_%")
        row["pl_d_num"] = _as_float_or_none(pl_d_val)
        row["chg_pct_num"] = _as_float_or_none(chg_pct)
        chg_list.append(_as_float_or_none(chg_pct))

        hist = quotes_hist.get(sym.upper(), [])
        vals = [
            _as_float_or_none(h.get("chg_pct"))
            for h in hist[-SPARK_N:] if _as_float_or_none(h.get("chg_pct")) is not None
        ]
        row["Intradía %"] = vals if len(vals) >= 2 else None
        all_spark_vals.extend(vals)

        if show_usd and _as_float_or_none(ccl_rate):
            rate = float(ccl_rate)
            row["val_usd_num"] = (
                float(r["valor_actual"]) / rate if not _is_none_nan_inf(r["valor_actual"]) else None
            )
            row["costo_usd_num"] = (
                float(r["costo"]) / rate if not _is_none_nan_inf(r["costo"]) else None
            )
            row["pl_usd_num"] = (
                float(r["pl"]) / rate if not _is_none_nan_inf(r["pl"]) else None
            )

        fmt_rows.append(row)

    df_tbl = pd.DataFrame(fmt_rows)

    def _color_pl(col: pd.Series):
        styles = []
        for v in col:
            val = _as_float_or_none(v)
            if val is None or not np.isfinite(val):
                styles.append("")
            elif val < 0:
                styles.append(f"color: {palette.negative}; font-weight: 600;")
            else:
                styles.append(f"color: {palette.positive}; font-weight: 600;")
        return styles

    if all_spark_vals:
        span = max(abs(min(all_spark_vals)), abs(max(all_spark_vals)))
        span = max(2.0, min(span, 30.0))
        y_min, y_max = -span, span
    else:
        y_min, y_max = -10.0, 10.0

    rename_map = {
        "cantidad_num": "Cantidad",
        "ultimo_num": "Último precio",
        "valor_actual_num": "Valorizado",
        "costo_num": "Costo",
        "pl_num": "P/L Acum Valor",
        "pl_pct_num": "P/L Acum %",
        "pl_d_num": "P/L diario Valor",
        "chg_pct_num": "P/L diario %",
        "val_usd_num": "Val. (USD CCL)",
        "costo_usd_num": "Costo (USD CCL)",
        "pl_usd_num": "P/L (USD CCL)",
    }

    column_help = {
        "Símbolo": "Ticker del activo",
        "Tipo": "Clasificación del instrumento",
        "Cantidad": "Cantidad de títulos en cartera",
        "Último precio": "Última cotización disponible",
        "Valorizado": "Valor actual (cantidad * último precio)",
        "Costo": "Costo total de adquisición",
        "P/L Acum Valor": "Ganancia/Pérdida desde la compra",
        "P/L Acum %": "Ganancia/Pérdida porcentual desde la compra",
        "P/L diario Valor": "Ganancia/Pérdida de la rueda actual",
        "P/L diario %": "Variación porcentual de la rueda actual",
        "Val. (USD CCL)": "Valorizado en USD usando CCL",
        "Costo (USD CCL)": "Costo en USD usando CCL",
        "P/L (USD CCL)": "Ganancia/Pérdida en USD usando CCL",
    }

    column_format = {
        "Cantidad": "%d",
        "Último precio": "$%.2f",
        "Valorizado": "$%.2f",
        "Costo": "$%.2f",
        "P/L Acum Valor": "$%.2f",
        "P/L Acum %": "%.2f%%",
        "P/L diario Valor": "$%.2f",
        "P/L diario %": "%.2f%%",
        "Val. (USD CCL)": "$%.2f",
        "Costo (USD CCL)": "$%.2f",
        "P/L (USD CCL)": "$%.2f",
    }

    column_config: dict[str, st.column_config.Column] = {}
    for col in ["Símbolo", "Tipo", "Favorito"]:
        if col in df_tbl.columns:
            help_text = column_help.get(col, "Indica si el símbolo está marcado como favorito.")
            column_config[col] = st.column_config.Column(
                label="⭐" if col == "Favorito" else col,
                help=help_text,
                width="small" if col == "Favorito" else None,
            )

    for col, label in rename_map.items():
        if col in df_tbl.columns:
            column_config[col] = st.column_config.NumberColumn(
                label=label,
                help=column_help[label],
                format=column_format.get(label, "%.2f"),
            )

    has_intraday_history = bool(all_spark_vals)
    if has_intraday_history:
        column_config["Intradía %"] = st.column_config.LineChartColumn(
            label="Intradía %",
            width="small",
            y_min=y_min,
            y_max=y_max,
            help="Variación diaria (%) intradía — últimos puntos",
        )
    else:
        df_tbl.drop(columns=["Intradía %"], inplace=True, errors="ignore")

    st.subheader("Detalle por símbolo")
    df_export = df_tbl.rename(columns=rename_map)
    if "es_favorito" in df_tbl.columns:
        df_export["Favorito"] = df_tbl["es_favorito"].astype(bool)
        df_export.drop(columns=["es_favorito"], inplace=True, errors="ignore")
    download_csv(df_export, "portafolio.csv")

    page_size = st.number_input("Filas por página", min_value=5, max_value=100, value=20, step=5)
    total_pages = max(1, math.ceil(len(df_tbl) / page_size))
    page = st.number_input("Página", min_value=1, max_value=total_pages, value=1, step=1)
    start = (page - 1) * page_size
    end = start + page_size

    df_page = df_tbl.iloc[start:end]

    st.dataframe(
        df_page.style.apply(
            _color_pl,
            subset=["pl_num", "pl_d_num", "pl_pct_num", "chg_pct_num"],
        ),
        width="stretch",
        hide_index=True,
        height=420,
        column_config=column_config,
    )
    mark_fragment_ready("portfolio_table", source="backend_optimistic")
    emit_fragment_ready("portfolio_table")

    st.caption("Tabla con todas tus posiciones actuales. Te ayuda a ver cuánto tenés en cada activo y cómo viene rindiendo.")

    drop_cols = list(rename_map.keys()) + ["es_favorito"]
    df_tbl.drop(columns=drop_cols, inplace=True, errors="ignore")

    _register_visibility(True)
