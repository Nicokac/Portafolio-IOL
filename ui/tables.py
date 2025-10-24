from __future__ import annotations

import logging
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
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
from ui.utils.formatters import format_asset_type

from .export import download_csv
from .palette import get_active_palette

_TOTALS_LABEL = "Totales (suma de activos)"
_TOTALS_MARKER = "__is_totals_row__"

_SEARCH_WIDGET_KEY = "portfolio_table_search_input"
_SEARCH_DATASET_STATE_KEY = "__portfolio_table_search_dataset__"
_DATASET_HASH_STATE_KEY = "dataset_hash"


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

    cols_order = [
        "mercado",
        "simbolo",
        "tipo",
        "cantidad",
        "ultimo",
        "valor_actual",
        "costo",
        "pl",
        "pl_%",
        "pl_d",
        "chg_%",
    ]
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
        mask = df_sorted["simbolo"].astype(str).str.lower().str.contains(search) | df_sorted["tipo"].astype(
            str
        ).str.lower().str.contains(search)
        df_sorted = df_sorted[mask]

    def _weighted_average(values: pd.Series, weights: pd.Series) -> float:
        if values.empty or weights.empty:
            return float("nan")
        values_array = pd.to_numeric(values, errors="coerce").to_numpy()
        weights_array = pd.to_numeric(weights, errors="coerce").to_numpy()
        if values_array.size == 0 or weights_array.size == 0:
            return float("nan")
        mask = np.isfinite(values_array) & np.isfinite(weights_array)
        if not mask.any():
            return float("nan")
        masked_weights = weights_array[mask]
        if not masked_weights.size or np.isclose(masked_weights.sum(), 0.0):
            return float("nan")
        return float(np.average(values_array[mask], weights=masked_weights))

    def _build_totals_row(source_df: pd.DataFrame) -> dict[str, Any]:
        def numeric(name: str) -> pd.Series:
            if name in source_df.columns:
                return pd.to_numeric(source_df[name], errors="coerce")
            return pd.Series(index=source_df.index, dtype=float)

        valor_actual_series = numeric("valor_actual")
        costo_series = numeric("costo")
        pl_series = numeric("pl")
        pl_d_series = numeric("pl_d")

        valor_total = float(np.nansum(valor_actual_series.to_numpy()))
        costo_total = float(np.nansum(costo_series.to_numpy()))
        if pl_series.notna().any():
            pl_total = float(np.nansum(pl_series.to_numpy()))
        else:
            pl_total = valor_total - costo_total

        if pl_d_series.notna().any():
            pl_d_total = float(np.nansum(pl_d_series.to_numpy()))
        else:
            pl_d_total = float("nan")

        if np.isfinite(costo_total) and not np.isclose(costo_total, 0.0):
            pl_pct_total = (pl_total / costo_total) * 100.0
        else:
            pl_pct_total = float("nan")

        variation_col = "variacion_diaria" if "variacion_diaria" in source_df.columns else "chg_%"
        variation_series = numeric(variation_col)
        weighted_variation = _weighted_average(variation_series, valor_actual_series)

        totals_dict: dict[str, Any] = {
            "simbolo": _TOTALS_LABEL,
            "tipo": "",
            "mercado": "",
            "cantidad": np.nan,
            "ultimo": np.nan,
            "valor_actual": valor_total,
            "costo": costo_total,
            "pl": pl_total,
            "pl_%": pl_pct_total,
            "pl_d": pl_d_total,
            "chg_%": weighted_variation,
        }

        if variation_col == "variacion_diaria":
            totals_dict["variacion_diaria"] = weighted_variation

        return totals_dict

    totals_row = _build_totals_row(df_view)
    if totals_row:
        df_sorted = df_sorted.copy()
        df_sorted[_TOTALS_MARKER] = False
        totals_row[_TOTALS_MARKER] = True
        df_sorted = pd.concat([df_sorted, pd.DataFrame([totals_row])], ignore_index=True)

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
        is_totals_row = bool(r.get(_TOTALS_MARKER))
        sym = str(r["simbolo"]) if not is_totals_row else _TOTALS_LABEL
        is_favorite = False if is_totals_row else favorites.is_favorite(sym)

        row = {
            "Símbolo": sym,
            "Tipo": "" if is_totals_row else format_asset_type(r.get("tipo")),
            "Favorito": "" if is_totals_row else ("⭐" if is_favorite else ""),
            "es_favorito": is_favorite,
            "is_totals_row": is_totals_row,
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
        if not is_totals_row:
            chg_list.append(_as_float_or_none(chg_pct))

        if not is_totals_row:
            hist = quotes_hist.get(sym.upper(), [])
            vals = [
                _as_float_or_none(h.get("chg_pct"))
                for h in hist[-SPARK_N:]
                if _as_float_or_none(h.get("chg_pct")) is not None
            ]
            row["Intradía %"] = vals if len(vals) >= 2 else None
            all_spark_vals.extend(vals)
        else:
            row["Intradía %"] = None

        if show_usd and _as_float_or_none(ccl_rate):
            rate = float(ccl_rate)
            row["val_usd_num"] = float(r["valor_actual"]) / rate if not _is_none_nan_inf(r["valor_actual"]) else None
            row["costo_usd_num"] = float(r["costo"]) / rate if not _is_none_nan_inf(r["costo"]) else None
            row["pl_usd_num"] = float(r["pl"]) / rate if not _is_none_nan_inf(r["pl"]) else None

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
    df_export = df_tbl.drop(columns=["is_totals_row"], errors="ignore").rename(columns=rename_map)
    if "es_favorito" in df_tbl.columns:
        df_export["Favorito"] = df_tbl["es_favorito"].astype(bool)
        df_export.drop(columns=["es_favorito"], inplace=True, errors="ignore")
    download_csv(df_export, "portafolio.csv")

    totals_mask = df_tbl.get("is_totals_row", pd.Series(False, index=df_tbl.index)).astype(bool)
    df_page = df_tbl.drop(columns=["is_totals_row"], errors="ignore")

    def _highlight_totals(row: pd.Series) -> list[str]:
        if totals_mask.iloc[row.name]:
            return ["background-color: #f0f2f6; font-weight: 600;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_page.style.apply(
            _color_pl,
            subset=["pl_num", "pl_d_num", "pl_pct_num", "chg_pct_num"],
        ).apply(_highlight_totals, axis=1),
        use_container_width=True,
        hide_index=True,
        height=420,
        column_config=column_config,
    )
    mark_fragment_ready("portfolio_table", source="backend_optimistic")
    emit_fragment_ready("portfolio_table")

    st.caption(
        "Tabla con todas tus posiciones actuales. Te ayuda a ver cuánto tenés en cada activo y cómo viene rindiendo."
    )

    drop_cols = list(rename_map.keys()) + ["es_favorito", "is_totals_row", _TOTALS_MARKER]
    df_tbl.drop(columns=drop_cols, inplace=True, errors="ignore")

    _register_visibility(True)
