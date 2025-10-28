from __future__ import annotations

import logging
from collections.abc import MutableMapping
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from shared.telemetry import log_metric
from shared.utils import _as_float_or_none, _is_none_nan_inf
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
    initial_limit: int | None = None,
):
    ensure_fragment_ready_script("portfolio_table")

    start_time = perf_counter()
    total_rows = int(getattr(df_view, "shape", (0, 0))[0]) if df_view is not None else 0
    display_rows = 0
    advanced_loaded = False
    effective_limit: int | None = None
    status = "ok"

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

    try:
        if df_view is None or df_view.empty:
            st.info("Sin datos para mostrar.")
            status = "no_data"
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

        df_source = df_view.copy()
        for col in cols_order:
            if col not in df_source.columns:
                df_source[col] = np.nan

        try:
            df_sorted = df_source.sort_values(order_by, ascending=not desc).copy()
        except Exception as exc:  # pragma: no cover - defensive fallback
            logging.getLogger(__name__).warning("Ordenamiento falló: %s", exc)
            df_sorted = df_source.copy()

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

        totals_row = _build_totals_row(df_source)
        if totals_row:
            df_sorted = df_sorted.copy()
            df_sorted[_TOTALS_MARKER] = False
            totals_row[_TOTALS_MARKER] = True
            df_sorted = pd.concat([df_sorted, pd.DataFrame([totals_row])], ignore_index=True)

        if df_sorted.empty:
            st.info("Sin datos para mostrar.")
            status = "no_data"
            _register_visibility(False)
            return

        def _build_rows(source_df: pd.DataFrame, *, include_advanced: bool) -> tuple[pd.DataFrame, list[float]]:
            rows: list[dict[str, Any]] = []
            spark_values: list[float] = []
            quotes_hist: dict[str, Any] = st.session_state.get("quotes_hist", {}) if include_advanced else {}
            spark_len = 30
            rate: float | None = None
            if include_advanced and show_usd and _as_float_or_none(ccl_rate):
                rate = float(ccl_rate)  # type: ignore[arg-type]

            for _, row in source_df.iterrows():
                is_totals_row = bool(row.get(_TOTALS_MARKER))
                symbol = str(row.get("simbolo", "")) if not is_totals_row else _TOTALS_LABEL
                favorite_flag = False if is_totals_row else favorites.is_favorite(symbol)
                base_row: dict[str, Any] = {
                    "Símbolo": symbol,
                    "Tipo": "" if is_totals_row else format_asset_type(row.get("tipo")),
                    "Favorito": "" if is_totals_row else ("⭐" if favorite_flag else ""),
                    "es_favorito": favorite_flag,
                    "is_totals_row": is_totals_row,
                    "cantidad_num": _as_float_or_none(row.get("cantidad")),
                    "valor_actual_num": _as_float_or_none(row.get("valor_actual")),
                    "pl_num": _as_float_or_none(row.get("pl")),
                }

                if include_advanced:
                    base_row["ultimo_num"] = _as_float_or_none(row.get("ultimo"))
                    base_row["costo_num"] = _as_float_or_none(row.get("costo"))
                    base_row["pl_pct_num"] = _as_float_or_none(row.get("pl_%"))
                    base_row["pl_d_num"] = _as_float_or_none(row.get("pl_d"))
                    chg_value = _as_float_or_none(row.get("chg_%"))
                    if chg_value is None:
                        chg_value = _as_float_or_none(row.get("pld_%"))
                    base_row["chg_pct_num"] = chg_value

                    if rate and not is_totals_row:
                        valor_actual = row.get("valor_actual")
                        costo_val = row.get("costo")
                        pl_val = row.get("pl")
                        base_row["val_usd_num"] = (
                            float(valor_actual) / rate if not _is_none_nan_inf(valor_actual) else None
                        )
                        base_row["costo_usd_num"] = (
                            float(costo_val) / rate if not _is_none_nan_inf(costo_val) else None
                        )
                        base_row["pl_usd_num"] = float(pl_val) / rate if not _is_none_nan_inf(pl_val) else None

                    if not is_totals_row:
                        history = quotes_hist.get(symbol.upper(), [])
                        values = [
                            _as_float_or_none(item.get("chg_pct"))
                            for item in history[-spark_len:]
                            if _as_float_or_none(item.get("chg_pct")) is not None
                        ]
                        base_row["Intradía %"] = values if len(values) >= 2 else None
                        spark_values.extend(values)
                    else:
                        base_row["Intradía %"] = None

                rows.append(base_row)

            return pd.DataFrame(rows), spark_values

        base_df, _ = _build_rows(df_sorted, include_advanced=False)

        def _color_pl(col: pd.Series) -> list[str]:
            styles: list[str] = []
            for value in col:
                numeric_value = _as_float_or_none(value)
                if numeric_value is None or not np.isfinite(numeric_value):
                    styles.append("")
                elif numeric_value < 0:
                    styles.append(f"color: {palette.negative}; font-weight: 600;")
                else:
                    styles.append(f"color: {palette.positive}; font-weight: 600;")
            return styles

        def _highlight_totals(mask: pd.Series):
            def _apply(row: pd.Series) -> list[str]:
                if mask.iloc[row.name]:
                    return ["background-color: #f0f2f6; font-weight: 600;"] * len(row)
                return [""] * len(row)

            return _apply

        try:
            limit = int(initial_limit) if initial_limit is not None else 20
        except (TypeError, ValueError):
            limit = 20
        limit = max(limit, 1)
        effective_limit = limit

        toggle_key = f"portfolio_table_show_all_{dataset_token}" if dataset_token else "portfolio_table_show_all"
        show_all_rows = st.toggle(
            "Mostrar todas las posiciones",
            key=toggle_key,
            help="Activalo para listar todo el portafolio. La vista inicial muestra las posiciones más relevantes.",
        )

        base_totals_mask = base_df.get("is_totals_row", pd.Series(False, index=base_df.index)).astype(bool)
        if show_all_rows:
            limited_df = base_df.copy()
        else:
            limited_df = pd.concat(
                [
                    base_df.loc[~base_totals_mask].head(limit),
                    base_df.loc[base_totals_mask],
                ],
                ignore_index=True,
            )

        display_rows = len(limited_df)
        base_page = limited_df.drop(columns=["is_totals_row", "es_favorito"], errors="ignore")
        totals_mask = limited_df.get("is_totals_row", pd.Series(False, index=limited_df.index)).astype(bool)

        rename_map_basic = {
            "cantidad_num": "Cantidad",
            "valor_actual_num": "Valorizado",
            "pl_num": "P/L Acum Valor",
        }
        column_help = {
            "Símbolo": "Ticker del activo",
            "Tipo": "Clasificación del instrumento",
            "Cantidad": "Cantidad de títulos en cartera",
            "Valorizado": "Valor actual (cantidad * último precio)",
            "P/L Acum Valor": "Ganancia/Pérdida desde la compra",
        }
        column_format_basic = {
            "Cantidad": "%d",
            "Valorizado": "$%.2f",
            "P/L Acum Valor": "$%.2f",
        }

        column_config_basic: dict[str, st.column_config.Column] = {}
        for column in ["Símbolo", "Tipo", "Favorito"]:
            if column in base_page.columns:
                help_text = column_help.get(column, "Indica si el símbolo está marcado como favorito.")
                column_config_basic[column] = st.column_config.Column(
                    label="⭐" if column == "Favorito" else column,
                    help=help_text,
                    width="small" if column == "Favorito" else None,
                )

        for raw, label in rename_map_basic.items():
            if raw in base_page.columns:
                column_config_basic[raw] = st.column_config.NumberColumn(
                    label=label,
                    help=column_help[label],
                    format=column_format_basic.get(label, "%.2f"),
                )

        st.subheader("Detalle por símbolo")
        st.dataframe(
            base_page.style.apply(_color_pl, subset=["pl_num"]).apply(_highlight_totals(totals_mask), axis=1),
            use_container_width=True,
            hide_index=True,
            height=360,
            column_config=column_config_basic,
        )

        mark_fragment_ready("portfolio_table", source="backend_optimistic")
        emit_fragment_ready("portfolio_table")

        st.caption(
            "Tabla con todas tus posiciones actuales. Te ayuda a ver cuánto tenés en cada activo y cómo viene rindiendo."
        )

        advanced_toggle_key = (
            f"portfolio_table_show_advanced_{dataset_token}" if dataset_token else "portfolio_table_show_advanced"
        )
        show_advanced = st.checkbox(
            "Cargar métricas avanzadas",
            key=advanced_toggle_key,
            help="Incluye KPIs porcentuales, variaciones diarias, historial intradía y conversión a USD cuando lo necesites.",
        )

        if show_advanced:
            advanced_loaded = True
            with st.expander("Métricas avanzadas y exportaciones", expanded=True):
                advanced_df, spark_values = _build_rows(df_sorted, include_advanced=True)
                advanced_mask = advanced_df.get("is_totals_row", pd.Series(False, index=advanced_df.index)).astype(bool)
                advanced_page = advanced_df.drop(columns=["is_totals_row", "es_favorito"], errors="ignore")

                rename_map_full = {
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
                column_help_full = {
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
                column_format_full = {
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

                column_config_adv: dict[str, st.column_config.Column] = {}
                for column in ["Símbolo", "Tipo", "Favorito"]:
                    if column in advanced_page.columns:
                        help_text = column_help_full.get(column, "Indica si el símbolo está marcado como favorito.")
                        column_config_adv[column] = st.column_config.Column(
                            label="⭐" if column == "Favorito" else column,
                            help=help_text,
                            width="small" if column == "Favorito" else None,
                        )

                for raw, label in rename_map_full.items():
                    if raw in advanced_page.columns:
                        column_config_adv[raw] = st.column_config.NumberColumn(
                            label=label,
                            help=column_help_full[label],
                            format=column_format_full.get(label, "%.2f"),
                        )

                if spark_values:
                    span = max(abs(min(spark_values)), abs(max(spark_values)))
                    span = max(2.0, min(span, 30.0))
                    y_min, y_max = -span, span
                    column_config_adv["Intradía %"] = st.column_config.LineChartColumn(
                        label="Intradía %",
                        width="small",
                        y_min=y_min,
                        y_max=y_max,
                        help="Variación diaria (%) intradía — últimos puntos",
                    )
                else:
                    advanced_page.drop(columns=["Intradía %"], inplace=True, errors="ignore")

                st.dataframe(
                    advanced_page.style.apply(
                        _color_pl,
                        subset=[col for col in ["pl_num", "pl_d_num", "pl_pct_num", "chg_pct_num"] if col in advanced_page.columns],
                    ).apply(_highlight_totals(advanced_mask), axis=1),
                    use_container_width=True,
                    hide_index=True,
                    height=420,
                    column_config=column_config_adv,
                )

                export_df = advanced_page.rename(columns=rename_map_full)
                if "es_favorito" in advanced_df.columns:
                    export_df["Favorito"] = advanced_df["es_favorito"].astype(bool)
                download_csv(export_df, "portafolio.csv")

        _register_visibility(True)
    finally:
        duration_ms = (perf_counter() - start_time) * 1000.0
        context = {
            "rows": total_rows,
            "display_rows": display_rows,
            "show_usd": bool(show_usd),
            "initial_limit": effective_limit,
            "advanced_loaded": advanced_loaded,
        }
        log_metric(
            "portfolio.table.mount_ms",
            context,
            status=status,
            duration_ms=duration_ms,
        )
