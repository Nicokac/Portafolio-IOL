"""Render helpers for the risk analysis tab in the portfolio UI.

Parte de la capa controllers. No ejecutar código en import.
"""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from io import BytesIO, StringIO
from typing import Any, Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from application.benchmark_service import benchmark_analysis
from application.risk_service import (
    annualized_volatility,
    asset_risk_breakdown,
    beta,
    compute_returns,
    drawdown_series,
    expected_shortfall,
    historical_var,
    markowitz_optimize,
    max_drawdown,
    monte_carlo_simulation,
    rolling_correlations,
)
from services.cache.market_data_cache import get_market_data_cache
from services.health import record_tab_latency
from services.notifications import NotificationFlags
from services.performance_timer import ProfileBlockResult, profile_block
from shared.errors import AppError
from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.charts import _apply_layout, plot_correlation_heatmap, plot_factor_betas
from ui.export import PLOTLY_CONFIG
from ui.favorites import render_favorite_badges, render_favorite_toggle
from ui.notifications import render_risk_badge
from ui.utils.formatters import format_asset_type

logger = logging.getLogger(__name__)

_MONTE_CARLO_THRESHOLD = 5000


def _extend_unique_types(target: list[str], values) -> None:
    """Append canonical types from *values* preserving order."""

    if values is None:
        return

    if isinstance(values, str):
        iterator = [values]
    else:
        try:
            iterator = list(values)
        except TypeError:
            iterator = [values]

    for value in iterator:
        label = _raw_type_label(value)
        if label and label not in target:
            target.append(label)


def _order_types(types: Sequence[str]) -> list[str]:
    """Return canonical types sorted by preferred order preserving extras."""

    ordered: list[str] = []
    seen: set[str] = set()

    for value in types:
        label = _raw_type_label(value)
        if label and label not in seen:
            ordered.append(label)
            seen.add(label)

    return ordered


def _string_or_empty(value) -> str:
    """Return a trimmed string representation or ``""`` when not meaningful."""

    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
        return "" if not text or text.lower() in {"nan", "none"} else text
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if not text or text.lower() in {"nan", "none"} else text


def _raw_type_label(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)


def _fetch_history_resilient(
    tasvc,
    symbols: Sequence[str],
    *,
    period: str,
    benchmark: str | None = None,
):
    """Fetch historical prices skipping symbols that fail individually."""

    cache = get_market_data_cache()
    request = [str(sym).strip() for sym in symbols if str(sym).strip()]
    benchmark_key = str(benchmark).strip() if benchmark else ""

    if not request and not benchmark_key:
        return pd.DataFrame(), []

    combined = list(dict.fromkeys(request + ([benchmark_key] if benchmark_key else [])))
    collected: "OrderedDict[str, pd.Series]" = OrderedDict()
    skipped: list[str] = []

    def _load(target: Sequence[str]):
        return tasvc.portfolio_history(simbolos=list(target), period=period)

    try:
        history = cache.get_history(
            combined,
            loader=lambda values=tuple(combined): _load(values),
            period=period,
            benchmark=benchmark_key,
        )
    except Exception as exc:
        logger.debug(
            "Fallo historial combinado para %s (%s): %s",
            combined,
            period,
            exc,
        )
        history = pd.DataFrame()

    if isinstance(history, pd.DataFrame) and not history.empty:
        for col in history.columns:
            series = history[col]
            if isinstance(series, pd.Series) and not series.dropna().empty:
                collected[str(col)] = series

    for sym in request:
        if sym in collected and not collected[sym].dropna().empty:
            continue
        subset = list(dict.fromkeys([sym] + ([benchmark_key] if benchmark_key else [])))
        try:
            subset_history = cache.get_history(
                subset,
                loader=lambda values=tuple(subset): _load(values),
                period=period,
                benchmark=benchmark_key,
            )
        except Exception as exc:
            logger.warning("Omitiendo %s por error en historial: %s", sym, exc)
            skipped.append(sym)
            continue
        if not isinstance(subset_history, pd.DataFrame) or sym not in subset_history.columns:
            skipped.append(sym)
            continue
        series = subset_history[sym]
        if series.dropna().empty:
            skipped.append(sym)
            continue
        collected[sym] = series

    if benchmark_key and benchmark_key not in collected:
        try:
            bench_history = cache.get_history(
                [benchmark_key],
                loader=lambda value=benchmark_key: _load([value]),
                period=period,
                benchmark=benchmark_key,
            )
        except Exception as exc:
            logger.debug("No se pudo obtener benchmark %s: %s", benchmark_key, exc)
        else:
            if (
                isinstance(bench_history, pd.DataFrame)
                and benchmark_key in bench_history.columns
                and not bench_history[benchmark_key].dropna().empty
            ):
                collected[benchmark_key] = bench_history[benchmark_key]

    if not collected:
        return pd.DataFrame(), skipped

    combined_df = pd.concat(collected.values(), axis=1)
    combined_df.columns = list(collected.keys())
    combined_df = combined_df.sort_index().ffill().dropna(how="all")
    return combined_df, skipped


def _normalize_symbol(symbol) -> str:
    text = _string_or_empty(symbol)
    return text.upper()


def _canonical_type(_symbol: str, raw_type) -> str:
    label = _raw_type_label(raw_type)
    return label or "N/D"


def _build_type_metadata(
    df: pd.DataFrame | None,
) -> tuple[pd.Series, dict[str, str], dict[str, str]]:
    """Return normalized type series, display labels and symbol→type map."""

    if df is None or df.empty or "simbolo" not in df.columns:
        empty_index = getattr(df, "index", pd.Index([])) if df is not None else pd.Index([])
        return pd.Series(pd.NA, index=empty_index, dtype="object"), {}, {}

    raw_types = df["tipo"] if "tipo" in df.columns else pd.Series(pd.NA, index=df.index)
    normalized_values: list[object] = []
    display_labels: dict[str, str] = {}
    symbol_type_map: dict[str, str] = {}

    for idx, symbol in enumerate(df["simbolo"].tolist()):
        raw_value = raw_types.iloc[idx] if idx < len(raw_types) else None
        canonical = _canonical_type(symbol, raw_value)
        normalized_values.append(canonical)
        sym_key = _normalize_symbol(symbol)
        if sym_key:
            symbol_type_map[sym_key] = canonical
        display_labels.setdefault(canonical, format_asset_type(canonical))

    series = pd.Series(normalized_values, index=df.index, dtype="object")
    return series, display_labels, symbol_type_map


def compute_risk_metrics(returns_df, bench_ret, weights, *, var_confidence: float = 0.95):
    """Compute core risk metrics for the portfolio."""
    port_ret = returns_df.mul(weights, axis=1).sum(axis=1)
    vol = annualized_volatility(port_ret)
    b = beta(port_ret, bench_ret)
    var_value = historical_var(port_ret, confidence=var_confidence)
    cvar_value = expected_shortfall(port_ret, confidence=var_confidence)
    opt_w = markowitz_optimize(returns_df)
    asset_vols, asset_drawdowns = asset_risk_breakdown(returns_df)
    port_drawdown = max_drawdown(port_ret)
    return (
        vol,
        b,
        var_value,
        cvar_value,
        opt_w,
        port_ret,
        asset_vols,
        asset_drawdowns,
        port_drawdown,
    )


def render_risk_analysis(
    df_view: pd.DataFrame,
    tasvc: Any,
    favorites: FavoriteSymbols | None = None,
    *,
    notifications: NotificationFlags | None = None,
    available_types: Sequence[str] | None = None,
) -> None:
    """Render correlation and risk analysis for the portfolio."""
    stage_profiles: dict[str, dict[str, float | None]] = {}

    def _record_stage(name: str, stage: ProfileBlockResult) -> None:
        stage_profiles[name] = {
            "ms": round(stage.duration_ms, 3),
            "cpu": None if stage.cpu_percent is None else round(stage.cpu_percent, 2),
            "mem": None if stage.ram_percent is None else round(stage.ram_percent, 2),
        }

    def _profile_stage(name: str, **extra: object):
        payload: dict[str, object] = {"stage": name}
        if extra:
            payload.update(extra)
        return profile_block(f"portfolio_risk.{name}", extra=payload)

    def _finalize(value=None):
        if stage_profiles:
            try:
                st.session_state["risk_stage_profiles"] = stage_profiles
            except Exception:
                pass
        return value

    favorites = favorites or get_persistent_favorites()
    st.subheader("Análisis de Correlación del Portafolio")
    symbols = sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()}) if not df_view.empty else []

    render_favorite_badges(
        favorites,
        empty_message="⭐ Marcá tus favoritos para seguirlos de cerca en el análisis de riesgo.",
    )
    if symbols:
        options = favorites.sort_options(symbols)
        selected_symbol = st.selectbox(
            "Gestionar favoritos",
            options=options,
            index=favorites.default_index(options),
            key="risk_favorite_select",
            format_func=favorites.format_symbol,
        )
        render_favorite_toggle(
            selected_symbol,
            favorites,
            key_prefix="risk",
            help_text="Tus favoritos se sincronizan entre portafolio, riesgo, técnico y fundamental.",
        )

    filtered_df = df_view.copy() if isinstance(df_view, pd.DataFrame) else df_view
    selected_type_filters = st.session_state.get("selected_asset_types") or []
    normalized_filters = [
        _raw_type_label(t) for t in selected_type_filters if isinstance(t, str) and _raw_type_label(t)
    ]

    normalized_series, type_display_map, symbol_type_map = _build_type_metadata(
        filtered_df if isinstance(filtered_df, pd.DataFrame) else None
    )

    if isinstance(filtered_df, pd.DataFrame):
        filtered_df["_normalized_type"] = normalized_series.reindex(filtered_df.index)
        if normalized_filters:
            filtered_df = filtered_df[filtered_df["_normalized_type"].isin(normalized_filters)].copy()

        # No se realizan sobrescrituras adicionales: se mantienen los tipos originales de IOL.
    else:
        type_display_map = {}
        symbol_type_map = {}

    if filtered_df.empty:
        st.warning("No hay datos para los tipos seleccionados.")
        st.info("Ajustá los filtros de tipo de activo para ver correlaciones y métricas de riesgo.")
        st.subheader("Análisis de Riesgo")
        st.info("No hay símbolos en el portafolio para analizar.")
        return _finalize()

    corr_period = st.selectbox(
        "Calcular correlación sobre el último período:",
        ["3mo", "6mo", "1y", "2y", "5y"],
        index=2,
    )
    portfolio_symbols: list[str] = []
    if isinstance(filtered_df, pd.DataFrame):
        for sym, canon in zip(
            filtered_df.get("simbolo", []),
            filtered_df.get("_normalized_type", []),
        ):
            sym_str = str(sym).strip()
            if not sym_str:
                continue
            sym_key = sym_str.upper()
            canon_str = symbol_type_map.get(sym_key) or _canonical_type(sym_key, canon)
            if normalized_filters and canon_str not in normalized_filters:
                continue
            portfolio_symbols.append(sym_str)
    else:
        portfolio_symbols = [
            str(sym).strip() for sym in getattr(filtered_df, "get", lambda *_: [])("simbolo", []) if str(sym).strip()
        ]
    data_warning_placeholder = st.empty()
    omitted_symbols: set[str] = set()

    if len(portfolio_symbols) >= 2:
        corr_latency: float | None = None
        with st.spinner(f"Calculando correlación ({corr_period})…"):
            start_time = time.perf_counter()
            try:
                unique_symbols = sorted(set(portfolio_symbols))
                with _profile_stage(
                    "fetch_history_corr",
                    symbols=len(unique_symbols),
                    period=corr_period,
                ) as stage_corr_fetch:
                    hist_df, skipped = _fetch_history_resilient(
                        tasvc,
                        unique_symbols,
                        period=corr_period,
                    )
                _record_stage("fetch_history_corr", stage_corr_fetch)
                if skipped:
                    omitted_symbols.update(skipped)
                    data_warning_placeholder.caption("⚠️ Datos incompletos")
                    st.warning("No pudimos descargar históricos para: " + ", ".join(sorted(skipped)))
            except AppError as err:
                corr_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", corr_latency, status="error")
                st.error(str(err))
                return _finalize()
            except Exception:
                logger.exception(
                    "Error al obtener históricos para correlación",
                )
                corr_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", corr_latency, status="error")
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return _finalize()
            corr_latency = (time.perf_counter() - start_time) * 1000.0
        record_tab_latency("riesgo", corr_latency, status="success")
        with _profile_stage("compute_returns_corr", columns=hist_df.shape[1]) as stage_returns_corr:
            returns_for_corr = compute_returns(hist_df)
        _record_stage("compute_returns_corr", stage_returns_corr)
        # Map each historical symbol to its canonical type so tabs remain aligned
        symbol_groups: dict[str, list[str]] = {}
        df_symbols_upper = pd.Series(dtype="object")
        df_types_series = pd.Series(dtype="object")
        if isinstance(filtered_df, pd.DataFrame) and "_normalized_type" in filtered_df:
            df_symbols_upper = filtered_df["simbolo"].astype(str).str.strip().str.upper()
            df_types_series = filtered_df["_normalized_type"]

        for sym in hist_df.columns:
            sym_str = str(sym).strip()
            if not sym_str:
                continue
            sym_key = sym_str.upper()
            canonical = symbol_type_map.get(sym_key)
            if not canonical and not df_symbols_upper.empty:
                matches = df_types_series.loc[df_symbols_upper == sym_key]
                if not matches.empty:
                    candidate = matches.iloc[0]
                    canonical = (
                        _raw_type_label(candidate)
                        if isinstance(candidate, str)
                        else _canonical_type(sym_key, None)
                    )
            if not canonical:
                canonical = _canonical_type(sym_key, None)
            if not canonical:
                continue
            if normalized_filters and canonical not in normalized_filters:
                continue
            symbol_groups.setdefault(canonical, []).append(sym_str)

        available_types_in_view: list[str] = []
        if isinstance(filtered_df, pd.DataFrame) and "_normalized_type" in filtered_df:
            _extend_unique_types(
                available_types_in_view,
                filtered_df["_normalized_type"].dropna().unique(),
            )
        _extend_unique_types(available_types_in_view, available_types)
        _extend_unique_types(available_types_in_view, symbol_groups.keys())

        if normalized_filters:
            ordered_types: list[str] = []
            for raw in normalized_filters:
                if raw not in ordered_types:
                    ordered_types.append(raw)
        else:
            base_types = available_types_in_view if available_types_in_view else list(symbol_groups.keys())
            ordered_types = _order_types(base_types)

        type_groups: list[tuple[str, pd.DataFrame, list[str]]] = []
        for type_name in ordered_types:
            if not isinstance(type_name, str) or not type_name:
                continue
            subset = (
                filtered_df[filtered_df["_normalized_type"] == type_name]
                if isinstance(filtered_df, pd.DataFrame)
                else pd.DataFrame()
            )
            symbols_for_type = sorted({sym for sym in symbol_groups.get(type_name, []) if sym in hist_df.columns})
            subset_df = subset.copy() if isinstance(subset, pd.DataFrame) else pd.DataFrame()
            type_groups.append((type_name, subset_df, symbols_for_type))

        if not type_groups:
            label = "Portafolio"
            fallback_df = filtered_df if isinstance(filtered_df, pd.DataFrame) else pd.DataFrame()
            fallback_symbols = [sym for sym in hist_df.columns if sym in portfolio_symbols]
            type_groups = [(label, fallback_df, fallback_symbols)]

        display_targets: list = []
        if len(type_groups) > 1:
            display_targets = st.tabs(
                [
                    type_display_map.get(
                        type_name,
                        format_asset_type(type_name),
                    )
                    for type_name, _, _ in type_groups
                ]
            )
        else:
            display_targets = [st.container()]

        for (type_name, subset_df, subset_symbols), display_host in zip(type_groups, display_targets):
            with display_host:
                display_label = type_display_map.get(
                    type_name,
                    format_asset_type(type_name),
                )
                if len(set(subset_symbols)) < 2:
                    st.warning(f"⚠️ No hay suficientes activos del tipo {display_label} para calcular correlaciones.")
                    continue

                subset_hist = hist_df[sorted(set(subset_symbols))]
                fig = plot_correlation_heatmap(
                    subset_hist,
                    title=f"Matriz de Correlación — {display_label}",
                )
                if fig:
                    st.plotly_chart(
                        fig,
                        key=f"corr_heatmap_{type_name.lower().replace(' ', '_')}",
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        """
                        Un heatmap de correlación muestra cómo se mueven los activos entre sí.
                        **Azul (cercano a 1)**: Se mueven juntos.
                        **Rojo (cercano a -1)**: Se mueven en direcciones opuestas.
                        **Blanco (cercano a 0)**: No tienen relación.
                        Una buena diversificación busca valores bajos (cercanos a 0 o negativos).
                        """
                    )
                else:
                    st.warning(
                        f"⚠️ No hay suficientes datos históricos para calcular la correlación de {display_label}."
                    )

        if returns_for_corr.shape[1] >= 2:
            window_options = {
                "1 mes (21)": 21,
                "3 meses (63)": 63,
                "6 meses (126)": 126,
            }
            selected_window_label = st.selectbox(
                "Ventana para correlaciones móviles",
                list(window_options.keys()),
                index=1,
                key="rolling_corr_window",
            )
            selected_window = window_options[selected_window_label]
            with _profile_stage("rolling_correlations", window=selected_window) as stage_rolling:
                rolling_df = rolling_correlations(returns_for_corr, selected_window)
            _record_stage("rolling_correlations", stage_rolling)
            if not rolling_df.empty:
                roll_fig = px.line(
                    rolling_df,
                    labels={
                        "index": "Fecha",
                        "value": "Correlación",
                        "variable": "Par",
                    },
                )
                roll_fig = _apply_layout(
                    roll_fig,
                    title=f"Correlaciones móviles ({selected_window} ruedas)",
                )
                st.plotly_chart(
                    roll_fig,
                    key="rolling_corr_chart",
                    config=PLOTLY_CONFIG,
                )
                latest = rolling_df.dropna(how="all").tail(1)
                if not latest.empty:
                    latest_tidy = latest.T.reset_index().rename(
                        columns={"index": "Par", latest.index[-1]: "Correlación"}
                    )
                    st.markdown(
                        latest_tidy.to_html(index=False, float_format="{:.2f}".format),
                        unsafe_allow_html=True,
                    )
            else:
                st.info("No se pudieron calcular correlaciones móviles con la ventana seleccionada.")
        else:
            st.warning(
                "No se pudieron obtener suficientes datos históricos para el período "
                f"'{corr_period}' para calcular la correlación."
            )
    else:
        st.info(
            "Necesitas al menos 2 activos en tu portafolio (después de aplicar filtros) para calcular la correlación."
        )

    if isinstance(filtered_df, pd.DataFrame) and "_normalized_type" in filtered_df:
        filtered_df = filtered_df.drop(columns="_normalized_type")

    st.subheader("Análisis de Riesgo")
    if notifications and notifications.risk_alert:
        render_risk_badge(
            help_text="Se detectaron eventos de riesgo relevantes para tus posiciones recientes.",
        )
    if portfolio_symbols:
        risk_latency: float | None = None
        with st.spinner("Descargando históricos…"):
            start_time = time.perf_counter()
            try:
                with _profile_stage("fetch_history_risk", symbols=len(portfolio_symbols)) as stage_risk_fetch:
                    prices_df, skipped = _fetch_history_resilient(
                        tasvc,
                        portfolio_symbols,
                        period="1y",
                    )
                _record_stage("fetch_history_risk", stage_risk_fetch)
                if skipped:
                    omitted_symbols.update(skipped)
                    data_warning_placeholder.caption("⚠️ Datos incompletos")
                    st.warning("Se omitieron del análisis de riesgo por falta de datos: " + ", ".join(sorted(skipped)))
            except AppError as err:
                risk_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", risk_latency, status="error")
                st.error(str(err))
                return _finalize()
            except Exception:
                logger.exception(
                    "Error al obtener históricos para análisis de riesgo",
                )
                risk_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", risk_latency, status="error")
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return _finalize()
            risk_latency = (time.perf_counter() - start_time) * 1000.0
        record_tab_latency("riesgo", risk_latency, status="success")
        if prices_df.empty:
            st.info("No se pudieron obtener datos históricos para calcular métricas de riesgo.")
        else:
            with _profile_stage("compute_returns_risk", columns=prices_df.shape[1]) as stage_returns_risk:
                returns_df = compute_returns(prices_df)
            _record_stage("compute_returns_risk", stage_returns_risk)
            weights = (
                filtered_df.set_index("simbolo")["valor_actual"].astype(float).reindex(returns_df.columns).dropna()
            )
            weights = weights / weights.sum() if not weights.empty else weights
            if weights.empty or returns_df.empty:
                st.info("No hay suficientes datos para calcular métricas de riesgo.")
            else:
                benchmark_labels = {
                    "S&P 500 (^GSPC)": "^GSPC",
                    "MERVAL": "MERVAL",
                    "Nasdaq (^IXIC)": "^IXIC",
                    "Dow Jones (^DJI)": "^DJI",
                }
                benchmark_choice = st.selectbox(
                    "Benchmark para beta y drawdown",
                    list(benchmark_labels.keys()),
                    index=0,
                    key="risk_benchmark_select",
                )
                benchmark_symbol = benchmark_labels[benchmark_choice]

                try:
                    with _profile_stage(
                        "fetch_history_benchmark", symbols=1, benchmark=benchmark_symbol
                    ) as stage_bench_fetch:
                        bench_df, skipped_bench = _fetch_history_resilient(
                            tasvc,
                            [benchmark_symbol],
                            period="1y",
                            benchmark=benchmark_symbol,
                        )
                    _record_stage("fetch_history_benchmark", stage_bench_fetch)
                    if skipped_bench:
                        omitted_symbols.update(skipped_bench)
                        data_warning_placeholder.caption("⚠️ Datos incompletos")
                        st.warning("Sin datos históricos suficientes para el benchmark seleccionado.")
                        return _finalize()
                except AppError as err:
                    st.error(str(err))
                    return _finalize()
                except Exception:
                    logger.exception(
                        "Error al obtener benchmark para análisis de riesgo",
                    )
                    st.error(
                        "No se pudieron obtener datos históricos para el benchmark seleccionado.",
                    )
                    return _finalize()

                with _profile_stage("compute_returns_benchmark", symbols=1) as stage_bench_ret:
                    bench_ret = compute_returns(bench_df).squeeze()
                _record_stage("compute_returns_benchmark", stage_bench_ret)
                if bench_ret.empty:
                    st.info("El benchmark seleccionado no tiene datos suficientes para calcular beta.")
                    return _finalize()

                factors_df = None
                factor_fetcher = getattr(tasvc, "factor_history", None)
                if callable(factor_fetcher):
                    try:
                        with _profile_stage(
                            "factor_history",
                            benchmark=benchmark_symbol,
                        ) as stage_factor:
                            try:
                                factors_df = factor_fetcher(benchmark=benchmark_symbol, period="1y")
                            except TypeError:
                                try:
                                    factors_df = factor_fetcher(period="1y")
                                except TypeError:
                                    factors_df = factor_fetcher()
                        _record_stage("factor_history", stage_factor)
                    except AppError as err:
                        st.warning(f"⚠️ No se pudieron obtener factores para el benchmark seleccionado: {err}")
                    except Exception:
                        logger.exception("Error al obtener factores para análisis de benchmark")
                        st.warning("⚠️ No se pudieron obtener factores para el análisis de benchmark.")

                confidence_options = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
                selected_conf_label = st.selectbox(
                    "Nivel de confianza para VaR/CVaR",
                    list(confidence_options.keys()),
                    index=1,
                    key="var_confidence_select",
                )
                var_confidence = confidence_options[selected_conf_label]

                with _profile_stage(
                    "compute_risk_metrics",
                    assets=len(weights),
                    confidence=var_confidence,
                ) as stage_risk_metrics:
                    (
                        vol,
                        b,
                        var_value,
                        cvar_value,
                        opt_w,
                        port_ret,
                        asset_vols,
                        asset_drawdowns,
                        port_drawdown,
                    ) = compute_risk_metrics(
                        returns_df,
                        bench_ret,
                        weights,
                        var_confidence=var_confidence,
                    )
                _record_stage("compute_risk_metrics", stage_risk_metrics)

                with _profile_stage("benchmark_analysis", benchmark=benchmark_symbol) as stage_benchmark:
                    factor_results = benchmark_analysis(port_ret, bench_ret, factors_df=factors_df)
                _record_stage("benchmark_analysis", stage_benchmark)

                c1, c2, c3, c4, c5 = st.columns(5)
                c1.metric(
                    "Volatilidad anualizada",
                    f"{vol:.2%}" if vol == vol else "N/A",
                )
                c2.metric(
                    f"Beta vs {benchmark_choice}",
                    f"{b:.2f}" if b == b else "N/A",
                )
                tail_pct = (1 - var_confidence) * 100
                c3.metric(
                    f"VaR {tail_pct:.0f}%",
                    f"{var_value:.2%}" if var_value == var_value else "N/A",
                )
                c4.metric(
                    f"CVaR {tail_pct:.0f}%",
                    f"{cvar_value:.2%}" if cvar_value == cvar_value else "N/A",
                )
                c5.metric(
                    "Drawdown máximo",
                    f"{port_drawdown:.2%}" if port_drawdown == port_drawdown else "N/A",
                )

                vol_draw_cols = st.columns(2)
                if not asset_vols.empty:
                    vol_df = (
                        asset_vols.sort_values(ascending=False)
                        .rename("Volatilidad")
                        .reset_index()
                        .rename(columns={"index": "Símbolo"})
                    )
                    fig_vol_dist = px.bar(
                        vol_df,
                        x="Símbolo",
                        y="Volatilidad",
                        color="Símbolo",
                    )
                    fig_vol_dist = _apply_layout(
                        fig_vol_dist,
                        title="Distribución de volatilidades",
                        show_legend=False,
                    )
                    vol_draw_cols[0].plotly_chart(
                        fig_vol_dist,
                        config=PLOTLY_CONFIG,
                    )
                else:
                    vol_draw_cols[0].info(
                        "No hay datos suficientes para calcular volatilidad por activo.",
                    )

                drawdown_series_port = drawdown_series(port_ret)
                if not drawdown_series_port.empty:
                    fig_drawdown = px.line(
                        drawdown_series_port,
                        labels={
                            "index": "Fecha",
                            "value": "Drawdown",
                        },
                    )
                    fig_drawdown = _apply_layout(
                        fig_drawdown,
                        title="Evolución del drawdown del portafolio",
                        show_legend=False,
                        y0_line=True,
                    )
                    vol_draw_cols[1].plotly_chart(
                        fig_drawdown,
                        config=PLOTLY_CONFIG,
                    )
                else:
                    vol_draw_cols[1].info(
                        "No hay datos suficientes para calcular drawdown del portafolio.",
                    )

                with st.expander("Volatilidad - evolución"):
                    rolling_vol = port_ret.rolling(30).std() * np.sqrt(252)
                    fig_vol = px.line(
                        rolling_vol,
                        labels={
                            "index": "Fecha",
                            "value": "Volatilidad anualizada",
                        },
                    )
                    st.plotly_chart(
                        fig_vol,
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La volatilidad refleja la variabilidad de los retornos; aquí se muestra "
                        "en una ventana móvil de 30 días."
                    )

                with st.expander("Distribución de retornos, VaR y CVaR"):
                    var_threshold = np.quantile(port_ret, 1 - var_confidence)
                    cvar_threshold = -cvar_value
                    fig_var = px.histogram(port_ret, nbins=50)
                    fig_var.add_vline(
                        x=var_threshold,
                        line_color="red",
                        annotation_text=f"VaR {tail_pct:.0f}%",
                        annotation_position="top left",
                    )
                    fig_var.add_vline(
                        x=cvar_threshold,
                        line_color="orange",
                        annotation_text=f"CVaR {tail_pct:.0f}%",
                        annotation_position="top right",
                    )
                    st.plotly_chart(
                        fig_var,
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        (
                            "Las líneas indican los niveles de pérdida esperada: VaR (rojo) y CVaR (naranja)"
                            f" para un nivel de confianza del {var_confidence:.0%}."
                        )
                    )

                with st.expander("Beta vs retorno"):
                    scatter_cols = st.columns(1)
                    if returns_df.empty or bench_ret.empty:
                        scatter_cols[0].info("No hay datos suficientes para calcular beta por activo.")
                    else:
                        avg_returns = returns_df.mean().fillna(0.0) * 252
                        betas = returns_df.apply(lambda s: beta(s, bench_ret), axis=0)
                        scatter_df = pd.DataFrame(
                            {
                                "Beta": betas,
                                "Retorno anualizado": avg_returns,
                            }
                        )
                        scatter_df["Símbolo"] = scatter_df.index
                        if favorites:
                            favorite_set = {favorites.normalize(sym) for sym in favorites.list()}
                        else:
                            favorite_set = set()
                        scatter_df["Favorito"] = scatter_df["Símbolo"].apply(lambda sym: sym in favorite_set)
                        scatter_df["Tamaño"] = np.where(
                            scatter_df["Favorito"],
                            18,
                            10,
                        )
                        scatter_df = scatter_df.replace([np.inf, -np.inf], np.nan).dropna()
                        if scatter_df.empty:
                            scatter_cols[0].info(
                                "No hay datos suficientes para graficar beta vs retorno.",
                            )
                        else:
                            fig_scatter = px.scatter(
                                scatter_df,
                                x="Beta",
                                y="Retorno anualizado",
                                size="Tamaño",
                                color="Favorito",
                                hover_name="Símbolo",
                                text="Símbolo",
                                color_discrete_map={
                                    True: "gold",
                                    False: "#636EFA",
                                },
                            )
                            fig_scatter.update_traces(
                                marker={"line": {"width": 1, "color": "#2a2a2a"}}
                            )
                            fig_scatter = _apply_layout(
                                fig_scatter,
                                title="Beta vs retorno anualizado",
                            )
                            scatter_cols[0].plotly_chart(
                                fig_scatter,
                                config=PLOTLY_CONFIG,
                            )

                with st.expander("Optimización de portafolio (Markowitz)"):
                    opt_df = pd.DataFrame({"ticker": opt_w.index, "weight": opt_w.values})
                    st.bar_chart(
                        opt_df,
                        x="ticker",
                        y="weight",
                        sort="descending",
                    )
                    st.caption(
                        "Barras con la proporción que el modelo recomienda invertir en cada activo "
                        "para equilibrar riesgo y retorno."
                    )

                with st.expander("Simulación Monte Carlo"):
                    dataset_scale = returns_df.shape[0] * returns_df.shape[1]
                    lazy_key = "risk_montecarlo_ready"
                    requires_lazy_sim = dataset_scale > _MONTE_CARLO_THRESHOLD
                    run_simulation = not requires_lazy_sim or st.session_state.get(lazy_key, False)
                    if requires_lazy_sim and not run_simulation:
                        if st.button(
                            "Generar simulación Monte Carlo",
                            key="run_monte_carlo_btn",
                        ):
                            st.session_state[lazy_key] = True
                            run_simulation = True
                        else:
                            st.info("Ejecutá la simulación bajo demanda para evitar cálculos pesados en cada recarga.")
                    if run_simulation:
                        sims = st.number_input(
                            "Nº de simulaciones",
                            min_value=100,
                            max_value=10000,
                            value=1000,
                            step=100,
                        )
                        horizon = st.number_input(
                            "Horizonte (días)",
                            min_value=30,
                            max_value=365,
                            value=252,
                            step=30,
                        )
                        with st.spinner("Generando análisis avanzado…"):
                            final_prices = monte_carlo_simulation(
                                returns_df,
                                weights,
                                n_sims=sims,
                                horizon=horizon,
                            )
                        st.line_chart(final_prices)
                        st.caption(
                            "Simula muchos escenarios posibles para estimar cómo podría variar "
                            "el valor de la cartera en el futuro."
                        )

                st.subheader("Análisis de Factores y Benchmark")

                tracking_error = factor_results.get("tracking_error") if factor_results else float("nan")
                active_return = factor_results.get("active_return") if factor_results else float("nan")
                information_ratio = factor_results.get("information_ratio") if factor_results else float("nan")

                metrics_cols = st.columns(3)
                metrics_cols[0].metric(
                    "Tracking Error",
                    f"{tracking_error:.2%}" if pd.notna(tracking_error) else "N/A",
                )
                metrics_cols[1].metric(
                    "Active Return",
                    f"{active_return:.2%}" if pd.notna(active_return) else "N/A",
                )
                metrics_cols[2].metric(
                    "Information Ratio",
                    f"{information_ratio:.2f}" if pd.notna(information_ratio) else "N/A",
                )

                betas: dict[str, float] = {}
                r_squared = float("nan")
                if factor_results:
                    betas = factor_results.get("factor_betas", {}) or {}
                    r_squared_raw = factor_results.get("r_squared", float("nan"))
                    if isinstance(r_squared_raw, (int, float, np.floating)):
                        r_squared = float(r_squared_raw)

                if betas:
                    fig_betas = plot_factor_betas(betas, r_squared)
                    st.plotly_chart(fig_betas, config=PLOTLY_CONFIG)
                    if pd.notna(r_squared):
                        st.caption(f"R² del modelo: {r_squared:.2%}")
                else:
                    if factors_df is None or (isinstance(factors_df, pd.DataFrame) and factors_df.empty):
                        st.info("No hay factores disponibles para estimar betas contra el benchmark seleccionado.")
                    else:
                        st.info(
                            "No se pudieron estimar betas estadísticamente significativas con los factores provistos."
                        )

                metrics_export = [
                    {"metric": "Tracking Error", "value": tracking_error},
                    {"metric": "Active Return", "value": active_return},
                    {"metric": "Information Ratio", "value": information_ratio},
                ]
                if pd.notna(r_squared):
                    metrics_export.append({"metric": "R_squared", "value": r_squared})

                metrics_df = pd.DataFrame(metrics_export)
                betas_df = pd.DataFrame(
                    [(factor, beta) for factor, beta in betas.items()],
                    columns=["factor", "beta"],
                )

                csv_buffer = StringIO()
                metrics_df.to_csv(csv_buffer, index=False)
                if not betas_df.empty:
                    csv_buffer.write("\n")
                    betas_df.to_csv(csv_buffer, index=False)
                csv_bytes = csv_buffer.getvalue().encode("utf-8")

                st.download_button(
                    "⬇️ Descargar análisis (CSV)",
                    csv_bytes,
                    file_name="factor_benchmark_analysis.csv",
                    mime="text/csv",
                    key="factor_analysis_csv",
                )

                excel_buffer = BytesIO()
                with pd.ExcelWriter(excel_buffer, engine="xlsxwriter") as writer:
                    metrics_df.to_excel(writer, index=False, sheet_name="metricas")
                    if not betas_df.empty:
                        betas_df.to_excel(writer, index=False, sheet_name="betas")
                excel_bytes = excel_buffer.getvalue()

                st.download_button(
                    "⬇️ Descargar análisis (XLSX)",
                    excel_bytes,
                    file_name="factor_benchmark_analysis.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    key="factor_analysis_xlsx",
                )
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
    return _finalize()
