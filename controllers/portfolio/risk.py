import logging
import re
import unicodedata
from typing import Sequence

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import time

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from application.risk_service import (
    compute_returns,
    annualized_volatility,
    beta,
    historical_var,
    expected_shortfall,
    rolling_correlations,
    markowitz_optimize,
    monte_carlo_simulation,
    apply_stress,
    asset_risk_breakdown,
    max_drawdown,
    drawdown_series,
)
from application.portfolio_service import classify_symbol
from services.notifications import NotificationFlags
from ui.charts import plot_correlation_heatmap, _apply_layout
from ui.export import PLOTLY_CONFIG
from ui.notifications import render_risk_badge
from shared.errors import AppError
from services.health import record_tab_latency


logger = logging.getLogger(__name__)

_LOCAL_SYMBOL_BLACKLIST = {"LOMA", "YPFD", "TECO2"}

_TYPE_ALIASES = {
    "ACCION": "ACCION_LOCAL",
    "ACCIONES": "ACCION_LOCAL",
    "ACCION LOCAL": "ACCION_LOCAL",
    "ACCION ARG": "ACCION_LOCAL",
    "ACCION ARGENTINA": "ACCION_LOCAL",
    "ACCION ARGENTINAS": "ACCION_LOCAL",
    "ACCION_LOCAL": "ACCION_LOCAL",
    "ACCIONES LOCALES": "ACCION_LOCAL",
    "ACCIONES NACIONALES": "ACCION_LOCAL",
    "BONO": "BONO",
    "BONOS": "BONO",
    "BONOS SOBERANOS": "BONO",
    "BONOS CORPORATIVOS": "BONO",
    "LETRA": "LETRA",
    "LETRAS": "LETRA",
    "CEDEAR": "CEDEAR",
    "CEDEARS": "CEDEAR",
    "ETF": "ETF",
    "ETFS": "ETF",
    "FONDO": "FCI",
    "FONDOS": "FCI",
    "FONDO COMUN": "FCI",
    "FONDO COMUN DE INVERSION": "FCI",
    "FONDO DE INVERSION": "FCI",
    "FCI": "FCI",
    "OTRO": "OTRO",
    "OTROS": "OTRO",
}

_TYPE_DISPLAY_OVERRIDES = {
    "ACCION_LOCAL": "Acciones locales",
    "CEDEAR": "CEDEARs",
    "BONO": "Bonos",
    "LETRA": "Letras",
    "ETF": "ETFs",
    "FCI": "Fondos comunes (FCI)",
    "OTRO": "Otros",
}

_SYMBOL_TYPE_OVERRIDES = {sym: "ACCION_LOCAL" for sym in _LOCAL_SYMBOL_BLACKLIST}


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


def _normalize_type_token(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = normalized.replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized.upper()


def _normalize_type_label(value) -> str:
    text = _string_or_empty(value)
    if not text:
        return ""
    token = _normalize_type_token(text)
    if not token:
        return ""
    return _TYPE_ALIASES.get(token, token)


def _normalize_symbol(symbol) -> str:
    text = _string_or_empty(symbol)
    return text.upper()


def _canonical_type(symbol: str, raw_type) -> str:
    sym_key = _normalize_symbol(symbol)
    override = _SYMBOL_TYPE_OVERRIDES.get(sym_key)
    if override:
        return override

    normalized = _normalize_type_label(raw_type)
    if normalized:
        return normalized

    if sym_key:
        fallback = classify_symbol(sym_key)
        normalized = _normalize_type_label(fallback)
        if normalized:
            return _SYMBOL_TYPE_OVERRIDES.get(sym_key, normalized)
    return ""


def _build_type_metadata(df: pd.DataFrame | None) -> tuple[pd.Series, dict[str, str], dict[str, str]]:
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
        normalized_values.append(canonical if canonical else pd.NA)
        if canonical:
            sym_key = _normalize_symbol(symbol)
            symbol_type_map[sym_key] = canonical
            if sym_key not in _SYMBOL_TYPE_OVERRIDES:
                raw_label = _string_or_empty(raw_value)
                if raw_label and canonical not in display_labels:
                    display_labels[canonical] = raw_label

    for canonical, label in _TYPE_DISPLAY_OVERRIDES.items():
        display_labels.setdefault(canonical, label)

    for canonical in symbol_type_map.values():
        display_labels.setdefault(canonical, canonical.replace("_", " ").title())

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
    df_view,
    tasvc,
    favorites: FavoriteSymbols | None = None,
    *,
    notifications: NotificationFlags | None = None,
    available_types: Sequence[str] | None = None,
):
    """Render correlation and risk analysis for the portfolio."""
    favorites = favorites or get_persistent_favorites()
    st.subheader("Análisis de Correlación del Portafolio")
    symbols = (
        sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()})
        if not df_view.empty
        else []
    )

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
        _normalize_type_label(t)
        for t in selected_type_filters
        if isinstance(t, str) and _normalize_type_label(t)
    ]

    normalized_series, type_display_map, symbol_type_map = _build_type_metadata(
        filtered_df if isinstance(filtered_df, pd.DataFrame) else None
    )

    if isinstance(filtered_df, pd.DataFrame):
        filtered_df["_normalized_type"] = normalized_series.reindex(
            filtered_df.index
        )
        if normalized_filters:
            filtered_df = filtered_df[
                filtered_df["_normalized_type"].isin(normalized_filters)
            ].copy()

        if not filtered_df.empty:
            normalized_symbols = (
                filtered_df["simbolo"].astype(str).str.strip().str.upper()
            )
            blacklist_mask = (
                filtered_df["_normalized_type"].eq("CEDEAR")
                & normalized_symbols.isin(_LOCAL_SYMBOL_BLACKLIST)
            )
            if blacklist_mask.any():
                filtered_df = filtered_df.loc[~blacklist_mask].copy()
    else:
        type_display_map = {}
        symbol_type_map = {}

    if filtered_df.empty:
        st.warning("No hay datos para los tipos seleccionados.")
        st.info("Ajustá los filtros de tipo de activo para ver correlaciones y métricas de riesgo.")
        st.subheader("Análisis de Riesgo")
        st.info("No hay símbolos en el portafolio para analizar.")
        return

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
            canon_str = symbol_type_map.get(sym_key)
            if not canon_str:
                canon_str = (
                    _normalize_type_label(canon)
                    if isinstance(canon, str)
                    else _canonical_type(sym_key, None)
                )
            if normalized_filters and canon_str not in normalized_filters:
                continue
            if canon_str == "CEDEAR" and sym_str.upper() in _LOCAL_SYMBOL_BLACKLIST:
                continue
            portfolio_symbols.append(sym_str)
    else:
        portfolio_symbols = [
            str(sym).strip()
            for sym in getattr(filtered_df, "get", lambda *_: [])("simbolo", [])
            if str(sym).strip()
        ]
    if len(portfolio_symbols) >= 2:
        corr_latency: float | None = None
        with st.spinner(f"Calculando correlación ({corr_period})…"):
            start_time = time.perf_counter()
            try:
                unique_symbols = sorted(set(portfolio_symbols))
                hist_df = tasvc.portfolio_history(
                    simbolos=unique_symbols, period=corr_period
                )
            except AppError as err:
                corr_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", corr_latency, status="error")
                st.error(str(err))
                return
            except Exception:
                logger.exception(
                    "Error al obtener históricos para correlación",
                )
                corr_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", corr_latency, status="error")
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return
            corr_latency = (time.perf_counter() - start_time) * 1000.0
        record_tab_latency("riesgo", corr_latency, status="success")
        returns_for_corr = compute_returns(hist_df)
        available_types_in_view: list[str] = []
        if isinstance(filtered_df, pd.DataFrame) and "_normalized_type" in filtered_df:
            available_types_in_view = [
                str(t)
                for t in filtered_df["_normalized_type"].dropna().unique()
                if isinstance(t, str) and str(t)
            ]

        if normalized_filters:
            ordered_types = [t for t in normalized_filters if t in available_types_in_view]
        else:
            ordered_types = sorted(available_types_in_view)

        type_groups: list[tuple[str, pd.DataFrame]] = []
        for type_name in ordered_types:
            subset = (
                filtered_df[filtered_df["_normalized_type"] == type_name]
                if isinstance(filtered_df, pd.DataFrame)
                else pd.DataFrame()
            )
            if isinstance(subset, pd.DataFrame) and not subset.empty:
                type_groups.append((type_name, subset.copy()))

        if not type_groups:
            label = "Portafolio"
            type_groups = [(label, filtered_df if isinstance(filtered_df, pd.DataFrame) else pd.DataFrame())]

        display_targets: list = []
        if len(type_groups) > 1:
            display_targets = st.tabs(
                [
                    type_display_map.get(
                        type_name,
                        type_name.replace("_", " ").title(),
                    )
                    for type_name, _ in type_groups
                ]
            )
        else:
            display_targets = [st.container()]

        for (type_name, subset_df), display_host in zip(type_groups, display_targets):
            with display_host:
                display_label = type_display_map.get(
                    type_name,
                    type_name.replace("_", " ").title(),
                )
                subset_symbols = [
                    str(sym).strip()
                    for sym in subset_df.get("simbolo", [])
                    if str(sym).strip()
                ]
                if type_name == "CEDEAR":
                    subset_symbols = [
                        sym
                        for sym in subset_symbols
                        if sym.upper() not in _LOCAL_SYMBOL_BLACKLIST
                    ]
                subset_symbols = [
                    sym for sym in subset_symbols if sym in hist_df.columns
                ]
                if len(set(subset_symbols)) < 2:
                    st.warning(
                        f"⚠️ No hay suficientes activos del tipo {display_label} para calcular correlaciones."
                    )
                    continue

                subset_hist = hist_df[sorted(set(subset_symbols))]
                fig = plot_correlation_heatmap(
                    subset_hist,
                    title=f"Matriz de Correlación — {display_label}",
                )
                if fig:
                    st.plotly_chart(
                        fig,
                        width="stretch",
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
            window_options = {"1 mes (21)": 21, "3 meses (63)": 63, "6 meses (126)": 126}
            selected_window_label = st.selectbox(
                "Ventana para correlaciones móviles",
                list(window_options.keys()),
                index=1,
                key="rolling_corr_window",
            )
            selected_window = window_options[selected_window_label]
            rolling_df = rolling_correlations(returns_for_corr, selected_window)
            if not rolling_df.empty:
                roll_fig = px.line(
                    rolling_df,
                    labels={"index": "Fecha", "value": "Correlación", "variable": "Par"},
                )
                roll_fig = _apply_layout(
                    roll_fig,
                    title=f"Correlaciones móviles ({selected_window} ruedas)",
                )
                st.plotly_chart(
                    roll_fig,
                    width="stretch",
                    key="rolling_corr_chart",
                    config=PLOTLY_CONFIG,
                )
                latest = rolling_df.dropna(how="all").tail(1)
                if not latest.empty:
                    latest_tidy = (
                        latest.T.reset_index().rename(columns={"index": "Par", latest.index[-1]: "Correlación"})
                    )
                    st.markdown(
                        latest_tidy.to_html(index=False, float_format="{:.2f}".format),
                        unsafe_allow_html=True,
                    )
            else:
                st.info(
                    "No se pudieron calcular correlaciones móviles con la ventana seleccionada."
                )
        else:
            st.warning(
                f"No se pudieron obtener suficientes datos históricos para el período '{corr_period}' para calcular la correlación."
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
                prices_df = tasvc.portfolio_history(
                    simbolos=portfolio_symbols, period="1y"
                )
            except AppError as err:
                risk_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", risk_latency, status="error")
                st.error(str(err))
                return
            except Exception:
                logger.exception(
                    "Error al obtener históricos para análisis de riesgo",
                )
                risk_latency = (time.perf_counter() - start_time) * 1000.0
                record_tab_latency("riesgo", risk_latency, status="error")
                st.error(
                    "No se pudieron obtener datos históricos, intente nuevamente más tarde",
                )
                return
            risk_latency = (time.perf_counter() - start_time) * 1000.0
        record_tab_latency("riesgo", risk_latency, status="success")
        if prices_df.empty:
            st.info(
                "No se pudieron obtener datos históricos para calcular métricas de riesgo."
            )
        else:
            returns_df = compute_returns(prices_df)
            weights = (
                filtered_df.set_index("simbolo")["valor_actual"].astype(float)
                .reindex(returns_df.columns)
                .dropna()
            )
            weights = weights / weights.sum() if not weights.empty else weights
            if weights.empty or returns_df.empty:
                st.info(
                    "No hay suficientes datos para calcular métricas de riesgo."
                )
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
                    bench_df = tasvc.portfolio_history(
                        simbolos=[benchmark_symbol], period="1y"
                    )
                except AppError as err:
                    st.error(str(err))
                    return
                except Exception:
                    logger.exception(
                        "Error al obtener benchmark para análisis de riesgo",
                    )
                    st.error(
                        "No se pudieron obtener datos históricos para el benchmark seleccionado.",
                    )
                    return

                bench_ret = compute_returns(bench_df).squeeze()
                if bench_ret.empty:
                    st.info(
                        "El benchmark seleccionado no tiene datos suficientes para calcular beta."
                    )
                    return

                confidence_options = {"90%": 0.90, "95%": 0.95, "99%": 0.99}
                selected_conf_label = st.selectbox(
                    "Nivel de confianza para VaR/CVaR",
                    list(confidence_options.keys()),
                    index=1,
                    key="var_confidence_select",
                )
                var_confidence = confidence_options[selected_conf_label]

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
                        width="stretch",
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
                        width="stretch",
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
                        width="stretch",
                        config=PLOTLY_CONFIG,
                    )
                    st.caption(
                        "La volatilidad refleja la variabilidad de los retornos; aquí se muestra en una ventana móvil de 30 días."
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
                        width="stretch",
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
                        scatter_cols[0].info(
                            "No hay datos suficientes para calcular beta por activo."
                        )
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
                        scatter_df["Favorito"] = scatter_df["Símbolo"].apply(
                            lambda sym: sym in favorite_set
                        )
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
                            fig_scatter.update_traces(marker=dict(line=dict(width=1, color="#2a2a2a")))
                            fig_scatter = _apply_layout(
                                fig_scatter,
                                title="Beta vs retorno anualizado",
                            )
                            scatter_cols[0].plotly_chart(
                                fig_scatter,
                                width="stretch",
                                config=PLOTLY_CONFIG,
                            )

                with st.expander("Optimización de portafolio (Markowitz)"):
                    opt_df = pd.DataFrame({"ticker": opt_w.index, "weight": opt_w.values})
                    st.bar_chart(opt_df, x="ticker", y="weight")
                    st.caption(
                        "Barras con la proporción que el modelo recomienda invertir en cada activo para equilibrar riesgo y retorno."
                    )

                with st.expander("Simulación Monte Carlo"):
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
                    final_prices = monte_carlo_simulation(
                        returns_df, weights, n_sims=sims, horizon=horizon
                    )
                    st.line_chart(final_prices)
                    st.caption(
                        "Simula muchos escenarios posibles para estimar cómo podría variar el valor de la cartera en el futuro."
                    )

                with st.expander("Aplicar shocks"):
                    templates = {"Leve": 0.03, "Moderado": 0.07, "Fuerte": 0.12}
                    tmpl = st.selectbox("Escenario", list(templates), index=0)
                    shocks = {sym: -templates[tmpl] for sym in returns_df.columns}
                    st.caption(
                        f"Aplicando un shock uniforme de {templates[tmpl]:.0%} a todos los activos."
                    )
                base_prices = pd.Series(1.0, index=weights.index)
                stressed_val = apply_stress(base_prices, weights, shocks)
                st.write(f"Retorno con shocks: {stressed_val - 1:.2%}")
    else:
        st.info("No hay símbolos en el portafolio para analizar.")
