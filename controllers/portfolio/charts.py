import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from shared.favorite_symbols import FavoriteSymbols, get_persistent_favorites
from ui.favorites import render_favorite_badges, render_favorite_toggle
from ui.tables import render_totals, render_table
from ui.export import PLOTLY_CONFIG
from services.portfolio_view import compute_symbol_risk_metrics
from ui.charts import (
    _apply_layout,
    plot_pl_topn,
    plot_donut_tipo,
    plot_dist_por_tipo,
    plot_pl_daily_topn,
    plot_bubble_pl_vs_costo,
    plot_heat_pl_pct,
    plot_portfolio_timeline,
    plot_contribution_heatmap,
)


def generate_basic_charts(df_view, top_n):
    """Generate basic portfolio charts."""
    return {
        "pl_topn": plot_pl_topn(df_view, n=top_n),
        "donut_tipo": plot_donut_tipo(df_view),
        "dist_tipo": plot_dist_por_tipo(df_view),
        "pl_diario": plot_pl_daily_topn(df_view, n=top_n),
    }


def render_basic_section(
    df_view,
    controls,
    ccl_rate,
    totals=None,
    favorites: FavoriteSymbols | None = None,
    historical_total=None,
    contribution_metrics=None,
):
    """Render totals, table and basic charts for the portfolio."""
    favorites = favorites or get_persistent_favorites()
    symbols = (
        sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()})
        if not df_view.empty
        else []
    )

    render_favorite_badges(
        favorites,
        empty_message="⭐ Marcá tus símbolos preferidos para destacarlos en todas las secciones.",
    )

    if symbols:
        options = favorites.sort_options(symbols)
        selected_symbol = st.selectbox(
            "Gestionar favoritos",
            options=options,
            index=favorites.default_index(options),
            key="portfolio_favorite_select",
            format_func=favorites.format_symbol,
        )
        render_favorite_toggle(
            selected_symbol,
            favorites,
            key_prefix="portfolio",
            help_text="La selección impacta todas las pestañas y exportaciones.",
        )

    if df_view.empty:
        st.info("No hay datos del portafolio para mostrar.")
        return

    render_totals(df_view, ccl_rate=ccl_rate, totals=totals)
    render_table(
        df_view,
        controls.order_by,
        controls.desc,
        ccl_rate=ccl_rate,
        show_usd=controls.show_usd,
        favorites=favorites,
    )

    charts = generate_basic_charts(df_view, controls.top_n)
    colA, colB = st.columns(2)
    with colA:
        st.subheader("P/L por símbolo (Top N)")
        fig = charts["pl_topn"]
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                key="pl_topn",
                config=PLOTLY_CONFIG,
            )
            st.caption(
                "Barras que muestran qué activos ganan o pierden más. Las más altas son las que más afectan tu resultado."
            )
        else:
            st.info("Sin datos para graficar P/L Top N.")
    with colB:
        st.subheader("Composición por tipo (Donut)")
        fig = charts["donut_tipo"]
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                key="donut_tipo",
                config=PLOTLY_CONFIG,
            )
            st.caption(
                "Indica qué porcentaje de tu inversión está en cada tipo de activo para ver si estás diversificando bien."
            )
        else:
            st.info("No hay datos para el donut por tipo.")

    st.subheader("Distribución por tipo (Valorizado)")
    fig = charts["dist_tipo"]
    if fig is not None:
        st.plotly_chart(
            fig,
            width="stretch",
            key="dist_tipo",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Compara cuánto dinero tenés en cada categoría de activos. Ayuda a detectar concentraciones."
        )
    else:
        st.info("No hay datos para la distribución por tipo.")

    st.subheader("P/L diario por símbolo (Top N)")
    fig = charts["pl_diario"]
    if fig is not None:
        st.plotly_chart(
            fig,
            width="stretch",
            key="pl_diario",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Muestra las ganancias o pérdidas del día para los activos con mayor movimiento."
        )
    else:
        st.info("Aún no hay datos de P/L diario.")

    st.subheader("Evolución histórica del portafolio")
    timeline_fig = plot_portfolio_timeline(historical_total)
    if timeline_fig is not None:
        st.plotly_chart(
            timeline_fig,
            width="stretch",
            key="portfolio_timeline",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Sigue cómo varían tus totales (valor, costo y P/L) con el tiempo para detectar tendencias y cambios relevantes."
        )
    else:
        st.info("Aún no hay suficientes datos históricos del portafolio.")

    st.subheader("Contribución por símbolo y tipo")
    by_symbol = getattr(contribution_metrics, "by_symbol", None)
    heatmap_fig = plot_contribution_heatmap(by_symbol)
    if heatmap_fig is not None:
        st.plotly_chart(
            heatmap_fig,
            width="stretch",
            key="portfolio_contribution_heatmap",
            config=PLOTLY_CONFIG,
        )
        st.caption(
            "Visualiza qué combinaciones de tipo y símbolo concentran mayor peso en tu cartera."
        )
    else:
        st.info("Sin datos de contribución por símbolo para mostrar el mapa de calor.")

    by_type = getattr(contribution_metrics, "by_type", None)
    if isinstance(by_type, pd.DataFrame) and not by_type.empty:
        display_cols = [
            col
            for col in ["tipo", "valor_actual", "valor_actual_pct", "pl", "pl_pct"]
            if col in by_type.columns
        ]
        df_table = by_type[display_cols].copy()
        for col in df_table.columns:
            if col == "tipo":
                df_table[col] = df_table[col].astype(str)
            elif col.endswith("_pct"):
                df_table[col] = df_table[col].apply(
                    lambda v: f"{float(v):.2f}%" if pd.notna(v) else "—"
                )
            else:
                df_table[col] = df_table[col].apply(
                    lambda v: f"{float(v):,.0f}" if pd.notna(v) else "—"
                )

        table_fig = go.Figure(
            data=[
                go.Table(
                    header=dict(
                        values=[col.replace("_", " ").title() for col in df_table.columns],
                        fill_color="rgba(0,0,0,0)",
                        align="left",
                    ),
                    cells=dict(
                        values=[df_table[col].tolist() for col in df_table.columns],
                        align="left",
                    ),
                )
            ]
        )
        table_fig = _apply_layout(table_fig, show_legend=False)
        st.plotly_chart(
            table_fig,
            width="stretch",
            key="portfolio_contribution_table",
            config=PLOTLY_CONFIG,
        )
    else:
        st.info("No hay datos agregados por tipo para mostrar en tabla.")


RISK_METRIC_OPTIONS = {
    "Volatilidad anualizada": "volatilidad",
    "Drawdown máximo": "drawdown",
    "Beta vs benchmark": "beta",
}


def _format_risk_value(metric: str, value: float, *, with_sign: bool = False) -> str:
    if value != value:
        return "N/A"
    if metric in {"volatilidad", "drawdown"}:
        return f"{value:+.2%}" if with_sign else f"{value:.2%}"
    return f"{value:+.2f}" if with_sign else f"{value:.2f}"


def render_advanced_analysis(df_view, tasvc, *, benchmark_choices=None):
    """Render advanced analysis charts (bubble and heatmap)."""
    st.subheader("Bubble Chart Interactivo")
    benchmark_choices = benchmark_choices or {
        "S&P 500 (^GSPC)": "^GSPC",
    }

    symbols = (
        sorted({str(sym) for sym in df_view.get("simbolo", []) if str(sym).strip()})
        if not df_view.empty
        else []
    )

    cols_controls = st.columns(3)
    with cols_controls[0]:
        period = st.selectbox(
            "Período de métricas",
            options=["3mo", "6mo", "1y", "2y"],
            index=2,
            key="bubble_period",
        )
    with cols_controls[1]:
        metric_label = st.selectbox(
            "Métrica de riesgo",
            options=list(RISK_METRIC_OPTIONS.keys()),
            index=0,
            key="bubble_risk_metric",
        )
    with cols_controls[2]:
        bench_label = st.selectbox(
            "Benchmark",
            options=list(benchmark_choices.keys()),
            index=0,
            key="bubble_benchmark",
        )
    benchmark_symbol = benchmark_choices[bench_label]

    metrics_df = compute_symbol_risk_metrics(
        tasvc,
        symbols,
        benchmark=benchmark_symbol,
        period=period,
    )

    assets_df = df_view.copy()
    benchmark_row = pd.DataFrame()
    if not metrics_df.empty:
        asset_metrics = metrics_df.loc[~metrics_df["es_benchmark"]].drop(
            columns=["es_benchmark"], errors="ignore"
        )
        assets_df = assets_df.merge(asset_metrics, on="simbolo", how="left")
        benchmark_row = metrics_df.loc[metrics_df["es_benchmark"]]

    axis_candidates = [
        c
        for c in ["costo", "pl", "pl_%", "valor_actual", "pl_d"]
        if c in assets_df.columns
    ]
    for c in RISK_METRIC_OPTIONS.values():
        if c in assets_df.columns:
            axis_candidates.append(c)

    axis_options = list(dict.fromkeys(axis_candidates))

    if not axis_options:
        st.info("No hay columnas disponibles para el gráfico bubble.")
    else:
        c1, c2 = st.columns(2)
        with c1:
            x_axis = st.selectbox(
                "Eje X",
                options=axis_options,
                index=axis_options.index("costo") if "costo" in axis_options else 0,
                key="bubble_x",
            )
            x_log = st.checkbox("Escala log X", key="bubble_x_log")
        with c2:
            y_axis = st.selectbox(
                "Eje Y",
                options=axis_options,
                index=axis_options.index("pl")
                if "pl" in axis_options
                else min(1, len(axis_options) - 1),
                key="bubble_y",
            )
            y_log = st.checkbox("Escala log Y", key="bubble_y_log")
        palette_opt = st.selectbox(
            "Paleta",
            ["Tema", "Plotly", "D3", "G10"],
            key="bubble_palette",
        )
        palette_map = {
            "Plotly": px.colors.qualitative.Plotly,
            "D3": px.colors.qualitative.D3,
            "G10": px.colors.qualitative.G10,
        }
        color_seq = palette_map.get(palette_opt) if palette_opt != "Tema" else None
        bubble_df = assets_df.copy()
        if not benchmark_row.empty:
            bench_data = benchmark_row.iloc[0].to_dict()
            bench_entry = {col: None for col in bubble_df.columns}
            bench_entry.update(bench_data)
            bench_entry.setdefault("simbolo", benchmark_symbol)
            if "valor_actual" in bubble_df.columns:
                avg_val = bubble_df["valor_actual"].dropna()
                bench_entry.setdefault(
                    "valor_actual", float(avg_val.mean()) if not avg_val.empty else 0.0
                )
            bench_entry.setdefault("tipo", "Benchmark")
            bubble_df = pd.concat(
                [bubble_df, pd.DataFrame([bench_entry])], ignore_index=True
            )

        if "es_benchmark" not in bubble_df.columns:
            if "simbolo" in bubble_df.columns:
                bubble_df["es_benchmark"] = bubble_df["simbolo"].eq(benchmark_symbol)
            else:
                bubble_df["es_benchmark"] = False
        else:
            bubble_df["es_benchmark"] = bubble_df["es_benchmark"].fillna(False)
        bubble_df["categoria"] = bubble_df["es_benchmark"].map(
            {True: "Benchmark", False: "Activo"}
        )

        fig = plot_bubble_pl_vs_costo(
            bubble_df,
            x_axis=x_axis,
            y_axis=y_axis,
            color_seq=color_seq,
            log_x=x_log,
            log_y=y_log,
            category_col="categoria",
            benchmark_col="es_benchmark",
        )
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                key="bubble_chart",
                config=PLOTLY_CONFIG,
            )

            metric_col = RISK_METRIC_OPTIONS[metric_label]
            if metric_col in bubble_df.columns:
                bench_value = bubble_df.loc[
                    bubble_df["es_benchmark"], metric_col
                ].dropna()
                asset_values = bubble_df.loc[
                    ~bubble_df["es_benchmark"], metric_col
                ].dropna()
                if not bench_value.empty and not asset_values.empty:
                    bench_val = float(bench_value.iloc[0])
                    avg_val = float(asset_values.mean())
                    delta = avg_val - bench_val
                    st.metric(
                        f"Promedio {metric_label}",
                        _format_risk_value(metric_col, avg_val),
                        delta=_format_risk_value(metric_col, delta, with_sign=True),
                        help=f"Benchmark: {bench_label} ({_format_risk_value(metric_col, bench_val)})",
                    )

            with st.expander("Descripción"):
                st.caption(
                    "Cada burbuja representa un símbolo; el tamaño refleja el valor actual. Cambia ejes, escalas y paleta para explorar distintos ángulos."
                )
        else:
            st.info("No hay datos suficientes para el gráfico bubble.")

        st.subheader("Heatmap de rendimiento (%) por símbolo")
        heat_scale = st.selectbox(
            "Escala de color",
            ["RdBu", "Viridis", "Plasma", "Cividis", "Turbo"],
            key="heat_scale",
        )
        fig = plot_heat_pl_pct(df_view, color_scale=heat_scale)
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                key="heatmap_chart",
                config=PLOTLY_CONFIG,
            )
            with st.expander("Descripción"):
                st.caption(
                    "El color indica la variación porcentual; prueba diferentes escalas para resaltar ganancias o pérdidas."
                )
        else:
            st.info("No hay datos suficientes para el heatmap.")
