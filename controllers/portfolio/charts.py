import plotly.express as px
import streamlit as st

from ui.tables import render_totals, render_table
from ui.export import PLOTLY_CONFIG
from ui.charts import (
    plot_pl_topn,
    plot_donut_tipo,
    plot_dist_por_tipo,
    plot_pl_daily_topn,
    plot_bubble_pl_vs_costo,
    plot_heat_pl_pct,
)


def generate_basic_charts(df_view, top_n):
    """Generate basic portfolio charts."""
    return {
        "pl_topn": plot_pl_topn(df_view, n=top_n),
        "donut_tipo": plot_donut_tipo(df_view),
        "dist_tipo": plot_dist_por_tipo(df_view),
        "pl_diario": plot_pl_daily_topn(df_view, n=top_n),
    }


def render_basic_section(df_view, controls, ccl_rate):
    """Render totals, table and basic charts for the portfolio."""
    if df_view.empty:
        st.info("No hay datos del portafolio para mostrar.")
        return

    render_totals(df_view, ccl_rate=ccl_rate)
    render_table(
        df_view,
        controls.order_by,
        controls.desc,
        ccl_rate=ccl_rate,
        show_usd=controls.show_usd,
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


def render_advanced_analysis(df_view):
    """Render advanced analysis charts (bubble and heatmap)."""
    st.subheader("Bubble Chart Interactivo")
    axis_options = [
        c
        for c in ["costo", "pl", "pl_%", "valor_actual", "pl_d"]
        if c in df_view.columns
    ]
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
        fig = plot_bubble_pl_vs_costo(
            df_view,
            x_axis=x_axis,
            y_axis=y_axis,
            color_seq=color_seq,
            log_x=x_log,
            log_y=y_log,
        )
        if fig is not None:
            st.plotly_chart(
                fig,
                width="stretch",
                key="bubble_chart",
                config=PLOTLY_CONFIG,
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
