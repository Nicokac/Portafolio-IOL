# ui\sidebar_controls.py
from __future__ import annotations
from dataclasses import asdict
from textwrap import shorten

import html
import streamlit as st

from domain.models import Controls

_CHIP_STYLE_KEY = "_sidebar_filter_chip_css"


def _ensure_chip_styles(container) -> None:
    if st.session_state.get(_CHIP_STYLE_KEY):
        return
    container.markdown(
        """
        <style>
            .sidebar-chip-row {
                display: flex;
                flex-wrap: wrap;
                gap: 0.35rem;
                margin: 0.25rem 0 1rem;
            }
            .sidebar-chip {
                background: rgba(16, 163, 127, 0.12);
                color: rgb(7, 89, 73);
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid rgba(16, 163, 127, 0.35);
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                white-space: nowrap;
            }
            .sidebar-chip__label {
                line-height: 1.1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CHIP_STYLE_KEY] = True


def _active_filter_chips(
    *,
    hide_cash: bool,
    show_usd: bool,
    symbol_query: str,
    selected_syms: list[str],
    all_symbols: list[str],
    selected_types: list[str],
    available_types: list[str],
) -> list[str]:
    chips: list[str] = []
    total_symbols = len(all_symbols)
    total_types = len(available_types)

    if symbol_query:
        chips.append(f"🔎 {shorten(symbol_query.upper(), width=18, placeholder='…')}")
    if total_symbols and selected_syms and len(selected_syms) < total_symbols:
        chips.append(f"🎯 {len(selected_syms)}/{total_symbols} símbolos")
    if total_types and selected_types and len(selected_types) < total_types:
        chips.append(f"🏷️ {len(selected_types)}/{total_types} tipos")
    if hide_cash:
        chips.append("💸 Sin efectivo")
    if show_usd:
        chips.append("💵 USD CCL")
    return chips


def _render_filter_overview(container, chips: list[str]) -> None:
    if not chips:
        container.caption("Mostrando todos los activos disponibles.")
        return

    _ensure_chip_styles(container)
    chip_html = "".join(
        "<span class='sidebar-chip'>"
        "<span class='sidebar-chip__label'>{label}</span>"
        "</span>".format(label=html.escape(label))
        for label in chips
    )
    container.caption("Filtros activos")
    container.markdown(
        "<div class='sidebar-chip-row'>{chips}</div>".format(chips=chip_html),
        unsafe_allow_html=True,
    )


def render_sidebar(all_symbols: list[str], available_types: list[str]) -> Controls:
    st.sidebar.header("🎛️ Controles")

    all_symbols = list(all_symbols or [])
    available_types = list(available_types or [])

    defaults = {
        "refresh_secs": st.session_state.get("refresh_secs", 30),
        "hide_cash":    st.session_state.get("hide_cash", True),
        "show_usd":     st.session_state.get("show_usd", False),
        "order_by":     st.session_state.get("order_by", "valor_actual"),
        "desc":         st.session_state.get("desc", True),
        "top_n":        st.session_state.get("top_n", 20),
        "selected_syms": st.session_state.get("selected_syms", all_symbols),
        "selected_types": st.session_state.get("selected_types", available_types),
        "symbol_query": st.session_state.get("symbol_query", ""),
    }

    order_options = ["valor_actual", "pl", "pl_%", "pl_d", "chg_%", "costo", "ultimo", "cantidad", "simbolo"]
    order_index = order_options.index(defaults["order_by"]) if defaults["order_by"] in order_options else 0

    with st.sidebar.form("controls_form"):
        st.markdown("### ⏱️ Actualización")
        st.caption("Controlá cada cuánto se refrescan tablas, totales y gráficos.")
        refresh_secs = st.slider(
            "Intervalo (seg)",
            5,
            120,
            defaults["refresh_secs"],
            step=5,
            help="Un intervalo menor mantiene los datos frescos pero puede aumentar el uso de recursos.",
        )

        st.markdown("### 🔍 Filtros")
        st.caption("Limitá la vista para enfocarte en activos específicos o categorías.")
        hide_cash = st.checkbox(
            "Ocultar IOLPORA / PARKING",
            value=defaults["hide_cash"],
            help="Quita el efectivo de las tablas y métricas para concentrarte en posiciones invertidas.",
        )
        symbol_query = st.text_input(
            "Buscar símbolo",
            value=defaults["symbol_query"],
            placeholder="p.ej. NVDA",
            help="Filtra dinámicamente la tabla principal y los gráficos según coincidencias con el ticker.",
        )
        selected_syms = st.multiselect(
            "Filtrar por símbolo",
            all_symbols,
            default=[s for s in (defaults["selected_syms"] or []) if s in all_symbols] or all_symbols,
            help="Los símbolos seleccionados se utilizarán en tablas, rankings y comparativas visuales.",
        )
        selected_types = st.multiselect(
            "Filtrar por tipo",
            available_types,
            default=[t for t in (defaults["selected_types"] or available_types) if t in available_types],
            help="Restringe la vista a clases de activo específicas, afectando gráficos y totales.",
        )

        st.markdown("### 💱 Moneda")
        st.caption("Cambiá la moneda para comparar contra USD CCL en todas las visualizaciones.")
        show_usd = st.toggle(
            "Mostrar valores en USD CCL",
            value=defaults["show_usd"],
            help="Transforma los importes a dólares CCL en tablas, métricas y exportaciones.",
        )

        _render_filter_overview(
            st.sidebar,
            _active_filter_chips(
                hide_cash=hide_cash,
                show_usd=show_usd,
                symbol_query=symbol_query,
                selected_syms=selected_syms,
                all_symbols=all_symbols,
                selected_types=selected_types,
                available_types=available_types,
            ),
        )

        st.markdown("### ↕️ Orden")
        st.caption("Definí cómo ordenarás la tabla de posiciones y rankings asociados.")
        order_by = st.selectbox(
            "Ordenar por",
            order_options,
            index=order_index,
            help="Aplica el criterio seleccionado tanto en la tabla principal como en exportaciones.",
        )
        desc = st.checkbox(
            "Descendente",
            value=defaults["desc"],
            help="Mostrá primero los valores más altos (o más bajos si se desactiva).",
        )

        st.markdown("### 📈 Gráficos")
        st.caption("Controlá cuántos elementos se visualizan en rankings y gráficos destacados.")
        top_n = st.slider(
            "Top N",
            5,
            50,
            defaults["top_n"],
            step=5,
            help="Determina la cantidad de barras o puntos que verás en los gráficos comparativos.",
        )

        c1, c2 = st.columns(2)
        apply_btn = c1.form_submit_button("Aplicar")
        reset_btn = c2.form_submit_button("Reset")

    controls = Controls(
        refresh_secs=refresh_secs,
        hide_cash=hide_cash,
        show_usd=show_usd,
        order_by=order_by,
        desc=desc,
        top_n=top_n,
        selected_syms=selected_syms,
        selected_types=selected_types,
        symbol_query=(symbol_query or "").strip().upper(),
    )

    if reset_btn:
        for k in asdict(controls).keys():
            st.session_state.pop(k, None)
        st.session_state["controls_snapshot"] = None
        st.rerun()

    if apply_btn:
        st.session_state.update(asdict(controls))
        st.session_state["controls_snapshot"] = asdict(controls)

    snap = st.session_state.get("controls_snapshot")
    if snap:
        return Controls(**snap)
    return controls
