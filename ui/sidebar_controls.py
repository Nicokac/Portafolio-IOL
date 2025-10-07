# ui\sidebar_controls.py
from __future__ import annotations
from dataclasses import asdict
from textwrap import shorten

import html
import streamlit as st

from domain.models import Controls

_CHIP_STYLE_KEY = "_sidebar_filter_chip_css"
_FLASH_STYLE_KEY = "_sidebar_controls_flash_css"
_FLASH_FLAG_KEY = "_sidebar_controls_flash_flag"


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
                background: color-mix(in srgb, var(--color-accent) 15%, transparent);
                color: color-mix(in srgb, var(--color-accent) 70%, var(--color-text) 30%);
                border-radius: 999px;
                padding: 0.2rem 0.65rem;
                font-size: 0.78rem;
                font-weight: 600;
                border: 1px solid color-mix(in srgb, var(--color-accent) 32%, transparent);
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                white-space: nowrap;
                transition: background-color 150ms ease, border-color 150ms ease,
                    box-shadow 150ms ease, color 150ms ease, transform 120ms ease;
            }
            .sidebar-chip:hover,
            .sidebar-chip:focus-visible {
                background: color-mix(in srgb, var(--color-accent) 24%, var(--color-bg) 76%);
                border-color: color-mix(in srgb, var(--color-accent) 45%, transparent);
                color: color-mix(in srgb, var(--color-accent) 82%, var(--color-text) 18%);
            }
            .sidebar-chip:active {
                background: color-mix(in srgb, var(--color-accent) 32%, var(--color-bg) 68%);
                border-color: color-mix(in srgb, var(--color-accent) 58%, transparent);
                color: color-mix(in srgb, var(--color-accent) 88%, var(--color-text) 12%);
                transform: translateY(1px);
            }
            .sidebar-chip:focus-visible {
                outline: none;
                box-shadow: 0 0 0 0.18rem color-mix(in srgb, var(--color-accent) 35%, transparent);
            }
            .sidebar-chip__label {
                line-height: 1.1;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_CHIP_STYLE_KEY] = True


def _ensure_flash_styles() -> None:
    if st.session_state.get(_FLASH_STYLE_KEY):
        return
    st.markdown(
        """
        <style>
            .control-panel__section--flash {
                animation: sidebar-controls-flash 1.6s ease-out;
            }
            @keyframes sidebar-controls-flash {
                0% {
                    box-shadow: 0 0 0 0 rgba(16, 163, 127, 0.45);
                    background-color: rgba(16, 163, 127, 0.12);
                }
                60% {
                    box-shadow: 0 0 0 6px rgba(16, 163, 127, 0);
                    background-color: rgba(16, 163, 127, 0.06);
                }
                100% {
                    box-shadow: 0 0 0 0 rgba(16, 163, 127, 0);
                    background-color: transparent;
                }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_FLASH_STYLE_KEY] = True


def _show_apply_feedback() -> None:
    message = "Filtros aplicados"
    if hasattr(st, "toast"):
        st.toast(message, icon="✅")
    elif hasattr(st, "info"):
        st.info(message)


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
        "<span class='sidebar-chip' tabindex='0' role='status'>"
        "<span class='sidebar-chip__label'>{label}</span>"
        "</span>".format(label=html.escape(label))
        for label in chips
    )
    container.caption("Filtros activos")
    container.markdown(
        "<div class='sidebar-chip-row'>{chips}</div>".format(chips=chip_html),
        unsafe_allow_html=True,
    )


def render_sidebar(
    all_symbols: list[str],
    available_types: list[str],
    *,
    container=None,
) -> Controls:
    host = container if container is not None else st.sidebar

    all_symbols = list(all_symbols or [])
    available_types = list(available_types or [])

    flash_active = bool(st.session_state.get(_FLASH_FLAG_KEY))
    if flash_active:
        _ensure_flash_styles()

    defaults = {
        "refresh_secs": st.session_state.get("refresh_secs", 30),
        "hide_cash": st.session_state.get("hide_cash", True),
        "show_usd": st.session_state.get("show_usd", False),
        "order_by": st.session_state.get("order_by", "valor_actual"),
        "desc": st.session_state.get("desc", True),
        "top_n": st.session_state.get("top_n", 20),
        "selected_syms": st.session_state.get("selected_syms", all_symbols),
        "selected_types": st.session_state.get("selected_types", available_types),
        "symbol_query": st.session_state.get("symbol_query", ""),
    }

    order_options = [
        "valor_actual",
        "pl",
        "pl_%",
        "pl_d",
        "chg_%",
        "costo",
        "ultimo",
        "cantidad",
        "simbolo",
    ]
    order_index = (
        order_options.index(defaults["order_by"])
        if defaults["order_by"] in order_options
        else 0
    )

    body_opened = False
    wrapper_opened = False
    if hasattr(host, "markdown"):
        host.markdown(
            "<div class='control-panel__body control-panel__body--sidebar'>",
            unsafe_allow_html=True,
        )
        body_opened = True
        wrapper_classes = "control-panel__section control-panel__section--sidebar"
        if flash_active:
            wrapper_classes += " control-panel__section--flash"
        host.markdown(
            "<div class='{classes}'>".format(classes=wrapper_classes),
            unsafe_allow_html=True,
        )
        wrapper_opened = True
        host.markdown("### 🎛️ Controles")
        if hasattr(host, "caption"):
            host.caption("Configura filtros, orden y visualizaciones del portafolio.")

    form = host.form("controls_form") if hasattr(host, "form") else st.form("controls_form")

    with form:
        update_col, filter_col, currency_col, order_col, charts_col = form.columns(
            (1.6, 3.2, 1.6, 1.6, 1.6)
        )

        with update_col:
            update_col.markdown("### ⏱️ Actualización")
            update_col.caption("Controlá cada cuánto se refrescan tablas, totales y gráficos.")
            refresh_secs = update_col.slider(
                "Intervalo (seg)",
                5,
                120,
                defaults["refresh_secs"],
                step=5,
                help="Refresca tus datos cada N segundos. Intervalos cortos usan más recursos.",
            )

        with filter_col:
            filter_col.markdown("### 🔍 Filtros")
            filter_col.caption(
                "Limitá la vista para enfocarte en activos específicos o categorías."
            )
            hide_cash = filter_col.checkbox(
                "Ocultar IOLPORA / PARKING",
                value=defaults["hide_cash"],
                help="Oculta el efectivo para enfocarte en las posiciones invertidas.",
            )
            symbol_query = filter_col.text_input(
                "Buscar símbolo",
                value=defaults["symbol_query"],
                placeholder="p.ej. NVDA",
                help="Busca tickers y actualiza tablas y gráficos al instante.",
            )
            selected_syms = filter_col.multiselect(
                "Filtrar por símbolo",
                all_symbols,
                default=[
                    s
                    for s in (defaults["selected_syms"] or [])
                    if s in all_symbols
                ]
                or all_symbols,
                help="Aplica solo los símbolos elegidos en tablas, rankings y gráficos.",
            )
            selected_types = filter_col.multiselect(
                "Filtrar por tipo",
                available_types,
                default=[
                    t
                    for t in (defaults["selected_types"] or available_types)
                    if t in available_types
                ],
                help="Muestra únicamente las clases de activo seleccionadas.",
            )

        with currency_col:
            currency_col.markdown("### 💱 Moneda")
            currency_col.caption(
                "Cambiá la moneda para comparar contra USD CCL en todas las visualizaciones."
            )
            show_usd = currency_col.toggle(
                "Mostrar valores en USD CCL",
                value=defaults["show_usd"],
                help="Convierte todos los valores a dólares CCL.",
            )

        chips = _active_filter_chips(
            hide_cash=hide_cash,
            show_usd=show_usd,
            symbol_query=symbol_query,
            selected_syms=selected_syms,
            all_symbols=all_symbols,
            selected_types=selected_types,
            available_types=available_types,
        )

        with order_col:
            order_col.markdown("### ↕️ Orden")
            order_col.caption(
                "Definí cómo ordenarás la tabla de posiciones y rankings asociados."
            )
            order_by = order_col.selectbox(
                "Ordenar por",
                order_options,
                index=order_index,
                help="Ordena tablas y exportaciones con este criterio.",
            )
            desc = order_col.checkbox(
                "Descendente",
                value=defaults["desc"],
                help="Muestra primero los valores más altos. Desactivalo para invertir el orden.",
            )

        with charts_col:
            charts_col.markdown("### 📈 Gráficos")
            charts_col.caption(
                "Controlá cuántos elementos se visualizan en rankings y gráficos destacados."
            )
            top_n = charts_col.slider(
                "Top N",
                5,
                50,
                defaults["top_n"],
                step=5,
                help="Elige cuántos elementos ver en rankings y gráficos comparativos.",
            )

        _render_filter_overview(filter_col, chips)

        action_cols = form.columns(2)
        apply_btn = action_cols[0].form_submit_button("Aplicar")
        reset_btn = action_cols[1].form_submit_button("Reset")

    if wrapper_opened:
        host.markdown("</div>", unsafe_allow_html=True)
    if body_opened:
        host.markdown("</div>", unsafe_allow_html=True)
    if flash_active:
        st.session_state.pop(_FLASH_FLAG_KEY, None)

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
        st.session_state[_FLASH_FLAG_KEY] = True
        _show_apply_feedback()

    snap = st.session_state.get("controls_snapshot")
    if snap:
        return Controls(**snap)
    return controls
