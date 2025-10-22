# ui\sidebar_controls.py
from __future__ import annotations

import html
from contextlib import contextmanager
from dataclasses import asdict
from textwrap import shorten

import streamlit as st

from domain.models import Controls

_FLASH_FLAG_KEY = "_sidebar_controls_flash_flag"


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


@contextmanager
def _section_card(host, *, extra_classes: str = ""):
    classes = "control-panel__section control-panel__section--sidebar"
    if extra_classes:
        classes = f"{classes} {extra_classes}".strip()
    host.markdown(f"<div class='{classes}'>", unsafe_allow_html=True)
    container = host.container()
    with container:
        yield container
    host.markdown("</div>", unsafe_allow_html=True)


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
    selected_types_state = st.session_state.get("selected_types")
    if selected_types_state is None:
        selected_types_state = st.session_state.get("selected_asset_types")

    defaults = {
        "refresh_secs": st.session_state.get("refresh_secs", 30),
        "hide_cash": st.session_state.get("hide_cash", True),
        "show_usd": st.session_state.get("show_usd", False),
        "order_by": st.session_state.get("order_by", "valor_actual"),
        "desc": st.session_state.get("desc", True),
        "top_n": st.session_state.get("top_n", 20),
        "selected_syms": st.session_state.get("selected_syms", all_symbols),
        "selected_types": selected_types_state if selected_types_state is not None else available_types,
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
    order_index = order_options.index(defaults["order_by"]) if defaults["order_by"] in order_options else 0

    body_opened = False
    if hasattr(host, "markdown"):
        host.markdown(
            "<div class='control-panel__body control-panel__body--sidebar'>",
            unsafe_allow_html=True,
        )
        body_opened = True
        host.markdown("### 🎛️ Controles")
        if hasattr(host, "caption"):
            host.caption("Configura filtros, orden y visualizaciones del portafolio.")

    form = host.form("controls_form") if hasattr(host, "form") else st.form("controls_form")

    with form:
        with _section_card(form) as update_section:
            update_section.markdown("### ⏱️ Actualización")
            update_section.caption("Controlá cada cuánto se refrescan tablas, totales y gráficos.")
            refresh_secs = update_section.slider(
                "Intervalo (seg)",
                5,
                120,
                defaults["refresh_secs"],
                step=5,
                help="Refresca tus datos cada N segundos. Intervalos cortos usan más recursos.",
            )

        with _section_card(
            form,
            extra_classes="control-panel__section--flash" if flash_active else "",
        ) as filter_section:
            filter_section.markdown("### 🔍 Filtros")
            filter_section.caption("Limitá la vista para enfocarte en activos específicos o categorías.")
            hide_cash = filter_section.checkbox(
                "Ocultar IOLPORA / PARKING",
                value=defaults["hide_cash"],
                help="Oculta el efectivo para enfocarte en las posiciones invertidas.",
            )
            symbol_query = filter_section.text_input(
                "Buscar símbolo",
                value=defaults["symbol_query"],
                placeholder="p.ej. NVDA",
                help="Busca tickers y actualiza tablas y gráficos al instante.",
            )
            selected_syms = filter_section.multiselect(
                "Filtrar por símbolo",
                all_symbols,
                default=[s for s in (defaults["selected_syms"] or []) if s in all_symbols] or all_symbols,
                help="Aplica solo los símbolos elegidos en tablas, rankings y gráficos.",
            )
            selected_types = filter_section.multiselect(
                "Filtrar por tipo",
                available_types,
                default=[t for t in (defaults["selected_types"] or available_types) if t in available_types],
                help="Muestra únicamente las clases de activo seleccionadas.",
            )
            overview_container = filter_section.container()

        with _section_card(form) as currency_section:
            currency_section.markdown("### 💱 Moneda")
            currency_section.caption("Cambiá la moneda para comparar contra USD CCL en todas las visualizaciones.")
            show_usd = currency_section.toggle(
                "Mostrar valores en USD CCL",
                value=defaults["show_usd"],
                help="Convierte todos los valores a dólares CCL.",
            )

        with _section_card(form) as order_section:
            order_section.markdown("### ↕️ Orden")
            order_section.caption("Definí cómo ordenarás la tabla de posiciones y rankings asociados.")
            order_by = order_section.selectbox(
                "Ordenar por",
                order_options,
                index=order_index,
                help="Ordena tablas y exportaciones con este criterio.",
            )
            desc = order_section.checkbox(
                "Descendente",
                value=defaults["desc"],
                help="Muestra primero los valores más altos. Desactivalo para invertir el orden.",
            )

        with _section_card(form) as charts_section:
            charts_section.markdown("### 📈 Gráficos")
            charts_section.caption("Controlá cuántos elementos se visualizan en rankings y gráficos destacados.")
            top_n = charts_section.slider(
                "Top N",
                5,
                50,
                defaults["top_n"],
                step=5,
                help="Elige cuántos elementos ver en rankings y gráficos comparativos.",
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

        _render_filter_overview(overview_container, chips)

        with _section_card(form, extra_classes="control-panel__actions") as actions_section:
            actions_section.caption("Aplicá tus cambios o volvé a los valores originales.")
            action_cols = actions_section.columns(2)
            apply_btn = action_cols[0].form_submit_button("Aplicar")
            reset_btn = action_cols[1].form_submit_button("Reset")

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
        st.session_state.pop("selected_asset_types", None)
        st.rerun()

    if apply_btn:
        st.session_state.update(asdict(controls))
        st.session_state["controls_snapshot"] = asdict(controls)
        st.session_state[_FLASH_FLAG_KEY] = True
        st.session_state["selected_asset_types"] = list(controls.selected_types)
        _show_apply_feedback()

    snap = st.session_state.get("controls_snapshot")
    if snap:
        st.session_state["selected_asset_types"] = list(snap.get("selected_types", []))
        return Controls(**snap)
    st.session_state["selected_asset_types"] = list(controls.selected_types)
    return controls
