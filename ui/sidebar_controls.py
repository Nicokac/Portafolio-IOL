# ui\sidebar_controls.py
from __future__ import annotations

import html
import os
from contextlib import contextmanager
from dataclasses import asdict
from textwrap import shorten

import streamlit as st

from domain.models import Controls
from ui.actions import render_action_menu

_FLASH_FLAG_KEY = "_sidebar_controls_flash_flag"
_SYMBOLS_STATE_KEY = "_sidebar_all_symbols"
_TYPES_STATE_KEY = "_sidebar_available_types"


def _store_reference_data(symbols: list[str], types: list[str]) -> None:
    try:
        st.session_state[_SYMBOLS_STATE_KEY] = list(symbols)
        st.session_state[_TYPES_STATE_KEY] = list(types)
    except Exception:  # pragma: no cover - session state may be read-only
        pass


def _get_reference_data() -> tuple[list[str], list[str]]:
    try:
        symbols = list(st.session_state.get(_SYMBOLS_STATE_KEY, []))
    except Exception:  # pragma: no cover - defensive guard
        symbols = []
    try:
        types = list(st.session_state.get(_TYPES_STATE_KEY, []))
    except Exception:  # pragma: no cover - defensive guard
        types = []
    return symbols, types


def _build_control_defaults(
    all_symbols: list[str],
    available_types: list[str],
) -> dict[str, object]:
    selected_types_state = st.session_state.get("selected_types")
    if selected_types_state is None:
        selected_types_state = st.session_state.get("selected_asset_types")

    try:
        st.session_state["hide_cash"] = False
    except Exception:  # pragma: no cover - session state may be read-only
        try:
            st.session_state.pop("hide_cash", None)
        except Exception:  # pragma: no cover - defensive guard
            pass

    return {
        "refresh_secs": st.session_state.get("refresh_secs", 30),
        "show_usd": st.session_state.get("show_usd", False),
        "order_by": st.session_state.get("order_by", "valor_actual"),
        "desc": st.session_state.get("desc", True),
        "top_n": st.session_state.get("top_n", 20),
        "selected_syms": st.session_state.get("selected_syms", all_symbols),
        "selected_types": selected_types_state if selected_types_state is not None else available_types,
        "symbol_query": st.session_state.get("symbol_query", ""),
    }


def _finalize_controls(controls: Controls) -> Controls:
    snap = st.session_state.get("controls_snapshot")
    if isinstance(snap, dict):
        snap = dict(snap)
        snap["hide_cash"] = False
        selected = list(snap.get("selected_types", []))
        try:
            st.session_state["selected_asset_types"] = selected
        except Exception:  # pragma: no cover - defensive guard
            pass
        try:
            st.session_state["hide_cash"] = False
        except Exception:  # pragma: no cover - defensive guard
            pass
        return Controls(**snap)

    try:
        st.session_state["selected_asset_types"] = list(controls.selected_types)
    except Exception:  # pragma: no cover - defensive guard
        pass
    try:
        st.session_state["hide_cash"] = False
    except Exception:  # pragma: no cover - defensive guard
        pass
    return controls


def get_active_controls(
    all_symbols: list[str] | None = None,
    available_types: list[str] | None = None,
) -> Controls:
    stored_symbols, stored_types = _get_reference_data()
    symbols = list(all_symbols) if all_symbols is not None else stored_symbols
    types = list(available_types) if available_types is not None else stored_types
    defaults = _build_control_defaults(symbols, types)

    controls = Controls(
        refresh_secs=int(defaults["refresh_secs"]),
        hide_cash=False,
        show_usd=bool(defaults["show_usd"]),
        order_by=str(defaults["order_by"]),
        desc=bool(defaults["desc"]),
        top_n=int(defaults["top_n"]),
        selected_syms=list(defaults["selected_syms"] or symbols),
        selected_types=list(defaults["selected_types"] or types),
        symbol_query=str(defaults["symbol_query"] or "").strip().upper(),
    )
    return _finalize_controls(controls)


def get_controls_reference_data() -> tuple[list[str], list[str]]:
    """Return cached symbols/types used to build the controls panel."""

    return _get_reference_data()
_SYMBOLS_STATE_KEY = "_sidebar_all_symbols"
_TYPES_STATE_KEY = "_sidebar_available_types"


def _show_apply_feedback() -> None:
    message = "Filtros aplicados"
    if hasattr(st, "toast"):
        st.toast(message, icon="✅")
    elif hasattr(st, "info"):
        st.info(message)


def _active_filter_chips(
    *,
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


def render_controls_panel(
    all_symbols: list[str],
    available_types: list[str],
    *,
    container=None,
) -> Controls:
    host = container if container is not None else st.sidebar

    all_symbols = list(all_symbols or [])
    available_types = list(available_types or [])
    _store_reference_data(all_symbols, available_types)

    flash_active = bool(st.session_state.get(_FLASH_FLAG_KEY))
    defaults = _build_control_defaults(all_symbols, available_types)

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
        host.markdown("### 🎯 Controles del portafolio")
        if hasattr(host, "caption"):
            host.caption("Ajustá filtros, orden y visualizaciones desde el centro de monitoreo.")

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
        hide_cash=False,
        show_usd=show_usd,
        order_by=order_by,
        desc=desc,
        top_n=top_n,
        selected_syms=selected_syms,
        selected_types=selected_types,
        symbol_query=(symbol_query or "").strip().upper(),
    )

    if reset_btn:
        for key in asdict(controls).keys():
            st.session_state.pop(key, None)
        st.session_state["controls_snapshot"] = None
        st.session_state.pop("selected_asset_types", None)
        st.rerun()

    if apply_btn:
        snapshot = asdict(controls)
        st.session_state.update(snapshot)
        st.session_state["controls_snapshot"] = snapshot
        st.session_state[_FLASH_FLAG_KEY] = True
        st.session_state["selected_asset_types"] = list(controls.selected_types)
        _show_apply_feedback()

    return _finalize_controls(controls)


def _environment_label() -> str:
    for key in ("PORTAFOLIO_ENV", "APP_ENV", "ENVIRONMENT"):
        raw = os.environ.get(key)
        if raw:
            label = raw.replace("_", " ").strip()
            return label.title() if label else raw
    return "Operativo"


def _render_sidebar_shell(host) -> None:
    if hasattr(host, "markdown"):
        host.markdown("## 📈 Portafolio IOL")
        host.caption("Monitoreo en vivo en modo solo lectura.")

    username = st.session_state.get("IOL_USERNAME")
    session_id = st.session_state.get("session_id")

    if username:
        host.markdown(f"**Sesión activa:** {html.escape(str(username))}")
    else:
        host.markdown("_Sesión no autenticada._")

    env_label = _environment_label()
    host.caption(f"Entorno actual: {env_label}")
    if session_id:
        host.caption(f"ID de sesión: `{session_id}`")

    render_action_menu(container=host, show_refresh=False)


def render_sidebar(
    all_symbols: list[str],
    available_types: list[str],
    *,
    container=None,
) -> Controls:
    host = container if container is not None else st.sidebar
    all_symbols = list(all_symbols or [])
    available_types = list(available_types or [])
    _store_reference_data(all_symbols, available_types)
    _render_sidebar_shell(host)
    return get_active_controls(all_symbols, available_types)
