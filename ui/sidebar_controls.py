# ui\sidebar_controls.py
from __future__ import annotations
from dataclasses import asdict
import streamlit as st
from domain.models import Controls
from infrastructure.cache import cache

def render_sidebar(all_symbols: list[str], available_types: list[str]) -> Controls:
    st.sidebar.header("🎛️ Controles")

    all_symbols = list(all_symbols or [])
    available_types = list(available_types or [])

    defaults = {
        "refresh_secs": cache.session_state.get("refresh_secs", 30),
        "hide_cash":    cache.session_state.get("hide_cash", True),
        "show_usd":     cache.session_state.get("show_usd", False),
        "order_by":     cache.session_state.get("order_by", "valor_actual"),
        "desc":         cache.session_state.get("desc", True),
        "top_n":        cache.session_state.get("top_n", 20),
        "selected_syms": cache.session_state.get("selected_syms", all_symbols),
        "selected_types": cache.session_state.get("selected_types", available_types),
        "symbol_query": cache.session_state.get("symbol_query", ""),
    }

    order_options = ["valor_actual", "pl", "pl_%", "pl_d", "chg_%", "costo", "ultimo", "cantidad", "simbolo"]
    order_index = order_options.index(defaults["order_by"]) if defaults["order_by"] in order_options else 0

    with st.sidebar.form("controls_form"):
        st.markdown("### ⏱️ Actualización")
        st.caption("Configura la frecuencia de refresco.")
        refresh_secs = st.slider("Intervalo (seg)", 5, 120, defaults["refresh_secs"], step=5)

        st.markdown("### 🔍 Filtros")
        st.caption("Limita los activos que se muestran.")
        hide_cash = st.checkbox("Ocultar IOLPORA / PARKING", value=defaults["hide_cash"])
        symbol_query = st.text_input("Buscar símbolo", value=defaults["symbol_query"], placeholder="p.ej. NVDA")
        selected_syms = st.multiselect(
            "Filtrar por símbolo",
            all_symbols,
            default=[s for s in (defaults["selected_syms"] or []) if s in all_symbols] or all_symbols,
        )
        selected_types = st.multiselect(
            "Filtrar por tipo",
            available_types,
            default=[t for t in (defaults["selected_types"] or available_types) if t in available_types],
        )

        st.markdown("### 💱 Moneda")
        st.caption("Muestra valores en pesos o en USD CCL.")
        show_usd = st.toggle("Mostrar valores en USD CCL", value=defaults["show_usd"])

        st.markdown("### ↕️ Orden")
        st.caption("Define el orden de la tabla.")
        order_by = st.selectbox("Ordenar por", order_options, index=order_index)
        desc = st.checkbox("Descendente", value=defaults["desc"])

        st.markdown("### 📈 Gráficos")
        st.caption("Cantidad de elementos en las visualizaciones.")
        top_n = st.slider("Top N", 5, 50, defaults["top_n"], step=5)

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
            cache.session_state.pop(k, None)
        cache.session_state["controls_snapshot"] = None
        st.rerun()

    if apply_btn:
        cache.session_state.update(asdict(controls))
        cache.session_state["controls_snapshot"] = asdict(controls)

    snap = cache.session_state.get("controls_snapshot")
    if snap:
        return Controls(**snap)
    return controls
