"""UI helpers for the experimental opportunities tab."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

from shared.version import __version__


def _normalize_notes(notes: object) -> list[str]:
    if notes is None:
        return []
    if isinstance(notes, str):
        return [notes]
    if isinstance(notes, Mapping):
        return [str(value) for value in notes.values() if value]
    if isinstance(notes, Iterable):
        normalized: list[str] = []
        for item in notes:
            if item is None:
                continue
            normalized.append(str(item))
        return normalized
    return [str(notes)]


def _normalize_table(table: object) -> pd.DataFrame | None:
    if table is None:
        return None
    if isinstance(table, pd.DataFrame):
        return table
    try:
        return pd.DataFrame(table)
    except Exception:  # pragma: no cover - fallback for unexpected payloads
        return None


def _extract_result(result: object) -> tuple[pd.DataFrame | None, list[str]]:
    if isinstance(result, Mapping):
        table = result.get("table") or result.get("data") or result.get("df")
        notes = result.get("notes") or result.get("messages") or result.get("warnings")
        return _normalize_table(table), _normalize_notes(notes)
    if isinstance(result, Sequence) and len(result) == 2:
        table, notes = result  # type: ignore[assignment]
        return _normalize_table(table), _normalize_notes(notes)
    return _normalize_table(result), []


def render_opportunities_tab() -> None:
    """Renderiza la pestaña experimental de oportunidades."""
    required_attrs = (
        "header",
        "caption",
        "expander",
        "number_input",
        "checkbox",
        "button",
        "spinner",
        "dataframe",
        "info",
        "markdown",
    )
    if not all(hasattr(st, attr) for attr in required_attrs):  # pragma: no cover - only for test stubs
        return

    st.header(f"🚀 Empresas con oportunidad · beta {__version__}")
    st.caption(
        "Explorá screenings cuantitativos experimentales para detectar compañías "
        "que podrían presentar oportunidades de inversión."
    )

    with st.expander("Parámetros del screening", expanded=True):
        min_market_cap = st.number_input(
            "Capitalización mínima (US$ MM)",
            min_value=0,
            value=500,
            step=50,
            help="Filtra empresas con capitalización menor al umbral indicado.",
        )
        max_pe = st.number_input(
            "P/E máximo",
            min_value=0.0,
            value=25.0,
            step=0.5,
            help="Limita el ratio precio/ganancias máximo permitido.",
        )
        min_growth = st.number_input(
            "Crecimiento ingresos mínimo (%)",
            min_value=-100.0,
            value=5.0,
            step=1.0,
            help="Requiere un crecimiento anual de ingresos superior al valor indicado.",
        )
        include_latam = st.checkbox(
            "Incluir Latam",
            value=True,
            help="Extiende el screening a emisores listados en Latinoamérica.",
        )

    st.markdown(
        "Seleccioná los parámetros deseados y presioná **Buscar oportunidades** para ejecutar "
        "el análisis en modo beta."
    )

    if st.button("Buscar oportunidades", type="primary", use_container_width=True):
        try:
            from controllers.opportunities import generate_opportunities_report
        except ImportError as err:  # pragma: no cover - fallback when controller missing
            st.error(
                "El módulo de oportunidades aún no está disponible. "
                "Contactá al equipo si el problema persiste."
            )
            st.caption(f"Detalles técnicos: {err}")
            return

        params = {
            "min_market_cap": float(min_market_cap),
            "max_pe": float(max_pe),
            "min_revenue_growth": float(min_growth),
            "include_latam": bool(include_latam),
        }

        with st.spinner("Generando screening de oportunidades..."):
            result = generate_opportunities_report(params)

        table, notes = _extract_result(result)

        if table is None or table.empty:
            st.info("No se encontraron oportunidades con los filtros seleccionados.")
        else:
            st.subheader("Resultados del screening")
            st.dataframe(table, use_container_width=True)

        if notes:
            st.markdown("### Notas")
            for note in notes:
                st.markdown(f"- {note}")
    else:
        st.info("El screening se ejecuta manualmente para evitar demoras innecesarias.")
