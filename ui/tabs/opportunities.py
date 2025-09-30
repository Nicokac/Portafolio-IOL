"""UI helpers for the experimental opportunities tab."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import pandas as pd
import streamlit as st

import shared.settings as shared_settings
from shared.version import __version__
from shared.ui import notes as shared_notes

_SECTOR_OPTIONS: Sequence[str] = (
    "Basic Materials",
    "Communication Services",
    "Consumer Cyclical",
    "Consumer Defensive",
    "Energy",
    "Financial Services",
    "Healthcare",
    "Industrials",
    "Real Estate",
    "Technology",
    "Utilities",
)
def _format_note(note: str) -> str:
    """Format backend notes based on their severity category.

    Las categorías disponibles son:

    * ``warning`` → mensajes relevantes que requieren atención inmediata.
    * ``info`` → notas informativas sin énfasis.
    * ``success`` → confirmaciones o resultados positivos destacados.
    * ``error`` → fallas o interrupciones reportadas por el backend.

    Cada categoría define iconografía y énfasis para reutilizar el mismo
    comportamiento en otras pestañas que consuman el helper.
    """

    return shared_notes.format_note(note)


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


def _extract_result(result: object) -> tuple[pd.DataFrame | None, list[str], str]:
    source = "yahoo"
    if isinstance(result, Mapping):
        table = None
        for key in ("table", "data", "df"):
            if key in result and result[key] is not None:
                table = result[key]
                break
        notes = None
        for key in ("notes", "messages", "warnings"):
            if key in result and result[key]:
                notes = result[key]
                break
        if "source" in result and result["source"]:
            source = str(result["source"])
        return _normalize_table(table), _normalize_notes(notes), source
    if isinstance(result, Sequence) and not isinstance(result, (str, bytes, bytearray)) and len(result) >= 2:
        table, notes = result[:2]  # type: ignore[assignment]
        if len(result) >= 3 and result[2]:
            source = str(result[2])
        return _normalize_table(table), _normalize_notes(notes), source
    return _normalize_table(result), [], source


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
        max_payout = st.number_input(
            "Payout máximo (%)",
            min_value=0.0,
            max_value=200.0,
            value=80.0,
            step=1.0,
            help="Descarta empresas con payout ratio superior al valor indicado (predeterminado: 80%).",
        )
        min_div_streak = st.slider(
            "Racha mínima de dividendos (años)",
            min_value=0,
            max_value=30,
            value=5,
            help="Exige al menos la cantidad seleccionada de años consecutivos pagando dividendos (predeterminado: 5 años).",
        )
        min_cagr = st.number_input(
            "CAGR mínimo de dividendos (%)",
            min_value=-50.0,
            value=4.0,
            step=0.5,
            help="Filtra compañías con crecimiento anual compuesto inferior al valor indicado (predeterminado: 4%).",
        )
        min_eps_growth = st.number_input(
            "Crecimiento mínimo de EPS (%)",
            min_value=-100.0,
            value=0.0,
            step=0.5,
            help="Requiere que el EPS proyectado supere al actual por al menos el porcentaje indicado (predeterminado: 0%).",
        )
        min_buyback = st.number_input(
            "Buyback mínimo (%)",
            min_value=-100.0,
            value=0.0,
            step=0.5,
            help="Exige una reducción mínima en el flotante. Usa valores positivos para requerir recompras netas.",
        )
        include_latam = st.checkbox(
            "Incluir Latam",
            value=True,
            help="Extiende el screening a emisores listados en Latinoamérica.",
        )
        include_technicals = st.checkbox(
            "Incluir indicadores técnicos",
            value=False,
            help="Agrega columnas con RSI y medias móviles de 50 y 200 ruedas.",
        )
        default_score = shared_settings.min_score_threshold
        normalized_default_score = max(0, min(100, int(default_score)))
        min_score_threshold = st.slider(
            "Score mínimo",
            min_value=0,
            max_value=100,
            value=normalized_default_score,
            step=1,
            help="Define el puntaje mínimo requerido para considerar un candidato.",
        )
        max_results = st.number_input(
            "Máximo de resultados",
            min_value=1,
            value=int(shared_settings.max_results),
            step=1,
            help="Limita la cantidad de oportunidades mostradas al tope indicado.",
        )
        sectors = st.multiselect(
            "Sectores",
            options=_SECTOR_OPTIONS,
            help="Limitá los resultados a los sectores seleccionados.",
        )

    st.markdown(
        "Seleccioná los parámetros deseados y presioná **Buscar oportunidades** para ejecutar "
        "el análisis en modo beta."
    )

    if st.button(
        "Buscar oportunidades",
        key="search_opportunities",
        type="primary",
        use_container_width=True,
    ):
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
            "max_payout": float(max_payout),
            "min_div_streak": int(min_div_streak),
            "min_cagr": float(min_cagr),
            "min_eps_growth": float(min_eps_growth),
            "min_buyback": float(min_buyback),
            "include_latam": bool(include_latam),
            "include_technicals": bool(include_technicals),
            "min_score_threshold": float(min_score_threshold),
            "max_results": int(max_results),
        }
        if sectors:
            params["sectors"] = list(sectors)

        with st.spinner("Generando screening de oportunidades..."):
            result = generate_opportunities_report(params)

        table, notes, source = _extract_result(result)

        if table is None or table.empty:
            st.info("No se encontraron oportunidades con los filtros seleccionados.")
        else:
            st.subheader("Resultados del screening")
            st.dataframe(table, use_container_width=True)

        if source == "stub":
            st.caption(shared_notes.format_note("⚠️ Resultados simulados (Yahoo no disponible)"))
        else:
            st.caption("Resultados obtenidos de Yahoo Finance")
        st.caption(
            shared_notes.format_note(
                "ℹ️ Los filtros avanzados de capitalización, P/E, crecimiento de ingresos, payout, racha de dividendos, CAGR, crecimiento de EPS, buybacks e inclusión de Latam requieren datos en vivo de Yahoo."
            )
        )

        if notes:
            st.markdown("### Notas")
            for note in notes:
                st.markdown(f"- {_format_note(note)}")
    else:
        st.info("El screening se ejecuta manualmente para evitar demoras innecesarias.")
