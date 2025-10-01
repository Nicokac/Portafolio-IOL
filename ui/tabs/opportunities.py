"""UI helpers for the opportunities tab."""
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


PRESET_FILTERS: Mapping[str, Mapping[str, object]] = {
    "Dividendos defensivos": {
        "min_market_cap": 1500,
        "max_pe": 22.0,
        "min_revenue_growth": 0.0,
        "max_payout": 70.0,
        "min_div_streak": 10,
        "min_cagr": 5.0,
        "min_eps_growth": 2.0,
        "min_buyback": 0.0,
        "include_latam": False,
        "include_technicals": False,
        "min_score_threshold": 65,
        "max_results": int(shared_settings.max_results),
        "sectors": ["Consumer Defensive", "Utilities"],
    },
    "Crecimiento balanceado": {
        "min_market_cap": 1000,
        "max_pe": 28.0,
        "min_revenue_growth": 12.0,
        "max_payout": 60.0,
        "min_div_streak": 5,
        "min_cagr": 6.0,
        "min_eps_growth": 8.0,
        "min_buyback": 1.0,
        "include_latam": True,
        "include_technicals": False,
        "min_score_threshold": 72,
        "max_results": int(shared_settings.max_results),
        "sectors": ["Technology", "Healthcare"],
    },
    "Recompras agresivas": {
        "min_market_cap": 500,
        "max_pe": 25.0,
        "min_revenue_growth": 5.0,
        "max_payout": 50.0,
        "min_div_streak": 3,
        "min_cagr": 4.0,
        "min_eps_growth": 6.0,
        "min_buyback": 3.0,
        "include_latam": True,
        "include_technicals": True,
        "min_score_threshold": 68,
        "max_results": int(shared_settings.max_results),
        "sectors": ["Financial Services", "Technology", "Industrials"],
    },
}


_CUSTOM_PRESETS_STATE_KEY = "custom_presets"
_PENDING_PRESET_STATE_KEY = "opportunities_pending_preset"


_INT_STATE_KEYS = {"min_market_cap", "min_div_streak", "max_results", "min_score_threshold"}
_FLOAT_STATE_KEYS = {
    "max_pe",
    "min_revenue_growth",
    "max_payout",
    "min_cagr",
    "min_eps_growth",
    "min_buyback",
}
_BOOL_STATE_KEYS = {"include_latam", "include_technicals"}


def _compute_default_widget_values() -> dict[str, object]:
    default_score = shared_settings.min_score_threshold
    normalized_default_score = max(0, min(100, int(default_score)))
    return {
        "min_market_cap": 500,
        "max_pe": 25.0,
        "min_revenue_growth": 5.0,
        "max_payout": 80.0,
        "min_div_streak": 5,
        "min_cagr": 4.0,
        "min_eps_growth": 0.0,
        "min_buyback": 0.0,
        "include_latam": True,
        "include_technicals": False,
        "min_score_threshold": normalized_default_score,
        "max_results": int(shared_settings.max_results),
        "sectors": [],
    }


def _get_custom_presets() -> dict[str, Mapping[str, object]]:
    presets = st.session_state.setdefault(_CUSTOM_PRESETS_STATE_KEY, {})
    if not isinstance(presets, dict):
        presets = {}
        st.session_state[_CUSTOM_PRESETS_STATE_KEY] = presets
    return presets


def _iter_available_presets() -> Sequence[str]:
    base_options: list[str] = ["Personalizado", *PRESET_FILTERS.keys()]
    custom_presets = _get_custom_presets()
    if custom_presets:
        base_options.extend(sorted(custom_presets))
    return base_options


def _resolve_preset_definition(preset_name: str | None) -> Mapping[str, object] | None:
    if not preset_name or preset_name == "Personalizado":
        return None
    preset = PRESET_FILTERS.get(preset_name)
    if preset is None:
        preset = _get_custom_presets().get(preset_name)
    return preset


def _apply_preset(preset_name: str | None) -> None:
    if not preset_name:
        return
    preset = _resolve_preset_definition(preset_name)
    if not preset:
        return
    defaults = _compute_default_widget_values()
    for key, default_value in defaults.items():
        value = preset.get(key, default_value)
        if key in _INT_STATE_KEYS:
            if key == "min_score_threshold":
                normalized_value = max(0, min(100, int(value)))
                st.session_state[key] = normalized_value
            else:
                st.session_state[key] = int(value)
        elif key in _FLOAT_STATE_KEYS:
            st.session_state[key] = float(value)
        elif key in _BOOL_STATE_KEYS:
            st.session_state[key] = bool(value)
        elif key == "sectors":
            sectors = [
                sector
                for sector in value
                if sector in _SECTOR_OPTIONS
            ]
            st.session_state[key] = sectors
        else:
            st.session_state[key] = value


def _handle_preset_change() -> None:
    preset_name = st.session_state.get("opportunities_preset")
    _apply_preset(preset_name)
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


def _slugify_label(label: str) -> str:
    normalized = [
        character.lower() if character.isalnum() else "_"
        for character in str(label)
    ]
    slug = "".join(normalized).strip("_")
    return slug or "preset"


def _render_screening_block(
    result: object,
    *,
    heading: str | None = None,
    download_key: str = "download_opportunities_csv",
    empty_message: str | None = None,
) -> None:
    table, notes, source = _extract_result(result)
    if heading:
        st.subheader(heading)

    if table is None or table.empty:
        st.info(empty_message or "No se encontraron oportunidades con los filtros seleccionados.")
    else:
        display_table = table.copy()
        link_column = "Yahoo Finance Link"
        column_config: dict[str, st.column_config.Column | st.column_config.LinkColumn] | None = None
        column_order: list[str] | None = None

        required_columns = ("ticker", link_column)
        if set(required_columns).issubset(display_table.columns):

            def _resolve_yahoo_url(row: pd.Series) -> str | None:
                raw_url = row.get(link_column)
                ticker_value = row.get("ticker")

                url = str(raw_url).strip() if raw_url else ""
                if not url and ticker_value:
                    url = f"https://finance.yahoo.com/quote/{ticker_value}"
                return url or None

            display_table[link_column] = display_table.apply(
                _resolve_yahoo_url, axis=1
            )
            column_config = {
                link_column: st.column_config.LinkColumn(
                    label="Yahoo Finance Link",
                    help="Abrí la ficha del activo en Yahoo Finance.",
                    display_text=None,
                )
            }
            column_order = [
                *required_columns,
                *[
                    column
                    for column in display_table.columns
                    if column not in required_columns
                ],
            ]

        st.dataframe(
            display_table,
            use_container_width=True,
            column_config=column_config,
            column_order=column_order,
        )

        csv_payload = display_table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Descargar resultados (.csv)",
            data=csv_payload,
            file_name="oportunidades.csv",
            mime="text/csv",
            key=download_key,
        )

    has_stub_or_fallback_note = any(
        isinstance(note, str)
        and note.strip().startswith(
            (
                "⚠️ Yahoo no disponible — Causa:",
                "⚠️ Stub procesó",
                "ℹ️ Stub procesó",
            )
        )
        for note in notes
    )

    if notes:
        st.markdown("### Notas del screening")
        for note in notes:
            st.markdown(_format_note(note))

    if source == "stub":
        if not has_stub_or_fallback_note:
            st.caption(
                shared_notes.format_note(
                    "⚠️ Resultados simulados (Yahoo no disponible)"
                )
            )
    else:
        st.caption("Resultados obtenidos de Yahoo Finance")
    st.caption(
        shared_notes.format_note(
            "ℹ️ Los filtros avanzados de capitalización, P/E, crecimiento de ingresos, payout, racha de dividendos, CAGR, crecimiento de EPS, buybacks e inclusión de Latam requieren datos en vivo de Yahoo."
        )
    )


def render_opportunities_tab() -> None:
    """Renderiza la pestaña de oportunidades."""
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

    st.header(f"🚀 Empresas con oportunidad · v{__version__}")
    with st.expander("¿Qué significa cada métrica?"):
        st.markdown(
            """
            - **Payout:** porcentaje de las ganancias que se reparte como dividendo. Ejemplo: con un payout del 60 %, una empresa distribuye US$0,60 por cada dólar de utilidad.
            - **EPS (Earnings Per Share):** ganancias por acción. Si una firma genera US$5 millones y tiene 1 millón de acciones, su EPS es US$5.
            - **Crecimiento de ingresos:** variación interanual de ventas. Un aumento de US$100 a US$112 implica un crecimiento del 12 %.
            - **Racha de dividendos:** cantidad de años consecutivos pagando dividendos. Una racha de 7 significa pagos sin interrupciones durante siete ejercicios.
            - **CAGR de dividendos:** crecimiento anual compuesto del dividendo. Pasar de US$1 a US$1,50 en cinco años implica un CAGR cercano al 8 %.
            - **Buybacks:** recompras netas que reducen el flotante. Un buyback del 2 % indica que la empresa retiró 2 de cada 100 acciones en circulación.
            - **Score compuesto:** puntaje de 0 a 100 que combina valuación, crecimiento, dividendos y técnicos; por ejemplo, un score de 85 señala atributos superiores al umbral típico de 80.
            """
        )
    st.caption(
        "Explorá screenings cuantitativos para detectar compañías "
        "que podrían presentar oportunidades de inversión."
    )

    defaults = _compute_default_widget_values()
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)
    st.session_state.setdefault("opportunities_preset", "Personalizado")
    _get_custom_presets()

    pending_preset = st.session_state.pop(_PENDING_PRESET_STATE_KEY, None)
    if pending_preset:
        st.session_state["opportunities_preset"] = pending_preset
        _apply_preset(pending_preset)

    available_presets = list(_iter_available_presets())
    if st.session_state["opportunities_preset"] not in available_presets:
        st.session_state["opportunities_preset"] = "Personalizado"

    st.selectbox(
        "Perfil recomendado",
        options=available_presets,
        key="opportunities_preset",
        on_change=_handle_preset_change,
        help="Aplicá perfiles sugeridos con criterios preconfigurados para iniciar el screening.",
    )

    with st.expander("Parámetros del screening", expanded=True):
        min_market_cap = st.number_input(
            "Capitalización mínima (US$ MM)",
            min_value=0,
            value=int(st.session_state["min_market_cap"]),
            step=50,
            help="Filtra empresas con capitalización menor al umbral indicado.",
            key="min_market_cap",
        )
        max_pe = st.number_input(
            "P/E máximo",
            min_value=0.0,
            value=float(st.session_state["max_pe"]),
            step=0.5,
            help="Limita el ratio precio/ganancias máximo permitido.",
            key="max_pe",
        )
        min_growth = st.number_input(
            "Crecimiento ingresos mínimo (%)",
            min_value=-100.0,
            value=float(st.session_state["min_revenue_growth"]),
            step=1.0,
            help="Requiere un crecimiento anual de ingresos superior al valor indicado.",
            key="min_revenue_growth",
        )
        max_payout = st.number_input(
            "Payout máximo (%)",
            min_value=0.0,
            max_value=200.0,
            value=float(st.session_state["max_payout"]),
            step=1.0,
            help="Descarta empresas con payout ratio superior al valor indicado (predeterminado: 80%).",
            key="max_payout",
        )
        min_div_streak = st.slider(
            "Racha mínima de dividendos (años)",
            min_value=0,
            max_value=30,
            value=int(st.session_state["min_div_streak"]),
            help="Exige al menos la cantidad seleccionada de años consecutivos pagando dividendos (predeterminado: 5 años).",
            key="min_div_streak",
        )
        min_cagr = st.number_input(
            "CAGR mínimo de dividendos (%)",
            min_value=-50.0,
            value=float(st.session_state["min_cagr"]),
            step=0.5,
            help="Filtra compañías con crecimiento anual compuesto inferior al valor indicado (predeterminado: 4%).",
            key="min_cagr",
        )
        min_eps_growth = st.number_input(
            "Crecimiento mínimo de EPS (%)",
            min_value=-100.0,
            value=float(st.session_state["min_eps_growth"]),
            step=0.5,
            help="Requiere que el EPS proyectado supere al actual por al menos el porcentaje indicado (predeterminado: 0%).",
            key="min_eps_growth",
        )
        min_buyback = st.number_input(
            "Buyback mínimo (%)",
            min_value=-100.0,
            value=float(st.session_state["min_buyback"]),
            step=0.5,
            help="Exige una reducción mínima en el flotante. Usa valores positivos para requerir recompras netas.",
            key="min_buyback",
        )
        include_latam = st.checkbox(
            "Incluir Latam",
            value=bool(st.session_state["include_latam"]),
            help="Extiende el screening a emisores listados en Latinoamérica.",
            key="include_latam",
        )
        include_technicals = st.checkbox(
            "Incluir indicadores técnicos",
            value=bool(st.session_state["include_technicals"]),
            help="Agrega columnas con RSI y medias móviles de 50 y 200 ruedas.",
            key="include_technicals",
        )
        min_score_threshold = st.slider(
            "Score mínimo",
            min_value=0,
            max_value=100,
            value=int(st.session_state["min_score_threshold"]),
            step=1,
            help="Define el puntaje mínimo requerido para considerar un candidato.",
            key="min_score_threshold",
        )
        max_results = st.number_input(
            "Máximo de resultados",
            min_value=1,
            value=int(st.session_state["max_results"]),
            step=1,
            help="Limita la cantidad de oportunidades mostradas al tope indicado.",
            key="max_results",
        )
        sectors = st.multiselect(
            "Sectores",
            options=_SECTOR_OPTIONS,
            default=st.session_state.get("sectors", []),
            help="Limitá los resultados a los sectores seleccionados.",
            key="sectors",
        )

    current_filters: dict[str, object] = {
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
        "sectors": list(sectors),
    }

    st.markdown(
        "Seleccioná los parámetros deseados y presioná **Buscar oportunidades** para ejecutar "
        "el análisis con la configuración estable."
    )

    with st.expander("Gestionar presets personalizados", expanded=True):
        with st.form("custom_preset_save_form"):
            preset_name_input = st.text_input(
                "Nombre del preset",
                key="custom_preset_name_input",
                help="Guardá la configuración actual para reutilizarla en esta sesión.",
            )
            save_preset = st.form_submit_button("Guardar preset personalizado")
            if save_preset:
                preset_name = preset_name_input.strip()
                if not preset_name:
                    st.warning("Indicá un nombre para guardar el preset.")
                elif preset_name in PRESET_FILTERS:
                    st.warning("El nombre ingresado coincide con un preset predefinido. Elegí otro identificador.")
                else:
                    payload = dict(current_filters)
                    custom_presets = dict(_get_custom_presets())
                    custom_presets[preset_name] = payload
                    st.session_state[_CUSTOM_PRESETS_STATE_KEY] = custom_presets
                    st.session_state[_PENDING_PRESET_STATE_KEY] = preset_name
                    st.success(f"Preset '{preset_name}' guardado correctamente.")

        custom_presets = _get_custom_presets()
        if custom_presets:
            with st.form("custom_preset_select_form"):
                options = sorted(custom_presets)
                preset_selector_default = st.session_state.get("opportunities_preset")
                if preset_selector_default not in options:
                    preset_selector_default = options[0]
                selected_custom = st.selectbox(
                    "Preset personalizado guardado",
                    options=options,
                    index=options.index(preset_selector_default),
                    key="custom_preset_selector",
                )
                apply_preset = st.form_submit_button("Aplicar preset guardado")
                if apply_preset and selected_custom:
                    st.session_state[_PENDING_PRESET_STATE_KEY] = selected_custom
                    st.success(f"Preset '{selected_custom}' aplicado correctamente.")

    def _resolve_generate_callable() -> "Callable[[Mapping[str, object]], Mapping[str, object]] | None":
        try:
            from controllers.opportunities import generate_opportunities_report
        except ImportError as err:  # pragma: no cover - fallback when controller missing
            st.error(
                "El módulo de oportunidades aún no está disponible. "
                "Contactá al equipo si el problema persiste."
            )
            st.caption(f"Detalles técnicos: {err}")
            return None

        return generate_opportunities_report

    results_rendered = False

    if st.button(
        "Buscar oportunidades",
        key="search_opportunities",
        type="primary",
        use_container_width=True,
    ):
        generate_callable = _resolve_generate_callable()
        if generate_callable is None:
            return

        params = dict(current_filters)
        if not params.get("sectors"):
            params.pop("sectors", None)

        with st.spinner("Generando screening de oportunidades..."):
            result = generate_callable(params)

        _render_screening_block(result, heading="Resultados del screening")
        results_rendered = True

    available_presets = list(_iter_available_presets())
    st.session_state.setdefault("preset_comparison_left", "Personalizado")
    st.session_state.setdefault("preset_comparison_right", "Personalizado")
    if st.session_state["preset_comparison_left"] not in available_presets:
        st.session_state["preset_comparison_left"] = available_presets[0]
    if st.session_state["preset_comparison_right"] not in available_presets:
        st.session_state["preset_comparison_right"] = available_presets[0]

    comparison_results: list[tuple[str, Mapping[str, object], object]] = []
    with st.form("compare_presets_form"):
        left_selection = st.selectbox(
            "Preset columna izquierda",
            options=available_presets,
            key="preset_comparison_left",
            help="Elegí el preset que se mostrará en la primera columna.",
        )
        right_selection = st.selectbox(
            "Preset columna derecha",
            options=available_presets,
            key="preset_comparison_right",
            help="Elegí el preset que se mostrará en la segunda columna.",
        )
        compare_submit = st.form_submit_button("Comparar presets")

    if compare_submit:
        generate_callable = _resolve_generate_callable()
        if generate_callable is None:
            return

        selections = [
            ("Columna izquierda", left_selection),
            ("Columna derecha", right_selection),
        ]
        for index, (title, preset_name) in enumerate(selections):
            preset_definition = _resolve_preset_definition(preset_name)
            if preset_definition is None:
                filters_payload = dict(current_filters)
            else:
                filters_payload = dict(preset_definition)

            with st.spinner(
                f"Generando screening para '{preset_name or 'Personalizado'}'..."
            ):
                comparison_results.append(
                    (
                        preset_name or "Personalizado",
                        filters_payload,
                        generate_callable(filters_payload),
                    )
                )

        if comparison_results:
            st.markdown("### Comparativa de presets")
            columns = st.columns(len(comparison_results))
            for idx, (column, (preset_label, _, result)) in enumerate(
                zip(columns, comparison_results)
            ):
                with column:
                    slug = _slugify_label(preset_label)
                    _render_screening_block(
                        result,
                        heading=f"Resultados: {preset_label}",
                        download_key=f"download_opportunities_csv_{slug}_{idx}",
                        empty_message="No se encontraron oportunidades para este preset.",
                    )
            results_rendered = True

    if not results_rendered:
        st.info("El screening se ejecuta manualmente para evitar demoras innecesarias.")
