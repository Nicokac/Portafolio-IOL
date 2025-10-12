"""UI helpers for the opportunities tab."""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

import altair as alt
import pandas as pd
import streamlit as st

from infrastructure.market.yahoo_client import make_symbol_url
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
_ACTIVE_PRESET_STATE_KEY = "opportunities_active_preset"
_SUMMARY_STATE_KEY = "opportunities_summary"
_SUMMARY_EXPANDED_STATE_KEY = "opportunities_summary_expanded"
_SUMMARY_COMPACT_OVERRIDE_KEY = "opportunities_summary_compact_override"
_SUMMARY_COMPACT_BREAKPOINT = 768
_SUMMARY_SECTOR_SCROLL_THRESHOLD = 6
_LAST_RESULT_TABLE_STATE_KEY = "opportunities_last_result_table"
_LAST_MACRO_ENTRIES_STATE_KEY = "opportunities_last_macro_entries"


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


def _initialise_filter_state() -> None:
    defaults = _compute_default_widget_values()
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _get_custom_presets() -> dict[str, Mapping[str, object]]:
    presets = st.session_state.setdefault(_CUSTOM_PRESETS_STATE_KEY, {})
    if not isinstance(presets, dict):
        presets = {}
        st.session_state[_CUSTOM_PRESETS_STATE_KEY] = presets
    return presets


def _count_active_filters(filters: Mapping[str, object]) -> int:
    count = 0
    for value in filters.values():
        if value is None:
            continue
        if isinstance(value, Mapping) and not value:
            continue
        if isinstance(value, (list, tuple, set, frozenset)) and not value:
            continue
        count += 1
    return count


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
    if not preset_name or preset_name == "Personalizado":
        st.session_state[_ACTIVE_PRESET_STATE_KEY] = "Personalizado"
        return
    preset = _resolve_preset_definition(preset_name)
    if not preset:
        st.session_state[_ACTIVE_PRESET_STATE_KEY] = "Personalizado"
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
    st.session_state[_ACTIVE_PRESET_STATE_KEY] = preset_name


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


def _normalize_summary_value(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _normalize_summary_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalize_summary_value(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return _normalize_summary_value(value.item())
        except Exception:  # pragma: no cover - defensive path
            pass
    if isinstance(value, (int, float, str)):
        if isinstance(value, float) and pd.isna(value):
            return None
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:  # pragma: no cover - defensive path
        pass
    return value


def _normalize_summary_payload(
    summary: Mapping[str, object] | None,
) -> dict[str, object] | None:
    if not summary:
        return None
    return {str(key): _normalize_summary_value(value) for key, value in summary.items()}


def _extract_summary_payload(result: object) -> Mapping[str, object] | None:
    if isinstance(result, Mapping):
        summary = result.get("summary")
        if isinstance(summary, Mapping):
            return summary
        table = result.get("table")
    else:
        table = result
    if isinstance(table, pd.DataFrame):
        raw_summary = getattr(table, "attrs", {}).get("summary")
        if isinstance(raw_summary, Mapping):
            return raw_summary
    return None


def _format_integer(value: object) -> str:
    try:
        number = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    return f"{number:,}".replace(",", ".")


def _format_percentage(value: object) -> str:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "—"
    if pd.isna(numeric):
        return "—"
    return f"{numeric:.0%}"


def _resolve_compact_layout_override() -> bool | None:
    override = st.session_state.get(_SUMMARY_COMPACT_OVERRIDE_KEY)
    if isinstance(override, bool):
        return override
    return None


def _should_use_compact_layout() -> bool:
    override = _resolve_compact_layout_override()
    if override is not None:
        return override

    viewport_candidates = (
        "viewport_width",
        "client_viewport",
        "browser_info",
        "screen",
    )
    width: int | float | str | None = None
    for key in viewport_candidates:
        candidate = st.session_state.get(key)
        if isinstance(candidate, Mapping):
            for inner_key in ("width", "client_width", "page_width", "innerWidth"):
                if inner_key in candidate and candidate[inner_key] is not None:
                    width = candidate[inner_key]
                    break
            if width is not None:
                break
        elif candidate is not None:
            width = candidate
            break

    if width is not None:
        try:
            numeric_width = float(width)
        except (TypeError, ValueError):
            numeric_width = None
        if numeric_width is not None:
            return numeric_width <= _SUMMARY_COMPACT_BREAKPOINT

    return False


def _render_sector_score_chart(table: pd.DataFrame | None) -> None:
    if table is None or table.empty or not {"sector", "score_compuesto"}.issubset(table.columns):
        st.caption("Sin datos de score por sector disponibles.")
        return

    working = table[["sector", "score_compuesto"]].copy()
    working["sector"] = (
        working["sector"].astype("string").str.strip().replace("", pd.NA)
    )
    working["score_compuesto"] = pd.to_numeric(
        working["score_compuesto"], errors="coerce"
    )
    working = working.dropna(subset=["sector", "score_compuesto"])
    if working.empty:
        st.caption("Sin datos de score por sector disponibles.")
        return

    grouped = (
        working.groupby("sector", dropna=False)
        .agg(score_promedio=("score_compuesto", "mean"), cantidad=("score_compuesto", "size"))
        .reset_index()
    )
    grouped["score_promedio"] = grouped["score_promedio"].astype(float)
    grouped = grouped.sort_values("score_promedio", ascending=False)

    chart_builder = alt.Chart(grouped)
    if hasattr(chart_builder, "mark_bar"):
        chart_builder = chart_builder.mark_bar()
    else:  # pragma: no cover - fallback para dummies de testing
        chart_builder = chart_builder.mark_arc(innerRadius=40)

    if hasattr(alt, "X"):
        x_field = alt.X("score_promedio:Q", title="Score promedio")
        y_field = alt.Y("sector:N", sort="-x", title="Sector")
        color_field = alt.Color("sector:N", legend=None)
        tooltip_fields = [
            alt.Tooltip("sector:N", title="Sector"),
            alt.Tooltip("score_promedio:Q", title="Score promedio", format=".2f"),
            alt.Tooltip("cantidad:Q", title="Empresas"),
        ]
    else:  # pragma: no cover - fallback para entornos sin Altair completo
        x_field = "score_promedio:Q"
        y_field = {"field": "sector", "type": "nominal"}
        color_field = {"field": "sector", "type": "nominal"}
        tooltip_fields = [
            {"field": "sector", "type": "nominal", "title": "Sector"},
            {"field": "score_promedio", "type": "quantitative", "title": "Score promedio"},
            {"field": "cantidad", "type": "quantitative", "title": "Empresas"},
        ]

    chart = (
        chart_builder
        .encode(
            x=x_field,
            y=y_field,
            color=color_field,
            tooltip=tooltip_fields,
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True, key="opportunities_sector_chart")


def _render_macro_history_chart(entries: Sequence[Mapping[str, object]] | None) -> None:
    if not entries:
        st.caption("Sin historial macro disponible.")
        return

    df = pd.DataFrame(entries)
    if df.empty:
        st.caption("Sin historial macro disponible.")
        return

    df["value"] = pd.to_numeric(df.get("value"), errors="coerce")
    df = df.dropna(subset=["value"])
    if df.empty:
        st.caption("Sin historial macro disponible.")
        return

    df["sector"] = df.get("sector").astype("string").fillna("General")
    timestamp = pd.to_datetime(df.get("as_of"), errors="coerce")
    if "run_ts" in df.columns:
        fallback_ts = pd.to_datetime(df["run_ts"], unit="s", errors="coerce")
        timestamp = timestamp.fillna(fallback_ts)
    df["timestamp"] = timestamp
    df = df.dropna(subset=["timestamp"])
    if df.empty:
        st.caption("Sin historial macro disponible.")
        return

    df = df.sort_values("timestamp")
    chart_builder = alt.Chart(df)
    if hasattr(chart_builder, "mark_line"):
        chart_builder = chart_builder.mark_line(point=True)
    else:  # pragma: no cover - fallback para stubs sin soporte completo
        chart_builder = chart_builder.mark_arc(innerRadius=0)

    if hasattr(alt, "X"):
        x_field = alt.X("timestamp:T", title="Fecha de referencia")
        y_field = alt.Y("value:Q", title="Indicador promedio")
        color_field = alt.Color("sector:N", title="Sector")
        tooltip_fields = [
            alt.Tooltip("sector:N", title="Sector"),
            alt.Tooltip("value:Q", title="Valor", format=".2f"),
            alt.Tooltip("timestamp:T", title="Fecha"),
        ]
    else:  # pragma: no cover - fallback para stubs
        x_field = {"field": "timestamp", "type": "temporal"}
        y_field = {"field": "value", "type": "quantitative"}
        color_field = {"field": "sector", "type": "nominal"}
        tooltip_fields = [
            {"field": "sector", "type": "nominal", "title": "Sector"},
            {"field": "value", "type": "quantitative", "title": "Valor"},
            {"field": "timestamp", "type": "temporal", "title": "Fecha"},
        ]

    chart = (
        chart_builder
        .encode(
            x=x_field,
            y=y_field,
            color=color_field,
            tooltip=tooltip_fields,
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True, key="opportunities_macro_chart")


def _render_summary_block(
    summary: Mapping[str, object] | None,
    *,
    placeholder: st.delta_generator.DeltaGenerator | None = None,
    expanded: bool = True,
    table: pd.DataFrame | None = None,
    macro_entries: Sequence[Mapping[str, object]] | None = None,
) -> None:
    target = placeholder.empty() if placeholder is not None else st
    with target.container():
        st.markdown("### Resumen del screening")
        if not expanded:
            st.caption(
                "Resumen oculto. Activá la opción **Mostrar resumen del screening** para ver los KPIs."
            )
            return

        normalized = _normalize_summary_payload(summary)
        if not normalized:
            st.info("Ejecutá el screening para ver un resumen de universo y descartes.")
            return

        compact_layout = _should_use_compact_layout()
        if compact_layout:
            st.caption(
                "Modo compacto activado para facilitar la lectura de métricas en pantallas estrechas."
            )

        selected_sectors = normalized.get("selected_sectors") or []
        if not isinstance(selected_sectors, Sequence) or isinstance(
            selected_sectors, (str, bytes, bytearray)
        ):
            selected_sectors = [str(selected_sectors)] if selected_sectors else []
        sector_values = [str(value) for value in selected_sectors if value]
        sectors_count = len(sector_values)
        sectors_label = str(sectors_count) if sectors_count else "Todos"
        sectors_container_enabled = sectors_count > _SUMMARY_SECTOR_SCROLL_THRESHOLD

        metrics_columns: Sequence[st.delta_generator.DeltaGenerator]
        if compact_layout or sectors_container_enabled:
            metrics_columns = st.columns(3, gap="small")
        else:
            metrics_columns = st.columns(3)

        universe_label = _format_integer(normalized.get("universe_count"))
        result_label = _format_integer(normalized.get("result_count"))
        ratio_label = _format_percentage(normalized.get("discarded_ratio"))
        ratio_delta = None if ratio_label == "—" else f"{ratio_label} descartados"

        metrics_payload = (
            (
                metrics_columns[0],
                "Universo analizado",
                universe_label,
                None,
                "Cantidad total de empresas analizadas en el screening.",
            ),
            (
                metrics_columns[1],
                "Candidatos finales",
                result_label,
                ratio_delta,
                "Cantidad de compañías que superaron los filtros establecidos. El delta muestra el porcentaje descartado.",
            ),
            (
                metrics_columns[2],
                "Sectores activos",
                None,
                None,
                "Sectores con filtros activos. 'Todos' indica que no se aplicaron restricciones sectoriales.",
            ),
        )

        for column, label, value, delta, help_text in metrics_payload:
            with column:
                metric_value = value if value is not None else sectors_label
                column.metric(label, metric_value, delta=delta, help=help_text)
                column.caption(help_text)

        if sector_values:
            if sectors_container_enabled:
                st.caption(
                    "Sectores filtrados: abrí el panel para ver el detalle por sector."
                )
                with st.expander("Sectores filtrados activos", expanded=False):
                    sector_lines = "\n".join(f"- {sector}" for sector in sector_values)
                    st.markdown(sector_lines)
            else:
                st.caption("Sectores filtrados: " + ", ".join(sector_values))

        drop_summary = normalized.get("drop_summary")
        if drop_summary:
            st.caption(f"Resumen de descartes: {drop_summary}")

        elapsed_seconds = normalized.get("elapsed_seconds")
        if isinstance(elapsed_seconds, (int, float)) and not pd.isna(elapsed_seconds):
            st.caption(f"Tiempo de cómputo: {float(elapsed_seconds):.2f} s")

        filter_descriptions = normalized.get("filter_descriptions") or []
        if filter_descriptions:
            st.markdown("#### Impacto de filtros")
            for idx, description in enumerate(filter_descriptions):
                st.markdown(f"- {description}")

        st.markdown("#### Visualizaciones avanzadas")
        chart_options = ("Sector", "Tiempo")
        if hasattr(st, "radio"):
            chart_mode = st.radio(
                "Visualización del resumen",
                chart_options,
                index=0,
                horizontal=True,
                key="opportunities_summary_chart_mode",
                label_visibility="collapsed",
            )
        else:  # pragma: no cover - fallback for simplified stubs
            chart_mode = st.selectbox(
                "Visualización del resumen",
                chart_options,
                index=0,
                key="opportunities_summary_chart_mode_select",
            )
        st.caption(
            "Elegí comparar por sector o revisar la evolución de indicadores macro históricos."
        )

        macro_payload: Sequence[Mapping[str, object]] | None = macro_entries
        if macro_payload is None:
            candidate = normalized.get("macro_entries")
            if isinstance(candidate, Sequence):
                macro_payload = [entry for entry in candidate if isinstance(entry, Mapping)]

        if chart_mode == "Sector":
            _render_sector_score_chart(table)
        else:
            _render_macro_history_chart(macro_payload)

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

                if raw_url is not None and not pd.isna(raw_url):
                    url_candidate = str(raw_url).strip()
                    if url_candidate and url_candidate.upper() != "<NA>":
                        return url_candidate

                return make_symbol_url(ticker_value)

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

    _initialise_filter_state()
    st.session_state.setdefault("opportunities_preset", "Personalizado")
    _get_custom_presets()

    available_presets = list(_iter_available_presets())
    active_preset = st.session_state.get(_ACTIVE_PRESET_STATE_KEY) or st.session_state[
        "opportunities_preset"
    ]
    if active_preset not in available_presets:
        active_preset = "Personalizado"
    st.session_state[_ACTIVE_PRESET_STATE_KEY] = active_preset

    preset_selection = st.selectbox(
        "Perfil recomendado",
        options=available_presets,
        index=available_presets.index(active_preset),
        key="opportunities_preset",
        help="Aplicá perfiles sugeridos con criterios preconfigurados para iniciar el screening.",
    )
    if preset_selection != active_preset:
        _apply_preset(preset_selection)
        active_preset = preset_selection if preset_selection else "Personalizado"
    else:
        st.session_state[_ACTIVE_PRESET_STATE_KEY] = active_preset or "Personalizado"

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

    active_preset_label = st.session_state.get(_ACTIVE_PRESET_STATE_KEY, "Personalizado")
    active_filters_count = _count_active_filters(current_filters)
    badge_html = (
        "<span style='background-color:#1f77b4;color:white;padding:0.25rem 0.75rem;"
        "border-radius:999px;font-size:0.85rem;font-weight:600;display:inline-block;'>"
        + f"Preset activo: {active_preset_label} · {active_filters_count} filtros"
        + "</span>"
    )
    st.markdown(badge_html, unsafe_allow_html=True)

    st.session_state.setdefault(_SUMMARY_EXPANDED_STATE_KEY, True)
    summary_expanded = st.checkbox(
        "Mostrar resumen del screening",
        value=bool(st.session_state[_SUMMARY_EXPANDED_STATE_KEY]),
        key=_SUMMARY_EXPANDED_STATE_KEY,
        help="Alterná la visualización de los KPIs principales del screening.",
    )

    summary_placeholder = st.empty()
    stored_summary = st.session_state.get(_SUMMARY_STATE_KEY)
    initial_summary = stored_summary if isinstance(stored_summary, Mapping) else None
    stored_table = st.session_state.get(_LAST_RESULT_TABLE_STATE_KEY)
    initial_table = stored_table if isinstance(stored_table, pd.DataFrame) else None
    stored_macro = st.session_state.get(_LAST_MACRO_ENTRIES_STATE_KEY)
    initial_macro = (
        stored_macro if isinstance(stored_macro, Sequence) else None
    )
    _render_summary_block(
        initial_summary,
        placeholder=summary_placeholder,
        expanded=summary_expanded,
        table=initial_table,
        macro_entries=initial_macro,
    )

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
                    st.session_state["opportunities_preset"] = preset_name
                    _apply_preset(preset_name)
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
                    st.session_state["opportunities_preset"] = selected_custom
                    _apply_preset(selected_custom)
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

        summary_payload = _extract_summary_payload(result)
        normalized_summary = _normalize_summary_payload(summary_payload)
        if normalized_summary is not None:
            st.session_state[_SUMMARY_STATE_KEY] = normalized_summary
        else:
            st.session_state.pop(_SUMMARY_STATE_KEY, None)

        result_table: pd.DataFrame | None = None
        if isinstance(result, Mapping):
            table_candidate = result.get("table")
            if isinstance(table_candidate, pd.DataFrame):
                result_table = table_candidate
            else:
                extracted_table, _, _ = _extract_result(result)
                if isinstance(extracted_table, pd.DataFrame):
                    result_table = extracted_table
        elif isinstance(result, pd.DataFrame):
            result_table = result
        if result_table is not None:
            st.session_state[_LAST_RESULT_TABLE_STATE_KEY] = result_table
        else:
            st.session_state.pop(_LAST_RESULT_TABLE_STATE_KEY, None)

        macro_entries_payload: Sequence[Mapping[str, object]] | None = None
        if isinstance(result, Mapping):
            metrics_payload = result.get("metrics")
            if isinstance(metrics_payload, Mapping):
                entries_candidate = metrics_payload.get("macro_entries")
                if isinstance(entries_candidate, Sequence):
                    macro_entries_payload = [
                        entry for entry in entries_candidate if isinstance(entry, Mapping)
                    ]
        if macro_entries_payload:
            st.session_state[_LAST_MACRO_ENTRIES_STATE_KEY] = list(macro_entries_payload)
        else:
            st.session_state.pop(_LAST_MACRO_ENTRIES_STATE_KEY, None)

        _render_summary_block(
            normalized_summary,
            placeholder=summary_placeholder,
            expanded=st.session_state.get(_SUMMARY_EXPANDED_STATE_KEY, True),
            table=result_table,
            macro_entries=macro_entries_payload,
        )

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
