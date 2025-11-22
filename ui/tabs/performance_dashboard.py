from __future__ import annotations

import io
import json
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

from services.performance_timer import LOG_PATH, read_recent_entries
from shared.debug.rerun_trace import safe_rerun
from ui.helpers.preload import gate_scientific_view

_ALERT_DURATION_SECONDS = 5.0
_ALERT_CPU_PERCENT = 80.0
_ALERT_MEM_PERCENT = 70.0

_VIEW_MODE_STATE_KEY = "performance_dashboard_view_mode"
_VIEW_MODE_TOGGLE_KEY = "performance_dashboard_view_mode_toggle"
_SPARKLINE_FILENAME = "performance_sparkline.csv"

_GRADIENT_POSITIVE = ("#22c55e", "#15803d")
_GRADIENT_NEGATIVE = ("#ef4444", "#b91c1c")
_GRADIENT_NEUTRAL = ("#9ca3af", "#4b5563")


def _extras_to_text(extras: dict[str, str]) -> str:
    if not extras:
        return ""
    return ", ".join(f"{key}={value}" for key, value in sorted(extras.items()))


def _entries_to_dataframe(entries: Iterable) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for entry in entries:
        timestamp = pd.to_datetime(entry.timestamp, errors="coerce")
        extras = dict(entry.extras)
        records.append(
            {
                "entry": entry,
                "timestamp": timestamp,
                "timestamp_text": entry.timestamp,
                "label": entry.label,
                "duration_s": float(entry.duration_s),
                "cpu_percent": entry.cpu_percent,
                "mem_percent": entry.ram_percent,
                "success": bool(getattr(entry, "success", True)),
                "extras": extras,
                "extras_text": json.dumps(extras, ensure_ascii=False, sort_keys=True) if extras else "",
            }
        )
    if not records:
        return pd.DataFrame(
            columns=[
                "entry",
                "timestamp",
                "timestamp_text",
                "label",
                "duration_s",
                "cpu_percent",
                "mem_percent",
                "success",
                "extras",
                "extras_text",
            ]
        )
    return pd.DataFrame(records)


def _extract_numeric_extra(series: pd.Series, key: str) -> pd.Series:
    if series.empty:
        return pd.Series(dtype="float64")
    return pd.to_numeric(
        series.map(lambda extras: extras.get(key) if isinstance(extras, dict) else None),
        errors="coerce",
    )


def _build_percentile_summary(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["P50 (s)", "P95 (s)", "P99 (s)"])
    grouped = df.groupby("label")["duration_s"].quantile([0.5, 0.95, 0.99]).unstack()
    grouped = grouped.rename(columns={0.5: "P50 (s)", 0.95: "P95 (s)", 0.99: "P99 (s)"})
    return grouped.sort_values(by="P99 (s)", ascending=False)


def _build_alert_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    duration_alert = df["duration_s"] > _ALERT_DURATION_SECONDS
    cpu_alert = df["cpu_percent"].fillna(-1) > _ALERT_CPU_PERCENT
    mem_alert = df["mem_percent"].fillna(-1) > _ALERT_MEM_PERCENT
    alerts = df[duration_alert | cpu_alert | mem_alert].copy()
    if alerts.empty:
        return alerts

    def _describe(row):
        messages: list[str] = []
        if row.duration_s > _ALERT_DURATION_SECONDS:
            messages.append(f"duración>{_ALERT_DURATION_SECONDS:.0f}s")
        if pd.notna(row.cpu_percent) and row.cpu_percent > _ALERT_CPU_PERCENT:
            messages.append(f"cpu>{_ALERT_CPU_PERCENT:.0f}%")
        if pd.notna(row.mem_percent) and row.mem_percent > _ALERT_MEM_PERCENT:
            messages.append(f"memoria>{_ALERT_MEM_PERCENT:.0f}%")
        return ", ".join(messages)

    alerts["Alertas"] = alerts.apply(_describe, axis=1)
    return alerts


def _render_sparkline_metric(
    column,
    label: str,
    series: pd.Series,
    *,
    unit: str = "",
    decimals: int = 2,
    chart_type: str = "line",
    help_text: str | None = None,
    chart_gradient: Sequence[str] | None = None,
) -> None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return
    latest = float(cleaned.iloc[-1])
    previous = float(cleaned.iloc[-2]) if len(cleaned) > 1 else None
    formatted_value = f"{latest:.{decimals}f}{unit}"
    delta_text: str | None = None
    if previous is not None:
        delta_value = latest - previous
        delta_text = f"{delta_value:+.{decimals}f}{unit}"
    metric_kwargs = {
        "label": label,
        "value": formatted_value,
        "delta": delta_text,
        "help": help_text,
        "border": True,
        "chart_data": cleaned.reset_index(drop=True),
        "chart_type": chart_type,
    }
    if chart_gradient:
        metric_kwargs["chart_color"] = list(chart_gradient)
    column.metric(**metric_kwargs)


def _compute_duration_gradient(series: pd.Series) -> Sequence[str] | None:
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if len(cleaned) < 2:
        return None
    trend = cleaned.iloc[-1] - cleaned.iloc[0]
    if trend < 0:
        return _GRADIENT_POSITIVE
    if trend > 0:
        return _GRADIENT_NEGATIVE
    return _GRADIENT_NEUTRAL


def _build_ui_overhead_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    ui_entries = df[df["label"] == "ui_total_load"].copy()
    if ui_entries.empty:
        return pd.DataFrame()
    ui_entries = ui_entries.reset_index(drop=True)
    total_series = _extract_numeric_extra(ui_entries["extras"], "total_ms").to_numpy()
    logic_series = _extract_numeric_extra(ui_entries["extras"], "profile_block_total_ms").to_numpy()
    overhead_series = _extract_numeric_extra(ui_entries["extras"], "streamlit_overhead_ms").to_numpy()
    frame = pd.DataFrame(
        {
            "ui_total_load_ms": total_series,
            "profile_block_total_ms": logic_series,
            "streamlit_overhead_ms": overhead_series,
        },
        index=ui_entries["timestamp"],
    )
    return frame.dropna(how="all")


def _prepare_display(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    formatted = df.copy()
    formatted["Timestamp"] = formatted["timestamp_text"]
    formatted["Bloque"] = formatted["label"]
    formatted["Duración (s)"] = formatted["duration_s"].round(4)
    formatted["CPU (%)"] = formatted["cpu_percent"].map(lambda v: None if pd.isna(v) else round(v, 2))
    formatted["RAM (%)"] = formatted["mem_percent"].map(lambda v: None if pd.isna(v) else round(v, 2))
    formatted["Extras"] = formatted["extras"].map(_extras_to_text)
    formatted["Estado"] = formatted["success"].map(lambda value: "✅" if value else "❌")
    return formatted[
        [
            "Timestamp",
            "Bloque",
            "Duración (s)",
            "CPU (%)",
            "RAM (%)",
            "Estado",
            "Extras",
        ]
    ]


def _export_payload(df: pd.DataFrame) -> list[dict[str, object]]:
    entries = df.get("entry")
    if entries is None:
        return []
    payload: list[dict[str, object]] = []
    for entry in entries.tolist():
        payload.append(entry.as_dict(include_raw=False))
    return payload


def render_performance_dashboard_tab(limit: int = 200) -> None:
    """Render the performance telemetry dashboard for QA and diagnostics."""

    if not gate_scientific_view(st, view_id="performance_dashboard"):
        return

    st.header("⏱️ Observabilidad de performance")
    st.caption("Analizá la telemetría instrumentada para detectar cuellos de botella.")

    if st.button("🔄 Actualizar métricas", key="refresh_performance_dashboard"):
        safe_rerun("performance_dashboard_refresh")

    entries = read_recent_entries(limit=limit)
    if not entries:
        st.info(
            "Todavía no se registraron mediciones. Ejecutá un flujo de portafolio o predicciones para generar datos."
        )
        st.caption(f"Archivo de log: {LOG_PATH}")
        return

    df = _entries_to_dataframe(entries)

    current_mode = st.session_state.get(_VIEW_MODE_STATE_KEY, "latest")
    show_historical_default = current_mode == "historical"
    show_historical = st.toggle(
        "Promedio histórico",
        key=_VIEW_MODE_TOGGLE_KEY,
        value=show_historical_default,
        help="Alterná entre la última ejecución y el promedio histórico para los indicadores.",
    )
    current_mode = "historical" if show_historical else "latest"
    st.session_state[_VIEW_MODE_STATE_KEY] = current_mode
    st.caption(f"Modo seleccionado: {'Promedio histórico' if show_historical else 'Última ejecución'}")

    labels = sorted(df["label"].dropna().unique())
    if labels:
        selected_labels = st.multiselect("Filtrar por bloque", labels, default=labels)
        if selected_labels:
            df = df[df["label"].isin(selected_labels)]

    valid_timestamps = df["timestamp"].dropna()
    if not valid_timestamps.empty:
        min_ts = valid_timestamps.min().to_pydatetime()
        max_ts = valid_timestamps.max().to_pydatetime()
        if min_ts != max_ts:
            start, end = st.slider(
                "Rango temporal",
                min_value=min_ts,
                max_value=max_ts,
                value=(min_ts, max_ts),
            )
            df = df[(df["timestamp"] >= pd.to_datetime(start)) & (df["timestamp"] <= pd.to_datetime(end))]

    keyword = st.text_input("Buscar palabras clave en extras", value="")
    if keyword:
        df = df[df["extras_text"].str.contains(keyword, case=False, na=False)]

    if df.empty:
        st.warning("No hay registros que coincidan con los filtros seleccionados.")
        st.caption(f"Archivo de log: {LOG_PATH}")
        return

    st.subheader("Registros recientes")
    st.dataframe(_prepare_display(df), width="stretch", hide_index=True)

    alerts_df = _build_alert_rows(df)
    if not alerts_df.empty:
        st.subheader("Alertas")
        st.warning("⚠️ Se detectaron bloques con duración prolongada o alto consumo de CPU/RAM.")
        st.dataframe(
            _prepare_display(alerts_df.assign(success=alerts_df["success"])),
            width="stretch",
            hide_index=True,
        )

    percentiles = _build_percentile_summary(df)
    if not percentiles.empty:
        st.subheader("Percentiles de duración por bloque")
        st.dataframe(percentiles, width="stretch")

    sparkline_export_df = pd.DataFrame()
    timeline = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if not timeline.empty:
        ui_overhead_frame = _build_ui_overhead_frame(timeline)
        if not ui_overhead_frame.empty:
            st.subheader("Desglose de carga de la UI")
            ui_recent = ui_overhead_frame.tail(20)
            ui_columns = st.columns(3)
            _render_sparkline_metric(
                ui_columns[0],
                "Total carga UI",
                ui_recent["ui_total_load_ms"],
                unit=" ms",
                decimals=0,
                chart_type="area",
                help_text="Tiempo total registrado por `ui_total_load_ms`.",
                chart_gradient=_compute_duration_gradient(ui_recent["ui_total_load_ms"]),
            )
            _render_sparkline_metric(
                ui_columns[1],
                "Lógica instrumentada",
                ui_recent["profile_block_total_ms"],
                unit=" ms",
                decimals=0,
                chart_type="line",
                help_text="Suma de los bloques instrumentados reportados en la sesión.",
                chart_gradient=_compute_duration_gradient(ui_recent["profile_block_total_ms"]),
            )
            _render_sparkline_metric(
                ui_columns[2],
                "Overhead Streamlit",
                ui_recent["streamlit_overhead_ms"],
                unit=" ms",
                decimals=0,
                chart_type="line",
                help_text=("Latencia atribuida al layout: ui_total_load_ms menos la lógica instrumentada."),
                chart_gradient=_compute_duration_gradient(ui_recent["streamlit_overhead_ms"]),
            )
            st.markdown(
                """
**Sugerencias para optimizar el layout:**
- Reducir markdown dinámico innecesario en el render.
- Consolidar múltiples `st.write` en bloques estructurados.
- Usar `staticPlot=True` en gráficos sin interactividad.
- Mejorar el caching de componentes UI costosos.
"""
            )
        metrics_frame = timeline.set_index("timestamp")[["duration_s", "cpu_percent", "mem_percent"]]
        if current_mode == "historical":
            metrics_frame = metrics_frame.expanding().mean()
        metrics_frame = metrics_frame.dropna(how="all")
        if not metrics_frame.empty:
            recent = metrics_frame.tail(20)
            sparkline_export_df = recent.reset_index().rename(
                columns={
                    "timestamp": "timestamp",
                    "duration_s": "duration_s",
                    "cpu_percent": "cpu_percent",
                    "mem_percent": "mem_percent",
                }
            )
            sparkline_export_df.insert(1, "view_mode", current_mode)

            metric_cols = st.columns(3)
            duration_label = "Duración promedio (s)" if current_mode == "historical" else "Duración última (s)"
            duration_help = (
                "Promedio acumulado de la duración histórica de los bloques instrumentados."
                if current_mode == "historical"
                else "Tiempo del último bloque instrumentado."
            )
            _render_sparkline_metric(
                metric_cols[0],
                duration_label,
                recent["duration_s"],
                unit="s",
                decimals=2,
                chart_type="area",
                help_text=duration_help,
                chart_gradient=_compute_duration_gradient(recent["duration_s"]),
            )
            cpu_label = "CPU promedio (%)" if current_mode == "historical" else "CPU última (%)"
            cpu_help = (
                "Promedio acumulado de uso de CPU reportado históricamente."
                if current_mode == "historical"
                else "Uso de CPU reportado por la medición más reciente."
            )
            _render_sparkline_metric(
                metric_cols[1],
                cpu_label,
                recent["cpu_percent"],
                unit="%",
                decimals=1,
                chart_type="line",
                help_text=cpu_help,
            )
            ram_label = "RAM promedio (%)" if current_mode == "historical" else "RAM última (%)"
            ram_help = (
                "Promedio acumulado de uso de RAM reportado históricamente."
                if current_mode == "historical"
                else "Uso de RAM reportado por la medición más reciente."
            )
            _render_sparkline_metric(
                metric_cols[2],
                ram_label,
                recent["mem_percent"],
                unit="%",
                decimals=1,
                chart_type="line",
                help_text=ram_help,
            )

            duration_series = metrics_frame["duration_s"].dropna().rename("Duración (s)")
            if not duration_series.empty:
                st.line_chart(duration_series)
            cpu_mem = metrics_frame[["cpu_percent", "mem_percent"]].rename(
                columns={"cpu_percent": "CPU (%)", "mem_percent": "RAM (%)"}
            )
            if not cpu_mem.dropna(how="all").empty:
                st.line_chart(cpu_mem)

    export_records = _export_payload(df)
    if export_records:
        export_container = st.container()
        with export_container:
            st.subheader("Exportar")
            csv_buffer = io.StringIO()
            pd.DataFrame(export_records).to_csv(csv_buffer, index=False)
            st.download_button(
                "⬇️ CSV",
                data=csv_buffer.getvalue().encode("utf-8"),
                file_name="performance_metrics.csv",
                mime="text/csv",
            )
            st.download_button(
                "⬇️ JSON",
                data=json.dumps(export_records, ensure_ascii=False, indent=2).encode("utf-8"),
                file_name="performance_metrics.json",
                mime="application/json",
            )
            if not sparkline_export_df.empty:
                sparkline_csv = io.StringIO()
                sparkline_export_df.to_csv(sparkline_csv, index=False)
                st.download_button(
                    "⬇️ Sparkline CSV",
                    data=sparkline_csv.getvalue().encode("utf-8"),
                    file_name=_SPARKLINE_FILENAME,
                    mime="text/csv",
                )

    st.caption(f"Archivo de log: {LOG_PATH}")


__all__ = ["render_performance_dashboard_tab"]
