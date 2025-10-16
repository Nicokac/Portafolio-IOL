from __future__ import annotations

import io
import json
from typing import Iterable

import pandas as pd
import streamlit as st

from services.performance_timer import LOG_PATH, read_recent_entries

_ALERT_DURATION_SECONDS = 5.0
_ALERT_CPU_PERCENT = 80.0
_ALERT_MEM_PERCENT = 70.0


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
                "extras_text": json.dumps(extras, ensure_ascii=False, sort_keys=True)
                if extras
                else "",
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
    column.metric(
        label,
        formatted_value,
        delta=delta_text,
        help=help_text,
        border=True,
        chart_data=cleaned.reset_index(drop=True),
        chart_type=chart_type,
    )


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
        ["Timestamp", "Bloque", "Duración (s)", "CPU (%)", "RAM (%)", "Estado", "Extras"]
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

    st.header("⏱️ Observabilidad de performance")
    st.caption("Analizá la telemetría instrumentada para detectar cuellos de botella.")

    if st.button("🔄 Actualizar métricas", key="refresh_performance_dashboard"):
        st.experimental_rerun()

    entries = read_recent_entries(limit=limit)
    if not entries:
        st.info(
            "Todavía no se registraron mediciones. Ejecutá un flujo de portafolio o predicciones para generar datos."
        )
        st.caption(f"Archivo de log: {LOG_PATH}")
        return

    df = _entries_to_dataframe(entries)

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
            df = df[
                (df["timestamp"] >= pd.to_datetime(start))
                & (df["timestamp"] <= pd.to_datetime(end))
            ]

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

    timeline = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    if not timeline.empty:
        recent = timeline.tail(20)
        metric_cols = st.columns(3)
        _render_sparkline_metric(
            metric_cols[0],
            "Duración última (s)",
            recent["duration_s"],
            unit="s",
            decimals=2,
            chart_type="area",
            help_text="Tiempo del último bloque instrumentado.",
        )
        _render_sparkline_metric(
            metric_cols[1],
            "CPU última (%)",
            recent["cpu_percent"],
            unit="%",
            decimals=1,
            chart_type="line",
            help_text="Uso de CPU reportado por la medición más reciente.",
        )
        _render_sparkline_metric(
            metric_cols[2],
            "RAM última (%)",
            recent["mem_percent"],
            unit="%",
            decimals=1,
            chart_type="line",
            help_text="Uso de RAM reportado por la medición más reciente.",
        )
        series = timeline.set_index("timestamp")["duration_s"].rename("Duración (s)")
        st.line_chart(series)
        cpu_mem = timeline.set_index("timestamp")[
            ["cpu_percent", "mem_percent"]
        ].rename(columns={"cpu_percent": "CPU (%)", "mem_percent": "RAM (%)"})
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

    st.caption(f"Archivo de log: {LOG_PATH}")


__all__ = ["render_performance_dashboard_tab"]
