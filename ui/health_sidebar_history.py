"""Render helpers for the health sidebar history block."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import plotly.graph_objects as go
import streamlit as st

from shared.time_provider import TimeProvider, TimeSnapshot


@dataclass
class _HistoryEntry:
    ts: float | None
    elapsed_ms: float | None
    status: str
    detail: str | None
    environment: Sequence[str]

    @property
    def color(self) -> str:
        status = self.status.lower()
        if status in {"success", "ok", "hit"}:
            return "#2ca02c"
        if status in {"warning", "degraded"}:
            return "#ff7f0e"
        return "#d62728"

    @property
    def label(self) -> str:
        return self.status.upper()


def _coerce_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, Mapping):
        items: Iterable[Any] = value.values()
    elif isinstance(value, Iterable):
        items = value
    else:
        return []

    labels: list[str] = []
    for item in items:
        text = str(item or "").strip()
        if text:
            labels.append(text)
    return labels


def _normalise_history(
    history: Iterable[Mapping[str, Any]] | None,
) -> list[_HistoryEntry]:
    entries: list[_HistoryEntry] = []
    if history is None:
        return entries
    for raw in history:
        if not isinstance(raw, Mapping):
            continue
        status = str(raw.get("status") or raw.get("result") or "").strip() or "unknown"
        detail = raw.get("detail") or raw.get("label")
        try:
            ts = float(raw.get("ts")) if raw.get("ts") is not None else None
        except (TypeError, ValueError):
            ts = None
        try:
            elapsed = float(raw.get("elapsed_ms")) if raw.get("elapsed_ms") is not None else None
        except (TypeError, ValueError):
            elapsed = None
        environments = _coerce_sequence(raw.get("environment") or raw.get("environments"))
        entries.append(
            _HistoryEntry(
                ts=ts,
                elapsed_ms=elapsed,
                status=status,
                detail=str(detail).strip() if detail is not None else None,
                environment=environments,
            )
        )
    return entries


def _build_history_chart(entries: Sequence[_HistoryEntry]) -> go.Figure:
    figure = go.Figure()
    timestamps: list[str] = []
    elapsed: list[float] = []
    colors: list[str] = []
    hover: list[str] = []

    for entry in entries:
        snapshot: TimeSnapshot | None = TimeProvider.from_timestamp(entry.ts)
        if snapshot is not None:
            label = snapshot.text
        elif entry.ts is not None:
            label = str(entry.ts)
        else:
            label = "s/d"
        timestamps.append(label)
        elapsed.append(float(entry.elapsed_ms or 0.0))
        colors.append(entry.color)
        hover_detail = entry.detail or entry.status
        hover.append(hover_detail)

    figure.add_trace(
        go.Bar(
            x=timestamps,
            y=elapsed,
            marker_color=colors,
            hovertext=hover,
            name="elapsed_ms",
        )
    )
    figure.update_layout(
        height=220,
        margin=dict(t=10, b=10, l=0, r=0),
        yaxis_title="ms",
        showlegend=False,
    )
    return figure


def _format_environment_badges(environment: Any) -> str | None:
    labels = _coerce_sequence(environment)
    if not labels:
        return None
    badges = " ".join(f"`{label}`" for label in labels)
    return f"Ambiente: {badges}"


def _format_last_error(entry: Mapping[str, Any] | None) -> str | None:
    if not isinstance(entry, Mapping):
        return None
    label = str(entry.get("label") or entry.get("detail") or entry.get("error") or entry.get("message") or "").strip()
    ts_snapshot = TimeProvider.from_timestamp(entry.get("ts"))
    components: list[str] = []
    if label:
        components.append(label)
    if ts_snapshot is not None:
        components.append(ts_snapshot.text)
    if not components:
        return None
    return "Último error: " + " — ".join(components)


def render_history_sidebar(metrics: Mapping[str, Any]) -> None:
    """Render the health sidebar history section."""

    history_entries = _normalise_history(metrics.get("history"))
    if history_entries:
        st.sidebar.subheader("Historial de screenings")
        st.sidebar.plotly_chart(
            _build_history_chart(history_entries),
            width="stretch",
            config={"responsive": True},
        )
        for entry in history_entries[-5:]:
            snapshot = TimeProvider.from_timestamp(entry.ts)
            ts_label = snapshot.text if snapshot is not None else "s/d"
            env_label = ", ".join(entry.environment) if entry.environment else "sin ambiente"
            detail = entry.detail or entry.status
            st.sidebar.markdown(f"• {ts_label} · {detail} ({env_label})")
    else:
        st.sidebar.subheader("Historial de screenings")
        st.sidebar.markdown("_Sin historial disponible._")

    env_badges = _format_environment_badges(metrics.get("environment"))
    if env_badges:
        st.sidebar.markdown(env_badges)

    last_error = _format_last_error(metrics.get("last_error"))
    if last_error:
        st.sidebar.markdown(last_error)


__all__ = ["render_history_sidebar"]
