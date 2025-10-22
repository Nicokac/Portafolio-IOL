from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping

import pandas as pd
import streamlit as st

from services.system_status import (
    PrometheusSnapshot,
    TokenSnapshot,
    get_system_status_snapshot,
)
from shared.time_provider import TimeProvider

"""Render the system status dashboard with observability insights."""


_DOCS_PATH = Path("docs/operations.md")
_TROUBLESHOOT_PATH = Path("docs/troubleshooting.md")


def _format_duration(seconds: float | None) -> str:
    if seconds is None:
        return "s/d"
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: list[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs and not parts:
        parts.append(f"{secs}s")
    return " ".join(parts) or "0s"


def _format_number(value: float | None) -> str:
    if value is None:
        return "s/d"
    return f"{int(value):,}".replace(",", ".")


def _format_percent(value: float | None) -> str:
    if value is None:
        return "s/d"
    return f"{value * 100:.1f}%"


def _render_summary_cards(prometheus: PrometheusSnapshot) -> None:
    columns = st.columns(3)
    cards = [
        (
            "🕒 Uptime",
            _format_duration(prometheus.uptime_seconds),
            None,
            "Tiempo transcurrido desde el arranque del backend.",
        ),
        (
            "🔁 Refresh tokens",
            _format_number(prometheus.auth_refresh_total),
            None,
            "Total de renovaciones exitosas registradas por Prometheus.",
        ),
        (
            "📦 Hit ratio caché",
            _format_percent(prometheus.cache_hit_ratio),
            None,
            "Porcentaje de aciertos del caché predictivo reportado por métricas.",
        ),
    ]
    for column, (label, value, delta, help_text) in zip(columns, cards):
        column.metric(label, value, delta=delta, help=help_text)


def _filter_metrics(metrics: Mapping[str, float], prefixes: Iterable[str]) -> list[tuple[str, float]]:
    selected: list[tuple[str, float]] = []
    prefix_tuple = tuple(prefixes)
    for name, value in sorted(metrics.items()):
        if not prefix_tuple or name.startswith(prefix_tuple):
            selected.append((name, value))
    return selected


def _render_metrics_table(entries: list[tuple[str, float]]) -> None:
    if not entries:
        st.caption("No hay métricas registradas para esta categoría.")
        return
    frame = pd.DataFrame(entries, columns=["Métrica", "Valor"])
    st.dataframe(frame, width="stretch", hide_index=True)


def _render_token_status(token: TokenSnapshot) -> None:
    st.subheader("Token de autenticación")
    if not token.active:
        st.info("No hay un token activo registrado en la sesión actual.")
        return
    cols = st.columns(3)
    cols[0].metric("Usuario", token.username or "anon")
    cols[1].metric("TTL", _format_duration(token.ttl))
    cols[2].metric("TTL restante", _format_duration(token.remaining_ttl))

    detail_cols = st.columns(2)
    detail_cols[0].markdown(
        f"**Emitido:** {token.issued_label or 's/d'}",
    )
    detail_cols[1].markdown(
        f"**Expira:** {token.expires_label or 's/d'}",
    )
    if token.refreshed_at:
        st.caption(f"Último refresh manual: {token.refreshed_at}")


def _handle_manual_refresh() -> None:
    from services.auth import AuthTokenError, refresh_active_token

    token = st.session_state.get("auth_token")
    claims = st.session_state.get("auth_token_claims")
    if not token or not isinstance(claims, dict):
        st.warning("No hay un token activo para refrescar.")
        return
    try:
        response = refresh_active_token(claims)
    except AuthTokenError as exc:
        st.error(f"No se pudo refrescar el token: {exc}")
        return
    new_token = response.get("token")
    if isinstance(new_token, str) and new_token:
        st.session_state["auth_token"] = new_token
    new_claims = response.get("claims")
    if isinstance(new_claims, dict):
        st.session_state["auth_token_claims"] = dict(new_claims)
    st.session_state["auth_token_refreshed_at"] = TimeProvider.now()
    st.success("Token refrescado correctamente.")


def _render_performance_tab(prometheus: PrometheusSnapshot) -> None:
    st.subheader("Performance")
    performance_entries = _filter_metrics(
        prometheus.metrics,
        prefixes=(
            "performance_",
            "prediction_",
            "engine_",
            "portfolio_",
        ),
    )
    _render_metrics_table(performance_entries)


def _render_security_tab(prometheus: PrometheusSnapshot, token: TokenSnapshot) -> None:
    st.subheader("Seguridad")
    security_entries = _filter_metrics(prometheus.metrics, prefixes=("auth_", "security_"))
    _render_metrics_table(security_entries)
    _render_token_status(token)
    st.markdown("---")
    if st.button("🔄 Refrescar token", key="system_status_refresh_button"):
        _handle_manual_refresh()


def _render_cache_tab(prometheus: PrometheusSnapshot) -> None:
    st.subheader("Caché")
    cache_entries = _filter_metrics(prometheus.metrics, prefixes=("cache_", "fx_cache_"))
    _render_metrics_table(cache_entries)


def render_system_status_panel() -> None:
    """Render the system observability dashboard."""

    snapshot = get_system_status_snapshot()
    st.header("🔍 Estado del Sistema")

    links: list[str] = []
    if _DOCS_PATH.exists():
        links.append(f"[📘 Documentación operativa]({_DOCS_PATH.as_posix()})")
    if _TROUBLESHOOT_PATH.exists():
        links.append(f"[🛟 Troubleshooting]({_TROUBLESHOOT_PATH.as_posix()})")
    if links:
        st.markdown(" | ".join(links))

    _render_summary_cards(snapshot.prometheus)

    tabs = st.tabs(["⚡ Performance", "🔐 Seguridad", "🗃️ Caché"])
    with tabs[0]:
        _render_performance_tab(snapshot.prometheus)
    with tabs[1]:
        _render_security_tab(snapshot.prometheus, snapshot.token)
    with tabs[2]:
        _render_cache_tab(snapshot.prometheus)


__all__ = ["render_system_status_panel"]
