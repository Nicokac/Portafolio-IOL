"""Panel de diagnóstico para capturar y comparar payloads RAW de IOL."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Mapping, Sequence
from typing import Any, Iterable

import pandas as pd
import streamlit as st

from application.portfolio_service import capture_iol_raw_snapshot, to_iol_format
from shared.redact import redact_secrets
from shared.telemetry import log_metric
from shared.time_provider import TimeProvider

logger = logging.getLogger(__name__)

_SYMBOL = "BPOC7"
_MARKET = "bcba"
_COUNTRY = "argentina"
_NOTE = "Los datos aquí mostrados son crudos desde la API de IOL (sin reescalado)."

_MAX_INLINE_JSON_BYTES = 200_000
_TRUNCATE_LIMIT = 160


def _format_bytes(size: int | float | None) -> str:
    if size is None:
        return "—"
    try:
        value = float(size)
    except (TypeError, ValueError):
        return "—"
    if value <= 0:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    step = 1024.0
    idx = 0
    while value >= step and idx < len(units) - 1:
        value /= step
        idx += 1
    return f"{value:.1f} {units[idx]}"


def _chunk_text(text: str, chunk_size: int = _MAX_INLINE_JSON_BYTES) -> list[str]:
    if chunk_size <= 0:
        return [text]
    encoded = text.encode("utf-8")
    if len(encoded) <= chunk_size:
        return [text]
    chunks: list[str] = []
    start = 0
    while start < len(encoded):
        end = min(start + chunk_size, len(encoded))
        chunk = encoded[start:end].decode("utf-8", errors="ignore")
        chunks.append(chunk)
        start = end
    return chunks


def _truncate_value(text: str, limit: int = _TRUNCATE_LIMIT) -> str:
    if len(text) <= limit:
        return text
    return text[: max(limit - 1, 0)] + "…"


def _safe_sequence(value: Any) -> Sequence[Any] | None:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return value
    return None


def _json_default(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    if isinstance(value, (pd.Series, pd.Index)):
        return value.tolist()
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    try:
        return str(value)
    except Exception:  # pragma: no cover - defensive safeguard
        return ""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=_json_default)


def _summarize_value(value: Any) -> str:
    if isinstance(value, Mapping):
        keys = list(value.keys())
        sample = ", ".join(map(str, keys[:5]))
        suffix = "…" if len(keys) > 5 else ""
        return _truncate_value(f"{len(keys)} claves: {sample}{suffix}")
    sequence = _safe_sequence(value)
    if sequence is not None:
        sample_items = ", ".join(map(str, sequence[:3]))
        suffix = "…" if len(sequence) > 3 else ""
        return _truncate_value(f"{len(sequence)} ítems: {sample_items}{suffix}")
    if isinstance(value, str):
        return _truncate_value(value)
    return _truncate_value(str(value))


def _estimate_json_bytes(value: Any) -> int:
    try:
        return len(_json_dumps(value).encode("utf-8"))
    except Exception:  # pragma: no cover - defensive safeguard
        return 0


def _overview_rows(snapshot: Mapping[str, Any]) -> Iterable[dict[str, Any]]:
    for key, value in snapshot.items():
        size_bytes = _estimate_json_bytes(value)
        yield {
            "Clave": key,
            "Tipo": type(value).__name__,
            "Tamaño": _format_bytes(size_bytes),
            "Resumen": _summarize_value(value),
        }


def _render_json_section(label: str, value: Any) -> None:
    if value is None:
        st.info(f"{label} no disponible en la captura actual.")
        return

    try:
        json_text = _json_dumps(value)
    except Exception:
        st.warning(f"No se pudo serializar {label} a JSON.")
        return

    chunks = _chunk_text(json_text)
    size_bytes = len(json_text.encode("utf-8"))
    sequence = _safe_sequence(value)
    entry_label = f"{label} • {_format_bytes(size_bytes)}"
    with st.expander(entry_label):
        if sequence is not None:
            st.caption(f"Ítems: {len(sequence)}")
        if isinstance(value, Mapping):
            st.caption(f"Claves: {len(value)}")
        for idx, chunk in enumerate(chunks, start=1):
            if len(chunks) > 1:
                st.markdown(f"**Bloque {idx} / {len(chunks)}**")
            st.code(chunk, language="json")


def _coerce_ms(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if result < 0:
        return 0.0
    return result


def _get_positions_dataframe() -> pd.DataFrame:
    state = getattr(st, "session_state", {})
    viewmodel = state.get("portfolio_last_viewmodel")
    if viewmodel is not None:
        positions = getattr(viewmodel, "positions", None)
        if isinstance(positions, pd.DataFrame):
            return positions
    dataset = state.get("portfolio_last_positions")
    if isinstance(dataset, pd.DataFrame):
        return dataset
    return pd.DataFrame()


def _safe_filename(timestamp: str | None) -> str:
    if timestamp:
        safe = timestamp.replace(":", "").replace("/", "-").replace(" ", "T")
        return f"iol_raw_{_SYMBOL}_{safe}.json"
    fallback = TimeProvider.now_datetime().strftime("%Y%m%dT%H%M%S")
    return f"iol_raw_{_SYMBOL}_{fallback}.json"


def _prepare_download_payload(sanitized_snapshot: Mapping[str, Any]) -> tuple[bytes, str, str]:
    payload = _json_dumps(sanitized_snapshot)
    file_name = _safe_filename(str(sanitized_snapshot.get("ts")))
    return payload.encode("utf-8-sig"), file_name, payload


def render_iol_raw_debug_panel() -> None:
    """Renderiza el panel de auditoría RAW de IOL."""

    st.header("🔍 IOL RAW")
    st.caption(_NOTE)

    state = getattr(st, "session_state", {})
    last_snapshot = state.get("iol_raw_last_snapshot")
    fetch_ms = _coerce_ms(state.get("_iol_raw_last_fetch_ms"))
    parse_ms = _coerce_ms(state.get("_iol_raw_last_parse_ms"))

    if st.button("🔍 Capturar IOL RAW (BPOC7)"):
        cli = state.get("cli")
        if cli is None:
            st.warning("No hay un cliente IOL autenticado en la sesión actual.")
        else:
            try:
                start_fetch = time.perf_counter()
                with st.spinner("Consultando API de IOL..."):
                    snapshot = capture_iol_raw_snapshot(
                        cli,
                        symbol=_SYMBOL,
                        mercado=_MARKET,
                        country=_COUNTRY,
                    )
                fetch_ms = max((time.perf_counter() - start_fetch) * 1000.0, 0.0)
            except Exception:  # pragma: no cover - defensive guard
                st.error("No se pudo capturar el snapshot RAW de IOL.")
                logger.debug("Captura RAW de IOL fallida", exc_info=True)
            else:
                state["iol_raw_last_snapshot"] = snapshot
                state["_iol_raw_last_fetch_ms"] = fetch_ms
                last_snapshot = snapshot

    if not isinstance(last_snapshot, Mapping):
        st.info(
            "Aún no se capturó ningún snapshot RAW. Utilizá el botón para obtener uno reciente.",
        )
        return

    with st.spinner("Cargando RAW…"):
        parse_start = time.perf_counter()
        sanitized_snapshot = redact_secrets(last_snapshot)
        download_bytes, file_name, payload_text = _prepare_download_payload(sanitized_snapshot)
        payload_bytes = len(payload_text.encode("utf-8"))
        overview_df = pd.DataFrame(list(_overview_rows(sanitized_snapshot)))
        parse_ms = max((time.perf_counter() - parse_start) * 1000.0, 0.0)
        state["_iol_raw_last_parse_ms"] = parse_ms

    timestamp_value = sanitized_snapshot.get("ts")

    metric_cols = st.columns(3)
    metric_cols[0].metric("Tamaño del payload", _format_bytes(payload_bytes))
    metric_cols[1].metric(
        "Fetch API",
        f"{fetch_ms:.0f} ms" if fetch_ms is not None else "—",
    )
    metric_cols[2].metric(
        "Procesamiento",
        f"{parse_ms:.0f} ms" if parse_ms is not None else "—",
    )

    if timestamp_value:
        st.caption(f"Timestamp de captura: {timestamp_value}")

    st.download_button(
        "💾 Descargar snapshot RAW",
        data=download_bytes,
        file_name=file_name,
        mime="application/json",
    )

    if not overview_df.empty:
        st.subheader("📋 Resumen de secciones")
        st.dataframe(overview_df, hide_index=True)

    st.subheader("🧾 Contenido RAW paginado")
    preferred_order = [
        "portfolio_row",
        "portfolio_raw",
        "quote_raw",
        "quote_detail_raw",
    ]
    rendered_keys: set[str] = set()
    for key in preferred_order:
        if key in sanitized_snapshot:
            rendered_keys.add(key)
            _render_json_section(key, sanitized_snapshot.get(key))
    for key, value in sanitized_snapshot.items():
        if key in rendered_keys:
            continue
        _render_json_section(key, value)

    try:
        log_metric(
            "monitoring.raw_payload",
            context={
                "payload_bytes": int(payload_bytes),
                "fetch_ms": fetch_ms,
                "parse_ms": parse_ms,
            },
        )
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudo registrar telemetría monitoring.raw_payload", exc_info=True)

    df_positions = _get_positions_dataframe()
    if not df_positions.empty:
        df_formatted = to_iol_format(df_positions)
        activo_series = df_formatted.get("Activo")
        if activo_series is not None:
            mask = activo_series.astype(str).str.upper() == _SYMBOL
        else:
            mask = pd.Series([], dtype=bool)
        if getattr(mask, "any", lambda: False)():
            st.subheader("📊 Comparativa formateada (vista actual)")
            st.dataframe(df_formatted.loc[mask], hide_index=True)


__all__ = ["render_iol_raw_debug_panel"]
