from __future__ import annotations

"""Sidebar panel summarising recent data source health."""

from typing import Any, Iterable, Mapping, Optional, Sequence

import streamlit as st

from services.health import get_health_metrics
from shared.time_provider import TimeProvider
from shared.ui import notes as shared_notes
from shared.version import __version__

format_note = shared_notes.format_note


_YFINANCE_PROVIDER_LABELS = {
    "yfinance": "Datos de Yahoo Finance",
    "fallback": "Fallback local",
    "error": "Error o sin datos",
}

_YFINANCE_PROVIDER_SHORT = {
    "yfinance": "YF",
    "fallback": "FB",
    "error": "ERR",
}

_YFINANCE_STATUS_LABELS = {
    "success": "OK",
    "ok": "OK",
    "hit": "OK",
    "fallback": "Fallback",
    "error": "Error",
    "failed": "Error",
    "timeout": "Timeout",
    "missing": "Sin datos",
}

_YFINANCE_ERROR_STATUSES = {"error", "failed", "timeout"}


_TAB_LABELS = {
    "tecnico": "Técnico",
    "riesgo": "Riesgo",
    "fundamentales": "Fundamentales",
}


def _sanitize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _normalize_status_key(value: Any) -> str:
    text = _sanitize_text(value)
    return text.casefold() if text else ""


def _extract_yfinance_history(data: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    history_raw = data.get("history")
    entries: list[Mapping[str, Any]] = []
    if isinstance(history_raw, Iterable) and not isinstance(
        history_raw, (str, bytes, bytearray)
    ):
        for entry in history_raw:
            if isinstance(entry, Mapping):
                entries.append(entry)
    return entries


def _compress_yfinance_history(
    entries: Sequence[Mapping[str, Any]], *, limit: int = 5
) -> Optional[str]:
    if not entries:
        return None

    recent = entries[-limit:]
    tokens: list[str] = []
    for entry in recent:
        provider_value = _sanitize_text(entry.get("provider")) or "desconocido"
        provider_key = provider_value.casefold()
        label = _YFINANCE_PROVIDER_SHORT.get(provider_key, provider_value[:3].upper())
        status_key = _normalize_status_key(entry.get("result") or entry.get("status"))
        fallback_flag = bool(entry.get("fallback"))
        if status_key in _YFINANCE_ERROR_STATUSES:
            icon = "⚠️"
        elif fallback_flag:
            icon = "🛟"
        else:
            icon = "✅"
        tokens.append(f"{icon}{label}")

    if not tokens:
        return None

    return "Historial: " + " · ".join(tokens)


def _format_timestamp(ts: Optional[float]) -> str:
    if not ts:
        return "Sin registro"
    snapshot = TimeProvider.from_timestamp(ts)
    if snapshot is None:
        return "Sin registro"
    return snapshot.text


def _format_iol_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin actividad registrada._"
    status = data.get("status")
    label = "Refresh correcto" if status == "success" else "Error al refrescar"
    ts = _format_timestamp(data.get("ts"))
    detail = data.get("detail")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    return format_note(f"{prefix} {label} • {ts}{detail_txt}")


def _format_yfinance_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin consultas registradas._"
    history_entries = _extract_yfinance_history(data)
    latest_entry: Mapping[str, Any] | dict[str, Any]
    if history_entries:
        latest_entry = history_entries[-1]
    else:
        latest_entry = data

    provider_value = (
        _sanitize_text(latest_entry.get("provider"))
        or _sanitize_text(data.get("latest_provider"))
        or _sanitize_text(data.get("source"))
        or "desconocido"
    )
    provider_key = provider_value.casefold()
    provider_label = _YFINANCE_PROVIDER_LABELS.get(
        provider_key, f"Fuente: {provider_value}"
    )

    fallback_value = latest_entry.get("fallback")
    if fallback_value is None:
        fallback_value = data.get("fallback")
    if fallback_value is None:
        fallback_flag = provider_key != "yfinance"
    else:
        fallback_flag = bool(fallback_value)

    result_raw = (
        latest_entry.get("result")
        or latest_entry.get("status")
        or data.get("latest_result")
        or data.get("result")
        or data.get("status")
    )
    result_key = _normalize_status_key(result_raw)
    if not result_key:
        result_key = "fallback" if fallback_flag else "success"

    if result_key in _YFINANCE_ERROR_STATUSES:
        icon = "⚠️"
    elif fallback_flag:
        icon = "🛟"
    else:
        icon = "✅"

    status_label = _YFINANCE_STATUS_LABELS.get(
        result_key, result_key.title() if result_key else "Desconocido"
    )

    timestamp_value = latest_entry.get("ts") if isinstance(latest_entry, Mapping) else None
    if timestamp_value is None:
        timestamp_value = data.get("ts")
    ts_text = _format_timestamp(timestamp_value)

    detail_value = None
    if isinstance(latest_entry, Mapping):
        detail_value = latest_entry.get("detail")
    if detail_value is None:
        detail_value = data.get("detail")
    detail_text = _sanitize_text(detail_value)
    detail_suffix = f" — {detail_text}" if detail_text else ""

    fallback_badge = " [Fallback]" if fallback_flag else ""
    history_summary = _compress_yfinance_history(history_entries)

    parts = [
        f"{icon} {provider_label}{fallback_badge}",
        f"• {ts_text}",
        f"• Resultado: {status_label}{detail_suffix}",
    ]
    if history_summary:
        parts.append(f"• {history_summary}")

    return format_note(" ".join(parts))


def _format_fx_api_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin llamadas a la API FX._"
    status = data.get("status")
    label = "API FX OK" if status == "success" else "API FX con errores"
    ts = _format_timestamp(data.get("ts"))
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    detail = data.get("error")
    detail_txt = f" — {detail}" if detail else ""
    prefix = "✅" if status == "success" else "⚠️"
    return format_note(f"{prefix} {label} • {ts} ({elapsed_txt}){detail_txt}")


def _format_fx_cache_status(data: Optional[dict]) -> str:
    if not data:
        return "_Sin uso de caché registrado._"
    mode = data.get("mode")
    icon = "✅" if mode == "hit" else "ℹ️"
    label = "Uso de caché" if mode == "hit" else "Actualización"
    ts = _format_timestamp(data.get("ts"))
    age = data.get("age")
    age_txt = f"{float(age):.0f}s" if isinstance(age, (int, float)) else "s/d"
    return format_note(f"{icon} {label} • {ts} (edad {age_txt})")


def _format_latency_line(label: str, data: Optional[dict]) -> str:
    if not data:
        return f"- {label}: sin registro"
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    parts = [f"- {label}: {elapsed_txt}"]
    source = data.get("source")
    if source:
        parts.append(f"fuente: {source}")
    count = data.get("count")
    if count is not None:
        parts.append(f"items: {count}")
    detail = data.get("detail")
    if detail:
        parts.append(detail)
    parts.append(_format_timestamp(data.get("ts")))
    return " • ".join(parts)


def _format_fx_section(api_data: Optional[dict], cache_data: Optional[dict]) -> Iterable[str]:
    lines = []
    lines.append(_format_fx_api_status(api_data))
    lines.append(_format_fx_cache_status(cache_data))
    return lines


def _format_latency_section(portfolio: Optional[dict], quotes: Optional[dict]) -> Iterable[str]:
    return [
        _format_latency_line("Portafolio", portfolio),
        _format_latency_line("Cotizaciones", quotes),
    ]


def _format_macro_history(
    attempts: Optional[Iterable[Mapping[str, Any]]],
    *,
    limit: int = 5,
) -> Iterable[str]:
    if not attempts:
        return []

    recent: list[Mapping[str, Any]] = []
    for entry in attempts:
        if isinstance(entry, Mapping):
            recent.append(entry)
    if not recent:
        return []

    lines: list[str] = []
    for attempt in reversed(recent[-limit:]):
        status = str(
            attempt.get("status_normalized")
            or attempt.get("status")
            or "unknown"
        ).casefold()
        label = str(
            attempt.get("provider_label")
            or attempt.get("label")
            or attempt.get("provider")
            or "desconocido"
        )
        ts = _format_timestamp(attempt.get("ts"))
        elapsed = attempt.get("elapsed_ms")
        if isinstance(elapsed, (int, float)):
            elapsed_txt = f"{float(elapsed):.0f} ms"
        else:
            elapsed_txt = "s/d"
        detail = attempt.get("detail")
        detail_txt = f" — {detail}" if detail else ""
        missing_raw = attempt.get("missing_series")
        missing: list[str] = []
        if isinstance(missing_raw, str):
            candidate = missing_raw.strip()
            if candidate:
                missing = [candidate]
        elif isinstance(missing_raw, Iterable) and not isinstance(
            missing_raw, (bytes, bytearray, str)
        ):
            missing = [str(item).strip() for item in missing_raw if str(item).strip()]
        missing_txt = f" • sin series: {', '.join(missing)}" if missing else ""
        fallback_flag = bool(attempt.get("fallback"))

        icon_map = {
            "success": "✅",
            "error": "⚠️",
            "disabled": "⛔️",
            "unavailable": "❌",
        }
        icon = icon_map.get(status, "ℹ️")
        if fallback_flag and status == "success":
            icon = "ℹ️"
        if status == "success" and not fallback_flag:
            status_label = "OK"
        elif fallback_flag:
            status_label = "Fallback"
        else:
            status_label = status.title()

        lines.append(
            format_note(
                f"{icon} {label}: {status_label} • {ts} ({elapsed_txt})"
                f"{detail_txt}{missing_txt}"
            )
        )

    return lines


def _format_macro_latency(buckets: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not isinstance(buckets, Mapping):
        return None

    counts = buckets.get("counts")
    ratios = buckets.get("ratios")
    if not isinstance(counts, Mapping):
        return None

    parts: list[str] = []
    for key, label in (("fast", "rápido"), ("medium", "medio"), ("slow", "lento")):
        count = counts.get(key)
        if not isinstance(count, (int, float)):
            continue
        count_int = int(count)
        if count_int <= 0:
            continue
        ratio_val = ratios.get(key) if isinstance(ratios, Mapping) else None
        if isinstance(ratio_val, (int, float)):
            parts.append(f"{label} {ratio_val:.0%} ({count_int})")
        else:
            parts.append(f"{label} ({count_int})")

    missing = counts.get("missing")
    if isinstance(missing, (int, float)) and int(missing) > 0:
        parts.append(f"sin dato {int(missing)}")

    if not parts:
        return None

    return "Latencia: " + " | ".join(parts)


def _format_macro_provider(summary: Mapping[str, Any]) -> str:
    label = str(summary.get("label") or summary.get("provider") or "desconocido")
    latest = summary.get("latest") if isinstance(summary.get("latest"), Mapping) else {}
    status = str(latest.get("status") or "desconocido")
    fallback_flag = bool(latest.get("fallback"))
    elapsed = latest.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    ts = _format_timestamp(latest.get("ts"))
    detail = latest.get("detail")
    detail_txt = f" — {detail}" if detail else ""

    if status == "error":
        icon = "⚠️"
        status_label = "Error reciente"
    elif fallback_flag:
        icon = "ℹ️"
        status_label = "Fallback"
    elif status == "success":
        icon = "✅"
        status_label = "OK"
    else:
        icon = "ℹ️"
        status_label = status

    total = summary.get("count")
    total_int = int(total) if isinstance(total, (int, float)) else 0
    status_counts = (
        summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    )
    status_ratios = (
        summary.get("status_ratios") if isinstance(summary.get("status_ratios"), Mapping) else {}
    )

    parts: list[str] = []
    success_count = status_counts.get("success")
    success_ratio = status_ratios.get("success")
    if isinstance(success_count, (int, float)):
        if isinstance(success_ratio, (int, float)) and total_int:
            parts.append(f"éxitos {success_ratio:.0%} ({int(success_count)}/{total_int})")
        else:
            parts.append(f"éxitos {int(success_count)}")

    for key, count_val in status_counts.items():
        if key in {"success", "error", "fallback"}:
            continue
        if not isinstance(count_val, (int, float)):
            continue
        ratio_val = status_ratios.get(key)
        label_txt = str(key)
        if isinstance(ratio_val, (int, float)) and total_int:
            parts.append(f"{label_txt} {ratio_val:.0%} ({int(count_val)}/{total_int})")
        else:
            parts.append(f"{label_txt} {int(count_val)}")

    error_count = summary.get("error_count")
    error_ratio = summary.get("error_ratio")
    if isinstance(error_count, (int, float)) and total_int:
        if isinstance(error_ratio, (int, float)):
            parts.append(f"errores {error_ratio:.0%} ({int(error_count)}/{total_int})")
        else:
            parts.append(f"errores {int(error_count)}/{total_int}")

    fallback_count = summary.get("fallback_count")
    fallback_ratio = summary.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and total_int:
        if isinstance(fallback_ratio, (int, float)):
            parts.append(f"fallbacks {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")
        elif int(fallback_count):
            parts.append(f"fallbacks {int(fallback_count)}")

    latency_line = _format_macro_latency(summary.get("latency_buckets"))
    if latency_line:
        parts.append(latency_line)

    summary_txt = f" — {' • '.join(parts)}" if parts else ""
    return format_note(
        f"{icon} {label}: {status_label} • {ts} ({elapsed_txt}){detail_txt}{summary_txt}"
    )


def _format_macro_overall(summary: Mapping[str, Any]) -> Optional[str]:
    total = summary.get("count")
    if not isinstance(total, (int, float)) or int(total) <= 0:
        return None
    total_int = int(total)
    status_counts = (
        summary.get("status_counts") if isinstance(summary.get("status_counts"), Mapping) else {}
    )
    status_ratios = (
        summary.get("status_ratios") if isinstance(summary.get("status_ratios"), Mapping) else {}
    )

    parts: list[str] = []
    success_count = status_counts.get("success")
    success_ratio = status_ratios.get("success")
    if isinstance(success_count, (int, float)) and isinstance(success_ratio, (int, float)):
        parts.append(f"éxitos {success_ratio:.0%} ({int(success_count)}/{total_int})")

    error_count = summary.get("error_count")
    error_ratio = summary.get("error_ratio")
    if isinstance(error_count, (int, float)) and isinstance(error_ratio, (int, float)):
        parts.append(f"errores {error_ratio:.0%} ({int(error_count)}/{total_int})")

    fallback_count = summary.get("fallback_count")
    fallback_ratio = summary.get("fallback_ratio")
    if isinstance(fallback_count, (int, float)) and isinstance(fallback_ratio, (int, float)):
        parts.append(f"fallbacks {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})")

    latency_line = _format_macro_latency(summary.get("latency_buckets"))
    if latency_line:
        parts.append(latency_line)

    if not parts:
        return None

    return format_note(f"📊 Totales macro ({total_int}) — {' • '.join(parts)}")


def _format_macro_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin datos macro registrados._"]

    providers = data.get("providers")
    if not isinstance(providers, Mapping) or not providers:
        return ["_Sin datos macro registrados._"]

    lines: list[str] = []
    overall = data.get("overall")
    if isinstance(overall, Mapping):
        overall_line = _format_macro_overall(overall)
        if overall_line:
            lines.append(overall_line)

    for key in sorted(providers):
        summary = providers.get(key)
        if not isinstance(summary, Mapping):
            continue
        lines.append(_format_macro_provider(summary))

    history_lines = list(_format_macro_history(data.get("attempts")))
    if history_lines:
        lines.append("_Historial de intentos recientes:_")
        lines.extend(history_lines)

    return lines or ["_Sin datos macro registrados._"]


def _sidebar_expander(label: str, *, expanded: bool = False):
    expander_fn = getattr(st.sidebar, "expander", None)
    if callable(expander_fn):
        return expander_fn(label, expanded=expanded)
    return st.expander(label, expanded=expanded)


def _format_tab_latency_entry(key: str, stats: Mapping[str, Any]) -> str:
    label = _TAB_LABELS.get(key, str(stats.get("label") or key).title())
    parts: list[str] = []

    avg = stats.get("avg")
    if isinstance(avg, (int, float)):
        parts.append(f"μ {float(avg):.0f} ms")

    percentiles = stats.get("percentiles")
    if isinstance(percentiles, Mapping):
        for name, display in (("p50", "P50"), ("p90", "P90"), ("p95", "P95")):
            value = percentiles.get(name)
            if isinstance(value, (int, float)):
                parts.append(f"{display} {float(value):.0f} ms")

    total = stats.get("total")
    total_int = int(total) if isinstance(total, (int, float)) else 0
    status_counts = stats.get("status_counts") if isinstance(stats.get("status_counts"), Mapping) else {}
    status_ratios = stats.get("status_ratios") if isinstance(stats.get("status_ratios"), Mapping) else {}

    success_count = status_counts.get("success")
    if isinstance(success_count, (int, float)) and total_int:
        success_ratio = status_ratios.get("success")
        if isinstance(success_ratio, (int, float)):
            parts.append(f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})")
        else:
            parts.append(f"OK {int(success_count)}/{total_int}")

    error_count = stats.get("error_count")
    if isinstance(error_count, (int, float)) and total_int:
        error_ratio = stats.get("error_ratio")
        if isinstance(error_ratio, (int, float)):
            parts.append(f"errores {int(error_count)}/{total_int} ({error_ratio:.0%})")
        else:
            parts.append(f"errores {int(error_count)}/{total_int}")

    missing_count = stats.get("missing_count")
    if isinstance(missing_count, (int, float)) and int(missing_count) > 0:
        parts.append(f"sin dato {int(missing_count)}")

    if not parts:
        parts.append("sin métricas registradas")

    return format_note(f"⏱️ {label} — {' • '.join(parts)}")


def _format_tab_latency_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping) or not data:
        return ["_Sin latencias registradas._"]

    lines: list[str] = []
    for key in sorted(data):
        stats = data.get(key)
        if not isinstance(stats, Mapping):
            continue
        lines.append(_format_tab_latency_entry(key, stats))
    return lines or ["_Sin latencias registradas._"]


def _format_adapter_fallback_line(
    adapter_label: str, provider_stats: Mapping[str, Any]
) -> str:
    provider_label = str(provider_stats.get("label") or "desconocido")
    count = provider_stats.get("count")
    fallback_count = provider_stats.get("fallback_count")
    fallback_ratio = provider_stats.get("fallback_ratio")
    status_counts = (
        provider_stats.get("status_counts")
        if isinstance(provider_stats.get("status_counts"), Mapping)
        else {}
    )
    status_ratios = (
        provider_stats.get("status_ratios")
        if isinstance(provider_stats.get("status_ratios"), Mapping)
        else {}
    )

    total_int = int(count) if isinstance(count, (int, float)) else 0
    metrics: list[str] = []
    if isinstance(fallback_count, (int, float)) and total_int:
        if isinstance(fallback_ratio, (int, float)):
            metrics.append(
                f"fallback {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})"
            )
        elif int(fallback_count):
            metrics.append(f"fallback {int(fallback_count)}/{total_int}")

    success_count = status_counts.get("success")
    if isinstance(success_count, (int, float)) and total_int:
        success_ratio = status_ratios.get("success")
        if isinstance(success_ratio, (int, float)):
            metrics.append(f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})")
        else:
            metrics.append(f"OK {int(success_count)}/{total_int}")

    error_count = status_counts.get("error")
    if isinstance(error_count, (int, float)) and total_int:
        error_ratio = status_ratios.get("error")
        if isinstance(error_ratio, (int, float)):
            metrics.append(f"errores {int(error_count)}/{total_int} ({error_ratio:.0%})")
        else:
            metrics.append(f"errores {int(error_count)}/{total_int}")

    if not metrics:
        metrics.append("sin métricas registradas")

    return format_note(
        f"🛟 {adapter_label} → {provider_label} — {' • '.join(metrics)}"
    )


def _format_adapter_fallback_section(data: Optional[Mapping[str, Any]]) -> Iterable[str]:
    if not isinstance(data, Mapping):
        return ["_Sin registros de fallbacks._"]

    adapters = data.get("adapters")
    lines: list[str] = []
    if isinstance(adapters, Mapping):
        for key in sorted(adapters):
            adapter_entry = adapters.get(key)
            if not isinstance(adapter_entry, Mapping):
                continue
            label = str(adapter_entry.get("label") or key).strip() or str(key)
            providers = adapter_entry.get("providers")
            if not isinstance(providers, Mapping):
                continue
            for provider_key in sorted(providers):
                provider_stats = providers.get(provider_key)
                if not isinstance(provider_stats, Mapping):
                    continue
                lines.append(_format_adapter_fallback_line(label, provider_stats))

    if not lines:
        return ["_Sin registros de fallbacks._"]

    provider_totals = data.get("providers")
    if isinstance(provider_totals, Mapping):
        aggregated: list[str] = []
        for provider_key in sorted(provider_totals):
            stats = provider_totals.get(provider_key)
            if not isinstance(stats, Mapping):
                continue
            label = str(stats.get("label") or provider_key).strip() or str(provider_key)
            count = stats.get("count")
            total_int = int(count) if isinstance(count, (int, float)) else 0
            fallback_count = stats.get("fallback_count")
            fallback_ratio = stats.get("fallback_ratio")
            metrics: list[str] = []
            if isinstance(fallback_count, (int, float)) and total_int:
                if isinstance(fallback_ratio, (int, float)):
                    metrics.append(
                        f"fallback {fallback_ratio:.0%} ({int(fallback_count)}/{total_int})"
                    )
                elif int(fallback_count):
                    metrics.append(f"fallback {int(fallback_count)}/{total_int}")
            status_counts = (
                stats.get("status_counts")
                if isinstance(stats.get("status_counts"), Mapping)
                else {}
            )
            status_ratios = (
                stats.get("status_ratios")
                if isinstance(stats.get("status_ratios"), Mapping)
                else {}
            )
            success_count = status_counts.get("success")
            if isinstance(success_count, (int, float)) and total_int:
                success_ratio = status_ratios.get("success")
                if isinstance(success_ratio, (int, float)):
                    metrics.append(
                        f"OK {int(success_count)}/{total_int} ({success_ratio:.0%})"
                    )
                else:
                    metrics.append(f"OK {int(success_count)}/{total_int}")
            if metrics:
                aggregated.append(
                    format_note(f"Σ {label} — {' • '.join(metrics)}")
                )
        if aggregated:
            lines.append("_Totales por proveedor:_")
            lines.extend(aggregated)

    return lines


def _format_opportunities_status(
    data: Optional[dict],
    history: Optional[Sequence[Mapping[str, Any]]] = None,
    stats: Optional[Mapping[str, Any]] = None,
) -> str:
    if not data and history:
        for entry in reversed(history):
            if isinstance(entry, Mapping):
                data = dict(entry)
                break

    if not data:
        return "_Sin screenings recientes._"

    mode = data.get("mode")
    icon = "✅" if mode == "hit" else "⚙️"
    label = "Cache reutilizada" if mode == "hit" else "Ejecución completa"
    elapsed = data.get("elapsed_ms")
    elapsed_txt = f"{float(elapsed):.0f} ms" if isinstance(elapsed, (int, float)) else "s/d"
    baseline = data.get("cached_elapsed_ms")
    baseline_txt = (
        f" • previo {float(baseline):.0f} ms" if isinstance(baseline, (int, float)) else ""
    )
    ts = _format_timestamp(data.get("ts"))
    metrics_parts: list[str] = []

    def _coerce_int(value: Any) -> Optional[int]:
        try:
            if value is None:
                return None
            return int(value)
        except (TypeError, ValueError):
            return None

    def _coerce_float(value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except (TypeError, ValueError):
            return None

    initial = _coerce_int(data.get("universe_initial"))
    final = _coerce_int(data.get("universe_final"))
    if initial is not None and final is not None:
        metrics_parts.append(f"universo {initial}→{final}")
    elif initial is not None:
        metrics_parts.append(f"universo inicial {initial}")
    elif final is not None:
        metrics_parts.append(f"universo final {final}")

    ratio = _coerce_float(data.get("discard_ratio"))
    if ratio is not None:
        metrics_parts.append(f"descartes {ratio:.0%}")

    sectors_raw = data.get("highlighted_sectors")
    if isinstance(sectors_raw, str):
        sectors = [sectors_raw.strip()] if sectors_raw.strip() else []
    elif isinstance(sectors_raw, Iterable) and not isinstance(
        sectors_raw, (bytes, bytearray, str)
    ):
        sectors = [str(item).strip() for item in sectors_raw if str(item).strip()]
    else:
        sectors = []
    if sectors:
        metrics_parts.append("sectores: " + ", ".join(sectors))

    origin_raw = data.get("counts_by_origin")
    origin_parts: list[str] = []
    if isinstance(origin_raw, Mapping):
        for key, value in origin_raw.items():
            origin_label = str(key).strip()
            if not origin_label:
                continue
            numeric = _coerce_float(value)
            if numeric is None:
                continue
            if float(numeric).is_integer():
                origin_parts.append(f"{origin_label}={int(numeric)}")
            else:
                origin_parts.append(f"{origin_label}={numeric:g}")
    if origin_parts:
        metrics_parts.append("origen: " + ", ".join(origin_parts))

    if stats and isinstance(stats, Mapping):
        summary_parts: list[str] = []
        elapsed_summary = stats.get("elapsed") if isinstance(stats, Mapping) else None
        if isinstance(elapsed_summary, Mapping):
            avg = elapsed_summary.get("avg")
            count = elapsed_summary.get("count")
            stdev = elapsed_summary.get("stdev")
            if isinstance(avg, (int, float)) and isinstance(count, (int, float)):
                count_int = int(count)
                if isinstance(stdev, (int, float)) and stdev > 0:
                    summary_parts.append(
                        f"prom {float(avg):.0f}±{float(stdev):.0f} ms (n={count_int})"
                    )
                else:
                    summary_parts.append(f"prom {float(avg):.0f} ms (n={count_int})")
        hit_ratio = stats.get("hit_ratio") if isinstance(stats, Mapping) else None
        mode_counts = stats.get("mode_counts") if isinstance(stats, Mapping) else None
        total_modes = stats.get("mode_total") if isinstance(stats, Mapping) else None
        if isinstance(hit_ratio, (int, float)) and isinstance(mode_counts, Mapping):
            hit_count = mode_counts.get("hit", 0)
            try:
                hit_count_int = int(hit_count)
            except (TypeError, ValueError):
                hit_count_int = 0
            total = int(total_modes) if isinstance(total_modes, (int, float)) else None
            if total:
                summary_parts.append(
                    f"hits {hit_ratio:.0%} ({hit_count_int}/{total})"
                )
        improvement = stats.get("improvement") if isinstance(stats, Mapping) else None
        if isinstance(improvement, Mapping):
            win_ratio = improvement.get("win_ratio")
            wins = improvement.get("wins")
            count = improvement.get("count")
            delta_avg = improvement.get("avg_delta_ms")
            if isinstance(win_ratio, (int, float)) and isinstance(count, (int, float)):
                try:
                    wins_int = int(wins)
                except (TypeError, ValueError):
                    wins_int = 0
                summary_parts.append(
                    f"mejoras {win_ratio:.0%} ({wins_int}/{int(count)})"
                )
            if isinstance(delta_avg, (int, float)):
                summary_parts.append(f"Δ̄ {float(delta_avg):+0.0f} ms vs caché")
        if summary_parts:
            metrics_parts.append("tendencia: " + " • ".join(summary_parts))

    metrics_txt = f" — {' | '.join(metrics_parts)}" if metrics_parts else ""
    return format_note(
        f"{icon} {label} • {ts} ({elapsed_txt}{baseline_txt}){metrics_txt}"
    )


def _format_opportunities_history(
    history: Optional[Iterable[Mapping[str, Any]]],
    stats: Optional[Mapping[str, Any]] = None,
) -> Iterable[str]:
    if not history:
        return ["_Sin historial disponible._"]

    average_elapsed: Optional[float] = None
    elapsed_stats = None
    if isinstance(stats, Mapping):
        elapsed_stats = stats.get("elapsed")
    if isinstance(elapsed_stats, Mapping):
        avg_candidate = elapsed_stats.get("avg")
        if isinstance(avg_candidate, (int, float)):
            average_elapsed = float(avg_candidate)

    header = "| Momento | Modo | t (ms) | Δ prom | Base |\n| --- | --- | --- | --- | --- |"
    rows = [header]

    for entry in history:
        if not isinstance(entry, Mapping):
            continue
        mode = entry.get("mode")
        icon = "✅" if mode == "hit" else "⚙️"
        label = "hit" if mode == "hit" else "miss"
        ts = _format_timestamp(entry.get("ts"))
        elapsed = entry.get("elapsed_ms")
        elapsed_txt = (
            f"{float(elapsed):.0f}" if isinstance(elapsed, (int, float)) else "s/d"
        )
        baseline = entry.get("cached_elapsed_ms")
        baseline_txt = (
            f"{float(baseline):.0f}" if isinstance(baseline, (int, float)) else "s/d"
        )
        if isinstance(elapsed, (int, float)) and average_elapsed is not None:
            delta = float(elapsed) - average_elapsed
            delta_txt = f"{delta:+.0f}"
        else:
            delta_txt = "s/d"
        rows.append(
            f"| {ts} | {icon} {label} | {elapsed_txt} | {delta_txt} | {baseline_txt} |"
        )

    if len(rows) == 1:
        return ["_Sin historial disponible._"]

    table_block = "\n".join(rows)
    return [table_block]


def render_health_sidebar() -> None:
    """Render the health summary panel inside the sidebar."""
    metrics = get_health_metrics()
    sidebar = st.sidebar
    sidebar.header(f"🩺 Healthcheck (versión {__version__})")
    sidebar.caption("Monitorea la procedencia y el rendimiento de los datos cargados.")

    sidebar.markdown("#### 🔐 Conexión IOL")
    sidebar.markdown(_format_iol_status(metrics.get("iol_refresh")))

    sidebar.markdown("#### 📈 Yahoo Finance")
    sidebar.markdown(_format_yfinance_status(metrics.get("yfinance")))

    sidebar.markdown("#### 🌐 Macro / Datos externos")
    for line in _format_macro_section(metrics.get("macro_api")):
        sidebar.markdown(line)

    sidebar.markdown("#### 💱 FX")
    for line in _format_fx_section(metrics.get("fx_api"), metrics.get("fx_cache")):
        sidebar.markdown(line)

    sidebar.markdown("#### 🔎 Screening de oportunidades")
    sidebar.markdown(
        _format_opportunities_status(
            metrics.get("opportunities"),
            metrics.get("opportunities_history"),
            metrics.get("opportunities_stats"),
        )
    )
    sidebar.markdown("#### 🗂️ Historial de screenings")
    history_entries = metrics.get("opportunities_history") or []
    for line in _format_opportunities_history(
        reversed(history_entries), metrics.get("opportunities_stats")
    ):
        sidebar.markdown(line)

    sidebar.markdown("#### 🛰️ Observabilidad")
    with _sidebar_expander("Latencias por pestaña"):
        for line in _format_tab_latency_section(metrics.get("tab_latencies")):
            st.markdown(line)
    with _sidebar_expander("Fallbacks por adaptador"):
        for line in _format_adapter_fallback_section(metrics.get("adapter_fallbacks")):
            st.markdown(line)

    sidebar.markdown("#### ⏱️ Latencias")
    for line in _format_latency_section(metrics.get("portfolio"), metrics.get("quotes")):
        sidebar.markdown(format_note(line))


__all__ = ["render_health_sidebar"]
