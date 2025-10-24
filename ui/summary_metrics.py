from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np
import streamlit as st

from application.portfolio_service import PortfolioTotals, ValuationBreakdown, calculate_totals
from shared.time_provider import TimeProvider
from shared.utils import _as_float_or_none, format_money

_CURRENCY_STATE_KEY = "portfolio_summary_currency"
_ACTIVE_FX_LABEL_KEY = "portfolio_summary_fx_label"
_ACTIVE_FX_VALUE_KEY = "portfolio_summary_fx_value"

CURRENCY_STATE_KEY = _CURRENCY_STATE_KEY


@dataclass(frozen=True)
class _MetricValue:
    label: str
    value: float | None
    base_currency: str
    help_text: str | None = None


def _ensure_currency_state() -> str:
    try:
        state = st.session_state
    except Exception:  # pragma: no cover - defensive in tests without Streamlit
        return "ARS"
    if _CURRENCY_STATE_KEY not in state:
        state[_CURRENCY_STATE_KEY] = "ARS"
    raw = state.get(_CURRENCY_STATE_KEY)
    return raw if isinstance(raw, str) and raw in {"ARS", "USD"} else "ARS"


def get_active_summary_currency() -> str:
    """Return the active currency selected for the summary block."""

    return _ensure_currency_state()


def _read_cash_metadata(df_view: Any) -> Mapping[str, Any]:
    attrs = getattr(df_view, "attrs", {})
    if isinstance(attrs, Mapping):
        candidate = attrs.get("cash_balances")
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _extract_timestamp(meta: Mapping[str, Any]) -> str | None:
    candidates: Sequence[str] = ("ts", "timestamp", "updated_at")
    for key in candidates:
        ts_value = meta.get(key)
        snapshot = TimeProvider.from_timestamp(ts_value)
        if snapshot is not None:
            return snapshot.text
        if isinstance(ts_value, str) and ts_value.strip():
            return ts_value.strip()
    return None


def _infer_rate_label(rate_value: float | None) -> str:
    if rate_value is None or not np.isfinite(rate_value) or rate_value <= 0:
        return "Desconocido"
    try:
        state = st.session_state
    except Exception:  # pragma: no cover - defensive guard
        return "Desconocido"
    fx_rates: Mapping[str, Any] | None = None
    if isinstance(state, MutableMapping):
        fx_rates = state.get("fx_rates")
    candidates: list[tuple[str, float]] = []
    if isinstance(fx_rates, Mapping):
        oficial = _as_float_or_none(fx_rates.get("oficial"))
        mep = _as_float_or_none(fx_rates.get("mep"))
        if oficial is not None:
            candidates.append(("Oficial", float(oficial)))
        if mep is not None:
            candidates.append(("MEP", float(mep)))
    best_label = "Desconocido"
    best_diff: float | None = None
    for label, reference in candidates:
        diff = abs(reference - rate_value)
        tolerance = max(5.0, reference * 0.05)
        if diff <= tolerance and (best_diff is None or diff < best_diff):
            best_label = label
            best_diff = diff
    return best_label


def _build_estimation_help(breakdown: ValuationBreakdown | None) -> str | None:
    if not isinstance(breakdown, ValuationBreakdown):
        return None
    try:
        estimated_rows = int(getattr(breakdown, "estimated_rows", 0) or 0)
    except (TypeError, ValueError):
        estimated_rows = 0
    estimated_value = _as_float_or_none(getattr(breakdown, "estimated_value", None)) or 0.0
    if estimated_rows <= 0 or not np.isfinite(estimated_value) or estimated_value <= 0:
        return None
    impact = getattr(breakdown, "estimated_impact_pct", float("nan"))
    info_bits: list[str] = [f"{estimated_rows} activo{'s' if estimated_rows != 1 else ''} involucrado"]
    formatted_value = format_money(float(estimated_value))
    info_bits.append(f"≈ {formatted_value}")
    if isinstance(impact, (int, float)) and np.isfinite(float(impact)):
        info_bits.append(f"impacto estimado {float(impact):.2f}%")
    return "⚠️ Cotizaciones estimadas de proveedores externos. " + " • ".join(info_bits)


def _convert(value: float, *, source: str, target: str, rate: float | None) -> float | None:
    if not np.isfinite(value):
        return None
    if source == target:
        return value
    if rate is None or not np.isfinite(rate) or rate <= 0:
        return None
    if source == "ARS" and target == "USD":
        return value / rate
    if source == "USD" and target == "ARS":
        return value * rate
    return None


def _format_metric(
    metric: _MetricValue,
    *,
    target_currency: str,
    rate: float | None,
) -> tuple[str, str, dict[str, Any]]:
    converted = metric.value
    if metric.value is not None:
        converted = _convert(metric.value, source=metric.base_currency, target=target_currency, rate=rate)
        if converted is None and metric.base_currency == target_currency:
            converted = metric.value
    currency_kw = {"currency": target_currency} if converted is not None and target_currency == "USD" else {}
    if converted is None:
        display_value = "—"
    else:
        display_value = format_money(converted, **currency_kw)
    label_suffix = f" · {target_currency}"
    return metric.label + label_suffix, display_value, {"help": metric.help_text} if metric.help_text else {}


def _format_total_metric(label: str, value: float | None, *, currency: str, rate: float | None) -> tuple[str, str]:
    if value is None or not np.isfinite(value):
        return label, "—"
    if currency == "USD":
        converted = _convert(value, source="ARS", target="USD", rate=rate)
        if converted is None:
            return label, "—"
        return label, format_money(converted, currency="USD")
    return label, format_money(value)


def render_summary_metrics(
    df_view,
    *,
    totals: PortfolioTotals | None = None,
    ccl_rate: float | None = None,
) -> bool:
    """Render the consolidated summary metrics for the portfolio."""

    totals = totals or calculate_totals(df_view)
    if totals is None:
        totals = calculate_totals(df_view)

    currency_state = _ensure_currency_state()
    currency = st.radio(
        "Moneda base",
        options=("ARS", "USD"),
        key=_CURRENCY_STATE_KEY,
        horizontal=True,
        label_visibility="collapsed",
    )
    if currency not in {"ARS", "USD"}:
        currency = currency_state

    active_rate = _as_float_or_none(getattr(totals, "usd_rate", None))
    fallback_rate = _as_float_or_none(ccl_rate)
    rate = active_rate if active_rate is not None else fallback_rate
    rate_label = _infer_rate_label(rate)

    cash_meta = _read_cash_metadata(df_view)
    rate_ts = _extract_timestamp(cash_meta)
    rate_help_bits = [
        "Tipo de cambio aplicado según cotización vigente (Oficial o MEP).",
        "Fuente: /estadocuenta",
    ]
    if rate_label != "Desconocido":
        rate_help_bits.append(f"Referencia: {rate_label}")
    if rate_ts:
        rate_help_bits.append(f"Actualizado: {rate_ts}")
    rate_help = " • ".join(rate_help_bits)

    try:
        st.session_state[_ACTIVE_FX_LABEL_KEY] = rate_label
        st.session_state[_ACTIVE_FX_VALUE_KEY] = float(rate) if rate and np.isfinite(rate) else None
    except Exception:  # pragma: no cover - defensive in tests
        pass

    totals_block = st.container(border=True)
    with totals_block:
        st.markdown("#### 🟩 Totales del portafolio")
        row1 = st.columns(2)
        row2 = st.columns(2)
        estimation_help = _build_estimation_help(getattr(totals, "valuation_breakdown", None))
        label, value = _format_total_metric("Valorizado", totals.total_value, currency=currency, rate=rate)
        metric_kwargs: dict[str, Any] = {}
        if estimation_help:
            metric_kwargs["help"] = estimation_help
        row1[0].metric(label, value, **metric_kwargs)
        label, value = _format_total_metric("Costo", totals.total_cost, currency=currency, rate=rate)
        row1[1].metric(label, value)
        label, value = _format_total_metric("P/L", totals.total_pl, currency=currency, rate=rate)
        delta = None if not np.isfinite(totals.total_pl_pct) else f"{totals.total_pl_pct:.2f}%"
        row2[0].metric(label, value, delta=delta)
        row2[1].metric("P/L %", "—" if not np.isfinite(totals.total_pl_pct) else f"{totals.total_pl_pct:.2f}%")

    cash_metrics: list[_MetricValue] = [
        _MetricValue("Cash ARS", totals.total_cash_ars, "ARS", "Saldo líquido informado en pesos."),
        _MetricValue(
            "Cash USD",
            totals.total_cash_usd,
            "USD",
            "Saldo líquido informado en dólares.",
        ),
        _MetricValue(
            "Money Market",
            totals.total_cash,
            "ARS",
            "Incluye saldos de Money Market y valores de /estadocuenta.",
        ),
    ]

    cash_total_value: float | None = totals.total_cash_combined
    if currency == "USD":
        converted_cash: float | None = 0.0
        for metric in cash_metrics:
            base_value = metric.value if metric.value is not None else 0.0
            converted = _convert(base_value, source=metric.base_currency, target="USD", rate=rate)
            if converted is None:
                converted_cash = None
                break
            converted_cash += converted
        cash_total_value = converted_cash
    cash_total_help = "Incluye saldos de Money Market y valores de /estadocuenta."
    cash_metrics.append(_MetricValue("Cash total", cash_total_value, currency, cash_total_help))

    liquidity_block = st.container(border=True)
    with liquidity_block:
        st.markdown("#### 🟦 Efectivo y Money Market")
        row = st.columns(3)
        for idx, metric in enumerate(cash_metrics[:3]):
            label, display_value, extra = _format_metric(metric, target_currency=currency, rate=rate)
            row[idx % 3].metric(label, display_value, **extra)
        rate_value = "—"
        if rate and np.isfinite(rate):
            rate_value = format_money(float(rate))
        row_rate, row_total = st.columns(2)
        row_rate.metric("Tipo de cambio", rate_value, help=rate_help)
        cash_total_label, cash_total_display, total_extra = _format_metric(
            cash_metrics[-1], target_currency=currency, rate=rate
        )
        row_total.metric(cash_total_label, cash_total_display, **total_extra)

    st.caption(
        "Los saldos en efectivo incluyen posiciones Money Market y valores de /estadocuenta. "
        f"Moneda base seleccionada: {currency}."
    )
    return True
