"""Controller helpers for the opportunities screener."""
from __future__ import annotations

import logging
import math
import re
import time
from collections import OrderedDict
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from numbers import Number

import pandas as pd

from application.screener.opportunities import run_screener_stub

try:  # The Yahoo implementation may not be available in all environments.
    from application.screener.opportunities import run_screener_yahoo
except ImportError:  # pragma: no cover - fallback handled at runtime
    run_screener_yahoo = None  # type: ignore[assignment]

from shared.errors import AppError
from services.health import record_opportunities_report

_EXPECTED_COLUMNS: Sequence[str] = (
    "ticker",
    "sector",
    "payout_ratio",
    "dividend_streak",
    "cagr",
    "dividend_yield",
    "price",
    "Yahoo Finance Link",
    "score_compuesto",
)
_TECHNICAL_COLUMNS: Sequence[str] = ("rsi", "sma_50", "sma_200")

_CACHE_MAX_ENTRIES = 64
_CacheEntry = Tuple[Mapping[str, object], float]
_OPPORTUNITIES_CACHE: "OrderedDict[tuple, _CacheEntry]" = OrderedDict()
_OPPORTUNITIES_CACHE_LOCK = Lock()


def _normalize_for_key(value: Any) -> Any:
    if isinstance(value, Mapping):
        return tuple((k, _normalize_for_key(v)) for k, v in sorted(value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_normalize_for_key(v) for v in value)
    return value


def _build_cache_key(controller_args: Mapping[str, Any]) -> tuple:
    return tuple(
        (key, _normalize_for_key(controller_args.get(key)))
        for key in sorted(controller_args.keys())
    )


def _get_cached_result(key: tuple) -> Optional[_CacheEntry]:
    with _OPPORTUNITIES_CACHE_LOCK:
        entry = _OPPORTUNITIES_CACHE.get(key)
        if entry is None:
            return None
        _OPPORTUNITIES_CACHE.move_to_end(key)
        return entry


def _store_cached_result(key: tuple, result: Mapping[str, object], elapsed_ms: float) -> None:
    with _OPPORTUNITIES_CACHE_LOCK:
        _OPPORTUNITIES_CACHE[key] = (result, elapsed_ms)
        _OPPORTUNITIES_CACHE.move_to_end(key)
        while len(_OPPORTUNITIES_CACHE) > _CACHE_MAX_ENTRIES:
            _OPPORTUNITIES_CACHE.popitem(last=False)


def _clear_opportunities_cache() -> None:
    with _OPPORTUNITIES_CACHE_LOCK:
        _OPPORTUNITIES_CACHE.clear()


def _build_yahoo_link(ticker: object) -> str | pd.NA:
    if ticker is None:
        return pd.NA
    normalized = str(ticker).strip().upper()
    if not normalized:
        return pd.NA
    return f"https://finance.yahoo.com/quote/{normalized}"


def _summarize_active_filters(filters: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return a compact mapping with the filters that are effectively active."""

    summary: dict[str, Any] = {}
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, frozenset)) and not value:
            continue
        if isinstance(value, Mapping) and not value:
            continue
        summary[key] = value
    return summary


def _format_filter_value(key: str, value: Any) -> Optional[str]:
    """Format a single filter value into a human readable fragment."""

    def _format_percentage(raw: Any) -> Optional[str]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        return f"{numeric:g}%"

    def _format_number(raw: Any) -> Optional[str]:
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            return None
        if numeric.is_integer():
            return f"{int(numeric)}"
        return f"{numeric:g}"

    if key in {"manual_tickers", "exclude_tickers", "sectors"}:
        if isinstance(value, Iterable) and not isinstance(value, (str, bytes, bytearray)):
            items = [str(item) for item in value if item]
            if not items:
                return None
            label = {
                "manual_tickers": "tickers",
                "exclude_tickers": "excluye",
                "sectors": "sectores",
            }[key]
            return f"{label}: {', '.join(items)}"
        return None
    if key == "include_technicals":
        if bool(value):
            return "indicadores técnicos"
        return None
    if key == "include_latam":
        if value is True:
            return "incluye Latam"
        if value is False:
            return "excluye Latam"
        return None
    if key == "max_results":
        number = _format_number(value)
        if number is None:
            return None
        return f"máx resultados ≤{number}"
    if key == "max_payout":
        percentage = _format_percentage(value)
        if percentage is None:
            return None
        return f"payout ≤{percentage}"
    if key == "min_div_streak":
        try:
            years = int(value)
        except (TypeError, ValueError):
            return None
        return f"racha dividendos ≥{years} años"
    if key == "min_cagr":
        percentage = _format_percentage(value)
        if percentage is None:
            return None
        return f"CAGR dividendos ≥{percentage}"
    if key == "min_market_cap":
        number = _format_number(value)
        if number is None:
            return None
        return f"market cap ≥{number}"
    if key == "max_pe":
        number = _format_number(value)
        if number is None:
            return None
        return f"P/E ≤{number}"
    if key == "min_revenue_growth":
        percentage = _format_percentage(value)
        if percentage is None:
            return None
        return f"crecimiento ingresos ≥{percentage}"
    if key == "min_eps_growth":
        percentage = _format_percentage(value)
        if percentage is None:
            return None
        return f"crecimiento EPS ≥{percentage}"
    if key == "min_buyback":
        percentage = _format_percentage(value)
        if percentage is None:
            return None
        return f"buyback ≥{percentage}"
    if key == "min_score_threshold":
        number = _format_number(value)
        if number is None:
            return None
        return f"score ≥{number}"
    return None


def _build_filters_note(filters: Mapping[str, Any]) -> Optional[str]:
    """Convert active filters into a readable summary note."""

    fragments: List[str] = []
    for key, value in filters.items():
        formatted = _format_filter_value(key, value)
        if formatted:
            fragments.append(formatted)
    if not fragments:
        return None
    return "ℹ️ Filtros aplicados: " + ", ".join(fragments)


def _describe_exception(exc: BaseException) -> str:
    message = str(exc).strip()
    if message:
        return message
    return exc.__class__.__name__


def _clean_manual_tickers(manual_tickers: Optional[Iterable[str]]) -> List[str]:
    """Normalise manual tickers by stripping whitespace and uppercasing."""

    if not manual_tickers:
        return []
    if isinstance(manual_tickers, str):
        manual_tickers = [manual_tickers]
    cleaned: List[str] = []
    seen = set()
    for raw in manual_tickers:
        if raw is None:
            continue
        tickers = re.split(r"[\s,;]+", str(raw))
        for ticker in tickers:
            ticker_clean = ticker.strip().upper()
            if not ticker_clean or ticker_clean in seen:
                continue
            seen.add(ticker_clean)
            cleaned.append(ticker_clean)
    return cleaned


def _clean_sectors(sectors: Optional[Iterable[str]]) -> List[str]:
    if not sectors:
        return []
    cleaned: List[str] = []
    seen: set[str] = set()
    for raw in sectors:
        if raw is None:
            continue
        sector = str(raw).strip()
        if not sector:
            continue
        sector_title = sector.title()
        key = sector_title.casefold()
        if key in seen:
            continue
        seen.add(key)
        cleaned.append(sector_title)
    return cleaned


def _ensure_columns(df: pd.DataFrame, include_technicals: bool) -> pd.DataFrame:
    """Guarantee that the DataFrame exposes the expected schema."""

    preserved_attrs = dict(getattr(df, "attrs", {}))
    df = df.copy()
    expected_columns = list(_EXPECTED_COLUMNS)
    if include_technicals:
        expected_columns.extend(col for col in _TECHNICAL_COLUMNS)

    for column in expected_columns:
        if column not in df.columns:
            df[column] = pd.NA

    # Drop unexpected technical columns when the flag is disabled.
    if not include_technicals:
        df = df.drop(columns=[c for c in _TECHNICAL_COLUMNS if c in df.columns])

    # Reorder columns for consistency.
    ordered_cols = [c for c in expected_columns if c in df.columns]
    df = df[ordered_cols]

    if "ticker" in df.columns and "Yahoo Finance Link" in df.columns:
        links = df["ticker"].map(_build_yahoo_link)
        df.loc[:, "Yahoo Finance Link"] = links.astype("string")

    if preserved_attrs:
        df.attrs.update(preserved_attrs)

    return df


def run_opportunities_controller(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    exclude_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    sectors: Optional[Iterable[str]] = None,
    include_technicals: bool = False,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: Optional[bool] = None,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Run the opportunities screener and return the results, notes and source."""

    tickers = _clean_manual_tickers(manual_tickers)
    excluded = _clean_manual_tickers(exclude_tickers)
    excluded_set = set(excluded)

    selected_sectors = _clean_sectors(sectors)

    min_score_value: Optional[float] = None
    if min_score_threshold is not None:
        try:
            min_score_value = float(min_score_threshold)
        except (TypeError, ValueError):
            min_score_value = None

    max_results_value: Optional[int] = None
    if max_results is not None:
        try:
            max_results_value = int(max_results)
        except (TypeError, ValueError):
            max_results_value = None

    yahoo_kwargs: dict[str, Any] = {
        "manual_tickers": tickers or None,
        "include_technicals": include_technicals,
    }
    if excluded_set:
        yahoo_kwargs["exclude_tickers"] = sorted(excluded_set)
    if max_payout is not None:
        yahoo_kwargs["max_payout"] = float(max_payout)
    if min_div_streak is not None:
        yahoo_kwargs["min_div_streak"] = int(min_div_streak)
    if min_cagr is not None:
        yahoo_kwargs["min_cagr"] = float(min_cagr)
    if min_market_cap is not None:
        yahoo_kwargs["min_market_cap"] = float(min_market_cap)
    if max_pe is not None:
        yahoo_kwargs["max_pe"] = float(max_pe)
    if min_revenue_growth is not None:
        yahoo_kwargs["min_revenue_growth"] = float(min_revenue_growth)
    if include_latam is not None:
        yahoo_kwargs["include_latam"] = bool(include_latam)
    if min_eps_growth is not None:
        yahoo_kwargs["min_eps_growth"] = float(min_eps_growth)
    if min_buyback is not None:
        yahoo_kwargs["min_buyback"] = float(min_buyback)
    if min_score_value is not None:
        yahoo_kwargs["min_score_threshold"] = float(min_score_value)
    if max_results_value is not None:
        yahoo_kwargs["max_results"] = int(max_results_value)
    if selected_sectors:
        yahoo_kwargs["sectors"] = selected_sectors


    notes: List[str] = []
    fallback_used = False
    source = "yahoo"

    extra_notes: List[str] = []
    failure_reason: Optional[str] = None
    active_filters = _summarize_active_filters(yahoo_kwargs)
    filters_note = _build_filters_note(active_filters)

    if callable(run_screener_yahoo):
        try:
            raw_result = run_screener_yahoo(**yahoo_kwargs)
        except AppError as exc:  # pragma: no cover - protective branch
            failure_reason = _describe_exception(exc)
            logging.getLogger(__name__).warning(
                "Yahoo screener failed with AppError: %s | filtros activos: %s",
                failure_reason,
                active_filters,
            )
            fallback_used = True
        except Exception as exc:  # pragma: no cover - unexpected failure
            failure_reason = _describe_exception(exc)
            logging.getLogger(__name__).exception(
                "Unexpected error from Yahoo screener | filtros activos: %s",
                active_filters,
            )
            fallback_used = True
        else:
            df, extra_notes = _normalize_yahoo_response(
                raw_result, include_technicals
            )
    else:
        fallback_used = True

    if fallback_used:
        source = "stub"
        stub_result = run_screener_stub(
            manual_tickers=tickers,
            exclude_tickers=excluded or None,
            max_payout=max_payout,
            min_div_streak=min_div_streak,
            min_cagr=min_cagr,
            min_market_cap=min_market_cap,
            max_pe=max_pe,
            min_revenue_growth=min_revenue_growth,
            include_latam=True if include_latam is None else include_latam,
            include_technicals=include_technicals,
            min_eps_growth=min_eps_growth,
            min_buyback=min_buyback,
            min_score_threshold=min_score_value,
            max_results=max_results_value,
            sectors=selected_sectors or None,
        )
        stub_notes: List[str] = []
        if isinstance(stub_result, tuple) and len(stub_result) == 2:
            df, stub_notes = stub_result
        else:
            df = stub_result  # type: ignore[assignment]
        fallback_note: Optional[str] = None
        if failure_reason:
            fallback_note = f"⚠️ Yahoo no disponible — Causa: {failure_reason}"
        if fallback_note:
            notes.append(fallback_note)
        if filters_note:
            notes.append(filters_note)
        if stub_notes:
            notes.extend(_normalize_notes(stub_notes))
        df = _ensure_columns(df, include_technicals)
    else:
        if filters_note:
            notes.append(filters_note)
        notes.extend(extra_notes)

    if tickers:
        missing = []
        for ticker in tickers:
            if ticker in excluded_set:
                continue
            rows = df[df["ticker"] == ticker]
            if rows.empty:
                missing.append(ticker)
            else:
                rows = rows.drop(
                    columns=["ticker", "Yahoo Finance Link"], errors="ignore"
                )
                if rows.isna().all(axis=None):
                    missing.append(ticker)
        if missing:
            notes.append(
                "No se encontraron datos para: " + ", ".join(sorted(set(missing)))
            )

    return df, notes, source


def _normalize_yahoo_response(
    result: object, include_technicals: bool
) -> Tuple[pd.DataFrame, List[str]]:
    """Convert different Yahoo screener payloads into a DataFrame and notes."""

    notes: List[str] = []

    if isinstance(result, tuple) and len(result) == 2:
        table, raw_notes = result
        df, nested_notes = _normalize_yahoo_response(table, include_technicals)
        notes.extend(nested_notes)
        notes.extend(_normalize_notes(raw_notes))
        return df, notes

    if isinstance(result, Mapping):
        table = None
        for key in ("table", "data", "df"):
            if key in result:
                table = result[key]
                break
        if table is not None:
            df, nested_notes = _normalize_yahoo_response(table, include_technicals)
            notes.extend(nested_notes)
        else:
            df = pd.DataFrame()
        for key in ("notes", "messages", "warnings"):
            value = result.get(key)
            if value:
                notes.extend(_normalize_notes(value))
        return _ensure_columns(df, include_technicals), notes

    if isinstance(result, pd.DataFrame):
        return _ensure_columns(result, include_technicals), notes

    try:
        df = pd.DataFrame(result)
    except Exception:  # pragma: no cover - defensive guard
        df = pd.DataFrame()
    return _ensure_columns(df, include_technicals), notes


def _normalize_notes(value: object) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    if isinstance(value, Mapping):
        notes: List[str] = []
        for item in value.values():
            notes.extend(_normalize_notes(item))
        return notes
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        notes: List[str] = []
        for item in value:
            notes.extend(_normalize_notes(item))
        return notes
    return [str(value)]


def _as_optional_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_optional_int(value: Any) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_optional_bool(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "y", "t"}:
            return True
        if lowered in {"false", "0", "no", "n", "f"}:
            return False
    return None


def _extract_opportunities_metrics(payload: Mapping[str, Any]) -> Dict[str, Any]:
    metrics: Dict[str, Any] = {}
    raw_metrics = payload.get("metrics")
    if isinstance(raw_metrics, Mapping):
        metrics.update(raw_metrics)

    for key in (
        "universe_initial",
        "universe_final",
        "discard_ratio",
        "highlighted_sectors",
        "counts_by_origin",
    ):
        if key in payload and key not in metrics:
            metrics[key] = payload[key]

    return metrics


def _normalise_controller_payload(
    payload: Any,
) -> Tuple[pd.DataFrame, List[str], str, Dict[str, Any]]:
    if isinstance(payload, Mapping):
        table = payload.get("table")
        if isinstance(table, pd.DataFrame):
            df = table
        else:
            try:
                df = pd.DataFrame(table)
            except Exception:
                df = pd.DataFrame()

        notes = _normalize_notes(payload.get("notes"))
        source = str(payload.get("source") or "desconocido")
        metrics = _extract_opportunities_metrics(payload)
        return df, notes, source, metrics

    if isinstance(payload, tuple) and len(payload) == 3:
        df, notes, source = payload
        normalized_notes = _normalize_notes(notes)
        return df, normalized_notes, source, {}

    return pd.DataFrame(), [], "desconocido", {}
def _convert_summary_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _convert_summary_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_convert_summary_value(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, Number):
        try:
            return _convert_summary_value(value.item())
        except Exception:  # pragma: no cover - defensive branch
            pass
    if isinstance(value, Number):
        numeric = float(value)
        if math.isnan(numeric) or math.isinf(numeric):
            return None
        if float(numeric).is_integer():
            return int(round(numeric))
        return float(numeric)
    if isinstance(value, str):
        return value
    if value is None:
        return None
    try:
        if pd.isna(value):  # type: ignore[arg-type]
            return None
    except Exception:  # pragma: no cover - defensive branch
        pass
    return value


def _build_summary_payload(
    table: pd.DataFrame | None, filters: Mapping[str, Any]
) -> Mapping[str, Any]:
    summary: dict[str, Any] = {}
    if isinstance(table, pd.DataFrame):
        raw_summary = getattr(table, "attrs", {}).get("summary")
        if isinstance(raw_summary, Mapping):
            summary.update(raw_summary)

    if isinstance(table, pd.DataFrame):
        summary.setdefault("result_count", int(table.index.size))
        summary.setdefault("universe_count", summary.get("result_count"))
    else:
        summary.setdefault("result_count", 0)

    if "discarded_ratio" not in summary:
        try:
            universe = int(summary.get("universe_count") or 0)
            result_count = int(summary.get("result_count") or 0)
        except (TypeError, ValueError):
            universe = 0
            result_count = 0
        summary["discarded_ratio"] = (
            (max(universe - result_count, 0) / universe) if universe else 0.0
        )

    selected_sectors = summary.get("selected_sectors")
    if not selected_sectors:
        raw_sectors = filters.get("sectors") or []
        if isinstance(raw_sectors, (list, tuple, set, frozenset)):
            selected_sectors = [str(item) for item in raw_sectors if item]
        elif raw_sectors:
            selected_sectors = [str(raw_sectors)]
        else:
            selected_sectors = []
        summary["selected_sectors"] = selected_sectors

    normalized = {str(key): _convert_summary_value(value) for key, value in summary.items()}
    return normalized


def generate_opportunities_report(
    filters: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, object]:
    """Entry point used by the UI to produce the opportunities report."""

    filters = filters or {}
    manual = filters.get("manual_tickers") or filters.get("tickers")
    include_technicals_value = filters.get("include_technicals")
    include_technicals_parsed = _as_optional_bool(include_technicals_value)
    if include_technicals_parsed is not None:
        include_technicals = include_technicals_parsed
    elif isinstance(include_technicals_value, bool):
        include_technicals = include_technicals_value
    else:
        include_technicals = False

    controller_args = {
        "manual_tickers": manual,
        "exclude_tickers": filters.get("exclude_tickers"),
        "max_payout": _as_optional_float(filters.get("max_payout")),
        "min_div_streak": _as_optional_int(filters.get("min_div_streak")),
        "min_cagr": _as_optional_float(filters.get("min_cagr")),
        "sectors": filters.get("sectors"),
        "include_technicals": include_technicals,
        "min_market_cap": _as_optional_float(filters.get("min_market_cap")),
        "max_pe": _as_optional_float(filters.get("max_pe")),
        "min_revenue_growth": _as_optional_float(filters.get("min_revenue_growth")),
        "include_latam": _as_optional_bool(filters.get("include_latam")),
        "min_eps_growth": _as_optional_float(filters.get("min_eps_growth")),
        "min_buyback": _as_optional_float(filters.get("min_buyback")),
        "min_score_threshold": _as_optional_float(filters.get("min_score_threshold")),
        "max_results": _as_optional_int(filters.get("max_results")),
    }

    cache_key = _build_cache_key(controller_args)
    start = time.perf_counter()
    cached_entry = _get_cached_result(cache_key)
    if cached_entry is not None:
        cached_result, baseline_elapsed = cached_entry
        metrics_payload: Dict[str, Any] = {}
        if isinstance(cached_result, Mapping):
            metrics_payload = _extract_opportunities_metrics(cached_result)
        elapsed_ms = (time.perf_counter() - start) * 1000
        record_opportunities_report(
            mode="hit",
            elapsed_ms=elapsed_ms,
            cached_elapsed_ms=baseline_elapsed,
            **metrics_payload,
        )
        return cached_result

    compute_start = time.perf_counter()
    raw_payload = run_opportunities_controller(**controller_args)
    df, notes, source, metrics_payload = _normalise_controller_payload(raw_payload)
    elapsed_ms = (time.perf_counter() - compute_start) * 1000
    result: Dict[str, object] = {"table": df, "notes": notes, "source": source}
    if metrics_payload:
        result["metrics"] = metrics_payload
    summary = _build_summary_payload(df, filters)
    result: Mapping[str, object] = {
        "table": df,
        "notes": notes,
        "source": source,
        "summary": summary,
    }
    _store_cached_result(cache_key, result, elapsed_ms)
    record_opportunities_report(
        mode="miss",
        elapsed_ms=elapsed_ms,
        cached_elapsed_ms=None,
        **metrics_payload,
    )
    return result


__all__ = ["run_opportunities_controller", "generate_opportunities_report"]
