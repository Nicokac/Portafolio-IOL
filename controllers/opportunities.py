"""Controller helpers for the opportunities screener."""
from __future__ import annotations

import logging
import re
from typing import Any, Iterable, List, Mapping, Optional, Sequence, Tuple

import pandas as pd

from application.screener.opportunities import run_screener_stub

try:  # The Yahoo implementation may not be available in all environments.
    from application.screener.opportunities import run_screener_yahoo
except ImportError:  # pragma: no cover - fallback handled at runtime
    run_screener_yahoo = None  # type: ignore[assignment]

from shared.errors import AppError

_EXPECTED_COLUMNS: Sequence[str] = (
    "ticker",
    "payout_ratio",
    "dividend_streak",
    "cagr",
    "dividend_yield",
    "price",
    "score_compuesto",
)
_TECHNICAL_COLUMNS: Sequence[str] = ("rsi", "sma_50", "sma_200")


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


def _ensure_columns(df: pd.DataFrame, include_technicals: bool) -> pd.DataFrame:
    """Guarantee that the DataFrame exposes the expected schema."""

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
    return df[ordered_cols]


def run_opportunities_controller(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    include_technicals: bool = False,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: Optional[bool] = None,
) -> Tuple[pd.DataFrame, List[str], str]:
    """Run the opportunities screener and return the results, notes and source."""

    tickers = _clean_manual_tickers(manual_tickers)

    yahoo_kwargs: dict[str, Any] = {
        "manual_tickers": tickers or None,
        "include_technicals": include_technicals,
    }
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

    notes: List[str] = []
    fallback_used = False
    source = "yahoo"

    extra_notes: List[str] = []

    if callable(run_screener_yahoo):
        try:
            raw_result = run_screener_yahoo(**yahoo_kwargs)
        except AppError as exc:  # pragma: no cover - protective branch
            logging.getLogger(__name__).warning(
                "Yahoo screener failed with AppError: %s", exc
            )
            fallback_used = True
        except Exception as exc:  # pragma: no cover - unexpected failure
            logging.getLogger(__name__).exception(
                "Unexpected error from Yahoo screener"
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
        df = run_screener_stub(
            manual_tickers=tickers,
            max_payout=max_payout,
            min_div_streak=min_div_streak,
            min_cagr=min_cagr,
            min_market_cap=min_market_cap,
            max_pe=max_pe,
            min_revenue_growth=min_revenue_growth,
            include_latam=True if include_latam is None else include_latam,
            include_technicals=include_technicals,
        )
        notes.append("⚠️ Datos simulados (Yahoo no disponible)")
        df = _ensure_columns(df, include_technicals)
    else:
        notes.extend(extra_notes)

    if tickers:
        missing = []
        for ticker in tickers:
            rows = df[df["ticker"] == ticker]
            if rows.empty:
                missing.append(ticker)
            else:
                rows = rows.drop(columns=["ticker"], errors="ignore")
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


def generate_opportunities_report(
    filters: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, object]:
    """Entry point used by the UI to produce the opportunities report."""

    filters = filters or {}
    manual = filters.get("manual_tickers") or filters.get("tickers")
    include_technicals = bool(filters.get("include_technicals", False))

    df, notes, source = run_opportunities_controller(
        manual_tickers=manual,
        max_payout=_as_optional_float(filters.get("max_payout")),
        min_div_streak=_as_optional_int(filters.get("min_div_streak")),
        min_cagr=_as_optional_float(filters.get("min_cagr")),
        include_technicals=include_technicals,
        min_market_cap=_as_optional_float(filters.get("min_market_cap")),
        max_pe=_as_optional_float(filters.get("max_pe")),
        min_revenue_growth=_as_optional_float(filters.get("min_revenue_growth")),
        include_latam=_as_optional_bool(filters.get("include_latam")),
    )

    return {"table": df, "notes": notes, "source": source}


__all__ = ["run_opportunities_controller", "generate_opportunities_report"]
