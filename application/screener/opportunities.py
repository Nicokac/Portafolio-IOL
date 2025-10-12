"""Opportunities screener helpers for both stub data and Yahoo integration.

The module exposes a deterministic dataset used for local development as
well as the functions that orchestrate the Yahoo Finance client, allowing UI
components and controllers to rely on a single entry point regardless of the
backing data source.
"""
from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, List, Optional

import numpy as np
import pandas as pd

from infrastructure.market import YahooFinanceClient
from infrastructure.market.yahoo_client import make_symbol_url
from shared import config as shared_config
from shared.errors import AppError
from shared.settings import settings as shared_settings

LOGGER = logging.getLogger(__name__)

_MAX_TICKER_WORKERS = 8
_REQUEST_DELAY = float(getattr(shared_settings, "yahoo_request_delay", 0.0))

_FILTER_LABELS: dict[str, str] = {
    "max_payout": "max_payout",
    "min_div_streak": "min_div_streak",
    "min_cagr": "min_cagr",
    "min_market_cap": "min_market_cap",
    "max_pe": "max_pe",
    "min_revenue_growth": "min_revenue_growth",
    "min_eps_growth": "min_eps_growth",
    "min_buyback": "min_buyback",
    "min_score_threshold": "min_score_threshold",
    "include_latam": "include_latam",
    "sectors": "sectors",
    "exclude_tickers": "exclude_tickers",
    "restrict_to_tickers": "restrict_to_tickers",
    "max_results": "max_results",
}


def _summarize_filter_telemetry(
    filter_telemetry: Sequence[tuple[str, int, int]],
    *,
    include_latam: bool | None,
    max_results: Optional[int],
    sectors: Optional[Iterable[str]],
    manual_tickers: Optional[Iterable[str]],
    exclude_tickers: Optional[Iterable[str]],
) -> tuple[list[str], list[tuple[str, str, int]]]:
    """Return human readable summaries for the applied filters."""

    filter_metrics: list[str] = []
    drop_summary_entries: list[tuple[str, str, int]] = []

    for name, before, after in filter_telemetry:
        dropped = max(int(before) - int(after), 0)
        ratio = (dropped / before) if before else 0.0
        label = _FILTER_LABELS.get(name, name)
        if name == "include_latam":
            label = f"include_latam={include_latam}"
        elif name == "max_results" and max_results is not None:
            label = f"max_results={max_results}"
        elif name == "sectors" and sectors:
            label = "sectors"
        elif name == "restrict_to_tickers" and manual_tickers:
            label = "manual_tickers"
        elif name == "exclude_tickers" and exclude_tickers:
            label = "exclude_tickers"
        filter_metrics.append(f"{label}: {dropped}/{before} ({ratio:.0%})")
        if dropped > 0:
            drop_summary_entries.append((name, label, dropped))

    return filter_metrics, drop_summary_entries


def _compute_sector_distribution(df: pd.DataFrame) -> dict[str, int]:
    if "sector" not in df.columns:
        return {}
    series = (
        df["sector"].astype("string").fillna("Sin sector").str.strip().replace("", "Sin sector")
    )
    counts = series.value_counts()
    return {str(index): int(count) for index, count in counts.items()}


# Base dataset used to simulate screener output.
_BASE_FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "fixtures"
    / "base_opportunities.csv"
)
_BASE_OPPORTUNITIES: list[dict[str, object]] = []
_BASE_OPPORTUNITIES_LOADED_AT: pd.Timestamp | None = None


def _load_base_opportunities_fixture() -> tuple[list[dict[str, object]], pd.Timestamp]:
    """Load the deterministic opportunities fixture from disk."""

    if not _BASE_FIXTURE_PATH.exists():
        raise AppError(
            f"No se encontró el dataset base en {_BASE_FIXTURE_PATH}"
        )

    try:
        df = pd.read_csv(_BASE_FIXTURE_PATH)
    except FileNotFoundError as exc:
        raise AppError(
            "El dataset base de oportunidades no está disponible"
        ) from exc
    except Exception as exc:  # pragma: no cover - defensive guard
        LOGGER.exception(
            "Error al cargar el fixture de oportunidades desde %s",
            _BASE_FIXTURE_PATH,
        )
        raise AppError("No se pudo cargar el dataset base de oportunidades") from exc

    df = df.convert_dtypes()
    records = df.to_dict("records")
    loaded_at = pd.Timestamp.utcnow()
    LOGGER.info(
        "Fixture base_opportunities.csv cargado (%s filas) — timestamp=%s",
        len(records),
        loaded_at.isoformat(),
    )
    return records, loaded_at


def _ensure_base_opportunities_loaded() -> None:
    """Populate :data:`_BASE_OPPORTUNITIES` from the fixture when needed."""

    global _BASE_OPPORTUNITIES, _BASE_OPPORTUNITIES_LOADED_AT
    if _BASE_OPPORTUNITIES:
        return
    records, loaded_at = _load_base_opportunities_fixture()
    _BASE_OPPORTUNITIES = records
    _BASE_OPPORTUNITIES_LOADED_AT = loaded_at


_ensure_base_opportunities_loaded()


def _get_target_markets() -> list[str]:
    try:
        configured = getattr(shared_settings, "OPPORTUNITIES_TARGET_MARKETS", [])
    except Exception:  # pragma: no cover - defensive fallback
        configured = []

    if isinstance(configured, str):
        configured = [configured]

    markets: list[str] = []
    seen: set[str] = set()
    for entry in configured or []:
        market = str(entry or "").strip().upper()
        if not market or market in seen:
            continue
        markets.append(market)
        seen.add(market)

    if not markets:
        markets = ["NASDAQ", "NYSE", "AMEX"]

    return markets


def _normalise_tickers(manual_tickers: Optional[Sequence[str]]) -> List[str]:
    """Normalise tickers to uppercase strings without duplicates."""

    if not manual_tickers:
        return []
    if isinstance(manual_tickers, str):
        manual_tickers = [manual_tickers]
    seen = set()
    cleaned: List[str] = []
    for ticker in manual_tickers:
        if ticker is None:
            continue
        ticker_clean = str(ticker).strip().upper()
        if not ticker_clean or ticker_clean in seen:
            continue
        seen.add(ticker_clean)
        cleaned.append(ticker_clean)
    return cleaned


def _append_placeholder_rows(df: pd.DataFrame, tickers: Sequence[str]) -> pd.DataFrame:
    """Ensure that the DataFrame contains rows for each provided ticker."""

    if not tickers:
        return df
    df = df.set_index("ticker")
    df = df.reindex(tickers)
    df.index.name = "ticker"
    return df.reset_index()


def _format_yahoo_link(ticker: object) -> str | pd.NA:
    """Build the Yahoo Finance quote URL for ``ticker`` when possible."""

    url = make_symbol_url(ticker)
    return url if url is not None else pd.NA


def _normalise_score_column(
    df: pd.DataFrame, column: str = "score_compuesto"
) -> None:
    """Clamp ``column`` values to the 0-100 range in-place."""

    if column not in df.columns:
        return

    scores = pd.to_numeric(df[column], errors="coerce")
    if scores.isna().all():
        df[column] = pd.NA
        return

    clamped = scores.clip(lower=0.0, upper=100.0)
    df[column] = clamped.round(2).where(~scores.isna(), pd.NA)


def _apply_filters_and_finalize(
    df: pd.DataFrame,
    *,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,
    include_latam_flag: bool | None = True,
    include_technicals: bool,
    restrict_to_tickers: Sequence[str] | None = None,
    placeholder_tickers: Sequence[str] | None = None,
    exclude_tickers: Sequence[str] | None = None,
    market_cap_column: str = "market_cap",
    pe_ratio_column: str = "pe_ratio",
    revenue_growth_column: str = "revenue_growth",
    latam_column: str = "is_latam",
    trailing_eps_column: str = "trailing_eps",
    forward_eps_column: str = "forward_eps",
    buyback_column: str = "buyback",
    allowed_sectors: Sequence[str] | None = None,
    sector_column: str = "sector",
    allow_na_filters: bool = False,
    extra_drop_columns: Sequence[str] | None = None,
    missing_optional_labels: Mapping[str, bool] | None = None,
    filter_telemetry: list[tuple[str, int, int]] | None = None,
) -> pd.DataFrame:
    """Apply common filters and final adjustments for screener outputs."""

    result = df.copy()
    notes: list[str] = []
    critical_missing: dict[str, set[str]] = {}
    optional_missing: dict[str, set[str]] = {}
    optional_lookup = {
        str(label): bool(flag)
        for label, flag in (missing_optional_labels or {}).items()
    }

    def record_filter(name: str, before: int, after: int) -> None:
        if filter_telemetry is None:
            return
        filter_telemetry.append((name, int(before), int(after)))

    def register_missing(mask: pd.Series, label: str) -> None:
        """Record tickers with missing critical data when NA filters are allowed."""

        if not allow_na_filters or "ticker" not in result.columns:
            return
        normalized_mask = mask.fillna(False)
        if not normalized_mask.any():
            return
        tickers_series = (
            result.loc[normalized_mask, "ticker"].astype("string").str.strip().str.upper()
        )
        tickers = [ticker for ticker in tickers_series if ticker]
        if not tickers:
            return
        if optional_lookup.get(label, False):
            optional_missing.setdefault(label, set()).update(tickers)
        else:
            critical_missing.setdefault(label, set()).update(tickers)
    exclude_set: set[str] | None = None
    if exclude_tickers:
        exclude_set = {
            str(ticker).strip().upper()
            for ticker in exclude_tickers
            if str(ticker).strip()
        }
    normalized_allowed: set[str] | None = None
    if allowed_sectors:
        normalized_allowed = {str(value).casefold() for value in allowed_sectors}
    original_sector_series: pd.Series | None = None
    if sector_column in df.columns:
        original_sector_series = (
            df.set_index("ticker")[sector_column]
            .astype("string")
            .str.strip()
            .str.casefold()
        )

    if max_payout is not None and "payout_ratio" in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result["payout_ratio"], errors="coerce")
        if allow_na_filters:
            register_missing(series.isna(), "ratio de payout")
        mask = series.notna() & (series <= max_payout)
        result = result[mask]
        record_filter("max_payout", before_count, len(result))

    if min_div_streak is not None and "dividend_streak" in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result["dividend_streak"], errors="coerce")
        if allow_na_filters:
            register_missing(series.isna(), "racha de dividendos")
        mask = series.notna() & (series >= min_div_streak)
        result = result[mask]
        record_filter("min_div_streak", before_count, len(result))

    if min_cagr is not None and "cagr" in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result["cagr"], errors="coerce")
        if allow_na_filters:
            register_missing(series.isna(), "CAGR")
        mask = series.notna() & (series >= min_cagr)
        result = result[mask]
        record_filter("min_cagr", before_count, len(result))

    if min_market_cap is not None and market_cap_column in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result[market_cap_column], errors="coerce")
        register_missing(series.isna(), "capitalización bursátil")
        mask = series.notna() & (series >= float(min_market_cap))
        result = result[mask]
        record_filter("min_market_cap", before_count, len(result))

    if max_pe is not None and pe_ratio_column in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result[pe_ratio_column], errors="coerce")
        register_missing(series.isna(), "P/E")
        mask = series.notna() & (series <= float(max_pe))
        result = result[mask]
        record_filter("max_pe", before_count, len(result))

    if min_revenue_growth is not None and revenue_growth_column in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result[revenue_growth_column], errors="coerce")
        register_missing(series.isna(), "crecimiento de ingresos")
        mask = series.notna() & (series >= float(min_revenue_growth))
        result = result[mask]
        record_filter("min_revenue_growth", before_count, len(result))

    if trailing_eps_column in result.columns:
        series = pd.to_numeric(result[trailing_eps_column], errors="coerce")
        register_missing(series.isna(), "EPS trailing")
        mask = (series > 0) & series.notna()
        result = result[mask]

    if forward_eps_column in result.columns:
        series = pd.to_numeric(result[forward_eps_column], errors="coerce")
        register_missing(series.isna(), "EPS forward")
        mask = (series > 0) & series.notna()
        result = result[mask]

    if (
        min_eps_growth is not None
        and trailing_eps_column in result.columns
        and forward_eps_column in result.columns
    ):
        trailing = pd.to_numeric(result[trailing_eps_column], errors="coerce")
        forward = pd.to_numeric(result[forward_eps_column], errors="coerce")
        denominator = trailing.replace(0, pd.NA)
        growth = (forward - trailing) / denominator
        growth = growth.replace([np.inf, -np.inf], pd.NA) * 100.0
        mask = (trailing > 0) & (forward > 0) & (growth >= float(min_eps_growth))
        mask = mask.fillna(False)
        register_missing(growth.isna(), "crecimiento de EPS")
        before_count = len(result)
        result = result[mask]
        record_filter("min_eps_growth", before_count, len(result))

    if min_buyback is not None and buyback_column in result.columns:
        before_count = len(result)
        series = pd.to_numeric(result[buyback_column], errors="coerce")
        if allow_na_filters:
            register_missing(series.isna(), "recompras")
        mask = series.notna() & (series >= float(min_buyback))
        result = result[mask]
        record_filter("min_buyback", before_count, len(result))

    if min_score_threshold is not None and "score_compuesto" in result.columns:
        threshold = float(min_score_threshold)
        series = pd.to_numeric(result["score_compuesto"], errors="coerce")
        before_threshold = len(result)
        mask = series >= threshold
        if allow_na_filters:
            mask = series.isna() | mask
        result = result[mask]
        record_filter("min_score_threshold", before_threshold, len(result))
        if before_threshold > 0 and result.empty:
            notes.append(
                "Ningún ticker superó el puntaje mínimo de "
                f"{threshold:g}."
            )

    if include_latam_flag is False and latam_column in result.columns:
        before_count = len(result)
        result = result[~result[latam_column].fillna(False)]
        record_filter("include_latam", before_count, len(result))

    if normalized_allowed and sector_column in result.columns:
        before_count = len(result)
        result = result[
            result[sector_column]
            .astype("string")
            .str.strip()
            .str.casefold()
            .isin(normalized_allowed)
        ]
        record_filter("sectors", before_count, len(result))

    if exclude_set and "ticker" in result.columns:
        before_count = len(result)
        normalized_tickers = result["ticker"].astype("string").str.upper()
        result = result[~normalized_tickers.isin(exclude_set)]
        record_filter("exclude_tickers", before_count, len(result))

    if restrict_to_tickers:
        before_count = len(result)
        result = result[result["ticker"].isin(restrict_to_tickers)]
        record_filter("restrict_to_tickers", before_count, len(result))

    if placeholder_tickers:
        placeholders = list(placeholder_tickers)
        if exclude_set:
            placeholders = [
                ticker
                for ticker in placeholders
                if str(ticker).strip().upper() not in exclude_set
            ]
        if normalized_allowed and original_sector_series is not None:
            placeholders = [
                ticker
                for ticker in placeholders
                if ticker in original_sector_series.index
                and original_sector_series[ticker] in normalized_allowed
            ]
        if placeholders:
            result = _append_placeholder_rows(result, placeholders)

    if extra_drop_columns:
        to_drop = [col for col in extra_drop_columns if col in result.columns]
        if to_drop:
            result = result.drop(columns=to_drop)

    if not include_technicals:
        technical_columns = ["rsi", "sma_50", "sma_200"]
        to_drop = [col for col in technical_columns if col in result.columns]
        if to_drop:
            result = result.drop(columns=to_drop)

    result = result.reset_index(drop=True)

    if "score_compuesto" in result.columns:
        _normalise_score_column(result)
        result = result.sort_values(
            by="score_compuesto",
            ascending=False,
            na_position="last",
        )
    if max_results is not None:
        try:
            limit = int(max_results)
        except (TypeError, ValueError):
            limit = None
        if limit is not None and limit >= 0:
            before_truncate = len(result)
            if limit == 0:
                result = result.iloc[0:0]
            elif before_truncate > limit:
                result = result.iloc[:limit]
            if before_truncate > len(result):
                record_filter("max_results", before_truncate, len(result))
                notes.append(
                    "Se muestran "
                    f"{len(result)} resultados de {before_truncate} tras aplicar el máximo solicitado ({limit})."
                )

    target_min_results: Optional[int] = None
    for candidate in (
        getattr(shared_settings, "OPPORTUNITIES_MIN_RESULTS", None),
        getattr(shared_config.settings, "OPPORTUNITIES_MIN_RESULTS", None),
    ):
        if target_min_results is not None:
            break
        if candidate is None:
            continue
        try:
            target_min_results = int(candidate)
        except (TypeError, ValueError):
            target_min_results = None
    if target_min_results and len(result) < target_min_results:
        notes.append(
            "Solo se encontraron "
            f"{len(result)} oportunidades (mínimo esperado: {target_min_results})."
        )

    if critical_missing:
        for label, tickers in critical_missing.items():
            ordered = sorted(tickers)
            if not ordered:
                continue
            count = len(ordered)
            preview_list = ordered[:5]
            preview = ", ".join(preview_list)
            if count > len(preview_list):
                preview += f", … (+{count - len(preview_list)} más)"
            verb = "Se descartó" if count == 1 else "Se descartaron"
            noun = "el ticker" if count == 1 else "los tickers"
            if preview:
                message = (
                    f"{verb} {noun} {preview} por falta de datos críticos de {label}."
                )
            else:
                message = f"{verb} {noun} por falta de datos críticos de {label}."
            notes.append(message)

    if optional_missing:
        for label, tickers in optional_missing.items():
            ordered = sorted(tickers)
            if not ordered:
                continue
            count = len(ordered)
            preview_list = ordered[:5]
            preview = ", ".join(preview_list)
            if count > len(preview_list):
                preview += f", … (+{count - len(preview_list)} más)"
            noun = "el ticker" if count == 1 else "los tickers"
            if preview:
                message = (
                    f"Faltan datos opcionales de {label} para {noun} {preview}."
                )
            else:
                message = f"Faltan datos opcionales de {label}."
            notes.append(message)

    result = result.reset_index(drop=True)

    if "ticker" in result.columns:
        links = result["ticker"].map(_format_yahoo_link)
        result.loc[:, "Yahoo Finance Link"] = pd.Series(
            links, index=result.index, dtype="string"
        )

    if notes:
        result.attrs.setdefault("_notes", [])
        result.attrs["_notes"].extend(notes)

    return result

def run_screener_stub(
    *,
    manual_tickers: Optional[Iterable[str]] = None,
    exclude_tickers: Optional[Iterable[str]] = None,
    max_payout: Optional[float] = None,
    min_div_streak: Optional[int] = None,
    min_cagr: Optional[float] = None,
    min_market_cap: Optional[float] = None,
    max_pe: Optional[float] = None,
    min_revenue_growth: Optional[float] = None,
    include_latam: bool = True,
    include_technicals: bool = False,
    min_eps_growth: Optional[float] = None,
    min_buyback: Optional[float] = None,
    sectors: Optional[Iterable[str]] = None,
    min_score_threshold: Optional[float] = None,
    max_results: Optional[int] = None,

) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Return a filtered sample dataset that mimics a screener output.

    Parameters
    ----------
    manual_tickers:
        Optional iterable of tickers to focus on. When provided, the returned
        DataFrame will contain rows for these tickers (missing data will be
        represented with ``NaN`` values).
    exclude_tickers:
        Iterable opcional de símbolos que deben descartarse del resultado
        final incluso si aparecen en la base simulada o en la lista manual.
    max_payout:
        Maximum payout ratio percentage allowed. Rows exceeding this value are
        dropped.
    min_div_streak:
        Minimum number of consecutive years of dividend growth required.
    min_cagr:
        Minimum compound annual growth rate required.
    min_market_cap:
        Minimum market capitalisation (expressed in USD millions) required.
    max_pe:
        Maximum price to earnings ratio allowed.
    min_revenue_growth:
        Minimum year-over-year revenue growth percentage required.
    min_eps_growth:
        Minimum EPS growth percentage (forward vs trailing) required when
        available. When ``None`` the growth filter is skipped but EPS must be
        positive to remain in the dataset.
    min_buyback:
        Minimum buyback percentage (share reduction) required. ``None`` keeps
        companies regardless of their buyback ratio.
    min_score_threshold:
        Minimum composite score required. Rows below the threshold are
        discarded unless ``allow_na_filters`` is enabled.
    max_results:
        Maximum number of rows returned after applying all filters.
    include_latam:
        When ``False`` securities flagged as originating from Latin America are
        excluded from the results.
    include_technicals:
        When ``True`` technical indicators (e.g., RSI, SMAs) are included in the
        DataFrame. When ``False`` those columns are omitted.
    """

    loop_start = time.perf_counter()

    df = pd.DataFrame(_BASE_OPPORTUNITIES)
    df["Yahoo Finance Link"] = df["ticker"].map(_format_yahoo_link)
    manual = _normalise_tickers(manual_tickers)
    excluded = set(_normalise_tickers(exclude_tickers))

    if excluded:
        df = df[~df["ticker"].isin(excluded)]
        manual = [ticker for ticker in manual if ticker not in excluded]

    df = df.copy()

    def _score_from_row(row: pd.Series) -> float | pd.NA:
        metrics = {
            "payout_ratio": row.get("payout_ratio"),
            "dividend_streak": row.get("dividend_streak"),
            "cagr": row.get("cagr"),
            "buyback": row.get("buyback"),
            "rsi": row.get("rsi"),
        }
        return _compute_score(metrics)

    df["score_compuesto"] = df.apply(_score_from_row, axis=1)

    universe_count = int(df.index.size)
    filter_telemetry: list[tuple[str, int, int]] = []
    normalized_sectors = _normalize_sector_filters(sectors)

    result = _apply_filters_and_finalize(
        df,
        max_payout=max_payout,
        min_div_streak=min_div_streak,
        min_cagr=min_cagr,
        min_market_cap=min_market_cap,
        max_pe=max_pe,
        min_revenue_growth=min_revenue_growth,
        min_eps_growth=min_eps_growth,
        min_buyback=min_buyback,
        min_score_threshold=min_score_threshold,
        max_results=max_results,
        include_latam_flag=include_latam,
        include_technicals=include_technicals,
        restrict_to_tickers=manual or None,
        placeholder_tickers=manual or None,
        exclude_tickers=sorted(excluded) or None,
        market_cap_column="market_cap",
        pe_ratio_column="pe_ratio",
        revenue_growth_column="revenue_growth",
        latam_column="is_latam",
        extra_drop_columns=(
            "trailing_eps",
            "forward_eps",
            "buyback",
        ),
        allowed_sectors=normalized_sectors,
        filter_telemetry=filter_telemetry,
    )

    elapsed = time.perf_counter() - loop_start
    result_count = int(result.index.size)
    discarded_count = max(universe_count - result_count, 0)
    discarded_ratio = (discarded_count / universe_count) if universe_count else 0.0
    filter_metrics, drop_summary_entries = _summarize_filter_telemetry(
        filter_telemetry,
        include_latam=include_latam,
        max_results=max_results,
        sectors=sectors,
        manual_tickers=manual,
        exclude_tickers=exclude_tickers,
    )

    metrics_summary = "ningún filtro aplicado"
    if filter_metrics:
        metrics_summary = ", ".join(filter_metrics)

    drop_parts: list[str] = []
    for name, label, dropped in drop_summary_entries:
        percentage = (dropped / universe_count) if universe_count else 0.0
        drop_parts.append(f"{percentage:.0%} descartados por {label}")

    drop_summary = ", ".join(drop_parts) if drop_parts else "sin descartes"

    warn_threshold = getattr(
        shared_settings,
        "STUB_MAX_RUNTIME_WARN",
        getattr(shared_settings, "stub_max_runtime_warn", 0.25),
    )
    note_prefix = "⚠️" if elapsed > warn_threshold else "ℹ️"
    telemetry_note = (
        f"{note_prefix} Stub procesó {universe_count} tickers en {elapsed:.2f} segundos "
        f"({discarded_ratio:.0%} descartados, resultado: {result_count}; {drop_summary})"
    )

    LOGGER.info("%s. Descartes: %s", telemetry_note, metrics_summary)

    summary_payload = {
        "universe_count": universe_count,
        "result_count": result_count,
        "discarded_ratio": discarded_ratio,
        "elapsed_seconds": float(elapsed),
        "selected_sectors": list(normalized_sectors),
        "sector_distribution": _compute_sector_distribution(result),
        "filter_descriptions": list(filter_metrics),
        "drop_summary": drop_summary,
        "source": "stub",
    }

    result.attrs["summary"] = summary_payload

    notes_attr = result.attrs.setdefault("_notes", [])
    notes_attr.append(telemetry_note)
    if filter_metrics:
        for metric in filter_metrics:
            notes_attr.append(f"• {metric}")

    return result, list(notes_attr)


def _is_valid_number(value: float | int | pd.NA | None) -> bool:
    return value is not None and not pd.isna(value)


def _safe_round(value: float | pd.NA | None, digits: int = 2) -> float | pd.NA:
    if not _is_valid_number(value):
        return pd.NA
    return round(float(value), digits)


def _compute_cagr(price_history: pd.DataFrame | None, years: int) -> float | pd.NA:
    if price_history is None or price_history.empty:
        return pd.NA
    data = price_history.sort_values("date")["date"].reset_index(drop=True)
    prices = price_history.sort_values("date")["adj_close"].astype(float).reset_index(drop=True)
    end_price = prices.iloc[-1]
    if end_price <= 0:
        return pd.NA
    cutoff = data.iloc[-1] - pd.DateOffset(years=years)
    mask = data <= cutoff
    if not mask.any():
        return pd.NA
    start_price = prices[mask].iloc[-1]
    if start_price <= 0:
        return pd.NA
    cagr = (end_price / start_price) ** (1 / years) - 1
    return float(cagr * 100.0)


def _compute_dividend_streak(dividends: pd.DataFrame | None) -> int | pd.NA:
    if dividends is None or dividends.empty:
        return pd.NA
    df = dividends.copy()
    df["year"] = df["date"].dt.year
    annual = df.groupby("year")["amount"].sum().sort_index()
    if annual.empty:
        return pd.NA
    streak = 0
    previous = None
    for _, value in reversed(list(annual.items())):
        if value <= 0:
            break
        if previous is None:
            streak = 1
            previous = value
            continue
        if value >= previous:
            streak += 1
            previous = value
        else:
            break
    return streak or pd.NA


def _compute_buyback_ratio(shares: pd.DataFrame | None) -> float | pd.NA:
    if shares is None or shares.empty:
        return pd.NA
    df = shares.sort_values("date")
    start = float(df["shares"].iloc[0])
    end = float(df["shares"].iloc[-1])
    if start <= 0 or end <= 0:
        return pd.NA
    return float((start - end) / start * 100.0)


def _compute_rsi(price_history: pd.DataFrame | None, period: int = 14) -> float | pd.NA:
    if price_history is None or len(price_history) <= period:
        return pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    delta = closes.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    if avg_gain.empty or avg_loss.empty:
        return pd.NA
    last_gain = avg_gain.iloc[-1]
    last_loss = avg_loss.iloc[-1]
    if pd.isna(last_gain) or pd.isna(last_loss):
        return pd.NA
    if last_loss == 0:
        return 100.0
    rs = last_gain / last_loss
    return float(100 - (100 / (1 + rs)))


def _compute_macd(price_history: pd.DataFrame | None, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[float | pd.NA, float | pd.NA]:
    if price_history is None or len(price_history) < slow + signal:
        return pd.NA, pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    ema_fast = closes.ewm(span=fast, adjust=False).mean()
    ema_slow = closes.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    hist = macd_line - signal_line
    macd_value = macd_line.iloc[-1]
    hist_value = hist.iloc[-1]
    if pd.isna(macd_value) or pd.isna(hist_value):
        return pd.NA, pd.NA
    return float(macd_value), float(hist_value)


def _compute_sma(price_history: pd.DataFrame | None, window: int) -> float | pd.NA:
    if price_history is None or len(price_history) < window:
        return pd.NA
    closes = price_history.sort_values("date")["adj_close"].astype(float)
    sma = closes.rolling(window=window).mean().iloc[-1]
    return float(sma) if not pd.isna(sma) else pd.NA


def _compute_score(metrics: dict[str, float | int | pd.NA]) -> float | pd.NA:
    contributions: list[float] = []

    payout = metrics.get("payout_ratio")
    if _is_valid_number(payout):
        contributions.append(max(0.0, min(100.0, 100.0 - float(payout))) / 10.0)

    streak = metrics.get("dividend_streak")
    if _is_valid_number(streak):
        contributions.append(min(float(streak), 50.0) / 5.0)

    cagr = metrics.get("cagr")
    if _is_valid_number(cagr):
        contributions.append(max(0.0, min(25.0, float(cagr))) / 2.5)

    buyback = metrics.get("buyback")
    if _is_valid_number(buyback):
        contributions.append(max(0.0, min(20.0, float(buyback))) / 2.0)

    rsi = metrics.get("rsi")
    if _is_valid_number(rsi):
        contributions.append(max(0.0, 100.0 - abs(float(rsi) - 50.0) * 2.0) / 10.0)

    macd_hist = metrics.get("macd_hist")
    if _is_valid_number(macd_hist):
        contributions.append((max(-5.0, min(5.0, float(macd_hist))) + 5.0))

    if not contributions:
        return pd.NA

    score = sum(contributions) / len(contributions)
    return round(score * 10.0, 2)


def _fetch_with_warning(
    fetcher: Callable[[str], object], ticker: str, label: str
) -> object | None:
    try:
        return fetcher(ticker)
    except AppError as exc:
        LOGGER.warning("Faltan datos de %s para %s: %s", label, ticker, exc)
        return None


def _output_columns(include_technicals: bool) -> list[str]:
    columns = [
        "ticker",
        "sector",
        "payout_ratio",
        "dividend_streak",
        "cagr",
        "dividend_yield",
        "price",
        "Yahoo Finance Link",
        "rsi",
        "sma_50",
        "sma_200",
        "score_compuesto",
    ]
    if include_technicals:
        return columns
    return [c for c in columns if c not in {"rsi", "sma_50", "sma_200"}]


_LATAM_COUNTRIES = {
    "argentina",
    "bolivia",
    "brazil",
    "chile",
    "colombia",
    "costa rica",
    "cuba",
    "dominican republic",
    "ecuador",
    "el salvador",
    "guatemala",
    "honduras",
    "mexico",
    "nicaragua",
    "panama",
    "paraguay",
    "peru",
    "uruguay",
    "venezuela",
}


def _normalize_sector_name(value: object) -> str | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    text = str(value).strip()
    if not text:
        return pd.NA
    return text.title()


def _normalize_sector_filters(values: Optional[Iterable[str]]) -> list[str]:
    if not values:
        return []
    normalized: list[str] = []
    seen: set[str] = set()
    for raw in values:
        normalized_value = _normalize_sector_name(raw)
        if normalized_value is pd.NA:
            continue
        key = str(normalized_value).casefold()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(str(normalized_value))
    return normalized


def _precheck_symbols(
    tickers: Sequence[str],
    *,
    listings_meta: Mapping[str, Mapping[str, object]] | None,
    min_market_cap: float | None,
    max_pe: float | None,
    min_revenue_growth: float | None,
) -> tuple[list[str], dict[str, list[str]], float]:
    """Apply cheap heuristics before downloading historical data."""

    listings_meta = listings_meta or {}
    minimum_market_cap = float(min_market_cap if min_market_cap is not None else 500_000_000.0)
    maximum_pe = float(max_pe if max_pe is not None else 50.0)
    minimum_revenue_growth = float(min_revenue_growth if min_revenue_growth is not None else 0.0)

    kept: list[str] = []
    discarded: dict[str, list[str]] = {}

    for ticker in tickers:
        entry = listings_meta.get(ticker, {})
        if not isinstance(entry, Mapping):
            entry = {}

        reasons: list[str] = []

        market_cap = _pick_first_numeric(
            (
                entry.get("market_cap"),
                entry.get("marketCap"),
            )
        )
        if market_cap is not pd.NA and _is_valid_number(market_cap):
            if float(market_cap) < minimum_market_cap:
                reasons.append(f"market_cap<{minimum_market_cap:,.0f}")

        pe_ratio = _pick_first_numeric(
            (
                entry.get("pe_ratio"),
                entry.get("trailingPE"),
                entry.get("pe"),
            )
        )
        if pe_ratio is not pd.NA and _is_valid_number(pe_ratio):
            if float(pe_ratio) > maximum_pe:
                reasons.append(f"pe_ratio>{maximum_pe:,.0f}")

        revenue_growth = _pick_first_numeric((entry.get("revenue_growth"),))
        if revenue_growth is not pd.NA and _is_valid_number(revenue_growth):
            if float(revenue_growth) < minimum_revenue_growth:
                reasons.append(f"revenue_growth<{minimum_revenue_growth:,.2f}")

        if reasons:
            discarded[ticker] = reasons
        else:
            kept.append(ticker)

    total = len(tickers)
    ratio = (len(discarded) / total) if total else 0.0

    if discarded:
        LOGGER.info(
            "Precheck descartó %s de %s tickers antes del fetch detallado: %s",
            len(discarded),
            total,
            ", ".join(sorted(discarded)),
        )
    else:
        LOGGER.info("Precheck validó %s tickers sin descartes previos", total)

    return kept, discarded, ratio


def _as_optional_float(value: object) -> float | pd.NA:
    if value is None or pd.isna(value):
        return pd.NA
    try:
        return float(value)
    except (TypeError, ValueError):
        return pd.NA


def _pick_first_numeric(values: Iterable[object]) -> float | pd.NA:
    for raw in values:
        result = _as_optional_float(raw)
        if result is not pd.NA:
            return result
    return pd.NA


def _is_latam_country(country: object | None) -> bool:
    if not country:
        return False
    normalized = str(country).strip().lower()
    if not normalized:
        return False
    return normalized in _LATAM_COUNTRIES


def _build_ticker_row(
    ticker: str,
    *,
    fundamentals: Mapping[str, object] | None,
    dividends: pd.DataFrame | None,
    shares: pd.DataFrame | None,
    prices: pd.DataFrame | None,
    listings_meta: Mapping[str, Mapping[str, object]],
) -> dict[str, object]:
    payout_raw = fundamentals.get("payout_ratio") if fundamentals else pd.NA
    payout = _safe_round(payout_raw)
    dividend_yield = _safe_round(fundamentals["dividend_yield"] if fundamentals else pd.NA)
    dividend_streak = _compute_dividend_streak(dividends)

    cagr_values: list[float] = []
    for years in (3, 5):
        value = _compute_cagr(prices, years)
        if _is_valid_number(value):
            cagr_values.append(float(value))
    cagr = _safe_round(sum(cagr_values) / len(cagr_values) if cagr_values else pd.NA)

    price = pd.NA
    if prices is not None and not prices.empty:
        price = _safe_round(prices.sort_values("date")["close"].iloc[-1])

    rsi = _safe_round(_compute_rsi(prices))
    sma_50 = _safe_round(_compute_sma(prices, 50))
    sma_200 = _safe_round(_compute_sma(prices, 200))
    _, macd_hist = _compute_macd(prices)
    macd_hist = _safe_round(macd_hist, digits=4) if _is_valid_number(macd_hist) else pd.NA
    buyback = _safe_round(_compute_buyback_ratio(shares))

    metrics = {
        "payout_ratio": payout,
        "dividend_streak": dividend_streak,
        "cagr": cagr,
        "buyback": buyback,
        "rsi": rsi,
        "macd_hist": macd_hist,
    }
    score = _compute_score(metrics)

    listing_meta = listings_meta.get(ticker, {})
    if not isinstance(listing_meta, Mapping):
        listing_meta = {}

    market_cap = _pick_first_numeric(
        (
            fundamentals.get("market_cap") if fundamentals else None,
            fundamentals.get("marketCap") if fundamentals else None,
            listing_meta.get("market_cap"),
            listing_meta.get("marketCap"),
        )
    )
    pe_ratio = _pick_first_numeric(
        (
            fundamentals.get("pe_ratio") if fundamentals else None,
            fundamentals.get("trailingPE") if fundamentals else None,
            fundamentals.get("pe") if fundamentals else None,
            listing_meta.get("pe_ratio"),
            listing_meta.get("trailingPE"),
            listing_meta.get("pe"),
        )
    )
    revenue_growth = _pick_first_numeric(
        (
            fundamentals.get("revenue_growth") if fundamentals else None,
            listing_meta.get("revenue_growth"),
        )
    )
    trailing_eps = _as_optional_float(
        fundamentals.get("trailing_eps") if fundamentals else pd.NA
    )
    if trailing_eps is pd.NA and fundamentals and "trailingEps" in fundamentals:
        trailing_eps = _as_optional_float(fundamentals.get("trailingEps"))
    forward_eps = _as_optional_float(
        fundamentals.get("forward_eps") if fundamentals else pd.NA
    )
    if forward_eps is pd.NA and fundamentals and "forwardEps" in fundamentals:
        forward_eps = _as_optional_float(fundamentals.get("forwardEps"))
    is_latam = False
    country: object | None = None
    sector = pd.NA
    if fundamentals:
        country = fundamentals.get("country") or fundamentals.get("region")
        sector = _normalize_sector_name(fundamentals.get("sector"))
    if not country and listing_meta:
        country = listing_meta.get("country") or listing_meta.get("region")
    is_latam = _is_latam_country(country)

    return {
        "ticker": ticker,
        "sector": sector,
        "payout_ratio": payout,
        "dividend_streak": dividend_streak,
        "cagr": cagr,
        "dividend_yield": dividend_yield,
        "price": price,
        "Yahoo Finance Link": _format_yahoo_link(ticker),
        "rsi": rsi,
        "sma_50": sma_50,
        "sma_200": sma_200,
        "score_compuesto": score,
        "_meta_market_cap": market_cap,
        "_meta_pe_ratio": pe_ratio,
        "_meta_revenue_growth": revenue_growth,
        "_meta_is_latam": is_latam,
        "_meta_trailing_eps": trailing_eps,
        "_meta_forward_eps": forward_eps,
        "_meta_buyback": _as_optional_float(buyback),
    }


def run_screener_yahoo(
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
    client: YahooFinanceClient | None = None,
) -> pd.DataFrame | tuple[pd.DataFrame, list[str]]:
    """Run the Yahoo-based screener returning the same schema as the stub.

    Parameters
    ----------
    exclude_tickers:
        Iterable opcional de símbolos a excluir de los resultados finales aun
        cuando aparezcan en la respuesta del proveedor de datos.
    """

    manual_requested = _normalise_tickers(manual_tickers)
    tickers = list(manual_requested)
    excluded = set(_normalise_tickers(exclude_tickers))
    if excluded:
        tickers = [ticker for ticker in tickers if ticker not in excluded]
        manual_requested = [
            ticker for ticker in manual_requested if ticker not in excluded
        ]
    notes: list[str] = []
    using_default_universe = False
    listings_meta: dict[str, Mapping[str, object]] = {}
    target_markets: list[str] = []

    client = client or YahooFinanceClient()

    if not tickers:
        target_markets = _get_target_markets()
        listings = client.list_symbols_by_markets(target_markets)
        seen: set[str] = set()
        for entry in listings:
            if isinstance(entry, Mapping):
                metadata = dict(entry)
                ticker_value = metadata.get("ticker") or metadata.get("symbol")
            else:
                metadata = {}
                ticker_value = entry

            ticker_clean = str(ticker_value or "").strip().upper()
            if (
                not ticker_clean
                or ticker_clean in seen
                or ticker_clean in excluded
            ):
                continue

            metadata["ticker"] = ticker_clean
            listings_meta[ticker_clean] = metadata
            tickers.append(ticker_clean)
            seen.add(ticker_clean)

        using_default_universe = True

    precheck_initial_count = len(tickers)
    precheck_dropped: dict[str, list[str]] = {}
    precheck_ratio = 0.0
    if tickers:
        tickers, precheck_dropped, precheck_ratio = _precheck_symbols(
            tickers,
            listings_meta=listings_meta,
            min_market_cap=min_market_cap,
            max_pe=max_pe,
            min_revenue_growth=min_revenue_growth,
        )

    if not tickers:
        columns = _output_columns(include_technicals)
        df = pd.DataFrame(columns=columns)
        if using_default_universe:
            message = "No se encontraron símbolos que cumplan los filtros especificados."
            if target_markets:
                message += " Mercados consultados: " + ", ".join(target_markets)
            notes.append(message)
        if precheck_dropped:
            notes.append(
                f"Precheck descartó {len(precheck_dropped)} símbolos antes del análisis detallado."
            )
        return (df, notes) if notes else df
    rows: list[dict[str, object]] = []
    sector_filters = _normalize_sector_filters(sectors)

    loop_start = time.perf_counter()
    results: dict[str, dict[str, object]] = {}
    elapsed_per_ticker: dict[str, float] = {}
    errors_by_ticker: dict[str, list[str]] = {}

    def _process_ticker(symbol: str) -> tuple[dict[str, object] | None, float, list[str]]:
        start = time.perf_counter()
        issues: list[str] = []

        def _fetch(
            fetcher: Callable[[str], object],
            label: str,
        ) -> object | None:
            result = _fetch_with_warning(fetcher, symbol, label)
            if result is None:
                issues.append(label)
            if _REQUEST_DELAY > 0:
                time.sleep(_REQUEST_DELAY)
            return result

        try:
            fundamentals_obj = _fetch(client.get_fundamentals, "fundamentals")
            dividends_obj = _fetch(client.get_dividends, "dividendos")
            shares_obj = _fetch(client.get_shares_outstanding, "acciones")
            prices_obj = _fetch(client.get_price_history, "precios")
            row = _build_ticker_row(
                symbol,
                fundamentals=fundamentals_obj if isinstance(fundamentals_obj, Mapping) else fundamentals_obj,
                dividends=dividends_obj if isinstance(dividends_obj, pd.DataFrame) else None,
                shares=shares_obj if isinstance(shares_obj, pd.DataFrame) else None,
                prices=prices_obj if isinstance(prices_obj, pd.DataFrame) else None,
                listings_meta=listings_meta,
            )
        except Exception as exc:  # pragma: no cover - defensive branch
            LOGGER.exception("Error inesperado al procesar %s", symbol)
            issues.append(f"exception:{exc}")
            row = None
        elapsed_local = time.perf_counter() - start
        return row, elapsed_local, issues

    with ThreadPoolExecutor(max_workers=_MAX_TICKER_WORKERS) as executor:
        future_map = {executor.submit(_process_ticker, ticker): ticker for ticker in tickers}
        for future in as_completed(future_map):
            ticker = future_map[future]
            try:
                row, elapsed_seconds, issues = future.result()
            except Exception as exc:  # pragma: no cover - defensive branch
                LOGGER.exception("Error crítico al procesar %s", ticker)
                errors_by_ticker[ticker] = [f"exception:{exc}"]
                continue
            elapsed_per_ticker[ticker] = elapsed_seconds
            if issues:
                errors_by_ticker[ticker] = issues
            if row is not None:
                results[ticker] = row

    rows = [results[ticker] for ticker in tickers if ticker in results]
    elapsed = time.perf_counter() - loop_start
    universe_size = len(tickers)
    elapsed_map = {
        ticker: round(seconds, 3)
        for ticker, seconds in sorted(elapsed_per_ticker.items())
    }
    average_elapsed = (
        sum(elapsed_per_ticker.values()) / len(elapsed_per_ticker)
        if elapsed_per_ticker
        else 0.0
    )
    if elapsed_map:
        LOGGER.info(
            "Tiempo promedio por ticker: %.3fs (%s)",
            average_elapsed,
            elapsed_map,
        )
    if errors_by_ticker:
        LOGGER.warning(
            "Yahoo screener detectó datos faltantes en: %s",
            {
                ticker: issues
                for ticker, issues in sorted(errors_by_ticker.items())
            },
        )
    LOGGER.info(
        "Yahoo screener processed %s tickers in %.3f seconds",
        universe_size,
        elapsed,
    )

    telemetry_note = (
        "ℹ️ Yahoo procesó "
        + f"{universe_size} tickers en {elapsed:.2f} segundos"
        + (f" (promedio {average_elapsed:.2f}s)" if elapsed_map else "")
    )

    df = pd.DataFrame(rows).convert_dtypes()

    include_latam_flag = True if include_latam is None else bool(include_latam)

    manual_placeholders = manual_requested or None

    missing_optional_labels = {
        "ratio de payout": max_payout is None,
        "racha de dividendos": min_div_streak is None,
        "CAGR": min_cagr is None,
        "recompras": min_buyback is None,
        "EPS trailing": False,
        "EPS forward": False,
        "crecimiento de EPS": min_eps_growth is None,
    }

    filter_telemetry: list[tuple[str, int, int]] = []

    df = _apply_filters_and_finalize(
        df,
        max_payout=max_payout,
        min_div_streak=min_div_streak,
        min_cagr=min_cagr,
        min_market_cap=min_market_cap,
        max_pe=max_pe,
        min_revenue_growth=min_revenue_growth,
        min_eps_growth=min_eps_growth,
        min_buyback=min_buyback,
        min_score_threshold=min_score_threshold,
        max_results=max_results,
        include_latam_flag=include_latam_flag,
        include_technicals=include_technicals,
        restrict_to_tickers=manual_placeholders,
        placeholder_tickers=manual_placeholders,
        exclude_tickers=sorted(excluded) or None,
        market_cap_column="_meta_market_cap",
        pe_ratio_column="_meta_pe_ratio",
        revenue_growth_column="_meta_revenue_growth",
        latam_column="_meta_is_latam",
        trailing_eps_column="_meta_trailing_eps",
        forward_eps_column="_meta_forward_eps",
        buyback_column="_meta_buyback",
        allowed_sectors=sector_filters,
        allow_na_filters=True,
        extra_drop_columns=(
            "_meta_market_cap",
            "_meta_pe_ratio",
            "_meta_revenue_growth",
            "_meta_is_latam",
            "_meta_trailing_eps",
            "_meta_forward_eps",
            "_meta_buyback",
        ),
        missing_optional_labels=missing_optional_labels,
        filter_telemetry=filter_telemetry,
    )

    filter_notes = list(df.attrs.pop("_notes", []))
    if filter_notes:
        notes.extend(filter_notes)

    if precheck_dropped:
        notes.append(
            f"Precheck descartó {len(precheck_dropped)} símbolos antes del análisis detallado."
        )

    notes.append(telemetry_note)

    filter_metrics, drop_summary_entries = _summarize_filter_telemetry(
        filter_telemetry,
        include_latam=include_latam_flag,
        max_results=max_results,
        sectors=sectors,
        manual_tickers=manual_requested,
        exclude_tickers=excluded,
    )

    drop_parts: list[str] = []
    for _, label, dropped in drop_summary_entries:
        percentage = (dropped / universe_size) if universe_size else 0.0
        drop_parts.append(f"{percentage:.0%} descartados por {label}")

    drop_summary = ", ".join(drop_parts) if drop_parts else "sin descartes"

    summary_payload = {
        "universe_count": universe_size,
        "result_count": int(df.index.size),
        "discarded_ratio": (
            (max(universe_size - int(df.index.size), 0) / universe_size)
            if universe_size
            else 0.0
        ),
        "elapsed_seconds": float(elapsed),
        "elapsed_seconds_per_ticker": elapsed_map,
        "average_elapsed_seconds_per_ticker": round(average_elapsed, 3),
        "ticker_error_count": len(errors_by_ticker),
        "precheck_initial_count": precheck_initial_count,
        "precheck_discarded_count": len(precheck_dropped),
        "precheck_discard_ratio": precheck_ratio,
        "selected_sectors": list(sector_filters),
        "sector_distribution": _compute_sector_distribution(df),
        "filter_descriptions": list(filter_metrics),
        "drop_summary": drop_summary,
        "source": "yahoo",
    }

    if errors_by_ticker:
        summary_payload["ticker_errors"] = {
            ticker: list(issues)
            for ticker, issues in sorted(errors_by_ticker.items())
        }
    if precheck_dropped:
        summary_payload["precheck_discards"] = {
            ticker: list(reasons)
            for ticker, reasons in sorted(precheck_dropped.items())
        }

    df.attrs["summary"] = summary_payload

    if using_default_universe and tickers:
        filters_applied: list[str] = []
        if min_market_cap is not None:
            filters_applied.append(f"min_market_cap={min_market_cap}")
        if max_pe is not None:
            filters_applied.append(f"max_pe={max_pe}")
        if min_revenue_growth is not None:
            filters_applied.append(f"min_revenue_growth={min_revenue_growth}")
        if include_latam is not None:
            filters_applied.append(f"include_latam={bool(include_latam)}")

        message = f"📈 Analizando {len(tickers)} símbolos seleccionados automáticamente"
        if target_markets:
            message += " de " + ", ".join(target_markets)
        notes.append(message)
        if filters_applied:
            notes.append("Filtros aplicados: " + ", ".join(filters_applied))

    return (df, notes) if notes else df


__all__ = ["run_screener_stub", "run_screener_yahoo"]
