import logging
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from threading import Lock
from typing import Iterable, Sequence

import pandas as pd
import streamlit as st

from services.cache import fetch_portfolio, fetch_quotes_bulk
from services.cache.market_data_cache import (
    StaleWhileRevalidateCache,
    create_persistent_cache,
)
from services.performance_timer import (
    QUOTES_BATCH_LATENCY_SECONDS,
    QUOTES_SWR_SERVED_TOTAL,
    performance_timer,
)
from shared.errors import AppError
from shared.settings import (
    max_quote_workers,
    quotes_batch_size,
    quotes_swr_grace_seconds,
    quotes_swr_ttl_seconds,
)

logger = logging.getLogger(__name__)


@dataclass
class QuoteBatch:
    """Describe a batch of quotes grouped by asset type."""

    group: str
    pairs: list[tuple[str, str]]
    key: str


_QUOTES_SWR_CACHE: StaleWhileRevalidateCache | None = None
_QUOTES_SWR_LOCK = Lock()


def _get_quotes_swr_cache() -> StaleWhileRevalidateCache:
    global _QUOTES_SWR_CACHE
    if _QUOTES_SWR_CACHE is not None:
        return _QUOTES_SWR_CACHE
    with _QUOTES_SWR_LOCK:
        if _QUOTES_SWR_CACHE is None:
            base_cache = create_persistent_cache("quotes_refresh")
            _QUOTES_SWR_CACHE = StaleWhileRevalidateCache(
                base_cache,
                default_ttl=quotes_swr_ttl_seconds,
                grace_ttl=quotes_swr_grace_seconds,
                max_workers=max_quote_workers,
            )
    return _QUOTES_SWR_CACHE


def _normalize_pairs(df_pos: pd.DataFrame) -> list[tuple[str, str]]:
    if df_pos.empty:
        return []
    cols = [col for col in ("mercado", "simbolo") if col in df_pos.columns]
    if len(cols) < 2:
        return []
    data = (
        df_pos[cols]
        .dropna(subset=["simbolo"])
        .astype({"mercado": str, "simbolo": str})
    )
    data["mercado"] = data["mercado"].str.lower()
    data["mercado"] = data["mercado"].where(
        data["mercado"].str.strip().astype(bool), "bcba"
    )
    data["simbolo"] = data["simbolo"].str.upper()
    data = data.drop_duplicates()
    return list(data.itertuples(index=False, name=None))


def _resolve_asset_groups(df_pos: pd.DataFrame, psvc) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if "tipo" in df_pos.columns:
        subset = (
            df_pos[["simbolo", "tipo"]]
            .dropna(subset=["simbolo"])
            .astype({"simbolo": str})
        )
        subset["simbolo"] = subset["simbolo"].str.upper()
        subset["tipo"] = subset["tipo"].astype(str)
        mapping.update({row.simbolo: row.tipo for row in subset.itertuples(index=False)})
    symbols = {
        str(sym or "").strip().upper()
        for sym in df_pos.get("simbolo", [])
        if str(sym or "").strip()
    }
    for symbol in symbols:
        if symbol not in mapping:
            try:
                asset_type = psvc.classify_asset_cached(symbol)
            except Exception:
                asset_type = ""
            mapping[symbol] = asset_type or "otros"
    return mapping


def _chunk_pairs(pairs: Sequence[tuple[str, str]], size: int) -> Iterable[list[tuple[str, str]]]:
    size = max(int(size or 1), 1)
    chunk: list[tuple[str, str]] = []
    for pair in pairs:
        chunk.append(pair)
        if len(chunk) >= size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk


def build_quote_batches(
    df_pos: pd.DataFrame,
    psvc,
    *,
    batch_size: int | None = None,
) -> list[QuoteBatch]:
    """Group quote pairs by asset type and chunk the result into batches."""

    pairs = _normalize_pairs(df_pos)
    if not pairs:
        return []
    asset_groups = _resolve_asset_groups(df_pos, psvc)
    grouped: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for market, symbol in pairs:
        group = asset_groups.get(symbol, "otros") or "otros"
        grouped[group].append((market, symbol))
    target_size = max(int(batch_size or quotes_batch_size), 1)
    batches: list[QuoteBatch] = []
    for group in sorted(grouped):
        for chunk in _chunk_pairs(grouped[group], target_size):
            key_parts = [f"{m}:{s}" for m, s in chunk]
            batch_key = f"{group}|" + ",".join(sorted(key_parts))
            batches.append(QuoteBatch(group=group, pairs=list(chunk), key=batch_key))
    return batches


def refresh_quotes_pipeline(
    cli,
    df_pos: pd.DataFrame,
    psvc,
    *,
    fetcher=fetch_quotes_bulk,
    swr_cache: StaleWhileRevalidateCache | None = None,
    ttl: float | None = None,
    grace: float | None = None,
    batch_size: int | None = None,
    max_workers_override: int | None = None,
) -> tuple[dict[tuple[str, str], dict], list[dict[str, object]]]:
    """Fetch quotes using batching and stale-while-revalidate strategy."""

    batches = build_quote_batches(df_pos, psvc, batch_size=batch_size)
    if not batches:
        return {}, []

    cache = swr_cache or _get_quotes_swr_cache()
    ttl_value = ttl if ttl is not None else quotes_swr_ttl_seconds
    grace_value = grace if grace is not None else quotes_swr_grace_seconds
    diagnostics: list[dict[str, object]] = []
    combined: dict[tuple[str, str], dict] = {}
    max_workers_value = max_workers_override or max_quote_workers
    worker_count = max(1, min(int(max_workers_value or 1), len(batches)))

    def _process(batch: QuoteBatch) -> tuple[dict[tuple[str, str], dict], dict[str, object]]:
        symbols = [sym for _, sym in batch.pairs]

        def _loader() -> dict[tuple[str, str], dict]:
            payload = fetcher(cli, batch.pairs)
            return dict(payload or {})

        def _on_refresh(duration: float, _value: dict, background: bool) -> None:
            if QUOTES_BATCH_LATENCY_SECONDS is not None:
                try:
                    QUOTES_BATCH_LATENCY_SECONDS.labels(
                        batch_size=str(len(batch.pairs)),
                        background="true" if background else "false",
                    ).observe(duration)
                except Exception:  # pragma: no cover - optional metric backend
                    pass
            logger.info(
                "quotes batch refreshed",
                extra={
                    "quotes_batch": {
                        "group": batch.group,
                        "symbols": symbols,
                        "size": len(batch.pairs),
                        "duration": duration,
                        "background": background,
                    }
                },
            )

        result = cache.get_or_refresh(
            batch.key,
            _loader,
            ttl=ttl_value,
            grace=grace_value,
            on_refresh=_on_refresh,
        )

        served_mode = "fresh"
        if result.was_refreshed and result.duration is not None:
            served_mode = "refresh"
        elif result.is_stale:
            served_mode = "stale"
        if QUOTES_SWR_SERVED_TOTAL is not None:
            try:
                QUOTES_SWR_SERVED_TOTAL.labels(mode=served_mode).inc()
            except Exception:  # pragma: no cover - optional metric backend
                pass

        logger.info(
            "quotes batch served",
            extra={
                "quotes_batch": {
                    "group": batch.group,
                    "symbols": symbols,
                    "size": len(batch.pairs),
                    "mode": served_mode,
                    "stale": result.is_stale,
                    "refresh_scheduled": result.refresh_scheduled,
                }
            },
        )

        payload = dict(result.value or {})
        batch_diagnostic = {
            "group": batch.group,
            "symbols": symbols,
            "stale": bool(result.is_stale),
            "served_mode": served_mode,
            "refresh_scheduled": bool(result.refresh_scheduled),
        }
        return payload, batch_diagnostic

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_process, batch): batch for batch in batches}
        for future in as_completed(futures):
            try:
                payload, meta = future.result()
            except Exception:
                batch = futures[future]
                logger.exception(
                    "Quote batch failed",
                    extra={
                        "quotes_batch": {
                            "group": batch.group,
                            "symbols": [sym for _, sym in batch.pairs],
                            "size": len(batch.pairs),
                        }
                    },
                )
                continue
            combined.update(payload)
            diagnostics.append(meta)

    return combined, diagnostics


def _filters_active() -> bool:
    state = getattr(st, "session_state", {})
    symbol_query = str(state.get("symbol_query", "") or "").strip()
    if symbol_query:
        return True

    selected_syms = state.get("selected_syms")
    last_all = state.get("portfolio_last_all_symbols")
    if isinstance(selected_syms, list):
        cleaned = [str(sym).strip() for sym in selected_syms if str(sym).strip()]
        if not cleaned and (last_all or symbol_query):
            return True
        if last_all and isinstance(last_all, list) and 0 < len(cleaned) < len(last_all):
            return True

    selected_types = state.get("selected_types")
    last_types = state.get("portfolio_last_available_types")
    if isinstance(selected_types, list):
        cleaned_types = [str(t).strip() for t in selected_types if str(t).strip()]
        if not cleaned_types and last_types:
            return True
        if (
            isinstance(last_types, list)
            and last_types
            and 0 < len(cleaned_types) < len(last_types)
        ):
            return True

    return False


def load_portfolio_data(cli, psvc):
    """Fetch and normalize portfolio positions."""
    tokens_path = getattr(getattr(cli, "auth", None), "tokens_path", None)
    payload = None
    with st.spinner("Cargando y actualizando portafolio... ⏳"):
        telemetry: dict[str, object] = {"status": "success", "source": "api"}
        try:
            with performance_timer("portfolio_load_data", extra=telemetry):
                try:
                    payload = fetch_portfolio(cli)
                except AppError as err:
                    telemetry["status"] = "error"
                    telemetry["detail"] = err.__class__.__name__
                    raise
                except Exception:
                    telemetry["status"] = "error"
                    telemetry["detail"] = "exception"
                    raise
                if isinstance(payload, dict) and payload.get("_cached"):
                    telemetry["source"] = "cache"
        except AppError as err:
            st.error(str(err))
            st.stop()
        except Exception:  # pragma: no cover - streamlit error path
            logger.exception(
                "Error al consultar portafolio",
                extra={"tokens_file": tokens_path},
            )
            st.error("No se pudo cargar el portafolio, intente más tarde")
            st.stop()

    auth_error = False
    if isinstance(payload, dict):
        msg = str(payload.get("message", ""))
        auth_error = (
            payload.get("status") in (401, 403)
            or payload.get("code") in (401, 403)
            or "unauthorized" in msg.lower()
            or "no autorizado" in msg.lower()
        )

    if st.session_state.get("force_login") or auth_error:
        st.session_state["force_login"] = True
        st.error("No se pudo autenticar con IOL")
        st.stop()
    elif isinstance(payload, dict) and payload.get("_cached"):
        st.warning(
            "No se pudo contactar a IOL; mostrando datos del portafolio en caché."
        )

    if isinstance(payload, dict) and "message" in payload:
        st.info(f"ℹ️ Mensaje de IOL: \"{payload['message']}\"")
        st.stop()

    df_pos = psvc.normalize_positions(payload)
    if df_pos.empty:
        logger.info(
            "Portafolio vacío pero API respondió correctamente",
            extra={"tokens_file": tokens_path},
        )
        if _filters_active():
            st.info("No se encontraron activos que cumplan los filtros.")
        else:
            st.warning(
                "No se encontraron posiciones o no pudimos mapear la respuesta."
            )
        if isinstance(payload, dict) and "activos" in payload:
            st.dataframe(pd.DataFrame(payload["activos"]).head(20))
            st.caption("Ejemplo de datos recibidos del portafolio")
        st.stop()

    all_symbols = sorted(df_pos["simbolo"].astype(str).str.upper().unique())
    available_types = sorted(
        {
            psvc.classify_asset_cached(s)
            for s in all_symbols
            if psvc.classify_asset_cached(s)
        }
    )
    try:
        st.session_state["portfolio_last_all_symbols"] = all_symbols
        st.session_state["portfolio_last_available_types"] = available_types
    except Exception:  # pragma: no cover - defensive safeguard
        logger.debug("No se pudieron almacenar los metadatos de filtros", exc_info=True)
    return df_pos, all_symbols, available_types

