import time
import logging
import pandas as pd
import streamlit as st

from shared.config import settings
from services.cache import fetch_quotes_bulk
from services.performance_timer import (
    ProfileBlockResult,
    performance_timer,
    profile_block,
)
from shared.errors import AppError

logger = logging.getLogger(__name__)


def apply_filters(df_pos, controls, cli, psvc):
    """Apply user filters and enrich positions with quotes."""

    telemetry: dict[str, object] = {
        "status": "success",
        "initial_rows": int(len(df_pos)),
    }
    stage_profiles: dict[str, dict[str, float | None]] = {}

    def _record_stage(name: str, stage: ProfileBlockResult) -> None:
        stage_profiles[name] = {
            "ms": round(stage.duration_ms, 3),
            "cpu": None if stage.cpu_percent is None else round(stage.cpu_percent, 2),
            "mem": None if stage.ram_percent is None else round(stage.ram_percent, 2),
        }

    def _profile_stage(name: str, **extra: object):
        payload: dict[str, object] = {"stage": name}
        if extra:
            payload.update(extra)
        return profile_block(f"apply_filters.{name}", extra=payload)

    with performance_timer("apply_filters", extra=telemetry):
        selected_syms = getattr(controls, "selected_syms", []) or []
        with _profile_stage(
            "filter_positions",
            hide_cash=controls.hide_cash,
            selected=len(selected_syms),
        ) as stage_filter:
            if controls.hide_cash:
                df_pos = df_pos[
                    ~df_pos["simbolo"].isin(["IOLPORA", "PARKING"])
                ].copy()
            if selected_syms:
                df_pos = df_pos[df_pos["simbolo"].isin(selected_syms)].copy()
        _record_stage("filter_positions", stage_filter)
        telemetry["post_filter_rows"] = int(len(df_pos))

        pairs = list(
            df_pos[["mercado", "simbolo"]]
            .drop_duplicates()
            .astype({"mercado": str, "simbolo": str})
            .itertuples(index=False, name=None)
        )
        telemetry["pairs"] = int(len(pairs))
        fetch_stage: ProfileBlockResult | None = None
        try:
            with _profile_stage("fetch_quotes", pairs=len(pairs)) as stage_fetch:
                fetch_stage = stage_fetch
                quotes_map = fetch_quotes_bulk(cli, pairs)
        except AppError as err:
            telemetry["status"] = "error"
            telemetry["detail"] = err.__class__.__name__
            st.error(str(err))
            st.stop()
        except Exception:
            telemetry["status"] = "error"
            telemetry["detail"] = "exception"
            logger.exception("Error al obtener cotizaciones")
            st.error("No se pudieron obtener cotizaciones, intente más tarde")
            st.stop()
        else:
            if fetch_stage is not None:
                _record_stage("fetch_quotes", fetch_stage)

        chg_cnt = sum(
            1
            for v in quotes_map.values()
            if isinstance(v, dict) and v.get("chg_pct") is not None
        )
        telemetry["quotes_with_chg"] = int(chg_cnt)
        logger.info(
            "apply_filters solicitó %d pares; %d con chg_pct",
            len(pairs),
            chg_cnt,
        )

        with _profile_stage("calc_rows", rows=len(df_pos)) as stage_calc:
            df_view = psvc.calc_rows(
                lambda mercado, simbolo=None: quotes_map.get(
                    (str(mercado).lower(), str((simbolo or mercado)).upper()), {}
                ),
                df_pos,
                exclude_syms=[],
            )
        _record_stage("calc_rows", stage_calc)
        telemetry["result_rows"] = int(len(df_view))
        if df_view.empty:
            if stage_profiles:
                try:
                    st.session_state["apply_filters_profiles"] = stage_profiles
                except Exception:
                    pass
            return df_view

        with _profile_stage("classify_assets", rows=len(df_view)) as stage_classify:
            df_view["tipo"] = df_view["simbolo"].astype(str).map(
                psvc.classify_asset_cached
            )
        _record_stage("classify_assets", stage_classify)

        if controls.selected_types:
            with _profile_stage(
                "apply_type_filter", filters=len(controls.selected_types)
            ) as stage_type:
                df_view = df_view[df_view["tipo"].isin(controls.selected_types)].copy()
            _record_stage("apply_type_filter", stage_type)
            telemetry["filtered_rows"] = int(len(df_view))

        symbol_q = (controls.symbol_query or "").strip()
        if symbol_q:
            with _profile_stage("apply_symbol_query", query=symbol_q) as stage_query:
                df_view = df_view[
                    df_view["simbolo"].astype(str).str.contains(
                        symbol_q, case=False, na=False
                    )
                ].copy()
            _record_stage("apply_symbol_query", stage_query)
            telemetry["query_rows"] = int(len(df_view))

        chg_map = {k: v.get("chg_pct") for k, v in quotes_map.items()}
        with _profile_stage("map_changes", rows=len(df_view)) as stage_changes:
            map_keys = df_view.apply(
                lambda row: (
                    str(row["mercado"]).lower(),
                    str(row["simbolo"]).upper(),
                ),
                axis=1,
            )

            df_view["chg_%"] = map_keys.map(chg_map)
            df_view["chg_%"] = pd.to_numeric(df_view["chg_%"], errors="coerce")
        _record_stage("map_changes", stage_changes)

        with _profile_stage("update_history", entries=len(chg_map)) as stage_history:
            st.session_state.setdefault("quotes_hist", {})
            now_ts = int(time.time())
            for (mkt, sym), chg in chg_map.items():
                if isinstance(chg, (int, float)):
                    st.session_state["quotes_hist"].setdefault(sym, [])
                    if (
                        not st.session_state["quotes_hist"][sym]
                        or (
                            st.session_state["quotes_hist"][sym][-1].get("ts")
                            != now_ts
                        )
                    ):
                        st.session_state["quotes_hist"][sym].append(
                            {"ts": now_ts, "chg_pct": float(chg)}
                        )
                        maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                        st.session_state["quotes_hist"][sym] = st.session_state[
                            "quotes_hist"
                        ][sym][-maxlen:]
        _record_stage("update_history", stage_history)

        if stage_profiles:
            telemetry["stage_ms"] = {
                name: metrics["ms"] for name, metrics in stage_profiles.items()
            }
            try:
                st.session_state["apply_filters_profiles"] = stage_profiles
            except Exception:
                pass

        return df_view
