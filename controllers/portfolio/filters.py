import time
import logging
import pandas as pd
import streamlit as st

from shared.config import settings
from services.cache import fetch_quotes_bulk
from services.performance_timer import performance_timer
from shared.errors import AppError

logger = logging.getLogger(__name__)


def apply_filters(df_pos, controls, cli, psvc):
    """Apply user filters and enrich positions with quotes."""

    telemetry: dict[str, object] = {
        "status": "success",
        "initial_rows": int(len(df_pos)),
    }
    with performance_timer("apply_filters", extra=telemetry):
        if controls.hide_cash:
            df_pos = df_pos[~df_pos["simbolo"].isin(["IOLPORA", "PARKING"])].copy()
        if controls.selected_syms:
            df_pos = df_pos[df_pos["simbolo"].isin(controls.selected_syms)].copy()

        pairs = list(
            df_pos[["mercado", "simbolo"]]
            .drop_duplicates()
            .astype({"mercado": str, "simbolo": str})
            .itertuples(index=False, name=None)
        )
        telemetry["pairs"] = int(len(pairs))
        try:
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

        df_view = psvc.calc_rows(
            lambda mercado, simbolo=None: quotes_map.get(
                (str(mercado).lower(), str((simbolo or mercado)).upper()), {}
            ),
            df_pos,
            exclude_syms=[],
        )
        telemetry["result_rows"] = int(len(df_view))
        if df_view.empty:
            return df_view

        df_view["tipo"] = df_view["simbolo"].astype(str).map(psvc.classify_asset_cached)

        if controls.selected_types:
            df_view = df_view[df_view["tipo"].isin(controls.selected_types)].copy()
            telemetry["filtered_rows"] = int(len(df_view))

        symbol_q = (controls.symbol_query or "").strip()
        if symbol_q:
            df_view = df_view[
                df_view["simbolo"].astype(str).str.contains(symbol_q, case=False, na=False)
            ].copy()
            telemetry["query_rows"] = int(len(df_view))

        chg_map = {k: v.get("chg_pct") for k, v in quotes_map.items()}
        map_keys = df_view.apply(
            lambda row: (str(row["mercado"]).lower(), str(row["simbolo"]).upper()), axis=1
        )

        df_view["chg_%"] = map_keys.map(chg_map)
        df_view["chg_%"] = pd.to_numeric(df_view["chg_%"], errors="coerce")

        st.session_state.setdefault("quotes_hist", {})
        now_ts = int(time.time())
        for (mkt, sym), chg in chg_map.items():
            if isinstance(chg, (int, float)):
                st.session_state["quotes_hist"].setdefault(sym, [])
                if (
                    not st.session_state["quotes_hist"][sym]
                    or (st.session_state["quotes_hist"][sym][-1].get("ts") != now_ts)
                ):
                    st.session_state["quotes_hist"][sym].append({"ts": now_ts, "chg_pct": float(chg)})
                    maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                    st.session_state["quotes_hist"][sym] = st.session_state["quotes_hist"][sym][-maxlen:]

        return df_view
