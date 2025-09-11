import streamlit as st
import pandas as pd

from shared.config import settings
from ui.fx_panels import render_spreads, render_fx_history
from infrastructure.cache import cache


def render_fx_section(container, rates):
    """Render FX information in the given container."""
    with container:
        rates = rates or {}
        if not rates:
            st.warning("No se pudieron obtener las cotizaciones del dólar.")
        render_spreads(rates)
        c_ts = rates.get("_ts")
        if c_ts:
            rec = {
                "ts": c_ts,
                "ccl": rates.get("ccl"),
                "mep": rates.get("mep"),
                "blue": rates.get("blue"),
                "oficial": rates.get("oficial"),
            }
            cache.session_state.setdefault("fx_history", [])
            if not cache.session_state["fx_history"] or cache.session_state["fx_history"][-1].get("ts") != c_ts:
                cache.session_state["fx_history"].append(rec)
                maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                cache.session_state["fx_history"] = cache.session_state["fx_history"][-maxlen:]
            fx_hist_df = pd.DataFrame(cache.session_state["fx_history"])
            if not fx_hist_df.empty:
                fx_hist_df["ts_dt"] = pd.to_datetime(fx_hist_df["ts"], unit="s")
                render_fx_history(fx_hist_df)

