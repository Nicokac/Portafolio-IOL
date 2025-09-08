import pandas as pd
import streamlit as st

from shared.config import settings
from ui.fx_panels import render_spreads, render_fx_history


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
            st.session_state.setdefault("fx_history", [])
            if not st.session_state["fx_history"] or st.session_state["fx_history"][-1].get("ts") != c_ts:
                st.session_state["fx_history"].append(rec)
                maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                st.session_state["fx_history"] = st.session_state["fx_history"][-maxlen:]
            fx_hist_df = pd.DataFrame(st.session_state["fx_history"])
            if not fx_hist_df.empty:
                fx_hist_df["ts_dt"] = pd.to_datetime(fx_hist_df["ts"], unit="s")
                render_fx_history(fx_hist_df)