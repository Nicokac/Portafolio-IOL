# app.py
# Orquestación Streamlit + módulos

from __future__ import annotations
import os, time, hashlib
import pandas as pd
import streamlit as st
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from shared.config import settings
from domain.models import Controls

# UI
from ui.header import render_header
from ui.tables import render_totals, render_table
from ui.fx_panels import render_fx_panel, render_spreads, render_fx_history
from ui.sidebar_controls import render_sidebar
from ui.fundamentals import render_fundamental_data
from ui.ui_settings import init_ui, render_ui_controls
from ui.actions import render_action_menu
from ui.charts import (
    plot_pl_topn,
    plot_donut_tipo,
    plot_dist_por_tipo,
    plot_bubble_pl_vs_costo,
    plot_heat_pl_pct,
    plot_pl_daily_topn,
    plot_correlation_heatmap,
    plot_technical_analysis_chart,
)

# Infra: IOL + FX + cache quotes
from infrastructure.iol.client import build_iol_client
from infrastructure.iol.ports import IIOLProvider
from infrastructure.fx.provider import FXProviderAdapter
from infrastructure.cache.quote_cache import get_quote_cached

# App facades
from application.portfolio_service import PortfolioService
from application.ta_service import TAService  # <- también la clase

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Configuración de UI centralizada (tema y layout)
init_ui()

# === Cache de alto nivel ===
@st.cache_resource
def get_client_cached(cache_key: str, user: str, password: str) -> IIOLProvider:
    _ = cache_key
    return build_iol_client(user, password)

@st.cache_data(ttl=settings.cache_ttl_portfolio)
def fetch_portfolio(_cli: IIOLProvider):
    start = time.time()
    data = _cli.get_portfolio()
    logger.info("fetch_portfolio done in %.0fms", (time.time() - start) * 1000)
    return data

@st.cache_data(ttl=settings.cache_ttl_quotes)
def fetch_last_price(_cli: IIOLProvider, mercado: str, simbolo: str):
    try:
        return _cli.get_last_price(mercado=mercado, simbolo=simbolo)
    except Exception as e:
        logger.warning("get_last_price(%s, %s) falló: %s", mercado, simbolo, e)
        return None

@st.cache_resource
def get_fx_provider() -> FXProviderAdapter:
    return FXProviderAdapter()

@st.cache_data(ttl=settings.cache_ttl_fx)
def fetch_fx_rates():
    return get_fx_provider().get_rates()

def _build_client(user: str, password: str) -> IIOLProvider:
    salt = str(st.session_state.get("client_salt", ""))
    cache_key = hashlib.sha256(f"{user}:{password}:{salt}".encode()).hexdigest()
    return get_client_cached(cache_key, user, password)

def main():
    # Credenciales desde settings (cargadas de .env si existe)
    user = settings.IOL_USERNAME
    password = settings.IOL_PASSWORD
    if not user or not password:
        st.error("Falta IOL_USERNAME / IOL_PASSWORD en tu archivo .env")
        st.stop()

    # ===== HEADER =====
    render_header()
    _, hcol2 = st.columns([4, 1])
    with hcol2:
        now = datetime.now()
        st.caption(f"🕒 {now.strftime('%d/%m/%Y %H:%M:%S')}")
        render_action_menu(user, password)

    # ===== LAYOUT DOS COLUMNAS =====
    main_col, side_col = st.columns([4, 1])

    with side_col:
        rates = fetch_fx_rates() or {}
        render_fx_panel(rates)
        render_spreads(rates)

        c_ts = rates.get("_ts")
        if c_ts:
            rec = {"ts": c_ts, "ccl": rates.get("ccl"), "mep": rates.get("mep"),
                   "blue": rates.get("blue"), "oficial": rates.get("oficial")}
            st.session_state.setdefault("fx_history", [])
            if not st.session_state["fx_history"] or st.session_state["fx_history"][-1].get("ts") != c_ts:
                st.session_state["fx_history"].append(rec)
                maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                st.session_state["fx_history"] = st.session_state["fx_history"][-maxlen:]
            fx_hist_df = pd.DataFrame(st.session_state["fx_history"])
            if not fx_hist_df.empty:
                fx_hist_df["ts_dt"] = pd.to_datetime(fx_hist_df["ts"], unit="s")
                render_fx_history(fx_hist_df)

    # === LÓGICA PRINCIPAL Y TABS (DENTRO DE main_col) ===
    with main_col:
        psvc = PortfolioService()
        tasvc = TAService()

        # --- DATA PORTAFOLIO ---
        cli = _build_client(user, password)
        with st.spinner("Cargando y actualizando portafolio... ⏳"):
            try:
                payload = fetch_portfolio(cli)
            except Exception as e:
                st.error(f"Error al consultar portafolio: {e}")
                st.stop()

        if isinstance(payload, dict) and "message" in payload:
            st.info(f"ℹ️ Mensaje de IOL: \"{payload['message']}\"")
            st.stop()

        df_pos = psvc.normalize_positions(payload)
        if df_pos.empty:
            st.warning("No se encontraron posiciones o no pudimos mapear la respuesta.")
            if isinstance(payload, dict) and "activos" in payload:
                st.dataframe(pd.DataFrame(payload["activos"]).head(20))
            st.stop()

        all_symbols = sorted(df_pos["simbolo"].astype(str).str.upper().unique())
        available_types = sorted({ psvc.classify_asset_cached(s) for s in all_symbols if psvc.classify_asset_cached(s) })

        controls: Controls = render_sidebar(all_symbols, available_types)
        render_ui_controls()

        refresh_secs  = controls.refresh_secs
        hide_cash     = controls.hide_cash
        order_by      = controls.order_by
        desc          = controls.desc
        selected_syms = controls.selected_syms
        top_n         = controls.top_n
        show_usd      = controls.show_usd

        if hide_cash:
            df_pos = df_pos[~df_pos["simbolo"].isin(["IOLPORA", "PARKING"])].copy()
        if selected_syms:
            df_pos = df_pos[df_pos["simbolo"].isin(selected_syms)].copy()

        st.session_state.pop("last_price_errors", None)

        df_view = psvc.calc_rows(
            lambda mercado, simbolo=None: fetch_last_price(cli, mercado, simbolo or mercado),
            df_pos,
            exclude_syms=[],
        )

        if st.session_state.get("last_price_errors"):
            errs = st.session_state["last_price_errors"]
            sample = errs[:3]
            msg = "Se reautenticó datos de mercado. Si persiste, probá ‘🔄 Relogin’. "
            msg += "Errores ejemplo: " + ", ".join([f"{e['simbolo']}({e['mercado']}): {e['err']}" for e in sample])
            st.info(msg)
            st.session_state.pop("last_price_errors", None)

        if not df_view.empty:
            df_view["tipo"] = df_view["simbolo"].astype(str).map(psvc.classify_asset_cached)

            sel_types = controls.selected_types or []
            if sel_types:
                df_view = df_view[df_view["tipo"].isin(sel_types)].copy()

            symbol_q = (controls.symbol_query or "").strip()
            if symbol_q:
                df_view = df_view[df_view["simbolo"].astype(str).str.contains(symbol_q, case=False, na=False)].copy()

        # === Cotizaciones diarias en paralelo ===
        if not df_view.empty:
            st.session_state.setdefault("quotes_hist", {})
            now_ts = int(time.time())
            symbols_to_query = (
                df_view[["mercado", "simbolo"]]
                .drop_duplicates()
                .astype({"mercado": str, "simbolo": str})
                .to_dict("records")
            )
            quotes_cache = {}
            max_workers = getattr(settings, "max_quote_workers", 12)
            with ThreadPoolExecutor(max_workers=min(max_workers, len(symbols_to_query) or 1)) as ex:
                futures = {
                    ex.submit(get_quote_cached, cli, it["mercado"], it["simbolo"]): (it["mercado"].lower(), it["simbolo"].upper())
                    for it in symbols_to_query
                }
                for fut in as_completed(futures):
                    mkt, sym = futures[fut]
                    try:
                        quote = fut.result()
                        quotes_cache[(mkt, sym)] = quote.get("chg_pct")
                        chg = quote.get("chg_pct")
                        if isinstance(chg, (int, float)):
                            st.session_state["quotes_hist"].setdefault(sym, [])
                            if (not st.session_state["quotes_hist"][sym]) or (st.session_state["quotes_hist"][sym][-1].get("ts") != now_ts):
                                st.session_state["quotes_hist"][sym].append({"ts": now_ts, "chg_pct": float(chg)})
                                maxlen = getattr(settings, "quotes_hist_maxlen", 500)
                                st.session_state["quotes_hist"][sym] = st.session_state["quotes_hist"][sym][-maxlen:]
                    except Exception as e:
                        logger.warning("get_quote_cached failed for %s:%s -> %s", mkt, sym, e)

            map_keys = df_view.apply(lambda row: (str(row["mercado"]).lower(), str(row["simbolo"]).upper()), axis=1)
            df_view["chg_%"] = map_keys.map(quotes_cache)

            df_view["chg_%"] = pd.to_numeric(df_view["chg_%"], errors="coerce")
            df_view["valor_actual"] = pd.to_numeric(df_view["valor_actual"], errors="coerce")
            denom = 100.0 + df_view["chg_%"]
            mask = df_view["chg_%"].notna() & df_view["valor_actual"].notna() & (denom != 0)
            df_view.loc[mask, "pl_d"] = df_view.loc[mask, "valor_actual"] * df_view.loc[mask, "chg_%"] / denom

        ccl_rate = (fetch_fx_rates() or {}).get("ccl")

        # === TABS ===
        tabs = st.tabs(["📂 Portafolio", "📊 Análisis avanzado", "🎲 Análisis de Riesgo", "🔎 Análisis de activos"])

        # Pestaña 1
        with tabs[0]:
            if df_view.empty:
                st.info("No hay datos del portafolio para mostrar.")
            else:
                render_totals(df_view, ccl_rate=ccl_rate)
                render_table(df_view, order_by, desc, ccl_rate=ccl_rate, show_usd=show_usd)

                colA, colB = st.columns(2)
                with colA:
                    st.subheader("P/L por símbolo (Top N)")
                    fig = plot_pl_topn(df_view, n=top_n)
                    _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("Sin datos para graficar P/L Top N.")
                with colB:
                    st.subheader("Composición por tipo (Donut)")
                    fig = plot_donut_tipo(df_view)
                    _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("No hay datos para el donut por tipo.")

                st.subheader("Distribución por tipo (Valorizado)")
                fig = plot_dist_por_tipo(df_view)
                _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("No hay datos para la distribución por tipo.")

                st.subheader("P/L diario por símbolo (Top N)")
                fig = plot_pl_daily_topn(df_view, n=top_n)
                _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("Aún no hay datos de P/L diario.")

        # Pestaña 2
        with tabs[1]:
            st.subheader("Bubble Chart Interactivo")
            axis_options = [c for c in ["costo", "pl", "pl_%", "valor_actual", "pl_d"] if c in df_view.columns]
            if not axis_options:
                st.info("No hay columnas disponibles para el gráfico bubble.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    x_axis = st.selectbox("Eje X", options=axis_options, index=axis_options.index("costo") if "costo" in axis_options else 0)
                with c2:
                    y_axis = st.selectbox("Eje Y", options=axis_options, index=axis_options.index("pl") if "pl" in axis_options else min(1, len(axis_options)-1))

                fig = plot_bubble_pl_vs_costo(df_view, x_axis=x_axis, y_axis=y_axis)
                _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("No hay datos suficientes para el gráfico bubble.")

                st.subheader("Heatmap de rendimiento (%) por símbolo")
                fig = plot_heat_pl_pct(df_view)
                _ = st.plotly_chart(fig, use_container_width=True) if fig is not None else st.info("No hay datos suficientes para el heatmap.")

        # Pestaña 3
        with tabs[2]:
            st.subheader("Análisis de Correlación del Portafolio")
            corr_period = st.selectbox("Calcular correlación sobre el último período:", ["3mo", "6mo", "1y", "2y", "5y"], index=2)
            portfolio_symbols = df_view["simbolo"].tolist()
            if len(portfolio_symbols) >= 2:
                with st.spinner(f"Calculando correlación ({corr_period})…"):
                    hist_df = tasvc.portfolio_history(simbolos=portfolio_symbols, period=corr_period)
                fig = plot_correlation_heatmap(hist_df)
                if fig:
                    _ = st.plotly_chart(fig, use_container_width=True)
                    st.caption("""
                        Un heatmap de correlación muestra cómo se mueven los activos entre sí. 
                        **Azul (cercano a 1)**: Se mueven juntos. 
                        **Rojo (cercano a -1)**: Se mueven en direcciones opuestas. 
                        **Blanco (cercano a 0)**: No tienen relación. 
                        Una buena diversificación busca valores bajos (cercanos a 0 o negativos).
                    """)
                else:
                    st.warning(f"No se pudieron obtener suficientes datos históricos para el período '{corr_period}' para calcular la correlación.")
            else:
                st.info("Necesitas al menos 2 activos en tu portafolio (después de aplicar filtros) para calcular la correlación.")

        # Pestaña 4
        with tabs[3]:
            st.subheader("Indicadores técnicos por activo")
            if not all_symbols:
                st.info("No hay símbolos en el portafolio para analizar.")
            else:
                sym = st.selectbox("Seleccioná un símbolo (CEDEAR / ETF)", options=all_symbols, index=0, key="ta_symbol")
                if sym:
                    us_ticker = tasvc.map_to_us_ticker(sym)
                    if not us_ticker:
                        st.info("No se encontró ticker US para este activo.")
                    else:
                        fundamental_data = tasvc.fundamentals(us_ticker) or {}
                        render_fundamental_data(fundamental_data)

                        cols = st.columns([1, 1, 1, 1])
                        with cols[0]:
                            period = st.selectbox("Período", ["3mo", "6mo", "1y", "2y"], index=1)
                        with cols[1]:
                            interval = st.selectbox("Intervalo", ["1d", "1h", "30m"], index=0)
                        with cols[2]:
                            sma_fast = st.number_input("SMA corta", min_value=5, max_value=100, value=20, step=1)
                        with cols[3]:
                            sma_slow = st.number_input("SMA larga", min_value=10, max_value=250, value=50, step=5)

                        df_ind = tasvc.indicators_for(sym, period=period, interval=interval, sma_fast=sma_fast, sma_slow=sma_slow)
                        if df_ind.empty:
                            st.info("No se pudo descargar histórico para ese símbolo/periodo/intervalo.")
                        else:
                            fig = plot_technical_analysis_chart(df_ind, sma_fast, sma_slow)
                            _ = st.plotly_chart(fig, use_container_width=True)
                            alerts = tasvc.alerts_for(df_ind)
                            if alerts:
                                for a in alerts:
                                    al = a.lower()
                                    if "bajista" in al or "sobrecompra" in al:
                                        st.warning(a)
                                    elif "alcista" in al or "sobreventa" in al:
                                        st.success(a)
                                    else:
                                        st.info(a)
                            else:
                                st.caption("Sin alertas técnicas en la última vela.")

    # Auto-refresh
    if "last_refresh" not in st.session_state:
        st.session_state["last_refresh"] = time.time()
    try:
        do_refresh = (refresh_secs is not None) and (float(refresh_secs) > 0)
    except Exception:
        do_refresh = True
    if do_refresh and (time.time() - st.session_state["last_refresh"] >= float(refresh_secs)):
        st.session_state["last_refresh"] = time.time()
        st.session_state["show_refresh_toast"] = True
        st.rerun()

if __name__ == "__main__":
    main()
