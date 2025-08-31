# application\ta_service.py
from __future__ import annotations
import logging
from typing import Optional, List

import numpy as np
import pandas as pd
import streamlit as st

# yfinance para históricos
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "yfinance"])
    import yfinance as yf

# Usamos sólo RSI de la librería 'ta'
try:
    from ta.momentum import RSIIndicator
except ImportError:  # pragma: no cover
    import subprocess, sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "ta"])
    from ta.momentum import RSIIndicator

# === Import refactorizado: get_config / clean_symbol (ya no existe load_config)
from .portfolio_service import get_config, clean_symbol

logger = logging.getLogger(__name__)
CONFIG = get_config()


# -----------------------
# Utilidades internas
# -----------------------

def _to_float(x) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace(" ", "")
    if "," in s and s.count(",") == 1 and s.rfind(",") > s.rfind("."):
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def map_to_us_ticker(simbolo: str) -> Optional[str]:
    """
    Mapea un CEDEAR a su ticker US usando config['cedear_to_us'].
    Si no está en el mapa y parece un ticker US (3-5 letras), devuelve el mismo.
    """
    s = clean_symbol(simbolo)
    cedear_map = CONFIG.get("cedear_to_us", {}) or {}
    if s in cedear_map:
        return clean_symbol(cedear_map[s])
    if s.isalpha() and 3 <= len(s) <= 5:
        return s
    return None


def _flatten_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Asegura columnas 1D: si yfinance devuelve MultiIndex (('Close','AAPL')), lo aplana.
    También estandariza a ['Open','High','Low','Close','Volume'].
    """
    if isinstance(df.columns, pd.MultiIndex):
        # Si sólo hay un ticker, dropeamos el último nivel
        if len(df.columns.levels[-1]) == 1:
            df.columns = df.columns.droplevel(-1)
        else:
            df.columns = ["_".join([str(c) for c in col if c != ""]) for col in df.columns]
    df = df.rename(columns={
        "open": "Open", "high": "High", "low": "Low", "close": "Close",
        "volume": "Volume", "adj close": "Adj Close", "adj_close": "Adj Close",
        "Adj Close": "Adj Close"
    })
    return df


def _bollinger_pandas(close: pd.Series, window: int = 20, std: float = 2.0):
    """Bollinger Bands con pandas puro (evita 2D)."""
    ma = close.rolling(window).mean()
    sd = close.rolling(window).std(ddof=0)
    upper = ma + std * sd
    lower = ma - std * sd
    return lower, ma, upper


# -----------------------
# Fetch + indicadores
# -----------------------

@st.cache_data
def fetch_with_indicators(
    simbolo: str,
    period: str = "6mo",
    interval: str = "1d",
    sma_fast: int = 20,
    sma_slow: int = 50,
    ema_win: int = 21,
    bb_win: int = 20,
    bb_std: float = 2.0,
    rsi_win: int = 14,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con OHLCV + SMA/EMA (pandas) + Bollinger (pandas) + RSI (ta).
    Columnas: ['Open','High','Low','Close','Volume','SMA_FAST','SMA_SLOW','EMA','BB_L','BB_M','BB_U','RSI']
    """
    ticker = map_to_us_ticker(simbolo)
    if not ticker:
        logger.info("No se pudo mapear %s a ticker US utilizable.", simbolo)
        return pd.DataFrame()

    hist = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
        group_by="column",  # ayuda a evitar multiindex por ticker
        threads=False,
    )
    if hist is None or hist.empty:
        return pd.DataFrame()

    df = _flatten_ohlcv(hist.copy())

    # Validar columnas mínimas
    for col in ("Close", "Open", "High", "Low", "Volume"):
        if col not in df.columns:
            # Intento recuperar "Close_*" si vino aplanado
            close_like = [c for c in df.columns if str(c).lower().startswith("close")]
            if col == "Close" and close_like:
                df["Close"] = df[close_like[0]]
            else:
                return pd.DataFrame()

    # Series 1D float
    close = pd.to_numeric(pd.Series(df["Close"]).squeeze(), errors="coerce")

    # Indicadores
    df["SMA_FAST"] = close.rolling(int(sma_fast)).mean()
    df["SMA_SLOW"] = close.rolling(int(sma_slow)).mean()
    df["EMA"]      = close.ewm(span=int(ema_win), adjust=False).mean()

    bb_l, bb_m, bb_u = _bollinger_pandas(close, window=int(bb_win), std=float(bb_std))
    df["BB_L"], df["BB_M"], df["BB_U"] = bb_l, bb_m, bb_u

    try:
        df["RSI"] = RSIIndicator(close=close, window=int(rsi_win), fillna=False).rsi()
    except Exception:
        df["RSI"] = pd.Series(index=df.index, dtype="float64")

    # Limpieza de NaNs iniciales por ventanas
    df = df.dropna().copy()
    return df


def simple_alerts(df: pd.DataFrame) -> List[str]:
    """
    Alertas simples:
    - RSI > 70 / < 30
    - Cruce alcista/bajista (SMA_FAST vs SMA_SLOW)
    - Precio tocando bandas
    """
    alerts: List[str] = []
    if df is None or df.empty:
        return alerts

    last = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else None

    # RSI
    rsi = _to_float(last.get("RSI"))
    if rsi is not None:
        if rsi >= 70:
            alerts.append("🔴 RSI en sobrecompra (≥70).")
        elif rsi <= 30:
            alerts.append("🟢 RSI en sobreventa (≤30).")

    # Cruces de SMA
    if prev is not None and "SMA_FAST" in df.columns and "SMA_SLOW" in df.columns:
        fast_now, slow_now = _to_float(last["SMA_FAST"]), _to_float(last["SMA_SLOW"])
        fast_prev, slow_prev = _to_float(prev["SMA_FAST"]), _to_float(prev["SMA_SLOW"])
        if None not in (fast_now, slow_now, fast_prev, slow_prev):
            if fast_prev <= slow_prev and fast_now > slow_now:
                alerts.append("⚡ Cruce alcista: SMA corta cruzó por encima de la SMA larga.")
            elif fast_prev >= slow_prev and fast_now < slow_now:
                alerts.append("⚠️ Cruce bajista: SMA corta cruzó por debajo de la SMA larga.")

    # Bandas de Bollinger
    close_val = _to_float(last.get("Close"))
    bbu = _to_float(last.get("BB_U"))
    bbl = _to_float(last.get("BB_L"))
    if None not in (close_val, bbu, bbl):
        if close_val >= bbu:
            alerts.append("📈 Precio tocando/rompiendo banda superior de Bollinger.")
        elif close_val <= bbl:
            alerts.append("📉 Precio tocando/rompiendo banda inferior de Bollinger.")

    return alerts


@st.cache_data
def get_fundamental_data(ticker: str) -> dict:
    """
    Obtiene datos fundamentales clave con yfinance. Filtra dividend yields implausibles (>20%).
    """
    try:
        MAX_PLAUSIBLE_YIELD = 20.0  # %
        stock = yf.Ticker(ticker)
        info = stock.info

        if not info or info.get("marketCap") is None:
            return {"error": "No se encontraron datos fundamentales para este ticker."}

        dividend_rate = info.get("dividendRate")
        prev_close = info.get("previousClose")
        yield_val = 0.0
        if dividend_rate is not None and prev_close not in (None, 0):
            yield_val = (dividend_rate / prev_close) * 100

        plausible_yield = None if yield_val > MAX_PLAUSIBLE_YIELD else yield_val

        data = {
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "website": info.get("website", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": plausible_yield,
        }
        return data
    except Exception as e:
        logging.error(f"Error al obtener datos fundamentales para {ticker}: {e}")
        return {"error": f"No se pudo contactar a la API para {ticker}."}


@st.cache_data
def get_portfolio_history(simbolos: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Descarga el historial (Adj Close si está, sino Close) para los símbolos indicados.
    - Mapea CEDEAR → US ticker para yfinance.
    - Renombra las columnas al **símbolo original** para que los gráficos muestren tus tickers.
    """
    if not simbolos:
        return pd.DataFrame()

    # Mapeo símbolo original -> US ticker (ignoramos los que no podemos mapear)
    pairs = [(s, map_to_us_ticker(s)) for s in simbolos]
    pairs = [(clean_symbol(s), t) for (s, t) in pairs if t]
    if not pairs:
        return pd.DataFrame()

    # yfinance soporta múltiples tickers a la vez; deduplicamos US tickers si fueran repetidos
    us_unique = sorted(set(t for _, t in pairs))
    try:
        hist = yf.download(
            tickers=us_unique,
            period=period,
            interval="1d",
            auto_adjust=False,
            progress=False,
            group_by="ticker",
            threads=False,
        )
    except Exception as e:
        logger.error("yfinance error (portfolio history): %s", e)
        return pd.DataFrame()

    if hist is None or hist.empty:
        return pd.DataFrame()

    # Seleccionamos Adj Close si existe; si no, Close
    df = None
    try:
        df = hist.xs("Adj Close", level=1, axis=1)
    except Exception:
        try:
            df = hist.xs("Close", level=1, axis=1)
        except Exception:
            return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    # Renombramos columnas al símbolo original (cuando sea posible)
    # Si varios símbolos originales mapean al mismo US ticker, prevalece el primero.
    us_to_orig = {}
    for orig, us_tk in pairs:
        us_to_orig.setdefault(us_tk, orig)

    df = df.rename(columns={us: us_to_orig.get(us, us) for us in df.columns})
    df = df.sort_index().ffill().dropna(how="all")
    return df

# --- Agregar al final de application/ta_service.py ---

class TAService:
    """Fachada de análisis técnico que envuelve las funciones existentes."""
    def indicators_for(self, sym: str, *, period: str = "6mo", interval: str = "1d",
                       sma_fast: int = 20, sma_slow: int = 50):
        return fetch_with_indicators(sym, period=period, interval=interval,
                                     sma_fast=sma_fast, sma_slow=sma_slow)

    def alerts_for(self, df_ind):
        return simple_alerts(df_ind)

    def fundamentals(self, us_ticker: str) -> dict:
        return get_fundamental_data(us_ticker) or {}

    def portfolio_history(self, *, simbolos: list[str], period: str = "1y"):
        return get_portfolio_history(simbolos=simbolos, period=period)

    def map_to_us_ticker(self, sym: str) -> str | None:
        return map_to_us_ticker(sym)