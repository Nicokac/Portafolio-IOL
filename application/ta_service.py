# application\ta_service.py
from __future__ import annotations
import logging
from typing import Optional, List
from .portfolio_service import clean_symbol, map_to_us_ticker

import numpy as np
import pandas as pd
import streamlit as st
from requests.exceptions import HTTPError

# yfinance para históricos
try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None
    logging.warning("La librería yfinance no está instalada.")

try:  # pragma: no cover
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, IchimokuIndicator
    from ta.volatility import AverageTrueRange
except ImportError:  # Si la librería no está disponible, usamos stubs
    logging.warning("La librería ta no está instalada. Indicadores técnicos deshabilitados.")
    RSIIndicator = StochasticOscillator = MACD = IchimokuIndicator = AverageTrueRange = None

logger = logging.getLogger(__name__)


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
    sma_fast: int = 9,
    sma_slow: int = 20,
    ema_win: int = 21,
    bb_win: int = 20,
    bb_std: float = 2.0,
    rsi_win: int = 14,
    macd_fast: int = 12,
    macd_slow: int = 26,
    macd_signal: int = 9,
    atr_win: int = 14,
    stoch_win: int = 14,
    stoch_smooth: int = 3,
    ichi_conv: int = 9,
    ichi_base: int = 26,
    ichi_span: int = 52,
) -> pd.DataFrame:
    """
    Devuelve un DataFrame con OHLCV + indicadores técnicos.
    Columnas principales:
    ['Open','High','Low','Close','Volume','SMA_FAST','SMA_SLOW','EMA',
    'BB_L','BB_M','BB_U','RSI','MACD','MACD_SIGNAL','MACD_HIST','ATR',
    'STOCH_K','STOCH_D','ICHI_CONV','ICHI_BASE','ICHI_A','ICHI_B']
    """
    if yf is None or RSIIndicator is None:
        st.warning("Las librerías 'yfinance' y/o 'ta' no están disponibles.")
        return pd.DataFrame()

    ticker = map_to_us_ticker(simbolo)
    if not ticker:
        logger.info("No se pudo mapear %s a ticker US utilizable.", simbolo)
        return pd.DataFrame()

    try:
        hist = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            group_by="column",  # ayuda a evitar multiindex por ticker
            threads=False,
        )
    except Exception as e:
        logger.error("yfinance error (%s): %s", ticker, e)
        return pd.DataFrame()
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

    # MACD
    try:
        macd = MACD(close=close, window_slow=int(macd_slow), window_fast=int(macd_fast), window_sign=int(macd_signal))
        df["MACD"] = macd.macd()
        df["MACD_SIGNAL"] = macd.macd_signal()
        df["MACD_HIST"] = macd.macd_diff()
    except Exception:
        df["MACD"] = df["MACD_SIGNAL"] = df["MACD_HIST"] = pd.Series(index=df.index, dtype="float64")

    # ATR
    try:
        atr = AverageTrueRange(high=pd.Series(df["High"]), low=pd.Series(df["Low"]), close=close, window=int(atr_win))
        df["ATR"] = atr.average_true_range()
    except Exception:
        df["ATR"] = pd.Series(index=df.index, dtype="float64")

    # Estocástico
    try:
        stoch = StochasticOscillator(high=pd.Series(df["High"]), low=pd.Series(df["Low"]), close=close,
                                      window=int(stoch_win), smooth_window=int(stoch_smooth))
        df["STOCH_K"] = stoch.stoch()
        df["STOCH_D"] = stoch.stoch_signal()
    except Exception:
        df["STOCH_K"] = df["STOCH_D"] = pd.Series(index=df.index, dtype="float64")

    # Ichimoku
    try:
        ichi = IchimokuIndicator(high=pd.Series(df["High"]), low=pd.Series(df["Low"]),
                                 window1=int(ichi_conv), window2=int(ichi_base), window3=int(ichi_span))
        df["ICHI_CONV"] = ichi.ichimoku_conversion_line()
        df["ICHI_BASE"] = ichi.ichimoku_base_line()
        df["ICHI_A"] = ichi.ichimoku_a()
        df["ICHI_B"] = ichi.ichimoku_b()
    except Exception:
        df["ICHI_CONV"] = df["ICHI_BASE"] = df["ICHI_A"] = df["ICHI_B"] = pd.Series(index=df.index, dtype="float64")

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

def run_backtest(df: pd.DataFrame, strategy: str = "sma") -> pd.DataFrame:
    """Ejecuta un backtest simple según la estrategia indicada."""
    if df is None or df.empty or "Close" not in df:
        return pd.DataFrame()

    strat = (strategy or "").lower()
    sig = None
    if strat == "sma" and {"SMA_FAST", "SMA_SLOW"}.issubset(df.columns):
        sig = np.where(df["SMA_FAST"] > df["SMA_SLOW"], 1, -1)
    elif strat == "macd" and {"MACD", "MACD_SIGNAL"}.issubset(df.columns):
        sig = np.where(df["MACD"] > df["MACD_SIGNAL"], 1, -1)
    elif strat in {"estocastico", "stochastic"} and {"STOCH_K", "STOCH_D"}.issubset(df.columns):
        sig = np.where(df["STOCH_K"] > df["STOCH_D"], 1, -1)
    elif strat == "ichimoku" and {"ICHI_CONV", "ICHI_BASE"}.issubset(df.columns):
        sig = np.where(df["ICHI_CONV"] > df["ICHI_BASE"], 1, -1)
    if sig is None:
        return pd.DataFrame()

    res = pd.DataFrame(index=df.index)
    res["signal"] = sig
    res["ret"] = pd.Series(df["Close"]).pct_change()
    res["strategy_ret"] = res["signal"].shift(1) * res["ret"]
    res["equity"] = (1 + res["strategy_ret"]).cumprod()
    return res.dropna()

@st.cache_data
def get_fundamental_data(ticker: str) -> dict:
    """
    Obtiene datos fundamentales clave con yfinance. Filtra dividend yields implausibles (>20%).
    """
    if yf is None:
        st.warning("La librería 'yfinance' no está disponible.")
        return {}
    
    if ticker.startswith("^"):
        return {}

    try:
        if ticker.startswith("^"):
            return {}
        MAX_PLAUSIBLE_YIELD = 20.0  # %
        stock = yf.Ticker(ticker)
        try:
            info = stock.info
        except HTTPError:
            return {}

        if not info or info.get("marketCap") is None:
            return {"error": "No se encontraron datos fundamentales para este ticker."}

        dividend_rate = info.get("dividendRate")
        prev_close = info.get("previousClose")
        yield_val = 0.0
        if dividend_rate is not None and prev_close not in (None, 0):
            yield_val = (dividend_rate / prev_close) * 100

        plausible_yield = None if yield_val > MAX_PLAUSIBLE_YIELD else yield_val

        roe = info.get("returnOnEquity")
        margin = info.get("profitMargins")
        data = {
            "name": info.get("shortName", "N/A"),
            "sector": info.get("sector", "N/A"),
            "website": info.get("website", "N/A"),
            "market_cap": info.get("marketCap", 0),
            "pe_ratio": info.get("trailingPE"),
            "dividend_yield": plausible_yield,
            "price_to_book": info.get("priceToBook"),
            "return_on_equity": roe * 100 if roe is not None else None,
            "profit_margin": margin * 100 if margin is not None else None,
            "debt_to_equity": info.get("debtToEquity"),
        }
        return data
    except Exception as e:
        logging.error(f"Error al obtener datos fundamentales para {ticker}: {e}")
        return {"error": f"No se pudo contactar a la API para {ticker}."}

@st.cache_data
def portfolio_fundamentals(simbolos: List[str]) -> pd.DataFrame:
    """Devuelve un DataFrame con métricas fundamentales y ESG para cada símbolo."""
    if yf is None:
        st.warning("La librería 'yfinance' no está disponible.")
        return pd.DataFrame()

    rows = []
    for sym in simbolos:
        ticker = map_to_us_ticker(sym)
        if not ticker or ticker.startswith("^"):
            continue
        try:
            stock = yf.Ticker(ticker)
            try:
                info = stock.info or {}
            except HTTPError:
                info = {}
            sustain = getattr(stock, "sustainability", None)
            esg_score = None
            if isinstance(sustain, pd.DataFrame) and not sustain.empty:
                if "Value" in sustain.columns and "totalEsg" in sustain.index:
                    esg_score = _to_float(sustain.loc["totalEsg", "Value"])
            rows.append(
                {
                    "symbol": sym,
                    "name": info.get("shortName"),
                    "sector": info.get("sector"),
                    "market_cap": info.get("marketCap"),
                    "pe_ratio": info.get("trailingPE"),
                    "price_to_book": info.get("priceToBook"),
                    "return_on_equity": info.get("returnOnEquity"),
                    "profit_margin": info.get("profitMargins"),
                    "debt_to_equity": info.get("debtToEquity"),
                    "revenue_growth": info.get("revenueGrowth"),
                    "earnings_growth": info.get("earningsQuarterlyGrowth"),
                    "esg_score": esg_score,
                }
            )
        except Exception as e:
            logger.warning("fundamental error %s: %s", sym, e)
    return pd.DataFrame(rows)

@st.cache_data
def get_portfolio_history(simbolos: List[str], period: str = "1y") -> pd.DataFrame:
    """
    Descarga el historial (Adj Close si está, sino Close) para los símbolos indicados.
    - Mapea CEDEAR → US ticker para yfinance.
    - Renombra las columnas al **símbolo original** para que los gráficos muestren tus tickers.
    """
    if yf is None:
        st.warning("La librería 'yfinance' no está disponible.")
        return pd.DataFrame()

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

    def indicators_for(
        self,
        sym: str,
        *,
        period: str = "6mo",
        interval: str = "1d",
        sma_fast: int = 9,
        sma_slow: int = 20,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        atr_win: int = 14,
        stoch_win: int = 14,
        stoch_smooth: int = 3,
        ichi_conv: int = 9,
        ichi_base: int = 26,
        ichi_span: int = 52,
    ):
        return fetch_with_indicators(
            sym,
            period=period,
            interval=interval,
            sma_fast=sma_fast,
            sma_slow=sma_slow,
            macd_fast=macd_fast,
            macd_slow=macd_slow,
            macd_signal=macd_signal,
            atr_win=atr_win,
            stoch_win=stoch_win,
            stoch_smooth=stoch_smooth,
            ichi_conv=ichi_conv,
            ichi_base=ichi_base,
            ichi_span=ichi_span,
        )

    def alerts_for(self, df_ind):
        return simple_alerts(df_ind)

    def backtest(self, df_ind, *, strategy: str = "sma"):
        return run_backtest(df_ind, strategy=strategy)

    def fundamentals(self, us_ticker: str) -> dict:
        return get_fundamental_data(us_ticker) or {}

    def portfolio_history(self, *, simbolos: list[str], period: str = "1y"):
        return get_portfolio_history(simbolos=simbolos, period=period)

    def map_to_us_ticker(self, sym: str) -> str | None:
        return map_to_us_ticker(sym)
    
    def portfolio_fundamentals(self, simbolos: List[str]) -> pd.DataFrame:
        return portfolio_fundamentals(simbolos)
