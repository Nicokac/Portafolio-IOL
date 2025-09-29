"""Yahoo Finance market data client."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, Sequence, TypeVar

import pandas as pd
import requests
import yfinance as yf

from shared.cache import cached
from shared.errors import AppError
from shared.settings import YAHOO_FUNDAMENTALS_TTL, YAHOO_QUOTES_TTL

T = TypeVar("T")


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def _normalise_percentage(value: Any) -> float | pd.NA:
    if value is None:
        return pd.NA
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive
        raise AppError("Valor numérico inválido en datos de Yahoo Finance") from exc
    if pd.isna(number):
        return pd.NA
    if -1.0 <= number <= 1.0:
        number *= 100.0
    return float(number)


class YahooFinanceClient:
    """Wrapper around :mod:`yfinance` with consistent outputs."""

    _SCREENER_URL = "https://query2.finance.yahoo.com/v1/finance/screener"

    def _with_ticker(self, ticker: str, loader: Callable[[yf.Ticker], T]) -> T:
        session = requests.Session()
        symbol = _normalise_symbol(ticker)
        try:
            yft = yf.Ticker(symbol, session=session)
            return loader(yft)
        except AppError:
            raise
        except Exception as exc:  # pragma: no cover - network/parsing issues
            raise AppError(f"Error al obtener datos para {symbol} desde Yahoo Finance") from exc
        finally:
            session.close()

    @cached(ttl=YAHOO_QUOTES_TTL)
    def list_symbols_by_markets(
        self, markets: Sequence[str], *, size: int | None = None
    ) -> list[dict[str, Any]]:
        """Fetch a list of tickers for the requested exchanges."""

        normalized_markets: list[str] = []
        seen_markets: set[str] = set()
        for raw in markets:
            market = str(raw or "").strip().upper()
            if not market or market in seen_markets:
                continue
            normalized_markets.append(market)
            seen_markets.add(market)

        if not normalized_markets:
            return []

        limit = size if isinstance(size, int) and size > 0 else 250
        limit = min(limit, 250)

        session = requests.Session()
        try:
            aggregated: list[dict[str, Any]] = []
            seen_symbols: set[str] = set()

            for market in normalized_markets:
                payload = {
                    "offset": 0,
                    "size": limit,
                    "quoteType": "EQUITY",
                    "sortField": "intradaymarketcap",
                    "sortType": "DESC",
                    "query": {
                        "operator": "and",
                        "operands": [
                            {
                                "operator": "or",
                                "operands": [
                                    {"operator": "eq", "operands": ["exchange", market]},
                                    {"operator": "eq", "operands": ["market", market]},
                                ],
                            }
                        ],
                    },
                }

                try:
                    response = session.post(
                        self._SCREENER_URL, json=payload, timeout=15
                    )
                    response.raise_for_status()
                except requests.RequestException as exc:  # pragma: no cover - network issues
                    raise AppError(
                        f"No se pudo obtener el listado de {market} desde Yahoo Finance"
                    ) from exc

                try:
                    data = response.json()
                except ValueError as exc:  # pragma: no cover - invalid payload
                    raise AppError(
                        "Yahoo Finance devolvió una respuesta inválida para el screener"
                    ) from exc

                finance = data.get("finance") if isinstance(data, dict) else None
                results = finance.get("result") if isinstance(finance, dict) else None
                if not isinstance(results, Iterable):
                    continue

                quotes: list[dict[str, Any]] = []
                for item in results:
                    if not isinstance(item, dict):
                        continue
                    entries = item.get("quotes")
                    if isinstance(entries, list):
                        quotes.extend(q for q in entries if isinstance(q, dict))

                for quote in quotes:
                    symbol = quote.get("symbol")
                    ticker = _normalise_symbol(symbol)
                    if not ticker or ticker in seen_symbols:
                        continue

                    aggregated.append(
                        {
                            "ticker": ticker,
                            "market_cap": quote.get("marketCap"),
                            "pe_ratio": quote.get("trailingPE") or quote.get("peRatio"),
                            "revenue_growth": quote.get("revenueGrowth"),
                            "country": quote.get("country") or quote.get("region"),
                            "market": quote.get("market"),
                            "exchange": quote.get("exchange"),
                        }
                    )
                    seen_symbols.add(ticker)

            return aggregated
        finally:
            session.close()

    @cached(ttl=YAHOO_FUNDAMENTALS_TTL)
    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Return normalised fundamentals for a symbol."""

        def _load(yft: yf.Ticker) -> Dict[str, Any]:
            info: Dict[str, Any] = yft.get_info()
            if not info:
                raise AppError(f"Yahoo Finance no devolvió fundamentals para {ticker}")

            dividend_yield = _normalise_percentage(info.get("dividendYield"))
            payout_ratio = _normalise_percentage(info.get("payoutRatio"))

            if dividend_yield is pd.NA or payout_ratio is pd.NA:
                raise AppError(f"Faltan métricas esenciales de fundamentals para {ticker}")

            result = {
                "ticker": _normalise_symbol(ticker),
                "long_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "currency": info.get("financialCurrency"),
                "market_cap": info.get("marketCap"),
                "dividend_yield": float(dividend_yield),
                "payout_ratio": float(payout_ratio),
            }
            trailing_eps = info.get("trailingEps")
            forward_eps = info.get("forwardEps")
            if trailing_eps is not None:
                result["trailing_eps"] = float(trailing_eps)
            if forward_eps is not None:
                result["forward_eps"] = float(forward_eps)
            return result

        return self._with_ticker(ticker, _load)

    @cached(ttl=YAHOO_QUOTES_TTL)
    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Return historical dividends as a DataFrame."""

        def _load(yft: yf.Ticker) -> pd.DataFrame:
            div_series: pd.Series = yft.dividends
            if div_series is None or div_series.empty:
                raise AppError(f"Yahoo Finance no tiene dividendos para {ticker}")
            df = div_series.rename("amount").to_frame().reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["amount"] = df["amount"].astype(float)
            return df.sort_values("date").reset_index(drop=True)

        return self._with_ticker(ticker, _load)

    @cached(ttl=YAHOO_FUNDAMENTALS_TTL)
    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        """Return shares outstanding history."""

        def _load(yft: yf.Ticker) -> pd.DataFrame:
            shares = yft.get_shares_full(start="1900-01-01")
            if shares is None:
                raise AppError(f"Yahoo Finance no devolvió flotantes para {ticker}")
            if isinstance(shares, pd.Series):
                df = shares.rename("shares").to_frame()
            else:
                df = pd.DataFrame(shares)
                if "shares" not in df.columns and df.shape[1] == 1:
                    df.columns = ["shares"]
            if df.empty:
                raise AppError(f"Yahoo Finance no devolvió flotantes para {ticker}")
            df = df.reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["shares"] = df["shares"].astype(float)
            return df.sort_values("date").reset_index(drop=True)

        return self._with_ticker(ticker, _load)

    @cached(ttl=YAHOO_QUOTES_TTL)
    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """Return OHLC history with adjusted close values."""

        def _load(yft: yf.Ticker) -> pd.DataFrame:
            history = yft.history(period="max", auto_adjust=False)
            if history is None or history.empty:
                raise AppError(f"Yahoo Finance no devolvió precios para {ticker}")
            required = {"Close", "Adj Close", "Volume"}
            if not required.issubset(history.columns):
                missing = required.difference(history.columns)
                raise AppError(
                    f"Datos de precios incompletos para {ticker}: faltan columnas {sorted(missing)}"
                )
            df = history.reset_index()
            rename_map = {"Date": "date", "Close": "close", "Adj Close": "adj_close", "Volume": "volume"}
            if "index" in df.columns and "Date" not in df.columns:
                rename_map["index"] = "date"
            df = df.rename(columns=rename_map)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["close"] = df["close"].astype(float)
            df["adj_close"] = df["adj_close"].astype(float)
            df["volume"] = df["volume"].astype(float)
            return df.sort_values("date").reset_index(drop=True)

        return self._with_ticker(ticker, _load)


__all__ = ["YahooFinanceClient"]
