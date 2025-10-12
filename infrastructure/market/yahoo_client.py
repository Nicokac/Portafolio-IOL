"""Yahoo Finance market data client."""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, Iterable, Sequence, TypeVar

import pandas as pd
import requests
import yfinance as yf

from services.cache.market_data_cache import create_persistent_cache
from shared.errors import AppError

T = TypeVar("T")

LOGGER = logging.getLogger(__name__)


def _normalise_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def make_symbol_url(symbol: object, base_url: str = "https://finance.yahoo.com/quote") -> str | None:
    """Return the Yahoo Finance quote URL for ``symbol`` when possible."""

    if symbol is None:
        return None
    try:
        if pd.isna(symbol):  # type: ignore[arg-type]
            return None
    except Exception:  # pragma: no cover - defensive branch
        pass
    normalized = _normalise_symbol(str(symbol))
    if not normalized:
        return None
    return f"{base_url}/{normalized}"


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
    _CACHE_TTL = 6 * 60 * 60
    _FUNDAMENTALS_CACHE = create_persistent_cache("yahoo_fundamentals")
    _DIVIDENDS_CACHE = create_persistent_cache("yahoo_dividends")
    _SHARES_CACHE = create_persistent_cache("yahoo_shares")
    _PRICES_CACHE = create_persistent_cache("yahoo_prices")

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

    def get_fundamentals(self, ticker: str) -> Dict[str, Any]:
        """Return normalised fundamentals for a symbol."""

        cache_key = _normalise_symbol(ticker)

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

        sentinel = object()
        cached = self._FUNDAMENTALS_CACHE.get(cache_key, sentinel)
        if cached is not sentinel:
            return dict(cached)

        data = self._with_ticker(ticker, _load)
        self._FUNDAMENTALS_CACHE.set(cache_key, dict(data), ttl=self._CACHE_TTL)
        return data

    def get_dividends(self, ticker: str) -> pd.DataFrame:
        """Return historical dividends as a DataFrame."""

        cache_key = _normalise_symbol(ticker)

        def _load(yft: yf.Ticker) -> pd.DataFrame:
            div_series: pd.Series = yft.dividends
            if div_series is None or div_series.empty:
                raise AppError(f"Yahoo Finance no tiene dividendos para {ticker}")
            df = div_series.rename("amount").to_frame().reset_index()
            df = df.rename(columns={df.columns[0]: "date"})
            df["date"] = pd.to_datetime(df["date"], utc=True)
            df["amount"] = df["amount"].astype(float)
            return df.sort_values("date").reset_index(drop=True)

        sentinel = object()
        cached = self._DIVIDENDS_CACHE.get(cache_key, sentinel)
        if isinstance(cached, pd.DataFrame):
            return cached.copy(deep=True)

        df = self._with_ticker(ticker, _load)
        self._DIVIDENDS_CACHE.set(cache_key, df.copy(deep=True), ttl=self._CACHE_TTL)
        return df

    def get_shares_outstanding(self, ticker: str) -> pd.DataFrame:
        """Return shares outstanding history."""

        cache_key = _normalise_symbol(ticker)

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

        sentinel = object()
        cached = self._SHARES_CACHE.get(cache_key, sentinel)
        if isinstance(cached, pd.DataFrame):
            return cached.copy(deep=True)

        df = self._with_ticker(ticker, _load)
        self._SHARES_CACHE.set(cache_key, df.copy(deep=True), ttl=self._CACHE_TTL)
        return df

    def get_price_history(self, ticker: str) -> pd.DataFrame:
        """Return OHLC history with adjusted close values."""

        cache_key = _normalise_symbol(ticker)

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

        sentinel = object()
        cached = self._PRICES_CACHE.get(cache_key, sentinel)
        if isinstance(cached, pd.DataFrame):
            return cached.copy(deep=True)

        df = self._with_ticker(ticker, _load)
        self._PRICES_CACHE.set(cache_key, df.copy(deep=True), ttl=self._CACHE_TTL)
        return df

    @classmethod
    def clear_cache(cls) -> None:
        """Clear all persistent Yahoo caches (used in tests)."""

        for cache in (
            cls._FUNDAMENTALS_CACHE,
            cls._DIVIDENDS_CACHE,
            cls._SHARES_CACHE,
            cls._PRICES_CACHE,
        ):
            try:
                cache.clear()
            except Exception:  # pragma: no cover - defensive
                LOGGER.exception("No se pudo limpiar la caché de Yahoo")


__all__ = ["YahooFinanceClient"]
