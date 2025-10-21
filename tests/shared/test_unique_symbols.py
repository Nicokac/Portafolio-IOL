import pandas as pd

from shared.portfolio_utils import unique_symbols


def test_unique_symbols_returns_sorted_tuple() -> None:
    symbols = ["aapl", "msft", "AAPL", "  tsla  ", ""]
    assert unique_symbols(symbols) == ("AAPL", "MSFT", "TSLA")


def test_unique_symbols_count_mode() -> None:
    symbols = ["aapl", "msft", "AAPL", None, "tsla"]
    assert unique_symbols(symbols, return_count=True) == 3


def test_unique_symbols_handles_series() -> None:
    series = pd.Series([" aapl", "msft", "aapl", "tsla"])
    assert unique_symbols(series, sort=False) == ("AAPL", "MSFT", "TSLA")


def test_unique_symbols_empty_input() -> None:
    assert unique_symbols([]) == ()
    assert unique_symbols([], return_count=True) == 0


def test_unique_symbols_sort_flag() -> None:
    symbols = ["b", "a", "c", "a"]
    assert unique_symbols(symbols, sort=False) == ("B", "A", "C")
    assert unique_symbols(symbols) == ("A", "B", "C")
