"""Utilities to standardise symbol and sector identifiers."""

from __future__ import annotations

import pandas as pd


_SIN_SECTOR = "Sin sector"


def normalise_symbol_sector(frame: pd.DataFrame | None) -> pd.DataFrame:
    """Return a copy of *frame* with normalised ``symbol`` and ``sector`` columns.

    The function is resilient to missing values, whitespace differences and
    alternative column names such as ``ticker``. Symbols are upper-cased and
    trimmed while sectors keep their original casing but collapse any blank or
    synonymous values into ``"Sin sector"``.
    """

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return pd.DataFrame(columns=["symbol", "sector"])

    normalised = frame.copy()
    if "symbol" not in normalised.columns and "ticker" in normalised.columns:
        normalised = normalised.rename(columns={"ticker": "symbol"})

    if "symbol" not in normalised.columns:
        normalised["symbol"] = pd.Series(dtype="string")

    symbols = (
        normalised.get("symbol", pd.Series(dtype="string"))
        .astype("string")
        .fillna("")
        .str.upper()
        .str.strip()
    )
    normalised["symbol"] = symbols

    sectors = (
        normalised.get("sector", pd.Series(dtype="string"))
        .astype("string")
        .fillna("")
        .str.strip()
    )
    sectors = sectors.mask(sectors.str.len() == 0, _SIN_SECTOR)
    sectors = sectors.mask(sectors.str.casefold() == _SIN_SECTOR.casefold(), _SIN_SECTOR)
    normalised["sector"] = sectors.fillna(_SIN_SECTOR)

    valid = normalised["symbol"] != ""
    normalised = normalised[valid]
    if "sector" in normalised.columns:
        normalised.loc[:, "sector"] = normalised["sector"].fillna(_SIN_SECTOR)

    return normalised


__all__ = ["normalise_symbol_sector"]
