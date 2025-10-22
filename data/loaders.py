"""Simple dataframe loading helpers with conservative typing."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def ensure_dataframe(
    data: pd.DataFrame | Iterable[Mapping[str, Any]] | None,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Return a copy of *data* as a DataFrame with optional column alignment."""

    if isinstance(data, pd.DataFrame):
        frame = data.copy()
    elif data is None:
        frame = pd.DataFrame()
    else:
        frame = pd.DataFrame(list(data))

    if not columns:
        return frame

    frame = frame.copy()
    for column in columns:
        if column not in frame.columns:
            frame[column] = pd.Series(dtype="object")
    return frame.loc[:, list(columns)]


def load_csv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    dtype: Mapping[str, Any] | None = None,
    encoding: str | None = None,
) -> pd.DataFrame:
    """Read a CSV file into a DataFrame returning an empty frame on failure."""

    resolved = Path(path)
    if not resolved.exists():
        return ensure_dataframe(None, columns=columns)

    try:
        frame = pd.read_csv(resolved, dtype=dtype, encoding=encoding)
    except (OSError, ValueError, TypeError):
        return ensure_dataframe(None, columns=columns)

    return ensure_dataframe(frame, columns=columns)


def load_parquet(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Read a Parquet file returning an empty DataFrame when unavailable."""

    resolved = Path(path)
    if not resolved.exists():
        return ensure_dataframe(None, columns=columns)

    try:
        frame = pd.read_parquet(resolved, columns=list(columns) if columns else None)
    except (OSError, ValueError, TypeError, ImportError):
        return ensure_dataframe(None, columns=columns)

    return ensure_dataframe(frame, columns=columns)
