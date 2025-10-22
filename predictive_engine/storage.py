"""Persistence helpers for adaptive forecast history."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Protocol, cast

import pandas as pd

from predictive_engine.models import empty_history_frame

try:  # pragma: no cover - sqlite may be unavailable in some runtimes
    import sqlite3
except ImportError:  # pragma: no cover - defensive fallback
    sqlite3 = None  # type: ignore[assignment]


_DEFAULT_TABLE = "forecast_history"


if TYPE_CHECKING:  # pragma: no cover - typing only
    import sqlite3 as sqlite3_module

    SupportsClose = sqlite3_module.Connection
else:

    class SupportsClose(Protocol):
        def close(self) -> None:
            """Close the underlying connection."""


SQLiteFactory = Callable[[str | Path], SupportsClose]


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _normalise_path(path: str | Path) -> Path:
    resolved = Path(path)
    if not resolved.suffix:
        return resolved.with_suffix(".parquet")
    return resolved


def save_forecast_history(
    df: pd.DataFrame,
    path: str | Path = "./data/forecast_history.parquet",
    *,
    table_name: str = _DEFAULT_TABLE,
    sqlite_factory: SQLiteFactory | None = None,
) -> Path:
    """Persist the adaptive history to Parquet with SQLite fallback."""

    if not isinstance(df, pd.DataFrame):
        raise TypeError("History must be provided as a pandas.DataFrame")

    target = _normalise_path(path)
    _ensure_parent(target)

    if target.suffix.lower() in {".db", ".sqlite"} and sqlite3 is not None:
        factory = sqlite_factory or cast(SQLiteFactory, sqlite3.connect)
        connection = factory(target)
        connection_any = cast(Any, connection)
        try:
            df.to_sql(table_name, connection_any, if_exists="replace", index=False)
        finally:
            connection.close()
        return target

    try:
        df.to_parquet(target, index=False)
        return target
    except (ImportError, ValueError, TypeError):
        if sqlite3 is None:
            raise
        sqlite_target = target.with_suffix(".sqlite")
        _ensure_parent(sqlite_target)
        factory = sqlite_factory or cast(SQLiteFactory, sqlite3.connect)
        connection = factory(sqlite_target)
        connection_any = cast(Any, connection)
        try:
            df.to_sql(table_name, connection_any, if_exists="replace", index=False)
        finally:
            connection.close()
        return sqlite_target


def load_forecast_history(
    path: str | Path = "./data/forecast_history.parquet",
    *,
    table_name: str = _DEFAULT_TABLE,
    sqlite_factory: SQLiteFactory | None = None,
) -> pd.DataFrame:
    """Load persisted adaptive history, returning an empty frame on failure."""

    target = _normalise_path(path)
    if target.suffix.lower() in {".db", ".sqlite"}:
        if sqlite3 is None or not target.exists():
            return empty_history_frame()
        factory = sqlite_factory or cast(SQLiteFactory, sqlite3.connect)
        connection = factory(target)
        connection_any = cast(Any, connection)
        try:
            return pd.read_sql(table_name, connection_any)
        finally:
            connection.close()

    if target.exists():
        try:
            return pd.read_parquet(target)
        except (ImportError, ValueError, TypeError):
            pass

    sqlite_target = target.with_suffix(".sqlite")
    if sqlite3 is None or not sqlite_target.exists():
        return empty_history_frame()

    factory = sqlite_factory or cast(SQLiteFactory, sqlite3.connect)
    connection = factory(sqlite_target)
    connection_any = cast(Any, connection)
    try:
        return pd.read_sql(table_name, connection_any)
    finally:
        connection.close()
