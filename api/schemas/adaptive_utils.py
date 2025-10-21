"""Utilities for validating adaptive forecast payload constraints."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from fastapi import HTTPException, status


def _iter_history(history: Iterable[Any] | None) -> list[Any]:
    if history is None:
        return []
    if isinstance(history, list):
        return history
    return list(history)


def _extract_symbol(entry: Any) -> str | None:
    symbol = getattr(entry, "symbol", None)
    if symbol is None and isinstance(entry, dict):
        symbol = entry.get("symbol")
    if symbol is None:
        return None
    symbol_str = str(symbol).strip()
    return symbol_str.upper() if symbol_str else None


def validate_adaptive_limits(history: Iterable[Any] | None, max_size: int = 200) -> bool:
    """Ensure adaptive forecast history satisfies size and uniqueness constraints."""

    history_items = _iter_history(history)

    if len(history_items) > max_size:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"History length {len(history_items)} exceeds limit {max_size}",
        )

    seen: set[str] = set()
    for symbol in filter(None, (_extract_symbol(entry) for entry in history_items)):
        if symbol in seen:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Duplicate symbols detected in adaptive forecast history.",
            )
        seen.add(symbol)

    return True


__all__ = ["validate_adaptive_limits"]
