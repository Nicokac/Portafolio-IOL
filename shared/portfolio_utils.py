"""Common helpers for normalizing and deduplicating portfolio symbols."""

from __future__ import annotations

from typing import Iterable, Iterator, Tuple, Union


def _normalize_symbols(symbols: Iterable[str | None]) -> Iterator[str]:
    for raw in symbols:
        if raw is None:
            continue
        normalized = str(raw).strip().upper()
        if normalized:
            yield normalized


def unique_symbols(
    symbols: Iterable[str | None],
    *,
    return_count: bool = False,
    sort: bool = True,
) -> Union[Tuple[str, ...], int]:
    """Return unique portfolio symbols or their count.

    Args:
        symbols: Iterable containing the raw symbol values. ``None`` or empty
            strings are ignored and remaining values are normalized by stripping
            whitespace and converting to upper-case.
        return_count: When ``True`` the function returns the number of unique
            symbols after normalization instead of the collection itself.
        sort: When ``True`` the resulting sequence of symbols is sorted
            alphabetically. When ``False`` the insertion order of the input is
            preserved.

    Returns:
        Either a tuple of unique symbols or, when ``return_count`` is ``True``,
        an integer with the number of unique items.
    """

    seen: set[str] = set()
    ordered: list[str] = []

    for symbol in _normalize_symbols(symbols):
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)

    if return_count:
        return len(seen)

    if sort:
        return tuple(sorted(seen))

    return tuple(ordered)


__all__ = ["unique_symbols"]
