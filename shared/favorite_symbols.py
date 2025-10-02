"""Utilities to manage favorite symbols persisted in Streamlit session state."""
from __future__ import annotations

from typing import Iterable, MutableMapping, Sequence

import streamlit as st


class FavoriteSymbols:
    """Manage a shared list of favorite symbols stored in ``st.session_state``."""

    STATE_KEY = "favorite_symbols"

    def __init__(self, state: MutableMapping[str, object] | None = None) -> None:
        self._state = state if state is not None else st.session_state
        if self.STATE_KEY not in self._state:
            self._state[self.STATE_KEY] = []

    # ------------------------------------------------------------------
    # Internal helpers
    def _normalize(self, symbol: str | None) -> str:
        if symbol is None:
            return ""
        sym = str(symbol).strip().upper()
        return sym

    def normalize(self, symbol: str | None) -> str:
        """Public wrapper returning the canonical representation for ``symbol``."""
        return self._normalize(symbol)

    # ------------------------------------------------------------------
    # CRUD operations
    def list(self) -> list[str]:
        """Return the current list of favorite symbols."""
        return list(self._state.get(self.STATE_KEY, []))

    def is_favorite(self, symbol: str | None) -> bool:
        """Check whether the provided symbol is currently marked as favorite."""
        sym = self._normalize(symbol)
        if not sym:
            return False
        return sym in self._state[self.STATE_KEY]

    def add(self, symbol: str | None) -> None:
        """Add a symbol to the favorites list."""
        sym = self._normalize(symbol)
        if not sym:
            return
        favorites = self._state[self.STATE_KEY]
        if sym not in favorites:
            favorites.append(sym)

    def remove(self, symbol: str | None) -> None:
        """Remove a symbol from the favorites list."""
        sym = self._normalize(symbol)
        if not sym:
            return
        favorites = self._state[self.STATE_KEY]
        if sym in favorites:
            favorites.remove(sym)

    def replace(self, symbols: Iterable[str]) -> list[str]:
        """Replace the favorites list with the given iterable of symbols."""
        normalized: list[str] = []
        for symbol in symbols:
            sym = self._normalize(symbol)
            if sym and sym not in normalized:
                normalized.append(sym)
        self._state[self.STATE_KEY] = normalized
        return normalized

    def toggle(self, symbol: str | None, *, value: bool | None = None) -> bool:
        """Toggle the favorite state for ``symbol``.

        Args:
            symbol: Symbol to toggle.
            value: Optional explicit state. When ``None`` the value will
                be flipped.

        Returns:
            bool: The resulting favorite state.
        """
        sym = self._normalize(symbol)
        if not sym:
            return False
        current = self.is_favorite(sym)
        target = not current if value is None else bool(value)
        if target and not current:
            self.add(sym)
        elif not target and current:
            self.remove(sym)
        return target

    # ------------------------------------------------------------------
    # Helpers for UI integration
    def format_symbol(self, symbol: str | None) -> str:
        """Format a symbol for display, highlighting favorites."""
        sym = self._normalize(symbol)
        if not sym:
            return ""
        return f"⭐ {sym}" if self.is_favorite(sym) else sym

    def sort_options(self, options: Sequence[str]) -> list[str]:
        """Return options sorted with favorites first and alphabetical inside each group."""
        unique = []
        seen: set[str] = set()
        for opt in options:
            sym = self._normalize(opt)
            if sym and sym not in seen:
                seen.add(sym)
                unique.append(sym)
        favorites = set(self.list())
        fav_items = sorted([s for s in unique if s in favorites])
        rest = sorted([s for s in unique if s not in favorites])
        return fav_items + rest

    def default_index(self, options: Sequence[str]) -> int:
        """Return the default index prioritising the first favorite in ``options``."""
        favorites = self.list()
        normalized_options = [self._normalize(opt) for opt in options]
        for favorite in favorites:
            if favorite in normalized_options:
                return normalized_options.index(favorite)
        return 0 if options else 0


__all__ = ["FavoriteSymbols"]
