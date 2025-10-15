"""Utilities to manage favorite symbols persisted in Streamlit session state."""
from __future__ import annotations

import logging
from pathlib import Path
import json
from typing import Iterable, MutableMapping, Protocol, Sequence

import streamlit as st


logger = logging.getLogger(__name__)


DEFAULT_STORAGE_PATH = Path.home() / ".portafolio_iol" / "favorites.json"


class FavoriteStorage(Protocol):
    """Protocol that defines how favorites are persisted externally."""

    def load(self) -> list[str]:
        """Return the stored list of symbols."""

    def save(self, symbols: Sequence[str]) -> None:
        """Persist the provided list of symbols."""

    @property
    def last_error(self) -> str | None:
        """Return the last persistence error, if any."""


class JSONFavoriteStorage:
    """Persist favorites to a JSON file on disk."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path is not None else DEFAULT_STORAGE_PATH
        self._last_error: str | None = None

    @property
    def path(self) -> Path:
        """Return the file used to persist favorites."""

        return self._path

    @property
    def last_error(self) -> str | None:  # pragma: no cover - simple property
        return self._last_error

    def load(self) -> list[str]:
        """Load favorites from the configured JSON file."""

        self._last_error = None
        try:
            with self._path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except FileNotFoundError:
            return []
        except OSError as exc:  # pragma: no cover - hard to trigger consistently
            self._last_error = f"No se pudo leer favoritos: {exc}"
            return []
        except json.JSONDecodeError as exc:
            self._last_error = f"Favoritos con formato inválido: {exc}"
            return []

        if not isinstance(data, list):
            self._last_error = "Formato inválido: se esperaba una lista de símbolos"
            return []
        return [str(item) for item in data]

    def save(self, symbols: Sequence[str]) -> None:
        """Persist the favorites list as JSON, handling I/O errors gracefully."""

        self._last_error = None
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("w", encoding="utf-8") as fh:
                json.dump(list(symbols), fh, ensure_ascii=False, indent=2)
        except OSError as exc:
            self._last_error = f"No se pudo guardar favoritos: {exc}"


class FavoriteSymbols:
    """Manage a shared list of favorite symbols stored in ``st.session_state``."""

    STATE_KEY = "favorite_symbols"
    LOADED_FLAG_KEY = "_favorite_symbols_loaded"
    ERROR_KEY = "_favorite_symbols_error"

    def __init__(
        self,
        state: MutableMapping[str, object] | None = None,
        *,
        storage: FavoriteStorage | None = None,
    ) -> None:
        self._state = state if state is not None else st.session_state
        self._ensure_state_container()
        self._storage = storage
        if storage is not None:
            self._load_from_storage()

    # ------------------------------------------------------------------
    # Internal helpers
    def _normalize(self, symbol: str | None) -> str:
        if symbol is None:
            return ""
        sym = str(symbol).strip().upper()
        return sym

    def _normalize_many(self, symbols: Iterable[str]) -> list[str]:
        normalized: list[str] = []
        for symbol in symbols:
            sym = self._normalize(symbol)
            if sym and sym not in normalized:
                normalized.append(sym)
        return normalized

    def _ensure_state_container(self, *, log_on_create: bool = False) -> set[str]:
        favorites = self._state.get(self.STATE_KEY)
        if isinstance(favorites, set):
            return favorites
        if favorites is None:
            favorites = set()
            self._state[self.STATE_KEY] = favorites
            if log_on_create:
                logger.warning(
                    "favorite_symbols no estaba inicializado en session_state; "
                    "se crea automáticamente con un set vacío",
                )
            return favorites
        if isinstance(favorites, (list, tuple, set)):
            normalized = set(self._normalize_many(favorites))
            favorites_set: set[str] = set(normalized)
            self._state[self.STATE_KEY] = favorites_set
            return favorites_set
        logger.warning(
            "favorite_symbols tenía un tipo inesperado (%s); se re-inicializa",
            type(favorites).__name__,
        )
        favorites_set = set()
        self._state[self.STATE_KEY] = favorites_set
        return favorites_set

    def _sync_error_state(self) -> None:
        if not self._storage:
            return
        err = self._storage.last_error
        if err:
            self._state[self.ERROR_KEY] = err
        elif self.ERROR_KEY in self._state:
            del self._state[self.ERROR_KEY]

    def _load_from_storage(self) -> None:
        if not self._storage:
            return
        if self._state.get(self.LOADED_FLAG_KEY):
            return
        favorites = self._ensure_state_container()
        if favorites:
            self._state[self.LOADED_FLAG_KEY] = True
            return
        stored = self._storage.load()
        self._sync_error_state()
        favorites.clear()
        favorites.update(self._normalize_many(stored))
        self._state[self.STATE_KEY] = favorites
        self._state[self.LOADED_FLAG_KEY] = True

    def _persist(self) -> None:
        if not self._storage:
            return
        favorites = self._ensure_state_container()
        self._storage.save(sorted(favorites))
        self._sync_error_state()

    def normalize(self, symbol: str | None) -> str:
        """Public wrapper returning the canonical representation for ``symbol``."""
        return self._normalize(symbol)

    # ------------------------------------------------------------------
    # CRUD operations
    def list(self) -> list[str]:
        """Return the current list of favorite symbols."""
        favorites = self._ensure_state_container()
        return sorted(favorites)

    @property
    def last_error(self) -> str | None:
        """Return the last persistence error, if any."""

        err = self._state.get(self.ERROR_KEY)
        return str(err) if err else None

    def is_favorite(self, symbol: str | None) -> bool:
        """Check whether the provided symbol is currently marked as favorite."""
        sym = self._normalize(symbol)
        if not sym:
            return False
        favorites = self._ensure_state_container(log_on_create=True)
        return sym in favorites

    def add(self, symbol: str | None) -> None:
        """Add a symbol to the favorites list."""
        sym = self._normalize(symbol)
        if not sym:
            return
        favorites = self._ensure_state_container()
        if sym not in favorites:
            favorites.add(sym)
            self._persist()

    def remove(self, symbol: str | None) -> None:
        """Remove a symbol from the favorites list."""
        sym = self._normalize(symbol)
        if not sym:
            return
        favorites = self._ensure_state_container()
        if sym in favorites:
            favorites.remove(sym)
            self._persist()

    def replace(self, symbols: Iterable[str]) -> list[str]:
        """Replace the favorites list with the given iterable of symbols."""
        normalized: set[str] = set()
        for symbol in symbols:
            sym = self._normalize(symbol)
            if sym:
                normalized.add(sym)
        favorites = self._ensure_state_container()
        favorites.clear()
        favorites.update(normalized)
        self._state[self.STATE_KEY] = favorites
        self._persist()
        return sorted(favorites)

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
        favorites = sorted(self.list())
        normalized_options = [self._normalize(opt) for opt in options]
        for favorite in favorites:
            if favorite in normalized_options:
                return normalized_options.index(favorite)
        return 0 if options else 0


def get_persistent_favorites(
    state: MutableMapping[str, object] | None = None,
    *,
    storage: FavoriteStorage | None = None,
) -> FavoriteSymbols:
    """Return a ``FavoriteSymbols`` instance backed by on-disk persistence."""

    storage = storage or JSONFavoriteStorage()
    return FavoriteSymbols(state, storage=storage)


__all__ = [
    "FavoriteSymbols",
    "FavoriteStorage",
    "JSONFavoriteStorage",
    "get_persistent_favorites",
]
