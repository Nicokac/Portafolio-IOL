from __future__ import annotations

import streamlit as st

from shared.favorite_symbols import FavoriteSymbols

_BADGE_STYLE_KEY = "_favorite_badge_css"


def _ensure_badge_styles() -> None:
    if st.session_state.get(_BADGE_STYLE_KEY):
        return
    st.markdown(
        """
        <style>
        .favorite-badges { margin-bottom: 0.5rem; }
        .favorite-badge {
            display: inline-block;
            background-color: rgba(255, 215, 0, 0.15);
            color: #b8860b;
            border: 1px solid rgba(184, 134, 11, 0.4);
            border-radius: 999px;
            padding: 0.1rem 0.6rem;
            margin-right: 0.3rem;
            font-size: 0.85rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.session_state[_BADGE_STYLE_KEY] = True


def render_favorite_badges(
    favorites: FavoriteSymbols,
    *,
    title: str = "⭐ Favoritos",
    empty_message: str = "⭐ Aún no marcaste favoritos.",
) -> None:
    """Render a badge list with the current favorites."""
    _ensure_badge_styles()
    favs = favorites.list()
    if not favs:
        st.caption(empty_message)
        return
    badges = " ".join(f"<span class='favorite-badge'>⭐ {sym}</span>" for sym in favs)
    st.markdown(
        f"<div class='favorite-badges'><strong>{title}:</strong> {badges}</div>",
        unsafe_allow_html=True,
    )


def render_favorite_toggle(
    symbol: str | None,
    favorites: FavoriteSymbols,
    *,
    key_prefix: str,
    help_text: str | None = None,
) -> bool:
    """Render a toggle to mark/unmark the provided symbol as favorite."""
    if not symbol:
        st.caption("Seleccioná un símbolo para poder marcarlo como favorito.")
        return False

    toggle_key = f"favorite_toggle_{key_prefix}"
    last_symbol_key = f"favorite_toggle_last_{key_prefix}"

    current_symbol = favorites.normalize(symbol)
    last_symbol = st.session_state.get(last_symbol_key)
    current_state = favorites.is_favorite(current_symbol)

    if last_symbol != current_symbol:
        st.session_state[last_symbol_key] = current_symbol
        st.session_state[toggle_key] = current_state

    toggled = st.toggle(
        "⭐ Marcar como favorito",
        key=toggle_key,
        help=help_text,
    )

    if toggled and not current_state:
        favorites.add(current_symbol)
    elif not toggled and current_state:
        favorites.remove(current_symbol)

    return toggled


__all__ = [
    "render_favorite_badges",
    "render_favorite_toggle",
]
