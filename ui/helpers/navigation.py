"""Navigation helpers for Streamlit UI components."""

from __future__ import annotations

from typing import Callable

import streamlit as st


def safe_page_link(
    page: str,
    label: str,
    render_fallback: Callable[[], None] | None = None,
    *,
    prefer_inline: bool = False,
) -> None:
    """Safely render a Streamlit page link, with fallback if not registered.

    Parameters
    ----------
    page:
        The fully qualified name of the page to link to.
    label:
        The text to display for the navigation link or fallback button.
    render_fallback:
        Optional callable to render inline content when the page is unavailable.
    prefer_inline:
        When ``True`` the inline fallback will be preferred even if the page is
        registered in the Streamlit registry.
    """

    def _render_inline() -> None:
        if render_fallback is not None:
            if st.button(label):
                render_fallback()
        else:
            st.caption(f"({label} no disponible)")

    if prefer_inline and render_fallback is not None:
        _render_inline()
        return

    try:
        runtime = getattr(st, "runtime", None)
        registry = getattr(runtime, "_page_registry", None)
        if registry is not None and page not in registry:
            _render_inline()
            return

        if hasattr(st, "page_link") and render_fallback is None:
            st.page_link(page, label=label)
            return
        if hasattr(st, "page_link") and not prefer_inline:
            st.page_link(page, label=label)
            return
    except KeyError:
        _render_inline()
        return

    if not hasattr(st, "page_link"):
        _render_inline()
