"""Navigation helpers for Streamlit UI components."""

from __future__ import annotations

import streamlit as st


def safe_page_link(page: str, label: str, render_fallback=None) -> None:
    """Safely render a Streamlit page link, with fallback if not registered.

    Parameters
    ----------
    page:
        The fully qualified name of the page to link to.
    label:
        The text to display for the navigation link or fallback button.
    render_fallback:
        Optional callable to render inline content when the page is unavailable.
    """

    def _render_inline() -> None:
        if render_fallback is not None:
            if st.button(label):
                render_fallback()
        else:
            st.caption(f"({label} no disponible)")

    try:
        runtime = getattr(st, "runtime", None)
        registry = getattr(runtime, "_page_registry", None)
        if registry is not None and page not in registry:
            _render_inline()
            return

        if hasattr(st, "page_link"):
            st.page_link(page, label=label)
            return
    except KeyError:
        _render_inline()
        return

    if not hasattr(st, "page_link"):
        _render_inline()
