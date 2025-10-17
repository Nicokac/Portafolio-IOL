"""Performance regressions for container stability of lazy components."""

from __future__ import annotations

import tests.ui.test_streamlit_lazy_fix as _lazy_stubs  # noqa: F401 - ensure stubs are registered
from controllers.portfolio import portfolio as portfolio_mod
from tests.ui.test_portfolio_ui import FakeStreamlit


def test_lazy_container_placeholders_are_stable(monkeypatch) -> None:
    """Ensures lazy containers retain their placeholders across reruns."""

    fake_st = FakeStreamlit(radio_sequence=[0])
    monkeypatch.setattr(portfolio_mod, "st", fake_st)

    store: dict[str, dict[str, object]] = {}
    entry = portfolio_mod._ensure_component_entry(store, "table")

    trigger_placeholder = entry.get("trigger_placeholder")
    body_placeholder = entry.get("body_placeholder")
    container = entry.get("container")

    assert trigger_placeholder is not None
    assert body_placeholder is not None
    assert container is not None

    entry_again = portfolio_mod._ensure_component_entry(store, "table")
    assert entry_again.get("trigger_placeholder") is trigger_placeholder
    assert entry_again.get("body_placeholder") is body_placeholder
    assert entry_again.get("container") is container

    charts_entry = portfolio_mod._ensure_component_entry(store, "charts")
    charts_trigger = charts_entry.get("trigger_placeholder")
    charts_body = charts_entry.get("body_placeholder")
    assert charts_trigger is not None
    assert charts_body is not None
