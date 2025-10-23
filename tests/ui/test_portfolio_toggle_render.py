"""Tests ensuring the portfolio panel renders sin depender de un toggle."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

import ui.lazy.runtime as lazy_runtime
from tests.ui.test_portfolio_ui import FakeStreamlit, _ContextManager


@pytest.fixture
def _portfolio_setup(monkeypatch: pytest.MonkeyPatch):
    import controllers.portfolio.portfolio as portfolio_mod
    from tests.ui import test_portfolio_ui as portfolio_ui

    monkeypatch.setattr(portfolio_mod, "render_risk_analysis", lambda *a, **k: None, raising=False)
    return portfolio_ui._portfolio_setup._fixture_function(monkeypatch)


def test_portfolio_panel_renders_without_lazy_toggle(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    telemetry_events: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs):
        telemetry_events.append(kwargs)

    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    fake_st.experimental_rerun = MagicMock()
    fake_st.fragment_calls: list[str] = []

    def _fragment(name: str | None = None):  # noqa: D401 - mimic Streamlit signature
        fake_st.fragment_calls.append(str(name or ""))
        return _ContextManager(fake_st)

    fake_st.fragment = _fragment  # type: ignore[assignment]

    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_WARNING_EMITTED", False)
    monkeypatch.setattr(lazy_runtime, "_FRAGMENT_CONTEXT_RERUN_DATASETS", set())
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", _capture_telemetry, raising=False)
    guardian = type("_Guardian", (), {"wait_for_hydration": staticmethod(lambda *_, **__: True)})()
    monkeypatch.setattr(lazy_runtime, "get_fragment_state_guardian", lambda: guardian)
    lazy_runtime.fragment_context_ready = True

    (
        portfolio_mod,
        _basic,
        _advanced,
        _risk,
        _fundamental,
        _technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(portfolio_mod, "log_default_telemetry", lambda **_: None, raising=False)

    table_calls = MagicMock()
    charts_calls = MagicMock()

    monkeypatch.setattr(portfolio_mod, "render_table_section", lambda *a, **k: table_calls())
    monkeypatch.setattr(portfolio_mod, "render_charts_section", lambda *a, **k: charts_calls())

    portfolio_mod.render_portfolio_section(
        fake_st.container(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1
    assert len(fake_st.fragment_calls) == 2
    assert fake_st.experimental_rerun.called is False

    lazy_state = fake_st.session_state.get("lazy_blocks", {})
    assert lazy_state.get("table", {}).get("status") == "loaded"
    assert lazy_state.get("charts", {}).get("status") == "loaded"

    visibility_events = [event for event in telemetry_events if event.get("phase") == "portfolio.fragment_visibility"]
    assert len(visibility_events) >= 2
    components = {event.get("extra", {}).get("lazy_loaded_component") for event in visibility_events}
    assert components == {"table", "charts"}
    assert all(event.get("extra", {}).get("portfolio.fragment_visible") is True for event in visibility_events)
