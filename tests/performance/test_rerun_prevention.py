"""Regression tests ensuring lazy triggers avoid Streamlit reruns."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from controllers.portfolio import portfolio as portfolio_mod
from tests.ui.test_portfolio_ui import _DummyContainer, FakeStreamlit, _portfolio_setup


def _patch_renderers(monkeypatch: pytest.MonkeyPatch, portfolio_module):
    summary_calls = MagicMock(return_value=True)
    table_calls = MagicMock()
    charts_calls = MagicMock()

    monkeypatch.setattr(
        portfolio_module,
        "render_summary_section",
        lambda *args, **kwargs: summary_calls(),
    )
    monkeypatch.setattr(
        portfolio_module,
        "render_table_section",
        lambda *args, **kwargs: table_calls(),
    )
    monkeypatch.setattr(
        portfolio_module,
        "render_charts_section",
        lambda *args, **kwargs: charts_calls(),
    )
    monkeypatch.setattr(portfolio_module, "log_default_telemetry", MagicMock())
    monkeypatch.setattr(portfolio_module, "log_telemetry", MagicMock())

    return summary_calls, table_calls, charts_calls


def test_lazy_components_do_not_rerun_app(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    """Second render should reuse state without triggering extra checkboxes."""

    fake_st = FakeStreamlit(
        radio_sequence=[0, 0, 0],
        checkbox_values={"load_table": [False, True], "load_charts": [False, True]},
    )

    (
        portfolio_module,
        _basic,
        _advanced,
        _risk,
        _fundamental,
        _technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    summary_calls, table_calls, charts_calls = _patch_renderers(monkeypatch, portfolio_module)

    render_portfolio = portfolio_module.render_portfolio_section

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 0
    assert charts_calls.call_count == 0

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1

    lazy_state = fake_st.session_state.get("lazy_blocks", {})
    table_state = lazy_state["table"]
    charts_state = lazy_state["charts"]
    table_trigger = table_state["triggered_at"]
    charts_trigger = charts_state["triggered_at"]

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1
    assert table_state["triggered_at"] == table_trigger
    assert charts_state["triggered_at"] == charts_trigger
    assert len(fake_st.checkbox_calls) == 4

    assert fake_st.session_state.get("load_table") is True
    assert fake_st.session_state.get("load_charts") is True
