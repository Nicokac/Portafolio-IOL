"""Regression tests for lazy portfolio components (tables and charts)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from tests.ui.test_portfolio_ui import FakeStreamlit, _DummyContainer


def _get_checkbox_key(fake_st: FakeStreamlit, label_prefix: str) -> str:
    for call in fake_st.checkbox_calls:
        label = str(call.get("label"))
        if label.startswith(label_prefix):
            key = call.get("key")
            if key:
                return str(key)
    raise AssertionError(f"No checkbox rendered with prefix {label_prefix!r}")


def _patch_renderers(
    monkeypatch: pytest.MonkeyPatch,
    portfolio_mod,
) -> tuple[MagicMock, MagicMock, MagicMock]:
    summary_calls = MagicMock(return_value=True)
    table_calls = MagicMock()
    charts_calls = MagicMock()

    monkeypatch.setattr(
        portfolio_mod,
        "render_summary_section",
        lambda *args, **kwargs: summary_calls(),
    )
    monkeypatch.setattr(
        portfolio_mod,
        "render_table_section",
        lambda *args, **kwargs: table_calls(),
    )
    monkeypatch.setattr(
        portfolio_mod,
        "render_charts_section",
        lambda *args, **kwargs: charts_calls(),
    )
    monkeypatch.setattr(portfolio_mod, "log_default_telemetry", MagicMock())
    monkeypatch.setattr(portfolio_mod, "log_telemetry", MagicMock())

    return summary_calls, table_calls, charts_calls


def test_lazy_table_renders_once(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0, 0])

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

    mark_calls: list[str] = []

    def _mark_placeholder(label: str, *, placeholder=None):
        mark_calls.append(label)
        return placeholder

    monkeypatch.setattr(portfolio_module.skeletons, "mark_placeholder", _mark_placeholder)

    render_portfolio = portfolio_module.render_portfolio_section

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert summary_calls.call_count == 1
    assert table_calls.call_count == 0
    assert charts_calls.call_count == 0

    table_key = _get_checkbox_key(fake_st, "📊")
    fake_st._checkbox_values[table_key] = [True]

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 0
    assert fake_st.session_state.get("load_table") is True

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1, "tabla debe renderizarse solo una vez"
    assert charts_calls.call_count == 0

    lazy_state = fake_st.session_state.get("lazy_blocks", {})
    assert lazy_state["table"]["status"] == "loaded"

    flag_store = fake_st.session_state.get(portfolio_module._LAZY_FLAGS_STATE_KEY, {})
    table_flag = flag_store.get("load_table")
    assert isinstance(table_flag, dict)
    assert table_flag.get("dataset") == lazy_state["table"]["dataset_hash"]

    assert mark_calls.count("table") == 1


def test_lazy_charts_renders_once(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0, 0])

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

    mark_calls: list[str] = []

    def _mark_placeholder(label: str, *, placeholder=None):
        mark_calls.append(label)
        return placeholder

    monkeypatch.setattr(portfolio_module.skeletons, "mark_placeholder", _mark_placeholder)

    render_portfolio = portfolio_module.render_portfolio_section

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert summary_calls.call_count == 1
    assert charts_calls.call_count == 0

    chart_key = _get_checkbox_key(fake_st, "📈")
    fake_st._checkbox_values[chart_key] = [True]

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert charts_calls.call_count == 1
    assert fake_st.session_state.get("load_charts") is True
    assert table_calls.call_count == 0

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert charts_calls.call_count == 1, "gráficos deben renderizarse una sola vez"
    assert table_calls.call_count == 0

    lazy_state = fake_st.session_state.get("lazy_blocks", {})
    assert lazy_state["charts"]["status"] == "loaded"

    flag_store = fake_st.session_state.get(portfolio_module._LAZY_FLAGS_STATE_KEY, {})
    chart_flag = flag_store.get("load_charts")
    assert isinstance(chart_flag, dict)
    assert chart_flag.get("dataset") == lazy_state["charts"]["dataset_hash"]

    assert mark_calls.count("charts") == 1
