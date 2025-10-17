from unittest.mock import MagicMock

import pytest

from tests.ui.test_portfolio_ui import _DummyContainer, FakeStreamlit, _portfolio_setup


def test_lazy_components_emit_telemetry(
    monkeypatch: pytest.MonkeyPatch, _portfolio_setup
) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0])

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

    summary_calls = MagicMock(return_value=True)
    table_calls = MagicMock()
    charts_calls = MagicMock()

    monkeypatch.setattr(portfolio_mod, "render_summary_section", lambda *a, **k: summary_calls())
    monkeypatch.setattr(portfolio_mod, "render_table_section", lambda *a, **k: table_calls())
    monkeypatch.setattr(portfolio_mod, "render_charts_section", lambda *a, **k: charts_calls())

    telemetry_events: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs) -> None:
        telemetry_events.append(kwargs)

    monkeypatch.setattr(portfolio_mod, "log_default_telemetry", _capture_telemetry)

    render_portfolio = portfolio_mod.render_portfolio_section

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 0
    assert charts_calls.call_count == 0
    assert telemetry_events == []

    for call in fake_st.checkbox_calls:
        state_key = str(call["key"]) if call.get("key") is not None else str(call["label"])
        fake_st._checkbox_values[state_key] = [True]

    render_portfolio(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1
    assert len(telemetry_events) == 2

    components = {event["extra"]["lazy_loaded_component"] for event in telemetry_events}
    assert components == {"table", "chart"}
    for event in telemetry_events:
        lazy_ms = event["extra"]["lazy_load_ms"]
        assert isinstance(lazy_ms, (int, float))
        assert lazy_ms >= 0
