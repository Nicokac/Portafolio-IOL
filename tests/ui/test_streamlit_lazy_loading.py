from unittest.mock import MagicMock

import pytest

from tests.ui.test_portfolio_ui import FakeStreamlit


def test_table_and_charts_defer_until_user_action(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
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

    render_portfolio = portfolio_mod.render_portfolio_section

    render_portfolio(
        fake_st.container(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert summary_calls.call_count == 1
    assert table_calls.call_count == 0
    assert charts_calls.call_count == 0

    lazy_state = fake_st.session_state.get("lazy_blocks")
    assert lazy_state["table"]["status"] == "pending"
    assert lazy_state["charts"]["status"] == "pending"

    for call in fake_st.checkbox_calls:
        state_key = str(call["key"]) if call.get("key") is not None else str(call["label"])
        fake_st._checkbox_values[state_key] = [True]

    render_portfolio(
        fake_st.container(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1
    assert lazy_state["table"]["status"] == "loaded"
    assert lazy_state["charts"]["status"] == "loaded"
