import tests.ui.test_streamlit_cache_reuse as reuse_mod  # noqa: F401 - ensures module stubs

import pytest

from controllers.portfolio.portfolio import (
    _DATASET_HASH_STATE_KEY,
    _PORTFOLIO_LAST_USER_STATE_KEY,
    _VISUAL_CACHE_STATE_KEY,
    render_portfolio_section,
)
from tests.ui.test_streamlit_cache_reuse import _portfolio_setup
from tests.ui.test_portfolio_ui import FakeStreamlit, _DummyContainer


def test_visual_cache_resets_when_user_changes(
    monkeypatch: pytest.MonkeyPatch,
    _portfolio_setup,
) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    telemetry_calls: list[dict[str, object]] = []

    def _log_telemetry_stub(*_args, **kwargs) -> None:
        telemetry_calls.append(kwargs)

    monkeypatch.setattr("controllers.portfolio.portfolio.log_telemetry", _log_telemetry_stub)

    current_user = {"value": "user-1"}

    def _current_user() -> str:
        return current_user["value"]

    monkeypatch.setattr("controllers.portfolio.portfolio.get_current_user_id", _current_user)

    (
        _portfolio_mod,
        _summary,
        _table,
        _charts,
        view_model_service,
        notifications_service,
    ) = _portfolio_setup(fake_st)

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    assert fake_st.session_state.get(_PORTFOLIO_LAST_USER_STATE_KEY) == "user-1"

    fake_st.session_state[_VISUAL_CACHE_STATE_KEY] = {"stale": {"rendered": True}}
    fake_st.session_state[_DATASET_HASH_STATE_KEY] = "stale-hash"

    current_user["value"] = "user-2"

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    cache = fake_st.session_state.get(_VISUAL_CACHE_STATE_KEY)
    assert isinstance(cache, dict)
    assert "stale" not in cache
    assert fake_st.session_state.get(_DATASET_HASH_STATE_KEY) != "stale-hash"
    assert fake_st.session_state.get(_PORTFOLIO_LAST_USER_STATE_KEY) == "user-2"

    cache_events = [
        entry
        for entry in telemetry_calls
        if entry.get("phase") == "portfolio.visual_cache" and isinstance(entry.get("extra"), dict)
    ]
    assert cache_events, "Se esperaba registrar telemetría de caché visual"
    assert cache_events[-1]["extra"].get("visual_cache_cleared") is True
