"""Tests for deferred per-tab rendering and caching in the portfolio UI."""

from __future__ import annotations

from controllers.portfolio.portfolio import render_portfolio_section
from tests.ui.test_portfolio_ui import FakeStreamlit, _DummyContainer


def _run_portfolio(
    fake_st: FakeStreamlit,
    view_model_factory,
    notifications_factory,
) -> None:
    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={"ccl": 0.0},
        view_model_service_factory=view_model_factory,
        notifications_service_factory=notifications_factory,
    )


def test_deferred_render_only_renders_active_tab(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0])
    (
        _portfolio_mod,
        basic,
        advanced,
        risk,
        fundamental,
        _technical_badge,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    _run_portfolio(fake_st, view_model_factory, notifications_factory)

    assert basic.call_count == 3
    assert advanced.call_count == 0
    assert risk.call_count == 0
    assert fundamental.call_count == 0
    assert any("Actualizado" in caption for caption in fake_st.captions)
    cache = fake_st.session_state.get("render_cache", {})
    assert "portafolio" in cache
    entry = cache["portafolio"]
    assert entry.get("rendered") is True
    assert entry.get("last_source") == "fresh"
    assert fake_st.session_state.get("active_tab") == "portafolio"


def test_tab_uses_cached_placeholder_on_second_visit(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    (
        _portfolio_mod,
        basic,
        *_rest,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    _run_portfolio(fake_st, view_model_factory, notifications_factory)
    assert basic.call_count == 3

    _run_portfolio(fake_st, view_model_factory, notifications_factory)
    assert basic.call_count == 3, "Expected cached tab to skip re-render"

    cache = fake_st.session_state["render_cache"]["portafolio"]
    assert cache.get("last_source") == "cache"

    stats = fake_st.session_state.get("portfolio_fingerprint_cache_stats", {})
    assert stats.get("hits", 0) > 0, "Fingerprint cache should record hits on second render"
    assert stats.get("hit_ratio", 0.0) > 0, "Fingerprint cache hit ratio should be positive"


def test_tab_cache_invalidated_when_snapshot_changes(_portfolio_setup) -> None:
    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    (
        _portfolio_mod,
        basic,
        *_rest,
        view_model_factory,
        notifications_factory,
    ) = _portfolio_setup(fake_st)

    _run_portfolio(fake_st, view_model_factory, notifications_factory)
    assert basic.call_count == 3

    cache = fake_st.session_state["render_cache"]["portafolio"]
    components = cache.get("components", {})
    for entry in components.values():
        if isinstance(entry, dict):
            entry["signature"] = ("stale",)
    cache["signature"] = ("stale",)

    _run_portfolio(fake_st, view_model_factory, notifications_factory)
    assert basic.call_count == 6

    cache = fake_st.session_state["render_cache"]["portafolio"]
    assert cache.get("last_source") == "hot"
    assert cache.get("rendered") is True
