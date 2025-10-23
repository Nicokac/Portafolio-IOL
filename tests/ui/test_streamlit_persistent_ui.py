"""Regression tests ensuring lazy UI state persists across reruns."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import tests.ui.test_streamlit_lazy_fix as _lazy_stubs  # noqa: F401 - ensure stubs are registered
from controllers.portfolio import portfolio as portfolio_mod
from tests.ui.test_portfolio_ui import FakeStreamlit


def test_lazy_trigger_state_survives_reruns(monkeypatch: pytest.MonkeyPatch) -> None:
    """The lazy checkbox must remain checked across Streamlit reruns."""

    fake_st = FakeStreamlit(radio_sequence=[0], checkbox_values={"load_table": [False, True, True]})
    placeholder = fake_st.empty()
    monkeypatch.setattr(portfolio_mod, "st", fake_st)

    dataset_token = "dataset-token"
    block = {
        "status": "pending",
        "dataset_hash": dataset_token,
        "triggered_at": None,
        "loaded_at": None,
        "prompt_rendered": False,
    }

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla principal se cargará cuando la solicites.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )

    assert ready is False
    assert len(fake_st.checkbox_calls) == 1
    assert fake_st.session_state.get("load_table") is False

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla principal se cargará cuando la solicites.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )

    assert ready is True
    assert fake_st.session_state.get("load_table") is True
    assert len(fake_st.checkbox_calls) == 2

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla principal se cargará cuando la solicites.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )

    assert ready is True
    assert fake_st.session_state.get("load_table") is True
    assert len(fake_st.checkbox_calls) == 3


def test_ui_persist_metric_tracks_visibility(monkeypatch: pytest.MonkeyPatch) -> None:
    """The persistence metric should accumulate visibility duration."""

    fake_st = FakeStreamlit(radio_sequence=[0], checkbox_values={"load_table": [True, True]})
    placeholder = fake_st.empty()
    monkeypatch.setattr(portfolio_mod, "st", fake_st)

    clock = {"time": 100.0, "perf": 10.0}

    def _fake_time() -> float:
        return clock["time"]

    def _fake_perf() -> float:
        return clock["perf"]

    monkeypatch.setattr(portfolio_mod.time, "time", _fake_time)
    monkeypatch.setattr(portfolio_mod.time, "perf_counter", _fake_perf)

    dataset_token = "dataset-token"
    block = {
        "status": "pending",
        "dataset_hash": dataset_token,
        "triggered_at": None,
        "loaded_at": clock["time"] - 2.0,
        "prompt_rendered": False,
    }

    assert portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )

    persist_ms = fake_st.session_state.get(portfolio_mod._UI_PERSIST_STATE_KEY)
    assert isinstance(persist_ms, float)
    assert persist_ms >= 2000.0

    clock["time"] += 1.5
    clock["perf"] += 1.5

    assert portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )

    persist_ms = fake_st.session_state.get(portfolio_mod._UI_PERSIST_STATE_KEY)
    assert isinstance(persist_ms, float)
    assert persist_ms >= 3500.0


def test_visual_cache_telemetry_never_flips_false(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    """Once cleared, the visual cache telemetry should not emit a false flag for the same dataset."""

    fake_st = FakeStreamlit(radio_sequence=[0, 0])
    telemetry_calls: list[dict[str, object]] = []

    def _log_telemetry(files, *, phase, dataset_hash=None, extra=None, **kwargs) -> None:
        telemetry_calls.append(
            {
                "phase": phase,
                "dataset_hash": dataset_hash,
                "extra": dict(extra or {}),
            }
        )

    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr("controllers.portfolio.portfolio.log_telemetry", _log_telemetry)

    resets = iter([True, False])
    monkeypatch.setattr(
        portfolio_mod,
        "_maybe_reset_visual_cache_state",
        lambda: next(resets, False),
    )

    (
        _portfolio_mod,
        _summary,
        _table,
        _charts,
        view_model_service,
        notifications_service,
    ) = _portfolio_setup(fake_st)

    portfolio_mod.render_portfolio_section(
        fake_st.container(),
        cli=SimpleNamespace(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    portfolio_mod.render_portfolio_section(
        fake_st.container(),
        cli=SimpleNamespace(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    events = [
        entry
        for entry in telemetry_calls
        if entry["phase"] == "portfolio.visual_cache" and isinstance(entry.get("extra"), dict)
    ]
    assert len(events) >= 2
    assert events[0]["extra"].get("visual_cache_cleared") is True
    assert events[1]["extra"].get("visual_cache_cleared") is True
