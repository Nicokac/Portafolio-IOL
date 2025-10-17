"""Regression tests for skeleton fallback and lazy placeholders (hotfix patch)."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from controllers.portfolio import portfolio as portfolio_mod
from domain.models import Controls
from shared import skeletons
from shared.favorite_symbols import FavoriteSymbols
from tests.ui.test_streamlit_skeletons import _SimpleStreamlit


class _StubPlaceholder:
    """Minimal placeholder double exposing ``container`` for skeleton tests."""

    def __init__(self) -> None:
        self.container_called = False

    def container(self) -> str:  # noqa: D401 - simple marker
        self.container_called = True
        return "stub-container"


@pytest.fixture(autouse=True)
def _reset_skeleton_state(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the skeleton module uses an isolated session state."""

    fake_st = SimpleNamespace(session_state={})
    monkeypatch.setattr(skeletons, "st", fake_st)
    skeletons.initialize(0.0)
    fake_st.session_state.pop(getattr(skeletons, "_METRIC_KEY", "_ui_skeleton_render_ms"), None)
    fake_st.session_state.pop(getattr(skeletons, "_LABEL_KEY", "_ui_skeleton_label"), None)
    skeletons._FALLBACK_METRIC = None  # type: ignore[attr-defined]
    skeletons._FALLBACK_LABEL = None  # type: ignore[attr-defined]


def test_mark_placeholder_logs_and_returns_container(caplog: pytest.LogCaptureFixture) -> None:
    """mark_placeholder should log the render and return a valid container."""

    caplog.set_level(logging.INFO, logger="shared.skeletons")
    placeholder = _StubPlaceholder()

    container = skeletons.mark_placeholder("table", placeholder=placeholder)

    assert placeholder.container_called is True
    assert container == "stub-container"
    assert "🧩 Skeleton render called for table" in caplog.text


def test_lazy_blocks_follow_session_flags(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When the session flag flips to loaded the skeleton placeholders refresh automatically."""

    fake_st = _SimpleStreamlit()
    fake_st.session_state.setdefault(
        portfolio_mod._LAZY_BLOCKS_STATE_KEY,
        {
            "table": {
                "status": "pending",
                "dataset_hash": "none",
                "triggered_at": None,
                "loaded_at": None,
                "prompt_rendered": False,
            },
            "charts": {
                "status": "pending",
                "dataset_hash": "none",
                "triggered_at": None,
                "loaded_at": None,
                "prompt_rendered": False,
            },
        },
    )
    portfolio_mod.reset_portfolio_services()
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(portfolio_mod.skeletons, "st", fake_st)
    skeletons.initialize(0.0)

    table_calls = MagicMock()
    charts_calls = MagicMock()

    def _table_stub(*args, **kwargs):  # noqa: ANN001 - streamlit stub
        table_calls()
        return {}

    def _charts_stub(*args, **kwargs):  # noqa: ANN001 - streamlit stub
        charts_calls()
        return {}

    monkeypatch.setattr(portfolio_mod, "update_table_data", _table_stub)
    monkeypatch.setattr(portfolio_mod, "update_summary_section", lambda *a, **k: {})
    monkeypatch.setattr(portfolio_mod, "update_charts", _charts_stub)

    viewmodel = SimpleNamespace(
        controls=Controls(),
        metrics=SimpleNamespace(ccl_rate=None),
        positions=pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1200.0]}),
        totals=None,
        historical_total=None,
        contributions=None,
        pending_metrics=(),
    )

    tab_cache: dict[str, object] = {}
    timings: dict[str, float] = {}

    caplog.set_level(logging.INFO, logger="shared.skeletons")

    portfolio_mod.render_basic_tab(
        viewmodel,
        FavoriteSymbols(),
        snapshot=None,
        tab_slug="portafolio",
        tab_cache=tab_cache,
        timings=timings,
    )

    assert table_calls.call_count == 0
    assert charts_calls.call_count == 0
    assert "🧩 Skeleton render called for table" in caplog.text

    fake_st.session_state["load_table"] = True
    fake_st.session_state["load_charts"] = True

    portfolio_mod.render_basic_tab(
        viewmodel,
        FavoriteSymbols(),
        snapshot=None,
        tab_slug="portafolio",
        tab_cache=tab_cache,
        timings=timings,
    )

    assert table_calls.call_count == 1
    assert charts_calls.call_count == 1
    lazy_state = fake_st.session_state[portfolio_mod._LAZY_BLOCKS_STATE_KEY]
    assert lazy_state["table"]["status"] == "loaded"
    assert lazy_state["charts"]["status"] == "loaded"
