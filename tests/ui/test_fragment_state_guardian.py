"""Tests for the lazy fragment state guardian."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
from typing import Any

import sys

import pytest

if "fastapi" not in sys.modules:
    fake_fastapi = ModuleType("fastapi")
    fake_fastapi.Depends = lambda *args, **kwargs: None
    fake_fastapi.HTTPException = Exception
    fake_fastapi.status = SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
    sys.modules["fastapi"] = fake_fastapi
    for name in (
        "fastapi.routing",
        "fastapi.params",
        "fastapi.temp_pydantic_v1_params",
        "fastapi.openapi",
        "fastapi.openapi.models",
    ):
        sys.modules[name] = ModuleType(name)
    security_stub = ModuleType("fastapi.security")
    security_stub.HTTPAuthorizationCredentials = type(
        "HTTPAuthorizationCredentials",
        (),
        {},
    )
    security_stub.HTTPBearer = lambda *args, **kwargs: None
    sys.modules["fastapi.security"] = security_stub

if "statsmodels" not in sys.modules:
    statsmodels_mod = ModuleType("statsmodels")
    statsmodels_api = ModuleType("statsmodels.api")
    statsmodels_mod.api = statsmodels_api
    sys.modules["statsmodels"] = statsmodels_mod
    sys.modules["statsmodels.api"] = statsmodels_api

if "scipy" not in sys.modules:
    scipy_mod = ModuleType("scipy")
    scipy_stats = ModuleType("scipy.stats")
    scipy_sparse = ModuleType("scipy.sparse")
    scipy_stats_qmc = ModuleType("scipy.stats._qmc")
    scipy_stats_multicomp = ModuleType("scipy.stats._multicomp")
    scipy_sparse_csgraph = ModuleType("scipy.sparse.csgraph")
    scipy_sparse_shortest = ModuleType("scipy.sparse.csgraph._shortest_path")
    scipy_mod.stats = scipy_stats
    scipy_mod.sparse = scipy_sparse
    scipy_sparse.csgraph = scipy_sparse_csgraph  # type: ignore[attr-defined]
    sys.modules.update(
        {
            "scipy": scipy_mod,
            "scipy.stats": scipy_stats,
            "scipy.stats._qmc": scipy_stats_qmc,
            "scipy.stats._multicomp": scipy_stats_multicomp,
            "scipy.sparse": scipy_sparse,
            "scipy.sparse.csgraph": scipy_sparse_csgraph,
            "scipy.sparse.csgraph._shortest_path": scipy_sparse_shortest,
        }
    )

if "plotly" not in sys.modules:
    plotly_mod = ModuleType("plotly")
    plotly_express = ModuleType("plotly.express")
    plotly_graph_objects = ModuleType("plotly.graph_objects")
    plotly_graph_objs = ModuleType("plotly.graph_objs")
    plotly_subplots = ModuleType("plotly.subplots")
    plotly_io = ModuleType("plotly.io")

    def _noop_figure(*_args, **_kwargs):  # noqa: ANN001 - test stub
        return SimpleNamespace()

    plotly_express.line = _noop_figure  # type: ignore[attr-defined]
    plotly_express.area = _noop_figure  # type: ignore[attr-defined]
    plotly_express.bar = _noop_figure  # type: ignore[attr-defined]
    plotly_express.scatter = _noop_figure  # type: ignore[attr-defined]

    class _QualitativePalette:
        def __init__(self) -> None:
            self.Set2: list[str] = []
            self.Pastel: list[str] = []
            self.Light24: list[str] = []

        def __getattr__(self, _name: str) -> list[str]:  # noqa: D401 - dynamic palette
            return []

    plotly_express.colors = SimpleNamespace(  # type: ignore[attr-defined]
        qualitative=_QualitativePalette()
    )

    plotly_io.write_image = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
    plotly_io.to_image = lambda *_args, **_kwargs: b""  # type: ignore[attr-defined]

    class _StubFigure:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def update_layout(self, *_args, **_kwargs) -> None:  # noqa: ANN001 - test stub
            return None

        def add_trace(self, *_args, **_kwargs) -> None:  # noqa: ANN001 - test stub
            return None

    plotly_graph_objects.Figure = _StubFigure  # type: ignore[attr-defined]
    plotly_graph_objs.Figure = _StubFigure  # type: ignore[attr-defined]
    plotly_subplots.make_subplots = _noop_figure  # type: ignore[attr-defined]

    plotly_mod.express = plotly_express  # type: ignore[attr-defined]
    plotly_mod.graph_objects = plotly_graph_objects  # type: ignore[attr-defined]
    plotly_mod.graph_objs = plotly_graph_objs  # type: ignore[attr-defined]
    plotly_mod.subplots = plotly_subplots  # type: ignore[attr-defined]
    plotly_mod.io = plotly_io  # type: ignore[attr-defined]

    sys.modules.update(
        {
            "plotly": plotly_mod,
            "plotly.express": plotly_express,
            "plotly.graph_objects": plotly_graph_objects,
            "plotly.graph_objs": plotly_graph_objs,
            "plotly.subplots": plotly_subplots,
            "plotly.io": plotly_io,
        }
    )

from controllers.portfolio import portfolio as portfolio_mod
import shared.fragment_state as fragment_state
from tests.ui.test_portfolio_ui import FakeStreamlit


@pytest.fixture()
def guardian_test_setup(monkeypatch: pytest.MonkeyPatch):
    fake_st = FakeStreamlit(radio_sequence=[0])
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(fragment_state, "st", fake_st)

    events: list[tuple[str, Any, Any]] = []

    def _capture(action: str, detail: Any, *, dataset_hash: str | None = None, latency_ms: Any | None = None) -> None:
        events.append((action, detail, dataset_hash))

    monkeypatch.setattr(portfolio_mod, "log_user_action", _capture)
    monkeypatch.setattr(fragment_state, "log_user_action", _capture)
    fragment_state.reset_fragment_state_guardian()
    return fake_st, events


def _build_block(dataset_token: str) -> dict[str, Any]:
    return {
        "status": "pending",
        "dataset_hash": dataset_token,
        "triggered_at": None,
        "prompt_rendered": False,
    }


def test_guardian_rehydrates_missing_state(guardian_test_setup):
    fake_st, events = guardian_test_setup
    placeholder = fake_st.empty()
    dataset_token = "dataset-token"
    block = _build_block(dataset_token)

    fake_st._checkbox_values["load_table"] = [True]
    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla se cargará al activarla.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )
    assert ready is True
    assert block["status"] == "loaded"

    # Simulate rerun losing the underlying flags.
    fake_st.session_state.pop("positions_load_table", None)
    fake_st.session_state.pop("load_table", None)

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla se cargará al activarla.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )
    assert ready is True
    assert fake_st.session_state.get("positions_load_table") is True
    assert fake_st.session_state.get("load_table") is True

    rehydrate_events = [event for event in events if event[0] == "lazy_block_rehydrated"]
    assert rehydrate_events, "expected rehydration to be logged"
    assert rehydrate_events[0][2] == dataset_token


def test_guardian_respects_explicit_toggle(guardian_test_setup):
    fake_st, events = guardian_test_setup
    placeholder = fake_st.empty()
    dataset_token = "dataset-token"
    block = _build_block(dataset_token)

    fake_st._checkbox_values["load_table"] = [True, False, True]

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla se cargará al activarla.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )
    assert ready is True
    assert block["status"] == "loaded"

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla se cargará al activarla.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )
    assert ready is False
    assert block["status"] == "pending"
    assert fake_st.session_state.get("positions_load_table") is False
    assert fake_st.session_state.get("load_table") is False

    ready = portfolio_mod._prompt_lazy_block(
        block,
        placeholder=placeholder,
        button_label="📊 Cargar tabla del portafolio",
        info_message="La tabla se cargará al activarla.",
        key="positions_load_table",
        dataset_token=dataset_token,
        fallback_key="load_table",
    )
    assert ready is True
    assert any(event[0] == "lazy_block_rehydrated" for event in events) is False
