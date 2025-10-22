"""Regression tests for the lazy loading hotfix behaviour."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

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
from shared import export, skeletons
from tests.ui.test_portfolio_ui import FakeStreamlit


def test_lazy_trigger_persists_without_rerun(monkeypatch: pytest.MonkeyPatch) -> None:
    """Ensure the lazy table trigger relies on session state instead of reruns."""

    fake_st = FakeStreamlit(radio_sequence=[0])
    fake_st.experimental_rerun = lambda: (_ for _ in ()).throw(AssertionError("rerun invoked"))
    placeholder = fake_st.empty()
    monkeypatch.setattr(portfolio_mod, "st", fake_st)

    dataset_token = "dataset-token"
    block = {
        "status": "pending",
        "dataset_hash": dataset_token,
        "triggered_at": None,
        "prompt_rendered": False,
    }

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
    assert len(fake_st.checkbox_calls) == 1
    assert block["status"] == "pending"
    assert block["triggered_at"] is None
    assert fake_st.session_state.get("load_table") is False

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
    first_trigger = block["triggered_at"]
    assert isinstance(first_trigger, float)
    assert fake_st.session_state.get("load_table") is True

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
    assert block["triggered_at"] == first_trigger
    assert len(fake_st.checkbox_calls) >= 2

    flag_store = fake_st.session_state.get(portfolio_mod._LAZY_FLAGS_STATE_KEY, {})
    assert flag_store.get("load_table", {}).get("dataset") == dataset_token


def test_skeleton_initializes_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """initialize should only register the skeleton start timestamp once per session."""

    fake_state: dict[str, object] = {}
    fake_streamlit = SimpleNamespace(session_state=fake_state)
    monkeypatch.setattr(skeletons, "st", fake_streamlit)
    monkeypatch.setattr(skeletons, "_FALLBACK_INITIALIZED", False)

    first = skeletons.initialize(123.0)
    assert first is True
    assert fake_state["skeleton_initialized"] is True
    assert fake_state["_ui_skeleton_start"] == 123.0

    second = skeletons.initialize(456.0)
    assert second is False
    assert fake_state["_ui_skeleton_start"] == 123.0


def test_browser_renderer_does_not_call_kaleido(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When Kaleido is disabled, fig_to_png_bytes must not invoke the renderer."""

    monkeypatch.setattr(export, "_KALEIDO_AVAILABLE", False)
    monkeypatch.setattr(export, "_KALEIDO_IMPORTED", False)
    monkeypatch.setattr(export, "_KALEIDO_RUNTIME_AVAILABLE", None)
    monkeypatch.setattr(export, "_CHROMIUM_AVAILABLE", False)

    called = False

    def _fail_call(*_args, **_kwargs):  # noqa: ANN001 - test stub
        nonlocal called
        called = True
        raise AssertionError("pio.to_image should not be called when Kaleido is disabled")

    monkeypatch.setattr(export.pio, "to_image", _fail_call)

    result = export.fig_to_png_bytes(SimpleNamespace())

    assert result is None
    assert called is False
    assert export._KALEIDO_RUNTIME_AVAILABLE is False
