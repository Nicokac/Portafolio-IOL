from __future__ import annotations

from contextlib import contextmanager
from types import ModuleType, SimpleNamespace

import pytest

import sys

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
from ui.lazy import table_fragment
import ui.lazy.runtime as lazy_runtime


class _FormStreamlit:
    def __init__(self, submit_sequence: list[bool]) -> None:
        self.session_state: dict[str, object] = {}
        self._submit_sequence = list(submit_sequence)
        self.form_calls: list[str] = []
        self.stop_calls: int = 0

    @contextmanager
    def form(self, key: str):
        self.form_calls.append(key)
        yield

    def form_submit_button(self, label: str, key: str | None = None) -> bool:
        if not self._submit_sequence:
            return False
        return bool(self._submit_sequence.pop(0))

    def container(self):
        return _ContainerStub(self)

    def empty(self):  # pragma: no cover - compatibility shim
        return _PlaceholderStub(self)

    def stop(self) -> None:
        self.stop_calls += 1


class _FragmentStreamlit(_FormStreamlit):
    def __init__(self, checkbox_sequence: list[bool]) -> None:
        super().__init__(submit_sequence=[])
        self._checkbox_sequence = list(checkbox_sequence)
        self.fragment_calls: list[str] = []

    @contextmanager
    def fragment(self, name: str):
        self.fragment_calls.append(name)
        yield

    def checkbox(self, label: str, *, key: str | None = None) -> bool:
        if not self._checkbox_sequence:
            return False
        result = bool(self._checkbox_sequence.pop(0))
        if key is not None:
            self.session_state[key] = result
        return result


class _ContainerStub:
    def __init__(self, st):
        self._st = st

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def write(self, *_args, **_kwargs):  # pragma: no cover - placeholder compat
        return None

    def checkbox(self, *_args, **_kwargs):  # pragma: no cover - placeholder compat
        raise AssertionError("checkbox should not be used in form fallback")

    toggle = checkbox
    button = checkbox


class _PlaceholderStub(_ContainerStub):
    pass


def test_lazy_fragment_uses_form_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry_events: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs):
        telemetry_events.append(kwargs)

    fake_st = _FormStreamlit(submit_sequence=[False, True])
    placeholder = _ContainerStub(fake_st)
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", _capture_telemetry, raising=False)

    block = {"status": "pending", "triggered_at": None}
    dataset_token = "token-123"

    with table_fragment(dataset_token=dataset_token) as fragment_ctx:
        assert fragment_ctx.scope == "form"
        ready = portfolio_mod._prompt_lazy_block(
            block,
            placeholder=placeholder,
            button_label="Load",
            info_message="Info",
            key="positions_load_table",
            dataset_token=dataset_token,
            fallback_key="load_table",
        )
        assert ready is False
        assert fake_st.form_calls == ["portfolio_table__form"]
        assert fake_st.session_state.get("load_table") is False

    with table_fragment(dataset_token=dataset_token) as fragment_ctx:
        ready = portfolio_mod._prompt_lazy_block(
            block,
            placeholder=placeholder,
            button_label="Load",
            info_message="Info",
            key="positions_load_table",
            dataset_token=dataset_token,
            fallback_key="load_table",
        )
        assert ready is True
        fragment_ctx.stop()
        assert fake_st.stop_calls == 1
        assert fake_st.session_state.get("load_table") is True

    assert telemetry_events
    last_event = telemetry_events[-1]
    extra = last_event.get("extra", {})
    assert extra.get("ui_rerun_scope") == "form"
    assert extra.get("lazy_loaded_component") == "table"


def test_lazy_fragment_prefers_streamlit_fragment(monkeypatch: pytest.MonkeyPatch) -> None:
    telemetry_events: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs):
        telemetry_events.append(kwargs)

    fake_st = _FragmentStreamlit(checkbox_sequence=[False, True])
    placeholder = SimpleNamespace(
        checkbox=fake_st.checkbox,
        write=lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(portfolio_mod, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "st", fake_st)
    monkeypatch.setattr(lazy_runtime, "log_default_telemetry", _capture_telemetry, raising=False)

    block = {"status": "pending", "triggered_at": None}
    dataset_token = "token-456"

    with table_fragment(dataset_token=dataset_token) as fragment_ctx:
        assert fragment_ctx.scope == "fragment"
        ready = portfolio_mod._prompt_lazy_block(
            block,
            placeholder=placeholder,
            button_label="Load",
            info_message="Info",
            key="positions_load_table",
            dataset_token=dataset_token,
            fallback_key="load_table",
        )
        assert ready is False
        assert fake_st.fragment_calls == ["portfolio_table"]

    with table_fragment(dataset_token=dataset_token) as fragment_ctx:
        ready = portfolio_mod._prompt_lazy_block(
            block,
            placeholder=placeholder,
            button_label="Load",
            info_message="Info",
            key="positions_load_table",
            dataset_token=dataset_token,
            fallback_key="load_table",
        )
        assert ready is True
        fragment_ctx.stop()
        assert fake_st.stop_calls == 0

    assert telemetry_events
    last_event = telemetry_events[-1]
    extra = last_event.get("extra", {})
    assert extra.get("ui_rerun_scope") == "fragment"
    assert extra.get("lazy_loaded_component") == "table"
