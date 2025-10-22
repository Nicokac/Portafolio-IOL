"""Performance regression checks for sidebar overhead and first-frame latency."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

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


@pytest.fixture
def _fresh_app(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.setenv("IOL_ALLOW_PLAIN_TOKENS", "1")
    monkeypatch.setenv("FASTAPI_TOKENS_KEY", "MDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDAwMDA=")
    monkeypatch.setenv("IOL_TOKENS_KEY", "MTExMTExMTExMTExMTExMTExMTExMTExMTExMTExMTE=")
    sys.modules.pop("app", None)
    sys.modules.pop("shared.config", None)
    st = ModuleType("streamlit")
    st.session_state = {}
    st.set_page_config = MagicMock()
    st.markdown = MagicMock()
    st.stop = MagicMock(side_effect=RuntimeError("stop"))
    st.columns = MagicMock(return_value=[MagicMock(), MagicMock()])
    st.caption = MagicMock()
    st.warning = MagicMock()
    st.error = MagicMock()
    st.container = MagicMock()
    st.rerun = MagicMock()
    st.cache_resource = lambda func=None, **_kwargs: func
    st.cache_data = st.cache_resource
    st.title = MagicMock()

    def _auto_mock(name: str) -> MagicMock:
        mock = MagicMock()
        setattr(st, name, mock)
        return mock

    st.__getattr__ = _auto_mock  # type: ignore[attr-defined]

    class _SidebarExpander:
        def __enter__(self) -> "_SidebarExpander":
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001 - context signature
            return False

        def markdown(self, *_args, **_kwargs) -> None:  # noqa: ANN001 - stub helper
            return None

    class _Sidebar:
        def expander(self, *_args, **_kwargs) -> _SidebarExpander:  # noqa: ANN001 - stub helper
            return _SidebarExpander()

    st.sidebar = _Sidebar()
    st.session_state.clear()
    st.session_state.update({"IOL_USERNAME": "demo", "authenticated": True})
    st.session_state["iol_login_ok_ts"] = 999.0
    st.cache_resource = MagicMock()
    st.stop = MagicMock()
    sys.modules["streamlit"] = st
    return importlib.import_module("app")


def test_startup_render_portfolio_complete_under_ten_seconds(
    _fresh_app: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = _fresh_app
    monkeypatch.setattr(app, "get_fx_rates_cached", MagicMock(return_value=({}, None)))
    monkeypatch.setattr(app, "render_header", MagicMock())
    monkeypatch.setattr(app, "render_action_menu", MagicMock())
    monkeypatch.setattr(app, "render_footer", MagicMock(side_effect=SystemExit))
    monkeypatch.setattr(app, "render_health_monitor_tab", MagicMock())
    monkeypatch.setattr(app, "render_ui_controls", MagicMock())
    monkeypatch.setattr(app, "build_iol_client", MagicMock())
    monkeypatch.setattr(app, "render_login_page", MagicMock())
    monkeypatch.setattr(app, "start_preload_worker", MagicMock(return_value=True))
    monkeypatch.setattr(app, "resume_preload_worker", MagicMock(return_value=True))
    monkeypatch.setattr(app, "is_preload_complete", MagicMock(return_value=True))
    monkeypatch.setattr(app, "ensure_scientific_preload_ready", MagicMock(return_value=True))
    monkeypatch.setattr(app, "_schedule_post_login_initialization", MagicMock())
    monkeypatch.setattr(app, "_schedule_scientific_preload_resume", MagicMock())
    original_lazy_attr = app._lazy_attr

    def _fake_lazy_attr(module: str, attr: str):  # noqa: ANN001 - local override
        if module == "ui.controllers.portfolio_ui" and attr == "render_portfolio_ui":
            return MagicMock(return_value=0)
        if module == "ui.tabs.recommendations" and attr == "render_recommendations_tab":
            return MagicMock()
        return original_lazy_attr(module, attr)

    monkeypatch.setattr(app, "_lazy_attr", _fake_lazy_attr)

    telemetry_calls: list[dict[str, object]] = []

    def _capture_telemetry(**kwargs) -> None:
        telemetry_calls.append(kwargs)

    monkeypatch.setattr(app, "log_default_telemetry", _capture_telemetry)
    monkeypatch.setattr(app.time, "time", lambda: 1000.0)

    with pytest.raises(SystemExit):
        app.main([])

    assert telemetry_calls, "Expected startup telemetry to be recorded."
    telemetry = telemetry_calls[-1]
    assert telemetry.get("phase") == "startup.render_portfolio_complete"
    assert telemetry.get("ui_total_load_ms") < 10_000
    extra = telemetry.get("extra") or {}
    if "streamlit_overhead_ms" in extra:
        assert float(extra["streamlit_overhead_ms"]) >= 0.0
