"""Tests ensuring visual skeleton placeholders render early."""

from __future__ import annotations

from types import ModuleType, SimpleNamespace
import sys

import pandas as pd
import pytest

# Stub heavy FastAPI dependency before importing portfolio module.
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
from domain.models import Controls
from shared.favorite_symbols import FavoriteSymbols


class _DummyContext:
    def __enter__(self):  # noqa: D401 - minimal context manager
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - minimal context manager
        return None


class _Column:
    def __init__(self, owner: "_SimpleStreamlit") -> None:
        self._owner = owner

    def metric(self, *args, **kwargs) -> None:  # noqa: ANN002 - test stub
        self._owner.metrics.append((args, kwargs))


class _Placeholder:
    def __init__(self, owner: "_SimpleStreamlit") -> None:
        self._owner = owner

    def container(self) -> _DummyContext:
        return _DummyContext()

    def write(self, body: str) -> None:
        self._owner.markdowns.append({"body": body, "placeholder": True, "write": True})

    def info(self, message: str) -> None:
        self._owner.infos.append(message)

    def caption(self, text: str) -> None:
        self._owner.captions.append(text)


class _SimpleStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.markdowns: list[dict[str, object]] = []
        self.captions: list[str] = []
        self.metrics: list[tuple[tuple[object, ...], dict[str, object]]] = []
        self.infos: list[str] = []

    def empty(self) -> _Placeholder:
        return _Placeholder(self)

    def markdown(self, body: str, *, unsafe_allow_html: bool = False) -> None:
        self.markdowns.append({"body": body, "unsafe": unsafe_allow_html})

    def caption(self, text: str) -> None:
        self.captions.append(text)

    def columns(self, count: int) -> list[_Column]:
        return [_Column(self) for _ in range(count)]

    def metric(self, *args, **kwargs) -> None:  # noqa: ANN002 - stubbed signature
        self.metrics.append((args, kwargs))

    def info(self, message: str) -> None:
        self.infos.append(message)

    def spinner(self, _message: str):  # noqa: ANN001 - minimal stub
        return _DummyContext()

    def radio(self, *_args, **_kwargs) -> int:
        return 0


@pytest.mark.parametrize("lazy_metrics", [False])
def test_table_placeholder_precedes_render(monkeypatch: pytest.MonkeyPatch, lazy_metrics: bool) -> None:
    """The table placeholder should be emitted before the heavy render occurs."""

    fake_st = _SimpleStreamlit()
    fake_st.session_state.setdefault(
        portfolio_mod._LAZY_BLOCKS_STATE_KEY,
        {
            "table": {
                "status": "loaded",
                "dataset_hash": "none",
                "triggered_at": None,
                "loaded_at": None,
                "prompt_rendered": True,
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
    monkeypatch.setattr(portfolio_mod, "st", fake_st)

    call_order: list[tuple[str, str]] = []

    def fake_mark_placeholder(label: str) -> None:
        call_order.append(("skeleton", label))

    monkeypatch.setattr(portfolio_mod.skeletons, "mark_placeholder", fake_mark_placeholder)

    def fake_update_table_data(*_, **__):
        call_order.append(("render", "table"))
        return {}

    monkeypatch.setattr(portfolio_mod, "update_table_data", fake_update_table_data)
    monkeypatch.setattr(portfolio_mod, "update_summary_section", lambda *a, **k: {})
    monkeypatch.setattr(portfolio_mod, "update_charts", lambda *a, **k: {})

    viewmodel = SimpleNamespace(
        controls=Controls(),
        metrics=SimpleNamespace(ccl_rate=None),
        positions=pd.DataFrame({"simbolo": ["ALUA"], "valor_actual": [100.0]}),
        totals=None,
        historical_total=None,
        contributions=None,
        pending_metrics=(),
    )

    portfolio_mod.render_basic_tab(
        viewmodel,
        FavoriteSymbols(),
        snapshot=None,
        tab_slug="portafolio",
        tab_cache={},
        timings={},
        lazy_metrics=lazy_metrics,
    )

    assert call_order[0] == ("skeleton", "table")
    assert ("render", "table") in call_order

    placeholder_bodies = [
        entry["body"]
        for entry in fake_st.markdowns
        if entry.get("placeholder") and entry.get("write")
    ]
    assert any("⏳ Cargando tabla" in body for body in placeholder_bodies)
