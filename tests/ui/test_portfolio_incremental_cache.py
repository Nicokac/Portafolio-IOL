import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pandas as pd

from domain.models import Controls

if "application.benchmark_service" not in sys.modules:
    fake_benchmark = ModuleType("application.benchmark_service")
    fake_benchmark.benchmark_analysis = lambda *args, **kwargs: None
    sys.modules["application.benchmark_service"] = fake_benchmark

if "controllers.portfolio.risk" not in sys.modules:
    fake_risk = ModuleType("controllers.portfolio.risk")
    fake_risk.compute_risk_metrics = lambda *args, **kwargs: None
    fake_risk.render_risk_analysis = lambda *args, **kwargs: None
    sys.modules["controllers.portfolio.risk"] = fake_risk

if "controllers.portfolio.fundamentals" not in sys.modules:
    fake_fundamentals = ModuleType("controllers.portfolio.fundamentals")
    fake_fundamentals.render_fundamental_analysis = lambda *args, **kwargs: None
    sys.modules["controllers.portfolio.fundamentals"] = fake_fundamentals

if "fastapi" not in sys.modules:
    fake_fastapi = ModuleType("fastapi")
    fake_fastapi.Depends = lambda *args, **kwargs: None
    fake_fastapi.HTTPException = Exception
    fake_fastapi.status = SimpleNamespace(HTTP_401_UNAUTHORIZED=401)
    sys.modules["fastapi"] = fake_fastapi
    for name in [
        "fastapi.routing",
        "fastapi.params",
        "fastapi.temp_pydantic_v1_params",
        "fastapi.openapi",
        "fastapi.openapi.models",
    ]:
        sys.modules[name] = ModuleType(name)

if "streamlit" not in sys.modules:
    fake_streamlit = ModuleType("streamlit")
    fake_streamlit.session_state = {}

    def _identity_decorator(*_args, **_kwargs):
        def _wrap(fn):
            return fn

        return _wrap

    fake_streamlit.cache_data = _identity_decorator
    fake_streamlit.cache_resource = _identity_decorator
    fake_streamlit.empty = lambda: SimpleNamespace(container=lambda: nullcontext(), empty=lambda: None)
    fake_streamlit.spinner = lambda *_a, **_k: nullcontext()
    fake_streamlit.radio = lambda *a, **k: 0
    fake_streamlit.selectbox = lambda *a, **k: None
    fake_streamlit.columns = lambda *a, **k: []
    fake_streamlit.sidebar = SimpleNamespace()
    fake_streamlit.subheader = lambda *a, **k: None
    fake_streamlit.warning = lambda *a, **k: None
    fake_streamlit.info = lambda *a, **k: None
    fake_streamlit.caption = lambda *a, **k: None
    fake_streamlit.button = lambda *a, **k: False
    fake_streamlit.markdown = lambda *a, **k: None
    sys.modules["streamlit"] = fake_streamlit

from controllers.portfolio import portfolio


class _Placeholder:
    def empty(self):
        return None

    def container(self):
        return nullcontext()


def _make_viewmodel(order_by: str = "valor_actual", df: pd.DataFrame | None = None) -> SimpleNamespace:
    controls = Controls(
        refresh_secs=30,
        hide_cash=False,
        show_usd=False,
        order_by=order_by,
        desc=True,
        top_n=5,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
    )
    metrics = SimpleNamespace(ccl_rate=100.0)
    positions = df if df is not None else pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1000.0]})
    return SimpleNamespace(
        positions=positions,
        controls=controls,
        metrics=metrics,
        totals=SimpleNamespace(),
        historical_total=pd.DataFrame(),
        contributions=SimpleNamespace(by_symbol=pd.DataFrame(), by_type=pd.DataFrame()),
        snapshot_id=None,
    )


def test_table_filters_only_rerender_table(monkeypatch):
    portfolio._INCREMENTAL_CACHE.clear()

    summary_calls = []
    table_calls = []
    chart_calls = []

    def _summary_stub(*args, **kwargs):
        summary_calls.append((args, kwargs))
        return True

    def _table_stub(*args, **kwargs):
        table_calls.append((args, kwargs))

    def _charts_stub(*args, **kwargs):
        chart_calls.append((args, kwargs))

    monkeypatch.setattr(
        portfolio,
        "st",
        SimpleNamespace(empty=lambda: _Placeholder(), caption=lambda *_: None),
    )
    monkeypatch.setattr(portfolio, "render_summary_section", _summary_stub)
    monkeypatch.setattr(portfolio, "render_table_section", _table_stub)
    monkeypatch.setattr(portfolio, "render_charts_section", _charts_stub)
    monkeypatch.setattr(portfolio, "_render_updated_caption", lambda *_: None)

    favorites = SimpleNamespace(list=lambda: [])
    snapshot = SimpleNamespace(storage_id=None)
    tab_cache: dict[str, dict] = {}

    viewmodel = _make_viewmodel()
    portfolio.render_basic_tab(viewmodel, favorites, snapshot, tab_slug="portafolio", tab_cache=tab_cache)

    assert len(summary_calls) == 1
    assert len(table_calls) == 1
    assert len(chart_calls) == 1

    portfolio.render_basic_tab(viewmodel, favorites, snapshot, tab_slug="portafolio", tab_cache=tab_cache)

    assert len(summary_calls) == 1
    assert len(table_calls) == 1
    assert len(chart_calls) == 1

    viewmodel_controls = _make_viewmodel(order_by="pl")
    portfolio.render_basic_tab(
        viewmodel_controls,
        favorites,
        snapshot,
        tab_slug="portafolio",
        tab_cache=tab_cache,
    )

    assert len(summary_calls) == 1
    assert len(table_calls) == 2
    assert len(chart_calls) == 1

    updated_df = pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1500.0]})
    viewmodel_updated = _make_viewmodel(order_by="pl", df=updated_df)
    portfolio.render_basic_tab(
        viewmodel_updated,
        favorites,
        snapshot,
        tab_slug="portafolio",
        tab_cache=tab_cache,
    )

    assert len(summary_calls) == 2
    assert len(table_calls) == 3
    assert len(chart_calls) == 2


def test_should_reset_rendered_flag_requires_matching_dataset() -> None:
    """Only reset the render flag when the dataset token matches the entry."""

    assert portfolio._should_reset_rendered_flag("dataset-a", "dataset-a", "pending") is True
    assert portfolio._should_reset_rendered_flag("dataset-a", "dataset-b", "pending") is False
    assert portfolio._should_reset_rendered_flag(None, "dataset-a", "pending") is False
    assert portfolio._should_reset_rendered_flag("dataset-a", "dataset-a", "loaded") is False
