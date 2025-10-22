from __future__ import annotations

import sys
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pandas as pd
import pytest

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
    fake_streamlit.cache_data = lambda *a, **k: (lambda fn: fn)
    fake_streamlit.cache_resource = lambda *a, **k: (lambda fn: fn)
    fake_streamlit.empty = lambda: SimpleNamespace(container=lambda: nullcontext(), empty=lambda: None)

    class _SidebarStub:
        def __enter__(self) -> "_SidebarStub":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
            return None

        def markdown(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
            return None

        def caption(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
            return None

        def container(self) -> "_SidebarStub":
            return _SidebarStub()

        def form(self, *args, **kwargs) -> "_SidebarStub":  # noqa: ANN002, ANN003
            return _SidebarStub()

        def slider(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
            if "value" in kwargs:
                return kwargs["value"]
            return args[2] if len(args) > 2 else 0

        def checkbox(self, *args, value=False, **kwargs):  # noqa: ANN002, ANN003
            return value

        def text_input(self, *args, value="", **kwargs):  # noqa: ANN002, ANN003
            return value

        def multiselect(self, label, options, default=None, **kwargs):
            if default is None:
                return list(options)
            if isinstance(default, list):
                return list(default)
            return list(options)

        def button(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
            return False

        def empty(self):  # noqa: ANN002, ANN003 - stub signature
            return _Placeholder()

        def radio(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
            return 0

    fake_streamlit.spinner = lambda *_a, **_k: nullcontext()
    fake_streamlit.radio = lambda *a, **k: 0
    fake_streamlit.selectbox = lambda *a, **k: None
    fake_streamlit.columns = lambda *a, **k: []
    fake_streamlit.sidebar = _SidebarStub()
    fake_streamlit.subheader = lambda *a, **k: None
    fake_streamlit.warning = lambda *a, **k: None
    fake_streamlit.info = lambda *a, **k: None
    fake_streamlit.caption = lambda *a, **k: None
    fake_streamlit.button = lambda *a, **k: False
    fake_streamlit.markdown = lambda *a, **k: None
    sys.modules["streamlit"] = fake_streamlit

from controllers.portfolio import portfolio
from domain.models import Controls
from services import portfolio_view


class _Placeholder:
    def __init__(self) -> None:
        self.messages: list[str] = []

    def empty(self) -> None:
        return None

    def container(self):  # noqa: ANN001 - mimic Streamlit container
        return nullcontext()

    def write(self, body: str) -> None:
        self.messages.append(body)

    def button(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return False


class _FakeStreamlit:
    def __init__(self) -> None:
        self.session_state: dict[str, object] = {}
        self.sidebar = _DummySidebar()

    def empty(self) -> _Placeholder:
        return _Placeholder()

    def form(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return _DummySidebar()

    def experimental_rerun(self) -> None:
        return None

    def spinner(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return nullcontext()

    def radio(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return 0

    def button(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return False

    def caption(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def info(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def warning(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def markdown(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def metric(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def success(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def error(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def write(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def toast(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None


class _DummySidebar:
    def __enter__(self) -> "_DummySidebar":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        return None

    def markdown(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def caption(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return None

    def container(self) -> "_DummySidebar":
        return _DummySidebar()

    def form(self, *args, **kwargs) -> "_DummySidebar":  # noqa: ANN002, ANN003
        return _DummySidebar()

    def slider(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        if "value" in kwargs:
            return kwargs["value"]
        return args[2] if len(args) > 2 else 0

    def checkbox(self, *args, value=False, **kwargs):  # noqa: ANN002, ANN003
        return value

    def text_input(self, *args, value="", **kwargs):  # noqa: ANN002, ANN003
        return value

    def multiselect(self, label, options, default=None, **kwargs):
        if default is None:
            return list(options)
        if isinstance(default, list):
            return list(default)
        return list(options)

    def button(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return False

    def empty(self) -> _Placeholder:
        return _Placeholder()

    def radio(self, *args, **kwargs):  # noqa: ANN002, ANN003 - stub signature
        return 0


class _Container:
    def __enter__(self) -> "_Container":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        return None

    def empty(self) -> _Placeholder:
        return _Placeholder()


class _DummyFragment:
    def __enter__(self) -> "_DummyFragment":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: D401 - standard signature
        return None

    def stop(self) -> None:
        return None


def _make_viewmodel(df: pd.DataFrame) -> SimpleNamespace:
    controls = Controls(
        refresh_secs=30,
        hide_cash=False,
        show_usd=False,
        order_by="valor_actual",
        desc=True,
        top_n=5,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
    )
    metrics = SimpleNamespace(ccl_rate=None)
    contributions = SimpleNamespace(by_symbol=pd.DataFrame(), by_type=pd.DataFrame())
    return SimpleNamespace(
        positions=df,
        controls=controls,
        metrics=metrics,
        totals=SimpleNamespace(),
        historical_total=pd.DataFrame(),
        contributions=contributions,
        pending_metrics=(),
        tab_options=("📂 Portafolio",),
    )


def test_soft_refresh_guard_preserves_table_fragment(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    df_pos = pd.DataFrame(
        {
            "simbolo": ["GGAL"],
            "mercado": ["BCBA"],
            "valor_actual": [100.0],
            "costo": [80.0],
        }
    )

    def fake_apply_filters(
        df_pos,
        controls,
        cli,
        psvc,
        *,
        dataset_hash=None,
        skip_invalidation=False,
    ):
        df = df_pos.copy()
        df["pl"] = df["valor_actual"] - df["costo"]
        df["pl_d"] = df["pl"]
        df["pl_pct"] = 10.0
        df["tipo"] = "Acción"
        return df

    monkeypatch.setattr(portfolio_view, "_apply_filters", fake_apply_filters)
    monkeypatch.setattr(
        portfolio_view,
        "calculate_totals",
        lambda df: portfolio_view.PortfolioTotals(
            float(df["valor_actual"].sum()),
            float(df["costo"].sum()),
            float((df["valor_actual"] - df["costo"]).sum()),
            0.0,
            0.0,
        ),
    )
    monkeypatch.setattr(
        portfolio_view,
        "_compute_contribution_metrics",
        lambda df: portfolio_view.PortfolioContributionMetrics.empty(),
    )

    invalidations: list[str | None] = []
    real_invalidate = portfolio_view.PortfolioViewModelService.invalidate_positions

    def tracking_invalidate(self, dataset_key: str | None = None):
        invalidations.append(dataset_key)
        return real_invalidate(self, dataset_key)

    monkeypatch.setattr(
        portfolio_view.PortfolioViewModelService,
        "invalidate_positions",
        tracking_invalidate,
    )

    service = portfolio_view.PortfolioViewModelService()
    controls_obj = Controls(
        refresh_secs=30,
        hide_cash=False,
        show_usd=False,
        order_by="valor_actual",
        desc=True,
        top_n=5,
        selected_syms=[],
        selected_types=[],
        symbol_query="",
    )

    snapshot_initial = service.get_portfolio_view(
        df_pos,
        controls_obj,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
        dataset_hash="hash-1",
    )
    snapshot_soft = service.get_portfolio_view(
        df_pos,
        controls_obj,
        cli=SimpleNamespace(),
        psvc=SimpleNamespace(),
        dataset_hash="hash-1",
        skip_invalidation=True,
    )

    assert len(invalidations) == 1
    assert snapshot_soft.soft_refresh_guard is True
    assert "[PortfolioView] Skipped early invalidate (dataset stable)" in caplog.text
    assert 'portfolio_view.skip_invalidation_guarded event="skip_invalidation_guarded"' in caplog.text

    fake_st = _FakeStreamlit()
    monkeypatch.setattr(portfolio, "st", fake_st)
    monkeypatch.setattr(portfolio, "_record_stage", lambda *a, **k: nullcontext())
    monkeypatch.setattr(portfolio, "_get_component_metadata", lambda *a, **k: None)
    monkeypatch.setattr(portfolio, "_store_component_metadata", lambda *a, **k: None)
    monkeypatch.setattr(portfolio, "_log_quotes_refresh_event", lambda *a, **k: None)
    monkeypatch.setattr(portfolio, "render_sidebar", lambda *a, **k: controls_obj)

    def fake_prompt_lazy_block(block, *a, **k):  # noqa: ANN001
        block["status"] = "loaded"
        return True

    monkeypatch.setattr(portfolio, "_prompt_lazy_block", fake_prompt_lazy_block)
    monkeypatch.setattr(portfolio, "table_fragment", lambda **_: _DummyFragment())
    monkeypatch.setattr(portfolio, "charts_fragment", lambda **_: _DummyFragment())
    monkeypatch.setattr(portfolio, "current_scope", lambda: "global")
    monkeypatch.setattr(portfolio, "current_component", lambda: "table")
    monkeypatch.setattr(portfolio, "log_user_action", lambda *a, **k: None)
    monkeypatch.setattr(
        portfolio,
        "visual_cache_registry",
        SimpleNamespace(record=lambda *a, **k: None, invalidate_dataset=lambda *a, **k: None),
    )
    monkeypatch.setattr(portfolio, "update_summary_section", lambda *a, **k: {"has_positions": True})
    monkeypatch.setattr(portfolio, "render_summary_section", lambda *a, **k: None)
    monkeypatch.setattr(portfolio, "update_table_data", lambda *a, **k: {"has_positions": True})
    monkeypatch.setattr(portfolio, "render_table_section", lambda *a, **k: None)
    monkeypatch.setattr(portfolio, "update_charts", lambda *a, **k: {})
    monkeypatch.setattr(portfolio, "render_charts_section", lambda *a, **k: None)

    skeleton_calls: list[str] = []

    def track_skeleton(label: str, *, placeholder=None):  # noqa: ANN001
        skeleton_calls.append(label)
        return placeholder

    monkeypatch.setattr(portfolio.skeletons, "mark_placeholder", track_skeleton)

    viewmodel_initial = _make_viewmodel(snapshot_initial.df_view)
    favorites = SimpleNamespace(list=lambda: [])
    tab_cache: dict[str, dict] = {}

    portfolio.render_basic_tab(
        viewmodel_initial,
        favorites,
        snapshot_initial,
        tab_slug="portafolio",
        tab_cache=tab_cache,
    )

    initial_skeleton_count = skeleton_calls.count("table")
    dataset_token = snapshot_soft.dataset_hash or "hash-1"
    component_store = tab_cache.setdefault("components", {})
    table_entry = component_store["table"]
    table_entry["skeleton_displayed"] = False
    lazy_blocks = portfolio._ensure_lazy_blocks(dataset_token)
    lazy_blocks["table"]["status"] = "pending"

    portfolio.render_basic_tab(
        _make_viewmodel(snapshot_soft.df_view),
        favorites,
        snapshot_soft,
        tab_slug="portafolio",
        tab_cache=tab_cache,
    )

    assert skeleton_calls.count("table") == initial_skeleton_count
    assert component_store["table"].get("rendered") is True

    fake_st.session_state["_dataset_skip_invalidation"] = True
    monkeypatch.setattr(portfolio, "_maybe_reset_visual_cache_state", lambda: False)
    monkeypatch.setattr(portfolio, "get_portfolio_service", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(portfolio, "get_ta_service", lambda *a, **k: SimpleNamespace())
    monkeypatch.setattr(portfolio, "get_portfolio_view_service", lambda factory=None: service)
    monkeypatch.setattr(
        portfolio,
        "get_notifications_service",
        lambda factory=None: SimpleNamespace(get_flags=lambda: {}),
    )
    monkeypatch.setattr(
        portfolio,
        "load_portfolio_data",
        lambda cli, psvc: (df_pos, ["GGAL"], ["Acción"]),
    )
    monkeypatch.setattr(portfolio, "_apply_tab_badges", lambda labels, flags: labels)
    monkeypatch.setattr(portfolio, "_get_cached_favorites", lambda: favorites)
    monkeypatch.setattr(
        portfolio,
        "build_portfolio_viewmodel",
        lambda snapshot, controls, fx_rates, all_symbols: _make_viewmodel(snapshot.df_view),
    )
    monkeypatch.setattr(
        portfolio,
        "fragment_state_soft_refresh",
        lambda dataset_hash=None: fake_st.session_state.setdefault("_fragment_soft_refresh", dataset_hash),
    )
    monkeypatch.setattr(
        portfolio,
        "snapshot_service",
        SimpleNamespace(
            is_null_backend=lambda: False,
            current_backend_name=lambda: "test",
        ),
    )

    refresh_secs = portfolio.render_portfolio_section(
        _Container(),
        cli=SimpleNamespace(),
        fx_rates={},
        view_model_service_factory=lambda: service,
        notifications_service_factory=lambda: SimpleNamespace(get_flags=lambda: {}),
        timings={},
        lazy_metrics=False,
    )

    assert refresh_secs == controls_obj.refresh_secs
    assert fake_st.session_state.get("_soft_refresh_applied") is True
    assert fake_st.session_state.get("_fragment_soft_refresh") == snapshot_soft.dataset_hash
    assert fake_st.session_state.pop("_dataset_skip_invalidation", False) is False
    assert len(invalidations) == 1
    assert "[Guardian] Prevented pre-render cache invalidation" in caplog.text
    assert 'portfolio_view.skip_invalidation_guarded event="skip_invalidation_guarded"' in caplog.text
    assert "🧩 Skeleton render called for table" not in caplog.text
