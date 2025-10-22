import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from application.portfolio_service import PortfolioTotals
from controllers.portfolio.portfolio import render_portfolio_section
from services.notifications import NotificationFlags
from services.portfolio_view import (
    PortfolioContributionMetrics,
    PortfolioViewSnapshot,
)
from tests.ui.test_portfolio_ui import FakeStreamlit, _DummyContainer

# Stub heavy optional dependencies used by portfolio controllers during import time.
statsmodels_mod = types.ModuleType("statsmodels")
statsmodels_api = types.ModuleType("statsmodels.api")
statsmodels_mod.api = statsmodels_api
sys.modules.setdefault("statsmodels", statsmodels_mod)
sys.modules.setdefault("statsmodels.api", statsmodels_api)

scipy_mod = types.ModuleType("scipy")
scipy_stats = types.ModuleType("scipy.stats")
scipy_sparse = types.ModuleType("scipy.sparse")
scipy_stats_qmc = types.ModuleType("scipy.stats._qmc")
scipy_stats_multicomp = types.ModuleType("scipy.stats._multicomp")
scipy_sparse_csgraph = types.ModuleType("scipy.sparse.csgraph")
scipy_sparse_shortest = types.ModuleType("scipy.sparse.csgraph._shortest_path")
scipy_mod.stats = scipy_stats
scipy_mod.sparse = scipy_sparse
scipy_sparse.csgraph = scipy_sparse_csgraph  # type: ignore[attr-defined]
sys.modules.setdefault("scipy", scipy_mod)
sys.modules.setdefault("scipy.stats", scipy_stats)
sys.modules.setdefault("scipy.stats._qmc", scipy_stats_qmc)
sys.modules.setdefault("scipy.stats._multicomp", scipy_stats_multicomp)
sys.modules.setdefault("scipy.sparse", scipy_sparse)
sys.modules.setdefault("scipy.sparse.csgraph", scipy_sparse_csgraph)
sys.modules.setdefault("scipy.sparse.csgraph._shortest_path", scipy_sparse_shortest)


@pytest.fixture
def _portfolio_setup(monkeypatch: pytest.MonkeyPatch):
    import controllers.portfolio.portfolio as portfolio_mod

    def _configure(fake_st: FakeStreamlit):
        portfolio_mod.st = fake_st
        portfolio_mod.reset_portfolio_services()
        portfolio_mod._INCREMENTAL_CACHE.clear()

        def _favorite_stub(*_args, **_kwargs):
            return None

        monkeypatch.setattr(portfolio_mod, "render_favorite_badges", _favorite_stub)
        monkeypatch.setattr(portfolio_mod, "render_favorite_toggle", _favorite_stub)

        df_positions = pd.DataFrame(
            {
                "simbolo": ["GGAL"],
                "mercado": ["bcba"],
                "cantidad": [10],
                "costo_unitario": [100.0],
            }
        )
        monkeypatch.setattr(
            portfolio_mod,
            "load_portfolio_data",
            lambda cli, svc: (df_positions.copy(), ["GGAL"], ["ACCION"]),
        )

        controls = types.SimpleNamespace(
            refresh_secs=30,
            hide_cash=True,
            show_usd=False,
            order_by="valor_actual",
            desc=True,
            top_n=10,
            selected_syms=["GGAL"],
            selected_types=["ACCION"],
            symbol_query="",
        )
        monkeypatch.setattr(portfolio_mod, "render_sidebar", lambda *a, **k: controls)

        df_view = pd.DataFrame({"simbolo": ["GGAL"], "valor_actual": [1200.0]})

        def _fake_snapshot(*_args, **_kwargs):
            return PortfolioViewSnapshot(
                df_view=df_view,
                totals=PortfolioTotals(0.0, 0.0, 0.0, float("nan"), 0.0),
                apply_elapsed=0.0,
                totals_elapsed=0.0,
                generated_at=0.0,
                historical_total=pd.DataFrame(),
                contribution_metrics=PortfolioContributionMetrics.empty(),
            )

        view_model_service = types.SimpleNamespace(get_portfolio_view=lambda *a, **k: _fake_snapshot())

        summary = MagicMock(return_value=True)
        table = MagicMock()
        charts = MagicMock()

        monkeypatch.setattr(portfolio_mod, "render_summary_section", summary)
        monkeypatch.setattr(portfolio_mod, "render_table_section", table)
        monkeypatch.setattr(portfolio_mod, "render_charts_section", charts)

        notifications = types.SimpleNamespace(get_flags=lambda: NotificationFlags())

        return portfolio_mod, summary, table, charts, view_model_service, notifications

    return _configure


def test_visual_sections_are_cached(monkeypatch: pytest.MonkeyPatch, _portfolio_setup) -> None:
    fake_st = FakeStreamlit(
        radio_sequence=[0, 0],
        button_clicks={
            "portafolio_load_table": [True, False],
            "portafolio_load_charts": [True, False],
        },
    )
    telemetry_calls: list[dict[str, object]] = []

    def _log_telemetry_stub(*_args, **kwargs):
        telemetry_calls.append(kwargs)

    monkeypatch.setattr("controllers.portfolio.portfolio.log_telemetry", _log_telemetry_stub)

    (
        portfolio_mod,
        summary,
        table,
        charts,
        view_model_service,
        notifications_service,
    ) = _portfolio_setup(fake_st)

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    assert summary.call_count == 1
    assert table.call_count == 1
    assert charts.call_count == 1

    render_portfolio_section(
        _DummyContainer(),
        cli=object(),
        fx_rates={},
        view_model_service_factory=lambda: view_model_service,
        notifications_service_factory=lambda: notifications_service,
    )

    assert summary.call_count == 1, "El resumen no debe re-renderizarse si el dataset no cambia"
    assert table.call_count == 1, "La tabla no debe re-renderizarse si el dataset no cambia"
    assert charts.call_count == 1, "Los gráficos no deben re-renderizarse si el dataset no cambia"

    dataset_hash = fake_st.session_state.get("dataset_hash")
    cache = fake_st.session_state.get("cached_render")
    assert isinstance(cache, dict)
    assert str(dataset_hash or "none") in cache
    assert fake_st.session_state.get("__portfolio_visual_cache_reused__") is True

    reuse_flags = [
        entry["extra"].get("reused_visual_cache")
        for entry in telemetry_calls
        if entry.get("phase") == "portfolio.visual_cache" and isinstance(entry.get("extra"), dict)
    ]
    assert reuse_flags and reuse_flags[-1] is True
